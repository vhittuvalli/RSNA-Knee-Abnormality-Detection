from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from dataset import KneeMRICache, KneeMRIDataset, build_splits, TARGETS
from model import KneeMRIModel

DEFAULT_PREPROCESSED_DIR = "preprocessed"
DEFAULT_LABELS_CSV = "report_labels_v1_regex.csv"
OUT_DIR = Path("training_output")
TIME_BUDGET_S = 8.5 * 3600
BATCH_SIZE = 16
NUM_WORKERS = 4
BACKBONE_LR = 1e-05
HEAD_LR = 0.001
WEIGHT_DECAY = 0.0001
MAX_EPOCHS = 40
PATIENCE = 10
WARMUP_EPOCHS = 2
GRAD_CLIP = 5.0


def warn_if_interactive():
    run_type = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "")
    if run_type and run_type.lower() != "batch":
        print(
            f"[WARNING] KAGGLE_KERNEL_RUN_TYPE={run_type!r} - this looks like an interactive Draft Session. /kaggle/working here is EPHEMERAL and will not persist. Use 'Save Version -> Save & Run All (Commit)' to get a run whose output actually survives.",
            flush=True,
        )


def collate_labeled(batch):
    uids, slots, mask, labels, weight = zip(*batch)
    return (
        list(uids),
        torch.stack(slots),
        torch.stack(mask),
        torch.stack(labels),
        torch.stack(weight),
    )


def make_loader(cache, uids, labels_df, batch_size, shuffle, augment=False):
    ds = KneeMRIDataset(cache, uids, labels_df, augment=augment)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_labeled,
        drop_last=False,
    )


def compute_pos_weight(labels_df, uids):
    sub = labels_df.set_index("StudyInstanceUID").loc[uids]
    pos = sub[TARGETS].sum(axis=0).values.astype(np.float32)
    neg = len(uids) - pos
    pos = np.clip(pos, 1, None)
    return torch.from_numpy(neg / pos)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = ([], [])
    for uids, slots, mask, labels, _ in loader:
        slots, mask = (slots.to(device), mask.to(device))
        logits = model(slots, mask)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())
    if not all_logits:
        return (float("nan"), {})
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    per_label_auc = {}
    for i, t in enumerate(TARGETS):
        y_hard = (labels[:, i] >= 0.5).astype(int)
        if len(np.unique(y_hard)) < 2:
            continue
        try:
            per_label_auc[t] = roc_auc_score(y_hard, logits[:, i])
        except ValueError:
            continue
    mean_auc = (
        float(np.mean(list(per_label_auc.values()))) if per_label_auc else float("nan")
    )
    return (mean_auc, per_label_auc)


def save_checkpoint(path, model, optimizer, scaler, epoch, best_auc, rng_state):
    tmp = Path(str(path) + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_auc": best_auc,
            "torch_rng_state": rng_state,
        },
        tmp,
    )
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    torch.set_rng_state(ckpt["torch_rng_state"])
    return (ckpt["epoch"], ckpt["best_auc"])


def aggregate_oof(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files matched {pattern!r}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    dup = df["StudyInstanceUID"].duplicated()
    if dup.any():
        raise ValueError(
            f"{dup.sum()} StudyInstanceUID(s) appear in more than one fold's held-out set - folds should partition the gold studies exactly once each. Check --gold-folds/--gold-fold matched across all runs that produced files matching {pattern!r}."
        )
    print(
        f"[INFO] pooled {len(df)} gold studies from {len(paths)} fold file(s)",
        flush=True,
    )
    per_label_auc = {}
    for t in TARGETS:
        y = df[f"true_{t}"].values
        p = df[f"pred_{t}"].values
        if len(np.unique(y)) < 2:
            continue
        per_label_auc[t] = roc_auc_score(y, p)
    mean_auc = (
        float(np.mean(list(per_label_auc.values()))) if per_label_auc else float("nan")
    )
    print(f"[AGGREGATE OOF GOLD] mean_auc={mean_auc:.4f}", flush=True)
    for t, a in per_label_auc.items():
        print(f"    {t}: {a:.4f}", flush=True)
    return (mean_auc, per_label_auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocessed-dir", default=DEFAULT_PREPROCESSED_DIR)
    ap.add_argument("--labels-csv", default=DEFAULT_LABELS_CSV)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--time-budget-s", type=float, default=TIME_BUDGET_S)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--pretrained", action="store_true", default=True)
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument(
        "--backbone",
        choices=["resnet50", "radimagenet_resnet50", "dinov2"],
        default="resnet50",
    )
    ap.add_argument(
        "--dinov2-variant", choices=["small", "base", "large"], default="small"
    )
    ap.add_argument("--dinov2-source", default=None)
    ap.add_argument("--dinov2-unfreeze-last", type=int, default=2)
    ap.add_argument(
        "--radimagenet-weights",
        default=None,
        help="path to RadImageNet-ResNet50_notop.onnx (converted via tf2onnx + onnx2torch), required when --backbone radimagenet_resnet50",
    )
    ap.add_argument("--radimagenet-unfreeze-last-frac", type=float, default=0.2)
    ap.add_argument("--gold-folds", type=int, default=5)
    ap.add_argument("--gold-fold", type=int, default=0)
    ap.add_argument("--aggregate-oof", default=None)
    args = ap.parse_args()
    if args.aggregate_oof:
        aggregate_oof(args.aggregate_oof)
        return
    warn_if_interactive()
    start_time = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}", flush=True)
    train_uids, val_gold_uids, val_silver_uids, labels_df = build_splits(
        args.preprocessed_dir,
        args.labels_csv,
        args.train_csv,
        gold_folds=args.gold_folds,
        gold_fold=args.gold_fold,
    )
    print(
        f"[INFO] gold_fold={args.gold_fold}/{args.gold_folds} train={len(train_uids)} val_gold={len(val_gold_uids)} val_silver={len(val_silver_uids)}",
        flush=True,
    )
    if len(val_gold_uids) == 0:
        print(
            "[WARNING] no gold studies available for validation - checkpoint selection will fall back to silver AUC, which is a noisier signal since silver labels are regex-derived, not radiologist-derived.",
            flush=True,
        )
    cache = KneeMRICache(args.preprocessed_dir, "train")
    train_loader = make_loader(
        cache, train_uids, labels_df, args.batch_size, shuffle=True, augment=True
    )
    val_gold_loader = (
        make_loader(cache, val_gold_uids, labels_df, args.batch_size, shuffle=False)
        if val_gold_uids
        else None
    )
    val_silver_loader = (
        make_loader(cache, val_silver_uids, labels_df, args.batch_size, shuffle=False)
        if val_silver_uids
        else None
    )
    pos_weight = compute_pos_weight(labels_df, train_uids).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    backbone_kwargs = {}
    if args.backbone == "dinov2":
        backbone_kwargs = {
            "variant": args.dinov2_variant,
            "unfreeze_last": args.dinov2_unfreeze_last,
            "source": args.dinov2_source,
        }
    elif args.backbone == "radimagenet_resnet50":
        backbone_kwargs = {
            "radimagenet_weights": args.radimagenet_weights,
            "unfreeze_last_frac": args.radimagenet_unfreeze_last_frac,
        }
    model = KneeMRIModel(
        n_labels=len(TARGETS),
        backbone=args.backbone,
        pretrained=args.pretrained,
        train_img_size=args.img_size,
        backbone_kwargs=backbone_kwargs,
    ).to(device)
    pretrained_submodule = (
        model.encoder.body
        if args.backbone in ("resnet50", "radimagenet_resnet50")
        else model.encoder.backbone
    )
    backbone_params = [p for p in pretrained_submodule.parameters() if p.requires_grad]
    head_params = list(model.encoder.proj.parameters()) + list(
        model.classifier.parameters()
    )
    n_backbone_trainable = sum((p.numel() for p in backbone_params))
    n_head_trainable = sum((p.numel() for p in head_params))
    print(
        f"[INFO] backbone={args.backbone} backbone_trainable_params={n_backbone_trainable / 1000000.0:.1f}M head_trainable_params={n_head_trainable / 1000000.0:.1f}M",
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": BACKBONE_LR},
            {"params": head_params, "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.max_epochs - WARMUP_EPOCHS)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    start_epoch = 0
    best_auc = -1.0
    ckpt_path = out_dir / "last.pt"
    best_path = out_dir / "best.pt"
    if args.resume and ckpt_path.is_file():
        start_epoch, best_auc = load_checkpoint(
            ckpt_path, model, optimizer, scaler, device
        )
        start_epoch += 1
        print(
            f"[INFO] resumed from epoch {start_epoch}, best_auc so far={best_auc:.4f}",
            flush=True,
        )
    epochs_since_improve = 0
    stopped_reason = None
    for epoch in range(start_epoch, args.max_epochs):
        if time.time() - start_time > args.time_budget_s:
            stopped_reason = "time_budget"
            print(
                f"[INFO] time budget ({args.time_budget_s}s) reached before epoch {epoch}, stopping gracefully",
                flush=True,
            )
            break
        if epoch < WARMUP_EPOCHS:
            warmup_scale = (epoch + 1) / WARMUP_EPOCHS
            for g, base_lr in zip(optimizer.param_groups, [BACKBONE_LR, HEAD_LR]):
                g["lr"] = base_lr * warmup_scale
        model.train()
        running_loss, n_batches = (0.0, 0)
        for uids, slots, mask, labels, weight in train_loader:
            slots, mask = (slots.to(device), mask.to(device))
            labels, weight = (labels.to(device), weight.to(device))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(slots, mask)
                per_elem = bce(logits, labels)
                loss = (per_elem * weight).sum() / weight.sum().clamp(min=1.0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            n_batches += 1
        if epoch >= WARMUP_EPOCHS:
            scheduler.step()
        train_loss = running_loss / max(1, n_batches)
        gold_auc, gold_per_label = (
            (float("nan"), {})
            if val_gold_loader is None
            else evaluate(model, val_gold_loader, device)
        )
        silver_auc, _ = (
            (float("nan"), {})
            if val_silver_loader is None
            else evaluate(model, val_silver_loader, device)
        )
        if val_gold_loader is not None and val_silver_loader is not None:
            selection_auc = 0.5 * gold_auc + 0.5 * silver_auc
        elif val_gold_loader is not None:
            selection_auc = gold_auc
        else:
            selection_auc = silver_auc
        elapsed = time.time() - start_time
        print(
            f"[epoch {epoch}] loss={train_loss:.4f} gold_auc={gold_auc:.4f} silver_auc={silver_auc:.4f} selection_auc={selection_auc:.4f} elapsed={elapsed / 60:.1f}min",
            flush=True,
        )
        with open(log_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "gold_auc": gold_auc,
                        "silver_auc": silver_auc,
                        "selection_auc": selection_auc,
                        "gold_per_label_auc": gold_per_label,
                        "elapsed_s": elapsed,
                    }
                )
                + "\n"
            )
        rng_state = torch.get_rng_state()
        save_checkpoint(ckpt_path, model, optimizer, scaler, epoch, best_auc, rng_state)
        if not np.isnan(selection_auc) and selection_auc > best_auc:
            best_auc = selection_auc
            epochs_since_improve = 0
            save_checkpoint(
                best_path, model, optimizer, scaler, epoch, best_auc, rng_state
            )
            print(
                f"[epoch {epoch}] new best (auc={best_auc:.4f}) -> saved {best_path}",
                flush=True,
            )
        else:
            epochs_since_improve += 1
        if epochs_since_improve >= PATIENCE:
            stopped_reason = "early_stop"
            print(f"[INFO] no improvement for {PATIENCE} epochs, stopping", flush=True)
            break
    else:
        stopped_reason = "max_epochs"
    if not best_path.is_file():
        raise RuntimeError(
            "training finished without ever saving best.pt - this means validation AUC was NaN every epoch (no usable val labels), which means the run produced no usable model. Check that val_gold_uids / labels_df are non-empty before trusting any downstream inference."
        )
    if val_gold_loader is not None:
        best_ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(best_ckpt["model"])
        model.eval()
        oof_rows = []
        with torch.no_grad():
            for uids, slots, mask, labels, _ in val_gold_loader:
                slots, mask = (slots.to(device), mask.to(device))
                probs = torch.sigmoid(model(slots, mask)).cpu().numpy()
                labels_np = labels.numpy()
                for uid, p, y in zip(uids, probs, labels_np):
                    row = {"StudyInstanceUID": uid, "gold_fold": args.gold_fold}
                    for i, t in enumerate(TARGETS):
                        row[f"pred_{t}"] = float(p[i])
                        row[f"true_{t}"] = float(y[i])
                    oof_rows.append(row)
        oof_path = out_dir / f"oof_gold_fold{args.gold_fold}.csv"
        pd.DataFrame(oof_rows).to_csv(oof_path, index=False)
        print(
            f"[INFO] wrote {len(oof_rows)} held-out gold predictions -> {oof_path} (pool across folds with --aggregate-oof once every fold is done)",
            flush=True,
        )
    summary = {
        "stopped_reason": stopped_reason,
        "best_auc": best_auc,
        "final_epoch": epoch,
        "elapsed_s": time.time() - start_time,
        "gold_fold": args.gold_fold,
        "gold_folds": args.gold_folds,
    }
    (out_dir / "_TRAIN_MANIFEST.json").write_text(json.dumps(summary, indent=2))
    print(f"[DONE] {summary}", flush=True)


if __name__ == "__main__":
    main()
