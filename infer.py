from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

sys.path.insert(0, str(Path(__file__).parent))
from dataset import KneeMRICache, TARGETS, window_slots
from model import KneeMRIModel

TTA_ANGLES = (0.0, -8.0, 8.0)


def rotate_slots_batch(slots, angle):
    if angle == 0.0:
        return slots
    b, s, c, h, w = slots.shape
    flat = slots.reshape(b * s, c, h, w).float()
    rotated = TF.rotate(flat, angle, interpolation=InterpolationMode.BILINEAR)
    return rotated.reshape(b, s, c, h, w)


def load_model(
    checkpoint, backbone, dinov2_variant, img_size, device, radimagenet_weights=None
):
    if backbone == "dinov2":
        backbone_kwargs = {"variant": dinov2_variant}
    elif backbone == "radimagenet_resnet50":
        if not radimagenet_weights:
            raise ValueError(
                "--radimagenet-weights is required when --backbone radimagenet_resnet50, even at inference - the onnx2torch body's architecture is parsed from the .onnx file, not just its weights."
            )
        backbone_kwargs = {"radimagenet_weights": radimagenet_weights}
    else:
        backbone_kwargs = {}
    model = KneeMRIModel(
        n_labels=len(TARGETS),
        backbone=backbone,
        pretrained=False,
        train_img_size=img_size,
        backbone_kwargs=backbone_kwargs,
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(
        f"[INFO] loaded {checkpoint} (epoch={ckpt.get('epoch')}, best_auc={ckpt.get('best_auc')})",
        flush=True,
    )
    return model


@torch.no_grad()
def run_inference(model, cache, available, batch_size, device, tta_angles=(0.0,)):
    preds = {}
    for start in range(0, len(available), batch_size):
        batch_uids = available[start : start + batch_size]
        idxs = [cache.study_to_idx[u] for u in batch_uids]
        slots = torch.from_numpy(np.asarray(cache.cache[idxs])).to(device)
        mask = torch.from_numpy(cache.mask[idxs].copy()).to(device)
        slots, mask = window_slots(slots, mask)
        probs_sum = None
        for angle in tta_angles:
            aug_slots = rotate_slots_batch(slots, angle)
            logits = model(aug_slots, mask)
            probs = torch.sigmoid(logits)
            probs_sum = probs if probs_sum is None else probs_sum + probs
        probs = (probs_sum / len(tta_angles)).cpu().numpy()
        for u, p in zip(batch_uids, probs):
            preds[u] = p
        if start % (batch_size * 10) == 0:
            print(
                f"[INFO] inferred {start + len(batch_uids)}/{len(available)}",
                flush=True,
            )
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocessed-dir", default="preprocessed")
    ap.add_argument(
        "--checkpoint",
        nargs="+",
        default=["training_output/best.pt"],
        help="one or more best.pt paths - pass multiple (one per fold) to average their predictions (fold ensembling)",
    )
    ap.add_argument(
        "--test-studies-csv",
        default="test.csv",
        help="the competition's test.csv (single StudyInstanceUID column) or sample_submission.csv - only the StudyInstanceUID column is used",
    )
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument(
        "--backbone",
        choices=["resnet50", "radimagenet_resnet50", "dinov2"],
        default="resnet50",
    )
    ap.add_argument(
        "--dinov2-variant", choices=["small", "base", "large"], default="small"
    )
    ap.add_argument(
        "--radimagenet-weights",
        default=None,
        help="path to RadImageNet-ResNet50_notop.onnx (converted via tf2onnx + onnx2torch), required when --backbone radimagenet_resnet50 - the checkpoint's trained weight VALUES overwrite this immediately, but the .onnx file is still needed to reconstruct the model's architecture",
    )
    ap.add_argument(
        "--tta",
        action="store_true",
        help=f"average predictions over small rotations ({TTA_ANGLES}) of each slot at inference time - no retraining required, stacks on top of fold ensembling",
    )
    args = ap.parse_args()
    tta_angles = TTA_ANGLES if args.tta else (0.0,)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}", flush=True)
    all_studies = pd.read_csv(args.test_studies_csv)["StudyInstanceUID"].tolist()
    print(f"[INFO] {len(all_studies)} studies required in submission", flush=True)
    cache = KneeMRICache(args.preprocessed_dir, "test")
    available = [s for s in all_studies if s in cache.study_to_idx]
    missing = [s for s in all_studies if s not in cache.study_to_idx]
    if missing:
        print(
            f"[WARNING] {len(missing)} studies have no preprocessed test cache entry (likely DICOM decode failures) - these rows will be filled with 0.5 per label rather than a real prediction: {missing[:5]}{('...' if len(missing) > 5 else '')}",
            flush=True,
        )
    for c in args.checkpoint:
        if not Path(c).is_file():
            raise FileNotFoundError(
                f"{c} not found - train.py must complete and save best.pt for every fold you're passing in before inference can run."
            )
    print(
        f"[INFO] ensembling {len(args.checkpoint)} checkpoint(s), tta={('on ' + str(tta_angles) if args.tta else 'off')}",
        flush=True,
    )
    sum_preds = {u: np.zeros(len(TARGETS), dtype=np.float64) for u in available}
    for ckpt_path in args.checkpoint:
        model = load_model(
            ckpt_path,
            args.backbone,
            args.dinov2_variant,
            args.img_size,
            device,
            radimagenet_weights=args.radimagenet_weights,
        )
        preds = run_inference(
            model, cache, available, args.batch_size, device, tta_angles=tta_angles
        )
        for u, p in preds.items():
            sum_preds[u] += p
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    n_ckpts = len(args.checkpoint)
    preds = {u: p / n_ckpts for u, p in sum_preds.items()}
    rows = []
    for uid in all_studies:
        if uid in preds:
            row = {"StudyInstanceUID": uid}
            row.update({t: float(preds[uid][i]) for i, t in enumerate(TARGETS)})
        else:
            row = {"StudyInstanceUID": uid}
            row.update({t: 0.5 for t in TARGETS})
        rows.append(row)
    sub = pd.DataFrame(rows, columns=["StudyInstanceUID"] + TARGETS)
    if len(sub) != len(all_studies):
        raise RuntimeError(
            f"submission has {len(sub)} rows but {len(all_studies)} studies were required - refusing to write a malformed file"
        )
    if sub.isnull().any().any():
        raise RuntimeError("submission contains null values - refusing to write")
    out_path = Path(args.out)
    tmp_path = Path(str(out_path) + ".tmp")
    sub.to_csv(tmp_path, index=False)
    tmp_path.replace(out_path)
    print(
        f"[DONE] wrote {out_path} with {len(sub)} rows ({len(missing)} filled with 0.5, averaged over {n_ckpts} checkpoint(s))",
        flush=True,
    )


if __name__ == "__main__":
    main()
