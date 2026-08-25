from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

N_SLOT = 6
CACHE_IMG = 224
WINDOW_SIZE = 3
TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
SILVER_LABEL_ACCURACY = {
    "ACL": 0.948,
    "MCL": 0.966,
    "Medial Meniscus": 0.879,
    "Lateral Meniscus": 0.879,
    "Medial OA": 0.897,
    "Lateral OA": 0.879,
    "PF OA": 0.931,
    "Effusion": 0.759,
    "Synovitis": 0.69,
    "Baker's": 0.931,
    "Contusion": 0.81,
    "Fracture": 0.793,
}
GOLD_WEIGHT_MULTIPLIER = 1.0


def infer_cache_shape(preprocessed_dir, split):
    preprocessed_dir = Path(preprocessed_dir)
    order_path = preprocessed_dir / f"{split}_study_order.csv"
    cache_path = preprocessed_dir / f"{split}_cache.dat"
    if not order_path.is_file():
        raise FileNotFoundError(f"{order_path} not found")
    if not cache_path.is_file():
        raise FileNotFoundError(f"{cache_path} not found")
    n_studies = len(pd.read_csv(order_path))
    cache_bytes = cache_path.stat().st_size
    per_study = N_SLOT * CACHE_IMG * CACHE_IMG
    if n_studies == 0 or cache_bytes % (n_studies * per_study) != 0:
        raise ValueError(
            f"{split}: {cache_path.name} is {cache_bytes} bytes, which doesn't divide evenly by {n_studies} studies x {per_study} bytes/study - the cache file looks truncated or corrupted"
        )
    cache_slices = cache_bytes // (n_studies * per_study)
    return (n_studies, N_SLOT, cache_slices, CACHE_IMG, CACHE_IMG)


class KneeMRICache:

    def __init__(self, preprocessed_dir, split):
        preprocessed_dir = Path(preprocessed_dir)
        shape = infer_cache_shape(preprocessed_dir, split)
        self.cache = np.memmap(
            preprocessed_dir / f"{split}_cache.dat",
            dtype=np.uint8,
            mode="r",
            shape=shape,
        )
        self.mask = np.load(preprocessed_dir / f"{split}_mask.npy")
        self.study_order = pd.read_csv(preprocessed_dir / f"{split}_study_order.csv")[
            "StudyInstanceUID"
        ].tolist()
        self.study_to_idx = {s: i for i, s in enumerate(self.study_order)}
        if self.mask.shape[0] != len(self.study_order):
            raise ValueError(
                f"{split}: mask has {self.mask.shape[0]} rows but study_order has {len(self.study_order)} entries"
            )


def augment_slots(slots):
    S = slots.shape[0]
    for i in range(S):
        img = slots[i]
        if random.random() < 0.5:
            angle = random.uniform(-12, 12)
            img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
        if random.random() < 0.5:
            img = TF.adjust_brightness(img, random.uniform(0.85, 1.15))
        if random.random() < 0.5:
            img = TF.adjust_contrast(img, random.uniform(0.85, 1.15))
        slots[i] = img
    return slots


def window_slots(slots, mask):
    *lead, n_slot, cache_slices, h, w = slots.shape
    if cache_slices < WINDOW_SIZE:
        raise ValueError(
            f"cache has {cache_slices} depth-slice(s) per slot, need at least {WINDOW_SIZE} (one fake-RGB window) - check the preprocessing cache was built with enough slices per slot"
        )
    n_windows = cache_slices // WINDOW_SIZE
    usable = n_windows * WINDOW_SIZE
    if usable != cache_slices:
        drop = cache_slices - usable
        start = drop // 2
        slots = slots[..., start : start + usable, :, :]
    slots = slots.reshape(*lead, n_slot, n_windows, WINDOW_SIZE, h, w)
    slots = slots.reshape(*lead, n_slot * n_windows, WINDOW_SIZE, h, w)
    mask = (
        mask.unsqueeze(-1)
        .expand(*lead, n_slot, n_windows)
        .reshape(*lead, n_slot * n_windows)
    )
    return (slots, mask)


class KneeMRIDataset(Dataset):

    def __init__(self, cache: KneeMRICache, study_uids, labels_df=None, augment=False):
        self.cache = cache
        self.study_uids = list(study_uids)
        self.labels_df = None
        self.weights = None
        self.augment = augment
        if labels_df is not None:
            self.labels_df = labels_df.set_index("StudyInstanceUID")
            reg_acc = np.array(
                [SILVER_LABEL_ACCURACY[t] for t in TARGETS], dtype=np.float32
            )
            self._silver_weight = reg_acc
            self._gold_weight = np.full(
                len(TARGETS), GOLD_WEIGHT_MULTIPLIER, dtype=np.float32
            )

    def __len__(self):
        return len(self.study_uids)

    def __getitem__(self, i):
        uid = self.study_uids[i]
        idx = self.cache.study_to_idx[uid]
        slots = torch.from_numpy(np.asarray(self.cache.cache[idx])).clone()
        mask = torch.from_numpy(self.cache.mask[idx].copy())
        slots, mask = window_slots(slots, mask)
        if self.augment:
            slots = augment_slots(slots)
        if self.labels_df is None:
            return (uid, slots, mask)
        row = self.labels_df.loc[uid]
        labels = torch.tensor([float(row[t]) for t in TARGETS], dtype=torch.float32)
        is_gold = str(row.get("label_source", "")).lower() == "gold"
        weight = self._gold_weight if is_gold else self._silver_weight
        weight = torch.from_numpy(weight.copy())
        return (uid, slots, mask, labels, weight)


def build_splits(
    preprocessed_dir,
    labels_csv,
    train_csv,
    val_silver_frac=0.1,
    seed=2026,
    gold_folds=5,
    gold_fold=0,
):
    if not 0 <= gold_fold < gold_folds:
        raise ValueError(f"gold_fold={gold_fold} must be in [0, {gold_folds})")
    labels_df = pd.read_csv(labels_csv)
    cache_study_set = set(
        pd.read_csv(Path(preprocessed_dir) / "train_study_order.csv")[
            "StudyInstanceUID"
        ]
    )
    labels_df = labels_df[labels_df["StudyInstanceUID"].isin(cache_study_set)].copy()
    reports = pd.read_csv(train_csv, usecols=["StudyInstanceUID", "Report"])
    labels_df = labels_df.merge(reports, on="StudyInstanceUID", how="left")
    labels_df["Report"] = labels_df["Report"].fillna("")
    group_of = dict(zip(labels_df["StudyInstanceUID"], labels_df["Report"]))
    source_of = dict(zip(labels_df["StudyInstanceUID"], labels_df["label_source"]))
    groups = {}
    for uid, rep in group_of.items():
        groups.setdefault(rep, []).append(uid)
    gold_groups, silver_only_groups = ([], [])
    for rep, uids in groups.items():
        if any((source_of[u] == "gold" for u in uids)):
            gold_groups.append(uids)
        else:
            silver_only_groups.append(uids)
    rng = np.random.RandomState(seed)
    rng.shuffle(gold_groups)
    rng.shuffle(silver_only_groups)
    val_gold_uids = []
    for i, uids in enumerate(gold_groups):
        if i % gold_folds == gold_fold:
            val_gold_uids.extend(uids)
    n_silver_total = sum((1 for u in group_of if source_of[u] != "gold"))
    silver_target = int(round(n_silver_total * val_silver_frac))
    val_silver_uids = []
    n_silver_in_val = 0
    for uids in silver_only_groups:
        if n_silver_in_val >= silver_target:
            break
        val_silver_uids.extend(uids)
        n_silver_in_val += len(uids)
    val_uids = set(val_gold_uids) | set(val_silver_uids)
    all_uids = set(group_of.keys())
    train_uids = list(all_uids - val_uids)
    rng.shuffle(train_uids)
    return (train_uids, val_gold_uids, val_silver_uids, labels_df)
