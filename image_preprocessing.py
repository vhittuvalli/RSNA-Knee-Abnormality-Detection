from __future__ import annotations
import argparse
import gc
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


CROP_MM = 130.0
CROP_MM_BY_SLOT = {"SAG_FLUID_FS": 90.0}
CACHE_IMG = 224
CACHE_SLICES = 9
SLICE_BAND = (0.2, 0.8)
LAT_MIN_OFFSET_MM = 20.0
HDR_THREADS = 16
PIX_THREADS = 12
ORDER_THREADS = 32
ORDER_BUDGET_S = 5400
FLUSH_EVERY = 500
SLOTS = [
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
]
N_SLOT = len(SLOTS)
FATSAT_OPTS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}
_SEP = re.compile("[_\\-.]")
_FATSAT_RX = re.compile(
    "\\bfs\\b|fatsat|fat sat|\\bstir\\b|\\bspair\\b|\\bspir\\b|\\bwe\\b|water excit|\\btirm\\b|\\bsting\\b|\\bfatsup\\b"
)
_T1_RX = re.compile("\\bt1\\b|\\bt1w\\b")
_T2_RX = re.compile("\\bt2\\b|\\bt2w\\b")
_PD_RX = re.compile("\\bpd\\b|\\bpdw\\b|proton|\\bdp\\b|dens")
HDR_TAGS = [
    "SeriesDescription",
    "SequenceName",
    "ScanOptions",
    "ScanningSequence",
    "RepetitionTime",
    "EchoTime",
    "Laterality",
    "PixelSpacing",
    "Rows",
    "Columns",
    "RescaleSlope",
    "RescaleIntercept",
    "ImagePositionPatient",
    "ImageOrientationPatient",
]
ORDER_TAGS = [(32, 50), (32, 55), (32, 19)]
DECODE_FAILED = []


def find_root():
    for c in [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
        Path("data"),
        Path("."),
    ]:
        if (c / "test.csv").is_file() and (c / "test_series").is_dir():
            return c
    base = Path("/kaggle/input")
    if base.is_dir():
        for depth1 in sorted((p for p in base.iterdir() if p.is_dir())):
            for cand in [depth1] + sorted((p for p in depth1.iterdir() if p.is_dir())):
                if (cand / "test.csv").is_file():
                    return cand
    raise FileNotFoundError(
        f"competition mount not found (cwd {Path.cwd()}); expected a directory holding test.csv and test_series/"
    )


def available_gb():
    try:
        with open("/proc/meminfo") as fh:
            info = {k.strip(): v for k, v in (l.split(":", 1) for l in fh if ":" in l)}
        return int(info["MemAvailable"].split()[0]) / 1024**2
    except Exception:
        return 16.0


def plan_cache(n_study, cache_fraction=0.45, budget_max_gb=24.0):
    avail = available_gb()
    budget = min(avail * cache_fraction, budget_max_gb)
    per_slice = n_study * N_SLOT * CACHE_IMG * CACHE_IMG
    afford = int(budget * 1024**3 // max(per_slice, 1))
    slices = max(1, min(CACHE_SLICES, afford))
    log(
        f"memory: {avail:.1f} GB available, {budget:.1f} GB budgeted for the cache -> {slices} slices/slot"
        + (f" (wanted {CACHE_SLICES})" if slices < CACHE_SLICES else "")
    )
    return slices


def _hdr_vec(s, n):
    if not isinstance(s, str):
        return None
    try:
        v = [float(x) for x in s.split("|")]
    except ValueError:
        return None
    return np.array(v) if len(v) >= n else None


def probe(item):
    split, study, series, path = item
    row = {
        "split": split,
        "StudyInstanceUID": study,
        "SeriesInstanceUID": series,
        "dir": path,
    }
    try:
        files = sorted((e.name for e in os.scandir(path) if e.name.endswith(".dcm")))
        row["files"] = files
        row["n_slices"] = len(files)
        if not files:
            return row
        ds = pydicom.dcmread(
            os.path.join(path, files[len(files) // 2]),
            stop_before_pixels=True,
            force=True,
        )
        for t in HDR_TAGS:
            v = getattr(ds, t, None)
            if v is None:
                row[t] = None
            elif isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
                row[t] = "|".join((str(x) for x in v))
            else:
                row[t] = str(v)
    except Exception as exc:
        row["err"] = str(exc)[:120]
    return row


def walk(root, split):
    base = root / f"{split}_series"
    items = []
    if not base.is_dir():
        return pd.DataFrame(
            columns=[
                "split",
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "dir",
                "files",
                "n_slices",
            ]
            + HDR_TAGS
        )
    for study in os.scandir(base):
        if study.is_dir():
            for series in os.scandir(study.path):
                if series.is_dir():
                    items.append((split, study.name, series.name, series.path))
    with ThreadPoolExecutor(max_workers=HDR_THREADS) as pool:
        rows = list(pool.map(probe, items))
    return pd.DataFrame(rows)


def annotate(df):
    desc = df["SeriesDescription"].fillna("") + " " + df["SequenceName"].fillna("")
    desc = desc.str.lower().str.replace(_SEP, " ", regex=True)
    opts = df["ScanOptions"].fillna("").str.upper().str.split("|")
    opts_fs = opts.apply(lambda ts: any((t.strip() in FATSAT_OPTS for t in ts)))
    df["fatsat"] = desc.str.contains(_FATSAT_RX) | opts_fs
    tr = pd.to_numeric(df["RepetitionTime"], errors="coerce")
    te = pd.to_numeric(df["EchoTime"], errors="coerce")
    gre = df["ScanningSequence"].fillna("").str.upper().str.contains("GR")
    t1, t2, pdw = (
        desc.str.contains(_T1_RX),
        desc.str.contains(_T2_RX),
        desc.str.contains(_PD_RX),
    )
    df["weight"] = np.where(
        t1 & ~t2 & ~pdw,
        "T1",
        np.where(
            t2 & ~pdw,
            "T2",
            np.where(
                pdw,
                "PD",
                np.where(
                    gre,
                    "GRE",
                    np.where(
                        tr < 800,
                        "T1",
                        np.where(te > 60, "T2", np.where(tr >= 800, "PD", "UNK")),
                    ),
                ),
            ),
        ),
    )
    df["fluid"] = np.isin(df["weight"], ["PD", "T2"])
    df["px"] = pd.to_numeric(
        df["PixelSpacing"].fillna("").str.split("|").str[0].replace("", np.nan),
        errors="coerce",
    )
    return df


def pick_slots(series_df, plane_map):
    series_df = series_df.copy()
    series_df["plane"] = series_df["SeriesInstanceUID"].map(plane_map)
    out = {}
    for study, g in series_df.groupby("StudyInstanceUID"):
        chosen = {}
        for name, plane, fluid, fs in SLOTS:
            sel = (g["plane"] == plane) & (g["fatsat"] == fs)
            if fluid is not None:
                sel &= g["fluid"] == fluid
            cand = g[sel]
            if len(cand):
                chosen[name] = cand.sort_values("n_slices", ascending=False).iloc[0]
        out[study] = chosen
    return out


def _natural_key(name):
    return tuple(
        (int(x) if x.isdigit() else x.lower() for x in re.split("(\\d+)", str(name)))
    )


def order_slices(rec):
    files, d = (rec["files"], rec["dir"])
    keyed = []
    for f in files:
        k = None
        try:
            ds = pydicom.dcmread(
                os.path.join(d, f),
                force=True,
                stop_before_pixels=True,
                specific_tags=ORDER_TAGS,
            )
            iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
            ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
            k = float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
        except Exception:
            try:
                k = float(ds.InstanceNumber)
            except Exception:
                k = None
        keyed.append((k, f))
    if any((k is None for k, _ in keyed)):
        return (files, False)
    return ([f for _, f in sorted(keyed, key=lambda t: t[0])], True)


def side_from_geometry(h):
    cx = {}
    for r in h.itertuples(index=False):
        ipp = _hdr_vec(getattr(r, "ImagePositionPatient", None), 3)
        iop = _hdr_vec(getattr(r, "ImageOrientationPatient", None), 6)
        ps = _hdr_vec(getattr(r, "PixelSpacing", None), 2)
        rows, cols = (getattr(r, "Rows", None), getattr(r, "Columns", None))
        if ipp is None or iop is None or ps is None or (not rows) or (not cols):
            continue
        try:
            c = (
                ipp[:3]
                + iop[:3] * ps[1] * float(cols) / 2
                + iop[3:6] * ps[0] * float(rows) / 2
            )
        except (TypeError, ValueError):
            continue
        cx.setdefault(r.StudyInstanceUID, []).append(float(c[0]))
    out = {}
    for st, xs in cx.items():
        m = float(np.median(xs))
        out[st] = None if abs(m) < LAT_MIN_OFFSET_MM else "R" if m < 0 else "L"
    return out


def lat_of(h, tag=""):
    geo = side_from_geometry(h)
    d, n_tag, n_geo, n_none, n_disagree = ({}, 0, 0, 0, 0)
    for st, g in h.groupby("StudyInstanceUID"):
        v = [str(x).strip().upper() for x in g["Laterality"].dropna()]
        v = [x[0] for x in v if x and x[0] in ("L", "R")]
        side = v[0] if v else None
        if side is not None:
            n_tag += 1
            if geo.get(st) is not None and geo[st] != side:
                n_disagree += 1
        else:
            side = geo.get(st)
            n_geo += side is not None
            n_none += side is None
        d[st] = side
    log(
        f"{tag}laterality: {n_tag} from the tag, {n_geo} from geometry, {n_none} unresolved; tag and geometry disagree on {n_disagree} ({n_disagree / max(n_tag, 1):.1%} of the tagged)"
    )
    return d


def normalise_laterality(img, plane, lat):
    if lat != "R":
        return img
    if plane in ("Coronal", "Axial"):
        return torch.flip(img, dims=[-1])
    return torch.flip(img, dims=[0])


def read_slot(rec, n_slice, out_size, crop_mm=CROP_MM):
    files, d, px = (rec.get("ordered") or rec["files"], rec["dir"], rec["px"])
    n = len(files)
    if n == 0:
        return None
    lo, hi = (int(SLICE_BAND[0] * (n - 1)), int(SLICE_BAND[1] * (n - 1)))
    idx = (
        np.unique(np.linspace(lo, hi, n_slice).astype(int))
        if hi > lo
        else np.array([n // 2])
    )
    while len(idx) < n_slice:
        idx = np.append(idx, idx[-1])
    planes, errors = ([], [])
    for i in idx[:n_slice]:
        try:
            ds = pydicom.dcmread(os.path.join(d, files[int(i)]), force=True)
            a = ds.pixel_array.astype(np.float32)
            sl = float(getattr(ds, "RescaleSlope", 1) or 1)
            ic = float(getattr(ds, "RescaleIntercept", 0) or 0)
            a = a * sl + ic
        except Exception as exc:
            a = None
            errors.append(f"{type(exc).__name__}: {str(exc)[:120]}")
        planes.append(a)
    got = [k for k, p in enumerate(planes) if p is not None]
    if not got:
        DECODE_FAILED.append(
            {"series": rec.get("SeriesInstanceUID", d), "errors": errors}
        )
        return None
    if len(got) < len(planes):
        DECODE_FAILED.append(
            {"series": rec.get("SeriesInstanceUID", d), "errors": errors}
        )
        for k, p in enumerate(planes):
            if p is None:
                planes[k] = planes[min(got, key=lambda j: abs(j - k))]
    shp = planes[0].shape
    planes = [p if p.shape == shp else np.zeros(shp, np.float32) for p in planes]
    vol = np.stack(planes)
    if px and np.isfinite(px) and (px > 0):
        want = int(round(crop_mm / px))
        h, w = shp
        if 16 < want < min(h, w):
            cy, cx = (h // 2, w // 2)
            half = want // 2
            vol = vol[:, max(0, cy - half) : cy + half, max(0, cx - half) : cx + half]
    lo_v, hi_v = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo_v) / max(hi_v - lo_v, 1e-06), 0, 1)
    t = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
    t = F.interpolate(
        t, size=(out_size, out_size), mode="bilinear", align_corners=False
    )
    return (t.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8)


def _atomic_save_npy(path, arr):
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def _atomic_save_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.json")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def warn_if_interactive():
    run_type = os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
    log("=" * 70)
    if run_type and run_type.lower() != "batch":
        log(f"RUN TYPE: {run_type!r} - this looks like an interactive Draft Session.")
        log("Files written here are NOT guaranteed to persist once this session ends.")
        log(
            "To keep this output for real, use: Save Version -> Save & Run All (Commit),"
        )
        log("then check the notebook's non-edit page for that version's Output panel.")
    elif run_type is None:
        log(
            "RUN TYPE: unknown (KAGGLE_KERNEL_RUN_TYPE not set) - cannot confirm this is a committed run."
        )
    else:
        log(
            f"RUN TYPE: {run_type!r} - this is a committed run; output should persist to this version's Output panel once it finishes."
        )
    log("=" * 70)


def verify_output(out_dir, split, cache_shape):
    cache_path = Path(out_dir) / f"{split}_cache.dat"
    mask_path = Path(out_dir) / f"{split}_mask.npy"
    order_path = Path(out_dir) / f"{split}_study_order.csv"
    expected_bytes = int(np.prod(cache_shape))
    problems = []
    if not cache_path.is_file():
        problems.append(f"{cache_path.name} missing")
    elif cache_path.stat().st_size != expected_bytes:
        problems.append(
            f"{cache_path.name} is {cache_path.stat().st_size} bytes, expected {expected_bytes}"
        )
    if not mask_path.is_file():
        problems.append(f"{mask_path.name} missing")
    if not order_path.is_file():
        problems.append(f"{order_path.name} missing")
    return problems


def open_or_create_memmap(path, shape, dtype=np.uint8):
    path = Path(path)
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if path.is_file() and path.stat().st_size == nbytes:
        log(f"resuming from existing {path.name} ({nbytes / 1024 ** 3:.2f} GB)")
        return (np.memmap(path, dtype=dtype, mode="r+", shape=shape), True)
    path.parent.mkdir(parents=True, exist_ok=True)
    log(f"creating new {path.name} ({nbytes / 1024 ** 3:.2f} GB)")
    return (np.memmap(path, dtype=dtype, mode="w+", shape=shape), False)


def build_cache(
    slot_map,
    cache_slices,
    img_size,
    tag,
    out_dir,
    order_budget_s=ORDER_BUDGET_S,
    order_cache_path=None,
    flush_every=FLUSH_EVERY,
):
    studies = sorted(slot_map)
    sidx = {s: i for i, s in enumerate(studies)}
    shape = (len(studies), N_SLOT, cache_slices, img_size, img_size)
    cache_path = Path(out_dir) / f"{tag}_cache.dat"
    mask_path = Path(out_dir) / f"{tag}_mask.npy"
    cache, resumed = open_or_create_memmap(cache_path, shape)
    if resumed and mask_path.is_file():
        prev_mask = np.load(mask_path)
        if prev_mask.shape == (len(studies), N_SLOT):
            mask = prev_mask.astype(np.float32)
        else:
            log(
                f"{tag}: mask shape mismatch on resume ({prev_mask.shape} vs {(len(studies), N_SLOT)}); restarting mask and cache together"
            )
            cache, resumed = open_or_create_memmap(cache_path, shape)
            cache[:] = 0
            mask = np.zeros((len(studies), N_SLOT), np.float32)
            resumed = False
    else:
        mask = np.zeros((len(studies), N_SLOT), np.float32)
    log(
        f"{tag}: cache {cache.shape} = {cache.nbytes / 1024 ** 3:.1f} GB"
        + (" (resumed)" if resumed else "")
    )
    all_jobs = [
        (st, k, plane, slot_map[st][name])
        for st in studies
        for k, (name, plane, _, _) in enumerate(SLOTS)
        if name in slot_map[st]
    ]
    n_job_total = len(all_jobs)
    jobs = [j for j in all_jobs if mask[sidx[j[0]], j[1]] < 0.5]
    if resumed:
        log(
            f"{tag}: {n_job_total - len(jobs)}/{n_job_total} slots already cached, {len(jobs)} remaining"
        )
    seen = {}
    if order_cache_path and Path(order_cache_path).is_file():
        try:
            seen = json.loads(Path(order_cache_path).read_text())
        except (OSError, ValueError):
            seen = {}
    t_ord = time.time()
    ok = 0
    hit = 0
    for _, _, _, rec in jobs:
        e = seen.get(rec["SeriesInstanceUID"])
        if e and len(e["files"]) == len(rec["files"]):
            rec["ordered"] = e["files"]
            ok += int(e["good"])
            hit += 1
    to_order = [j for j in jobs if "ordered" not in j[3]]
    log(
        f"{tag}: {hit} slot-series ordered from cache, {len(to_order)} to read ({sum((len(j[3]['files']) for j in to_order))} slice headers)"
    )
    done = 0
    last_log = time.time()
    with ThreadPoolExecutor(max_workers=ORDER_THREADS) as pool:
        futures = {pool.submit(order_slices, rec): rec for _, _, _, rec in to_order}
        for fut in as_completed(futures):
            rec = futures[fut]
            files, good = fut.result()
            rec["ordered"] = files
            ok += int(good)
            done += 1
            if order_cache_path:
                seen[rec["SeriesInstanceUID"]] = {"files": files, "good": bool(good)}
            now = time.time()
            if now - last_log > 15:
                rate = done / max(now - t_ord, 1e-06)
                remain = len(to_order) - done
                eta_s = remain / max(rate, 1e-06)
                log(
                    f"  {tag} ordering {done}/{len(to_order)} ({rate:.0f}/s, ~{eta_s / 60:.0f}min remaining)"
                )
                last_log = now
            if now - t_ord > order_budget_s:
                log(
                    f"{tag}: ordering budget spent at {done}/{len(to_order)}; rest keep file order"
                )
                for f in futures:
                    f.cancel()
                break
    if order_cache_path and done:
        tmp = Path(order_cache_path).with_suffix(".tmp")
        tmp.write_text(json.dumps(seen))
        tmp.replace(order_cache_path)
    log(
        f"{tag}: ordered {ok}/{len(jobs)} by geometry ({len(jobs) - ok} kept arbitrary) in {time.time() - t_ord:.0f}s"
    )
    log(f"{tag}: decoding {len(jobs)} slot-series")
    n_failed_before = len(DECODE_FAILED)
    done = 0
    since_flush = 0
    with ThreadPoolExecutor(max_workers=PIX_THREADS) as pool:
        futures = {
            pool.submit(
                read_slot,
                rec,
                cache_slices,
                img_size,
                CROP_MM_BY_SLOT.get(SLOTS[k][0], CROP_MM),
            ): (st, k, plane)
            for st, k, plane, rec in jobs
        }
        for fut in as_completed(futures):
            st, k, plane = futures[fut]
            done += 1
            since_flush += 1
            try:
                img = fut.result()
            except Exception as exc:
                DECODE_FAILED.append(
                    {
                        "series": st,
                        "errors": [f"{type(exc).__name__}: {str(exc)[:120]}"],
                    }
                )
                img = None
            if img is not None:
                cache[sidx[st], k] = normalise_laterality(
                    img, plane, _CURRENT_LAT.get(st)
                ).numpy()
                mask[sidx[st], k] = 1.0
            if since_flush >= flush_every:
                cache.flush()
                _atomic_save_npy(mask_path, mask)
                since_flush = 0
            if done % 4096 < 1:
                log(f"  {tag} {done}/{len(jobs)}")
    cache.flush()
    _atomic_save_npy(mask_path, mask)
    n_failed = len(DECODE_FAILED) - n_failed_before
    log(
        f"{tag}: {int(mask.sum())}/{n_job_total} slots filled"
        + (f"; {n_failed} series had a slice that would not decode" if n_failed else "")
    )
    gc.collect()
    return (studies, cache, mask)


def coverage_report(tag, studies, cache, mask, slot_map, lat_map):
    n_study = len(studies)
    log(f"=== {tag} coverage report ===")
    log(f"studies: {n_study}")
    log(f"overall slot fill rate: {mask.mean():.1%}")
    for k, (name, plane, fluid, fs) in enumerate(SLOTS):
        log(f"  {name:16s} {int(mask[:, k].sum())}/{n_study} ({mask[:, k].mean():.1%})")
    n_lat_resolved = sum((1 for st in studies if lat_map.get(st) is not None))
    log(
        f"laterality resolved: {n_lat_resolved}/{n_study} ({n_lat_resolved / max(n_study, 1):.1%})"
    )
    n_chosen = n_unscaled = 0
    for st in studies:
        for name, row in slot_map.get(st, {}).items():
            n_chosen += 1
            px = row.get("px")
            if px is None or not np.isfinite(px) or px <= 0:
                n_unscaled += 1
    if n_chosen:
        log(
            f"chosen slot-series without usable pixel spacing (not physically cropped): {n_unscaled}/{n_chosen} ({n_unscaled / n_chosen:.1%})"
        )
    if DECODE_FAILED:
        reasons = Counter()
        for entry in DECODE_FAILED:
            for e in entry["errors"]:
                reasons[e.split(":")[0]] += 1
        log(f"decode failures: {len(DECODE_FAILED)} series affected")
        for reason, n in reasons.most_common(10):
            log(f"  {reason}: {n}")
    else:
        log("decode failures: none")


_CURRENT_LAT = {}


def process_split(root, split, out_dir, n_slice_target, order_cache_path):
    log(f"walking {split} series directories + reading headers")
    h = walk(root, split)
    if h.empty:
        log(f"{split}: no series found, skipping")
        return None
    h = annotate(h)
    plane_map = (
        h.set_index("SeriesInstanceUID")["Anatomical_Plane"].to_dict()
        if "Anatomical_Plane" in h.columns
        else {}
    )
    if not plane_map:
        series_csv = root / f"{split}_series.csv"
        if series_csv.is_file():
            series_meta = pd.read_csv(series_csv)
            plane_map = series_meta.set_index("SeriesInstanceUID")[
                "Anatomical_Plane"
            ].to_dict()
    lat_map = lat_of(h, tag=f"{split} ")
    global _CURRENT_LAT
    _CURRENT_LAT = lat_map
    slot_map = pick_slots(h, plane_map)
    studies, cache, mask = build_cache(
        slot_map,
        cache_slices=n_slice_target,
        img_size=CACHE_IMG,
        tag=split,
        out_dir=out_dir,
        order_cache_path=order_cache_path,
    )
    pd.Series(studies, name="StudyInstanceUID").to_csv(
        Path(out_dir) / f"{split}_study_order.csv", index=False
    )
    coverage_report(split, studies, cache, mask, slot_map, lat_map)
    return (studies, cache, mask)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--splits",
        default="train,test",
        help="comma-separated splits to process. Pass '--splits test' to rebuild ONLY the test cache from whatever test_series/ is present right now - use this inside the final submission notebook so a swapped-in hidden test set at grading time actually gets processed, instead of reusing a frozen cache built earlier against the 3-study example test.csv.",
    )
    ap.add_argument(
        "--out-dir",
        default="preprocessed",
        help="directory to write {split}_cache.dat / _mask.npy / _study_order.csv / _MANIFEST.json into",
    )
    ap.add_argument(
        "--cache-slices",
        type=int,
        default=None,
        help="force the number of slices cached per slot. MUST match whatever value was used to build the *other* split's cache (check that run's log for '-> N slices/slot') - the model's input shape at inference has to match what it was trained on. If omitted, sized automatically from available memory, which is only safe when building train and test together in the same run.",
    )
    args = ap.parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    OUT_DIR = Path(args.out_dir)
    ORDER_CACHE_PATH = OUT_DIR / "order_cache.json"
    ROOT = find_root()
    log(f"input root: {ROOT}")
    warn_if_interactive()
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    train_df = pd.read_csv(ROOT / "train.csv")
    n_slice_target = (
        args.cache_slices
        if args.cache_slices is not None
        else plan_cache(len(train_df))
    )
    manifest_entries = {}
    problems_all = []
    for split in splits:
        result = process_split(ROOT, split, OUT_DIR, n_slice_target, ORDER_CACHE_PATH)
        if result is None:
            continue
        studies, cache, mask = result
        problems = verify_output(OUT_DIR, split, cache.shape)
        problems_all += problems
        cache_path = OUT_DIR / f"{split}_cache.dat"
        mask_path = OUT_DIR / f"{split}_mask.npy"
        manifest_entries[split] = {
            "n_studies": len(studies),
            "cache_shape": list(cache.shape),
            "cache_bytes": cache_path.stat().st_size if cache_path.is_file() else 0,
            "mask_bytes": mask_path.stat().st_size if mask_path.is_file() else 0,
            "verified": len(problems) == 0,
        }
        del cache, mask, studies, result
        gc.collect()
    _atomic_save_json(
        OUT_DIR / "_MANIFEST.json",
        {
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "kaggle_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "unknown"),
            "splits": manifest_entries,
        },
    )
    log("=" * 70)
    if problems_all:
        log("SAVE VERIFICATION FAILED:")
        for p in problems_all:
            log(f"  - {p}")
        log("=" * 70)
        raise RuntimeError(
            f"{len(problems_all)} output file(s) missing or the wrong size after the run finished - see the list above. Do not trust this run's output."
        )
    log("SAVE VERIFICATION PASSED - every expected file is on disk at the right size:")
    for split, info in manifest_entries.items():
        log(
            f"  {split}: {info['n_studies']} studies, cache {info['cache_bytes'] / 1024 ** 3:.2f} GB, verified OK"
        )
    warn_if_interactive()
    log("=" * 70)
