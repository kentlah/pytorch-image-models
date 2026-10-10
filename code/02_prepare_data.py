# 02_prepare_data.py - main task: EVA-02 feature CSVs to .npy, our validation split and a small training sample

import hashlib
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "output"
PROC.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 7
MODEL = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
VAL_PER_CLASS = 50        # our validation split, carved from train; the test sets are never used for tuning
SMALL_PER_CLASS = 100     # a quick sample for trying models before training on everything
CHUNK = 20_000


# 1. Convert
# the train CSV is 14 GB, so it is read in chunks and written straight into a float32 array on disk
def open_raw(name):
    for path in (RAW / f"{name}_{MODEL}.csv", RAW / f"{name}_{MODEL}.csv.zip"):
        if path.suffix == ".csv" and path.exists():
            return open(path, "rb")
        if path.suffix == ".zip" and path.exists():
            z = zipfile.ZipFile(path)
            return z.open([m for m in z.namelist() if m.endswith(".csv")][0])
    raise FileNotFoundError(f"{name}_{MODEL}.csv(.zip) not found in {RAW}")


def convert(name, out):
    if (PROC / f"{out}_X.npy").exists():
        print(out, "already converted")
        return
    t0 = time.time()
    with open_raw(name) as f:
        features = [c for c in f.readline().decode().strip().split(",") if c.isdigit()]
        n = sum(block.count(b"\n") for block in iter(lambda: f.read(2**26), b""))
    print(f"{out}: {n:,} rows x {len(features)} features")

    X = np.lib.format.open_memmap(PROC / f"{out}_X.npy", mode="w+", dtype=np.float32, shape=(n, len(features)))
    y = np.empty(n, dtype=np.int16)
    paths = []
    start = 0
    with open_raw(name) as f:
        dtypes = {c: np.float32 for c in features} | {"label": np.int16}
        for chunk in pd.read_csv(f, usecols=["path", "label"] + features, dtype=dtypes, chunksize=CHUNK):
            X[start:start + len(chunk)] = chunk[features].to_numpy()
            y[start:start + len(chunk)] = chunk["label"].to_numpy()
            paths += chunk["path"].tolist()
            start += len(chunk)
            if start % 200_000 < CHUNK:
                print(f"  {start:,} rows, {time.time() - t0:.0f}s", flush=True)
    X.flush()
    del X
    np.save(PROC / f"{out}_y.npy", y)
    pd.DataFrame({"path": paths, "label": y}).to_csv(PROC / f"{out}_meta.csv", index=False)
    print(f"{out} done in {time.time() - t0:.0f}s")


convert("v2", "test2")
convert("val", "test1")
convert("train", "train_full")


# 2. Split
# per class, 50 random training images become our validation set; 100 of the rest form train_small
y = np.load(PROC / "train_full_y.npy")
rng = np.random.default_rng(SEED)
val_idx, train_idx, small_idx = [], [], []
for c in range(1000):
    rows = rng.permutation(np.flatnonzero(y == c))
    val_idx.append(rows[:VAL_PER_CLASS])
    train_idx.append(rows[VAL_PER_CLASS:])
    small_idx.append(rows[VAL_PER_CLASS:VAL_PER_CLASS + SMALL_PER_CLASS])
splits = {"val": val_idx, "train": train_idx, "train_small": small_idx}
splits = {k: np.sort(np.concatenate(v)) for k, v in splits.items()}
for k, v in splits.items():
    np.save(PROC / f"split_{k}_idx.npy", v)
print({k: len(v) for k, v in splits.items()})

# val and train_small are small enough to save as their own arrays; train is used through the index
X_full = np.load(PROC / "train_full_X.npy", mmap_mode="r")
meta = pd.read_csv(PROC / "train_full_meta.csv")
for k in ("val", "train_small"):
    np.save(PROC / f"{k}_X.npy", np.asarray(X_full[splits[k]]))
    np.save(PROC / f"{k}_y.npy", y[splits[k]])
    meta.iloc[splits[k]].to_csv(PROC / f"{k}_meta.csv", index=False)


# 3. Check
# size, class balance and missing values per set; an image in both train and a test set would
# have an identical feature vector, so test rows are looked up among hashed train rows
def row_hashes(X):
    return {hashlib.md5(r.tobytes()).digest() for i in range(0, len(X), 100_000) for r in np.asarray(X[i:i + 100_000])}


train_hashes = row_hashes(X_full)
summary = []
for k in ("train_full", "train_small", "val", "test1", "test2"):
    X = np.load(PROC / f"{k}_X.npy", mmap_mode="r")
    counts = np.bincount(np.load(PROC / f"{k}_y.npy"), minlength=1000)
    summary.append({
        "set": k, "rows": len(X), "features": X.shape[1], "classes": int((counts > 0).sum()),
        "min_per_class": int(counts.min()), "max_per_class": int(counts.max()),
        "nan_values": int(sum(np.isnan(X[i:i + 100_000]).sum() for i in range(0, len(X), 100_000))),
        "rows_also_in_train": len(row_hashes(X) & train_hashes) if k.startswith("test") else "",
    })
summary = pd.DataFrame(summary)
summary.to_csv(OUT / "prepare_summary.csv", index=False)
print(summary.to_string(index=False))
