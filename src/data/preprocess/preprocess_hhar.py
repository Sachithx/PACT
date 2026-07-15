"""
preprocess_hhar.py — Preprocess HHAR raw CSV → per-user windowed .pt files

Reads Phones_accelerometer.csv from the HHAR Activity recognition exp directory.
Groups samples by (User, Device), sorts by Creation_Time, applies a sliding
window of 128 samples / 64-sample stride, keeps only windows with a single
uniform activity label, then does a stratified 70/15/15 train/val/test split
per user and saves:

  dataset/HHAR_processed/train_{0..8}.pt
  dataset/HHAR_processed/val_{0..8}.pt
  dataset/HHAR_processed/test_{0..8}.pt

Each file: {"samples": FloatTensor[N, 3, 128], "labels": LongTensor[N]}
  - axis order: [x, y, z]
  - domain ID: user a=0, b=1, ..., i=8
  - class ID:  bike=0, sit=1, stand=2, walk=3, stairsup=4, stairsdown=5

AdaTime HHAR scenarios (for reference, using these domain IDs):
  (0,6),(1,6),(2,7),(3,8),(4,5),(5,0),(6,1),(7,4),(8,3),(0,2)

Run:
  /home/AD/sachith/.conda/envs/entrope/bin/python preprocess_hhar.py
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

# ── Config ─────────────────────────────────────────────────────────────────────

RAW_CSV   = (Path(__file__).parent / "dataset" / "HHAR" /
             "Activity recognition exp" / "Phones_accelerometer.csv")
OUT_DIR   = Path(__file__).parent / "dataset" / "HHAR_processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW    = 128
STRIDE    = 64
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15          # test = remaining 0.15
SEED       = 42

USER_MAP  = {u: i for i, u in enumerate("abcdefghi")}  # a→0 … i→8
ACT_MAP   = {"bike": 0, "sit": 1, "stand": 2,
             "walk": 3, "stairsup": 4, "stairsdown": 5}


# ── Windowing ──────────────────────────────────────────────────────────────────

def extract_windows(group: pd.DataFrame):
    """Sliding window over one (User, Device) segment sorted by time.
    Returns arrays X[N, 3, 128] and y[N] for pure-label windows only."""
    group = group.sort_values("Creation_Time")
    xyz   = group[["x", "y", "z"]].to_numpy(dtype=np.float32)
    acts  = group["gt"].to_numpy()
    n     = len(xyz)

    X_list, y_list = [], []
    for start in range(0, n - WINDOW + 1, STRIDE):
        end   = start + WINDOW
        win_y = acts[start:end]
        # keep only uniform-label windows (no activity transitions)
        if len(set(win_y)) != 1:
            continue
        label = ACT_MAP.get(win_y[0], -1)
        if label == -1:
            continue
        X_list.append(xyz[start:end].T)   # [3, 128]
        y_list.append(label)

    if not X_list:
        return np.empty((0, 3, WINDOW), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.stack(X_list).astype(np.float32), np.array(y_list, dtype=np.int64)


# ── Stratified split ───────────────────────────────────────────────────────────

def stratified_split(X, y, train_frac, val_frac, seed):
    """Returns (X_tr, y_tr), (X_va, y_va), (X_te, y_te)."""
    test_frac = 1.0 - train_frac - val_frac

    # First carve off test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    tr_va_idx, te_idx = next(sss1.split(X, y))
    X_trva, y_trva = X[tr_va_idx], y[tr_va_idx]
    X_te,   y_te   = X[te_idx],   y[te_idx]

    # Then carve val from the remainder
    val_of_trva = val_frac / (train_frac + val_frac)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_of_trva, random_state=seed)
    tr_idx, va_idx = next(sss2.split(X_trva, y_trva))
    return (X_trva[tr_idx], y_trva[tr_idx],
            X_trva[va_idx], y_trva[va_idx],
            X_te,           y_te)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Reading {RAW_CSV} …")
    df = pd.read_csv(RAW_CSV, usecols=["Creation_Time", "x", "y", "z", "User", "Device", "gt"])

    # Drop null-class rows
    df = df[df["gt"].isin(ACT_MAP)].reset_index(drop=True)
    print(f"After dropping null:  {len(df):,} rows across users {sorted(df['User'].unique())}")

    summary_lines = ["HHAR preprocessing summary\n",
                     f"Window={WINDOW}  Stride={STRIDE}  "
                     f"Split={TRAIN_FRAC:.0%}/{VAL_FRAC:.0%}/{1-TRAIN_FRAC-VAL_FRAC:.0%}\n"]

    for user_char, domain_id in sorted(USER_MAP.items()):
        user_df = df[df["User"] == user_char]
        if user_df.empty:
            print(f"  User {user_char} (domain {domain_id}): no data — skipping")
            continue

        # Collect windows across all devices for this user
        X_parts, y_parts = [], []
        for device, grp in user_df.groupby("Device"):
            Xw, yw = extract_windows(grp)
            if len(Xw):
                X_parts.append(Xw)
                y_parts.append(yw)

        if not X_parts:
            print(f"  User {user_char} (domain {domain_id}): 0 windows after filtering")
            continue

        X_all = np.concatenate(X_parts, axis=0)
        y_all = np.concatenate(y_parts, axis=0)

        # Shuffle before splitting (different devices may be contiguous)
        rng  = np.random.default_rng(SEED)
        perm = rng.permutation(len(X_all))
        X_all, y_all = X_all[perm], y_all[perm]

        # Stratified train / val / test split
        X_tr, y_tr, X_va, y_va, X_te, y_te = stratified_split(
            X_all, y_all, TRAIN_FRAC, VAL_FRAC, SEED)

        # Save
        for split_name, Xs, ys in [("train", X_tr, y_tr),
                                    ("val",   X_va, y_va),
                                    ("test",  X_te, y_te)]:
            path = OUT_DIR / f"{split_name}_{domain_id}.pt"
            torch.save({"samples": torch.from_numpy(Xs),
                        "labels":  torch.from_numpy(ys)}, path)

        counts = {c: int((y_all == v).sum()) for c, v in ACT_MAP.items()}
        line = (f"  User {user_char} (domain {domain_id}):  "
                f"total={len(X_all):6,}  train={len(y_tr):5,}  "
                f"val={len(y_va):4,}  test={len(y_te):4,}  "
                f"classes={counts}")
        print(line)
        summary_lines.append(line)

    # Write summary
    (OUT_DIR / "preprocessing_summary.txt").write_text("\n".join(summary_lines))
    print(f"\nDone. Files in {OUT_DIR}")
    print(f"  Format: {{samples: [N, 3, 128] float32, labels: [N] int64}}")
    print(f"  Classes: {ACT_MAP}")
    print(f"  Domain IDs: {USER_MAP}")


if __name__ == "__main__":
    main()
