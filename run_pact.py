"""
run_pact.py — End-to-end PACT evaluation on FD / HHAR / Sleep-EDF

Steps:
  1. Load a pre-trained (frozen) source-domain backbone.
  2. Fit PACT on 5% labeled calibration data from the target domain.
  3. Evaluate source retention and target drift recovery on held-out test sets.

Usage:
  python run_pact.py --dataset hhar --src 0 --tgt 6 --calib_frac 0.05 \\
                     --backbone_ckpt checkpoints/tslanet_hhar_u0.pt \\
                     --k 4 --device cuda

  python run_pact.py --dataset fd --src FD-A --tgt FD-B --calib_frac 0.05 \\
                     --backbone_ckpt checkpoints/tslanet_fd_FD-A.pt

  python run_pact.py --dataset sleep --src 0 --tgt 11 \\
                     --backbone_ckpt checkpoints/tslanet_sleep_0.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from pact import PACT
from backbone.tslanet import load_tslanet
from datasets import FDDataset, HHARDataset, SleepEDFDataset


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PACT evaluation")
    p.add_argument("--dataset",       required=True,
                   choices=["fd", "hhar", "sleep"],
                   help="Dataset to use")
    p.add_argument("--src",           required=True,
                   help="Source domain ID (FD: 'FD-A'…'FD-D'; HHAR/Sleep: int)")
    p.add_argument("--tgt",           required=True,
                   help="Target domain ID (same format as --src)")
    p.add_argument("--data_root",     default="data",
                   help="Root directory containing processed datasets")
    p.add_argument("--backbone_ckpt", required=True,
                   help="Path to pre-trained source backbone checkpoint (.pt)")
    p.add_argument("--k",             type=int,   default=4,
                   help="Number of PACT concepts (default: 4)")
    p.add_argument("--calib_frac",    type=float, default=0.05,
                   help="Fraction of target train used for calibration (default: 0.05)")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--adapter_ep",    type=int,   default=200)
    p.add_argument("--gate_ep",       type=int,   default=200)
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pact_ckpt",     default=None,
                   help="Save fitted PACT state to this path")
    return p.parse_args()


# ── Dataset helpers ────────────────────────────────────────────────────────────

def load_splits(args):
    """Return (src_train, src_test, tgt_train, tgt_test) Dataset objects."""
    root = args.data_root
    ds   = args.dataset
    src  = args.src
    tgt  = args.tgt

    if ds == "fd":
        return (FDDataset(src, "train", root), FDDataset(src, "test", root),
                FDDataset(tgt, "train", root), FDDataset(tgt, "test", root))
    elif ds == "hhar":
        src_id, tgt_id = int(src), int(tgt)
        return (HHARDataset(src_id, "train", root), HHARDataset(src_id, "test", root),
                HHARDataset(tgt_id, "train", root), HHARDataset(tgt_id, "test", root))
    else:  # sleep
        src_id, tgt_id = int(src), int(tgt)
        from datasets import SleepEDFDataset
        return (SleepEDFDataset(src_id, "train", root), SleepEDFDataset(src_id, "test", root),
                SleepEDFDataset(tgt_id, "train", root), SleepEDFDataset(tgt_id, "test", root))


def make_loader(ds, batch=256, shuffle=False, drop_last=False):
    return torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=shuffle, drop_last=drop_last)


def calib_subset(tgt_train_ds, frac: float, seed: int):
    """Return a stratified fraction of the target train set as a Subset."""
    labels = torch.stack([tgt_train_ds[i][1] for i in range(len(tgt_train_ds))]).numpy()
    sss    = StratifiedShuffleSplit(n_splits=1, test_size=1 - frac, random_state=seed)
    idx, _ = next(sss.split(np.zeros(len(labels)), labels))
    return torch.utils.data.Subset(tgt_train_ds, idx)


BATCH = {"fd": 64, "hhar": 256, "sleep": 64}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device(args.device)
    batch  = BATCH[args.dataset]

    print(f"\n{'='*60}")
    print(f"  PACT  |  dataset={args.dataset}  src={args.src}  tgt={args.tgt}")
    print(f"  calib_frac={args.calib_frac:.0%}  K={args.k}  device={device}")
    print(f"{'='*60}\n")

    # ── Load backbone ──────────────────────────────────────────────────────────
    print("Loading backbone …")
    backbone = load_tslanet(args.backbone_ckpt, args.dataset, device=device)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading datasets …")
    src_train, src_test, tgt_train, tgt_test = load_splits(args)

    calib_ds  = calib_subset(tgt_train, args.calib_frac, args.seed)
    print(f"  Calibration: {len(calib_ds)} samples  "
          f"({args.calib_frac:.0%} of {len(tgt_train)} target train)")

    src_train_ld = make_loader(src_train, batch)
    src_test_ld  = make_loader(src_test,  batch)
    calib_ld     = make_loader(calib_ds,  batch)
    tgt_test_ld  = make_loader(tgt_test,  batch)

    # ── Baseline (no adaptation) ───────────────────────────────────────────────
    pact = PACT(backbone=backbone, k=args.k, emb_dim=128,
                device=device, seed=args.seed)

    f1_src_bare = pact.evaluate_backbone(src_test_ld)
    f1_tgt_bare = pact.evaluate_backbone(tgt_test_ld)
    print(f"\nNo adaptation:  src={f1_src_bare:.4f}  tgt={f1_tgt_bare:.4f}\n")

    # ── Fit PACT ───────────────────────────────────────────────────────────────
    pact.fit(src_train_ld, calib_ld,
             adapter_epochs=args.adapter_ep,
             gate_epochs=args.gate_ep)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    f1_src_pact = pact.evaluate(src_test_ld)
    f1_tgt_pact = pact.evaluate(tgt_test_ld)
    forgetting   = f1_src_bare - f1_src_pact
    improvement  = f1_tgt_pact - f1_tgt_bare

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'─'*60}")
    print(f"  Source retention:  {f1_src_pact:.4f}  "
          f"(forgetting={forgetting:+.4f})")
    print(f"  Target recovery:   {f1_tgt_pact:.4f}  "
          f"(improvement={improvement:+.4f})")
    print(f"{'='*60}\n")

    if args.pact_ckpt:
        pact.save(args.pact_ckpt)


if __name__ == "__main__":
    main()
