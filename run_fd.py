"""Reproduce the FD row of PACT in the main comparison table.

Frozen Sundial backbone, K = 8 charts, LoRA rank-8 transports, 5% labelled
calibration per target domain, learned soft gate (temperature 0.5), seed 42,
over the 4 fixed domain orders below. Reports AMF1 (= AA) and PRF after the
full stream as mean +/- std over the 4 orders.

    python run_fd.py --data_dir /path/to/FD
    python run_fd.py --data_dir /path/to/FD --orders 2,1,0,3     # one order only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pact.backbone import SundialBackbone
from pact.data import FD_NAMES, FD_NUM_CLASSES, FDDataset
from pact.method import PactConfig, extract_embs, run_stream

# The 4 domain orders used in the paper (0..3 = FD-A..FD-D; first = source).
FD_ORDERS = [[2, 1, 0, 3], [2, 1, 3, 0], [2, 0, 1, 3], [0, 2, 3, 1]]


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_source_head(model, source_ds, device, epochs=150, lr=1e-3, batch=64):
    """Fit the embedding standardizer and the linear head on the source
    domain, then freeze the head."""
    X, y = extract_embs(model, torch.utils.data.DataLoader(source_ds, batch_size=batch), device)
    mean, std = X.mean(0), X.std(0)
    model.set_standardizer(mean, std)
    Xs = torch.from_numpy((X - mean) / np.clip(std, 1e-6, None)).to(device)
    yt = torch.from_numpy(y).long().to(device)
    opt = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xs, yt),
                                     batch_size=256, shuffle=True)
    model.head.train()
    for _ in range(epochs):
        for eb, yb in dl:
            opt.zero_grad(); crit(model.head(eb), yb).backward(); opt.step()
    model.freeze_head()
    with torch.no_grad():
        return (model.head(Xs).argmax(1) == yt).float().mean().item()


def run_order(order, args, device):
    set_seed(args.seed)                  # every order starts from the same RNG state
    model = SundialBackbone(FD_NUM_CLASSES).to(device)
    make_dataset = lambda d, split: FDDataset(d, split, args.data_dir)
    acc = fit_source_head(model, make_dataset(order[0], "train"), device)
    print(f"  source head fitted on {FD_NAMES[order[0]]}: train acc={acc:.4f}")

    cfg = PactConfig(k=args.k, emb_dim=model.emb_dim, calib_frac=args.calib_frac,
                     seed=args.seed, temperature=args.temperature)
    A, rows = run_stream(model, make_dataset, order, FD_NAMES, device, cfg)
    return A, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True,
                   help="folder with train_{0..3}.pt and test_{0..3}.pt (AdaTime FD)")
    p.add_argument("--orders", nargs="*", default=None,
                   help="domain orders like 2,1,0,3 (default: the 4 paper orders)")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--calib_frac", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="results")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    orders = ([[int(v) for v in o.split(",")] for o in args.orders]
              if args.orders else FD_ORDERS)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    summary = []
    for order in orders:
        tag = ",".join(map(str, order))
        names = [FD_NAMES[d] for d in order]
        print(f"\n=== order {' -> '.join(names)}  (K={args.k}, seed={args.seed}) ===")
        t0 = time.time()
        A, rows = run_order(order, args, device)
        final = rows[-1]
        print(f"  final AMF1={final['AA']:.4f}  PRF={final['PRF']:.5f}  "
              f"({time.time() - t0:.0f}s)")
        run_dir = out / f"fd_order{tag}_K{args.k}_s{args.seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
        pd.DataFrame(A, index=names, columns=names).to_csv(run_dir / "f1_matrix.csv")
        summary.append(dict(order=tag, AMF1=final["AA"], PRF=final["PRF"]))

    df = pd.DataFrame(summary)
    df.to_csv(out / "fd_summary.csv", index=False)
    print("\n" + df.to_string(index=False))
    if len(df) > 1:
        # std with ddof=0 (population std), as in the paper's tables
        print(f"\nFD  AMF1 = {df.AMF1.mean():.3f} +/- {df.AMF1.std(ddof=0):.3f}   "
              f"PRF = {df.PRF.mean():.4f} +/- {df.PRF.std(ddof=0):.4f}   "
              f"(over {len(df)} orders)")
    with open(out / "fd_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
