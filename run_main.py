"""Reproduce the PACT row of the main comparison.

Frozen Sundial-base-128m, rank-8 transports, learned soft gate (temperature
0.5), seed 42, over the four fixed streams in the paper. Reports AMF1, RF,
and PRF as the mean over those streams.

    python download_data.py
    python run_main.py --dataset fd
    python run_main.py --dataset all
    python run_main.py --dataset hhar --orders 6,0,8,2,5,1,3,4,7
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
from pact.data import DomainDataset
from pact.method import PactConfig, extract_embs, run_stream

# Paper streams. The first id is the labelled source. K and the calibration
# budget are the dataset-specific settings used for the main comparison.
DATASETS = {
    "fd": dict(
        title="FD", folder="FD", num_classes=3, k=8, calib_frac=0.05, calib_n=None,
        names={0: "FD-A", 1: "FD-B", 2: "FD-C", 3: "FD-D"},
        orders=[[0, 2, 3, 1], [2, 0, 1, 3], [2, 1, 0, 3], [2, 1, 3, 0]],
    ),
    "wisdm": dict(
        title="WISDM", folder="WISDM", num_classes=6, k=4, calib_frac=0.05, calib_n=450,
        names={i: f"Subj-{i}" for i in range(36)},
        orders=[
            [10, 24, 20, 0, 3, 1, 22, 14, 2],
            [12, 23, 7, 9, 5, 15, 31, 28, 10],
            [24, 11, 0, 23, 13, 21, 16, 35, 3],
            [30, 3, 0, 31, 20, 24, 19, 13, 17],
        ],
    ),
    "hhar": dict(
        title="HHAR", folder="HHAR", num_classes=6, k=4, calib_frac=0.05, calib_n=None,
        names={i: f"User-{i}" for i in range(9)},
        orders=[
            [1, 6, 4, 7, 8, 5, 0, 2, 3],
            [5, 1, 6, 3, 2, 7, 4, 0, 8],
            [5, 8, 0, 3, 2, 6, 4, 1, 7],
            [6, 0, 8, 2, 5, 1, 3, 4, 7],
        ],
    ),
    "ucihar": dict(
        title="UCI-HAR", folder="UCIHAR", num_classes=6, k=4, calib_frac=0.05, calib_n=100,
        names={i: f"Subj-{i}" for i in range(1, 31)},
        orders=[
            [14, 24, 1, 27, 26, 10, 12, 7, 9],
            [19, 14, 6, 12, 4, 29, 9, 18, 21],
            [24, 14, 19, 17, 6, 5, 7, 3, 30],
            [6, 20, 18, 30, 12, 24, 25, 3, 17],
        ],
    ),
    "sleep": dict(
        title="Sleep-EDF", folder="Sleep-EDF", num_classes=5, k=1, calib_frac=0.05, calib_n=None,
        names={i: f"Subj-{i}" for i in range(20)},
        orders=[
            [0, 5, 18, 9, 3, 10, 6, 1, 14],
            [1, 5, 9, 0, 4, 3, 18, 8, 13],
            [15, 6, 17, 5, 8, 9, 18, 19, 2],
            [18, 5, 12, 7, 10, 17, 0, 3, 1],
        ],
    ),
}


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_data_dir(root, spec):
    """Prefer dataset/<folder>/. Also accept a flat folder of train_*.pt (FD)."""
    root = Path(root)
    nested = root / spec["folder"]
    if nested.is_dir() and any(nested.glob("train_*.pt")):
        return nested
    if any(root.glob("train_*.pt")):
        return root
    return nested


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
            opt.zero_grad()
            crit(model.head(eb), yb).backward()
            opt.step()
    model.freeze_head()
    with torch.no_grad():
        return (model.head(Xs).argmax(1) == yt).float().mean().item()


def run_order(order, spec, data_dir, args, device):
    set_seed(args.seed)
    model = SundialBackbone(spec["num_classes"]).to(device)
    make_dataset = lambda d, split: DomainDataset(d, split, data_dir)
    acc = fit_source_head(model, make_dataset(order[0], "train"), device, batch=args.batch)
    print(f"  source head fitted on {spec['names'][order[0]]}: train acc={acc:.4f}")
    cfg = PactConfig(
        k=args.k if args.k is not None else spec["k"],
        emb_dim=model.emb_dim,
        calib_frac=spec["calib_frac"],
        calib_n=spec["calib_n"],
        seed=args.seed,
        temperature=args.temperature,
        batch=args.batch,
    )
    return run_stream(model, make_dataset, order, spec["names"], device, cfg)


def run_dataset(name, args, device):
    spec = DATASETS[name]
    data_dir = resolve_data_dir(args.data_dir, spec)
    orders = ([[int(v) for v in o.split(",")] for o in args.orders]
              if args.orders else spec["orders"])
    k = args.k if args.k is not None else spec["k"]
    budget = f"n={spec['calib_n']}" if spec["calib_n"] else f"{spec['calib_frac']:.0%}"
    print(f"\n######## {spec['title']}  K={k}  calibration={budget}  "
          f"data={data_dir} ########")

    summary = []
    out = Path(args.out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    for order in orders:
        tag = ",".join(map(str, order))
        names = [spec["names"][d] for d in order]
        print(f"\n=== {spec['title']}  {' -> '.join(names)}  (K={k}, seed={args.seed}) ===")
        t0 = time.time()
        A, rows = run_order(order, spec, data_dir, args, device)
        final = rows[-1]
        print(f"  final AMF1={final['AA']:.4f}  RF={final['RF']:.4f}  "
              f"PRF={final['PRF']:.4f}  ({time.time() - t0:.0f}s)")
        run_dir = out / f"order{tag}_K{k}_s{args.seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
        pd.DataFrame(A, index=names, columns=names).to_csv(run_dir / "f1_matrix.csv")
        summary.append(dict(order=tag, AMF1=final["AA"], RF=final["RF"], PRF=final["PRF"]))

    df = pd.DataFrame(summary)
    df.to_csv(out / "summary.csv", index=False)
    print("\n" + df.to_string(index=False))
    if len(df) > 1:
        print(f"\n{spec['title']}  "
              f"AMF1 = {df.AMF1.mean():.4f} +/- {df.AMF1.std(ddof=0):.4f}   "
              f"RF = {df.RF.mean():.4f} +/- {df.RF.std(ddof=0):.4f}   "
              f"PRF = {df.PRF.mean():.4f} +/- {df.PRF.std(ddof=0):.4f}   "
              f"(over {len(df)} streams, seed {args.seed})")
    with open(out / "config.json", "w") as f:
        json.dump(dict(dataset=name, k=k, seed=args.seed, data_dir=str(data_dir),
                       calib_frac=spec["calib_frac"], calib_n=spec["calib_n"]), f, indent=2)
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="all",
                   choices=["all", *DATASETS],
                   help="which benchmark to run (default: all five)")
    p.add_argument("--data_dir", default="dataset",
                   help="root that contains FD/, WISDM/, HHAR/, UCIHAR/, Sleep-EDF/")
    p.add_argument("--orders", nargs="*", default=None,
                   help="override streams, e.g. --orders 2,1,0,3")
    p.add_argument("--k", type=int, default=None, help="override the paper chart count")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--out_dir", default="results")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    print(f"device={device}")
    for name in names:
        run_dataset(name, args, device)


if __name__ == "__main__":
    main()
