"""Run all gate variants (prototype, prototype_diag, prototype_lowrank, confidence)
on a given dataset/backbone and dump results for comparison."""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import DATA_ROOT, RESULTS_ROOT

from src.data import DATASET_NAMES, get_spec
from src.data.datasets import adatime_chain
from src.method import PactConfig, run_sequential
from src.method.pact import extract_embs
from src.models.tsfm import make_tsfm, TSFM_MODELS
from src.utils import get_device, set_seed


GATE_KINDS = ["prototype", "prototype_diag", "prototype_lowrank", "confidence"]


def fit_standardizer_and_head(model, spec, src_id, device, epochs=150, lr=1e-3, batch=64):
    ld = torch.utils.data.DataLoader(
        spec.get_dataset(src_id, "train", DATA_ROOT), batch_size=batch)
    X, y = extract_embs(model, ld, device)
    mean, std = X.mean(0), X.std(0)
    model.set_standardizer(mean, std)
    Xs = (X - mean) / np.clip(std, 1e-6, None)

    Xt = torch.from_numpy(Xs).to(device)
    yt = torch.from_numpy(y).long().to(device)
    opt = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, yt), batch_size=256, shuffle=True)
    model.head.train()
    for _ in range(epochs):
        for eb, yb in dl:
            opt.zero_grad()
            loss = crit(model.head(eb), yb)
            loss.backward(); opt.step()
    model.freeze_head()
    with torch.no_grad():
        acc = (model.head(Xt).argmax(1) == yt).float().mean().item()
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASET_NAMES, default="fd")
    p.add_argument("--tsfm", choices=list(TSFM_MODELS), default="moment")
    p.add_argument("--gate_kinds", nargs="+", default=GATE_KINDS)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--calib_frac", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    spec = get_spec(args.dataset)
    order, _ = adatime_chain(args.dataset)
    calib_tag = f"c{int(args.calib_frac*100)}"

    print(f"=== Gate Comparison: {args.dataset}/{args.tsfm}  K={args.k}  calib={calib_tag} ===")
    print(f"Gates to evaluate: {args.gate_kinds}")
    print(f"Device: {device}\n")

    model = make_tsfm(args.tsfm, spec.num_classes, device=str(device)).to(device)
    src_acc = fit_standardizer_and_head(model, spec, order[0], device)
    print(f"Source head fitted: train acc={src_acc:.4f}\n")

    results_summary = []
    out_root = RESULTS_ROOT / "gate_compare"
    out_root.mkdir(parents=True, exist_ok=True)

    for gk in args.gate_kinds:
        print(f"\n{'='*60}")
        print(f"  Running gate_kind = {gk}")
        print(f"{'='*60}")
        t0 = time.time()

        cfg = PactConfig(k=args.k, emb_dim=model.emb_dim, calib_frac=args.calib_frac,
                         seed=args.seed, batch=args.batch, gate_kind=gk)
        try:
            res = run_sequential(model, spec, order, DATA_ROOT, device, cfg,
                                 temperature=args.temperature, verbose=True)
        except Exception as e:
            print(f"  ** FAILED: {e}")
            results_summary.append(dict(gate_kind=gk, AA=float("nan"), F=float("nan"),
                                        elapsed=0, error=str(e)))
            continue

        elapsed = time.time() - t0
        final = res["metrics"][-1]
        results_summary.append(dict(
            gate_kind=gk, AA=final["AA"], F=final["F_all"],
            BWT=final["BWT"], RF=final["RF"], PRF=final["PRF"],
            elapsed=elapsed, error=""))

        tag = f"{args.dataset}_{args.tsfm}_adatime_K{args.k}_{calib_tag}_{gk}"
        out_dir = RESULTS_ROOT / "sequential_tsfm" / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(res["metrics"]).to_csv(out_dir / "metrics.csv", index=False)
        labels = res["labels"]
        pd.DataFrame(res["A_soft"], index=labels, columns=labels).to_csv(
            out_dir / "soft_f1_matrix.csv")
        print(f"  -> AA={final['AA']:.4f}  F={final['F_all']:.4f}  ({elapsed:.1f}s)")

    # Print comparison table
    print(f"\n\n{'='*70}")
    print(f"  GATE COMPARISON SUMMARY: {args.dataset} / {args.tsfm}")
    print(f"{'='*70}")
    print(f"{'Gate Kind':<20} {'AA':>8} {'F':>8} {'BWT':>8} {'RF':>8} {'Time(s)':>8}")
    print("-" * 70)
    for r in results_summary:
        print(f"{r['gate_kind']:<20} {r.get('AA',0):>8.4f} {r.get('F',0):>8.4f} "
              f"{r.get('BWT',0):>+8.4f} {r.get('RF',0):>8.4f} {r.get('elapsed',0):>8.1f}")

    df = pd.DataFrame(results_summary)
    df.to_csv(out_root / f"{args.dataset}_{args.tsfm}_{calib_tag}_gate_compare.csv",
              index=False)
    print(f"\nSaved -> {out_root / f'{args.dataset}_{args.tsfm}_{calib_tag}_gate_compare.csv'}")


if __name__ == "__main__":
    main()
