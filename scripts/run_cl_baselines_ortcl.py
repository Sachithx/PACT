"""ORTCL baseline on the domain-incremental chain (TSLANet only).

At each incoming domain t:
  1. Collect unlabeled features from all seen domains (0..t-1) and domain t.
  2. Apply orthogonal SVD rotation to the last ICB conv3 (label-free).
  3. Optionally fine-tune the linear head on 5% labelled calibration.

Methods:
  ortcl-lf      — rotation only (0 target labels beyond rotation features)
  ortcl-headft  — rotation + head fine-tune on 5% calib (Table 2A baseline)

Outputs match run_cl_baselines.py so aggregate_table2.py can ingest them.

Example:
  python scripts/run_cl_baselines_ortcl.py --dataset fd --method ortcl-headft
"""

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

import _bootstrap  # noqa: F401
from _bootstrap import CKPT_ROOT, DATA_ROOT, RESULTS_ROOT

from src.data import get_spec
from src.data.datasets import adatime_chain
from src.method import compute_cl_metrics
from src.method.ortcl import (apply_ortcl_rotation, finetune_head,
                              new_domain_loader, seen_train_loader)
from src.models.factory import load_backbone
from src.training import train_or_load_source
from src.utils import get_device, set_seed

METHODS = ["ortcl-lf", "ortcl-headft"]
METHOD_LABEL = {"ortcl-lf": "ORTCL (LF)", "ortcl-headft": "ORTCL+HeadFT"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["fd", "hhar", "sleep"], default="fd")
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--backbone", default="tslanet",
                   help="ORTCL rotation is implemented for TSLANet only")
    p.add_argument("--domain_order", default="adatime")
    p.add_argument("--epochs", type=int, default=20,
                   help="head fine-tune epochs (ortcl-headft only)")
    p.add_argument("--calib_frac", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--device", default=None)
    return p.parse_args()


@torch.no_grad()
def eval_f1(model, spec, did, device, batch):
    model.eval()
    ld = torch.utils.data.DataLoader(
        spec.get_dataset(did, "test", DATA_ROOT), batch_size=batch)
    ys, ps = [], []
    for xb, yb in ld:
        ps.append(model(xb.to(device)).argmax(1).cpu().numpy())
        ys.append(yb.numpy())
    return f1_score(np.concatenate(ys), np.concatenate(ps),
                    average="macro", zero_division=0)


def calib_subset(spec, did, calib_frac, seed):
    ds = spec.get_dataset(did, "train", DATA_ROOT)
    if calib_frac >= 1.0:
        return ds
    y = ds.y.numpy()
    idx, _ = next(StratifiedShuffleSplit(
        n_splits=1, test_size=1 - calib_frac, random_state=seed
    ).split(np.zeros(len(y)), y))
    return torch.utils.data.Subset(ds, idx)


def adapt_domain(model, spec, order, t, device, args):
    src_ld = seen_train_loader(spec, order, t, DATA_ROOT, args.batch)
    tgt_ld = new_domain_loader(spec, order[t], DATA_ROOT, args.batch)
    err = apply_ortcl_rotation(model, src_ld, tgt_ld, device, args.batch)
    print(f"    ORTCL rotation orthogonality error: {err:.2e}")
    if args.method == "ortcl-headft":
        calib_ds = calib_subset(spec, order[t], args.calib_frac, args.seed)
        finetune_head(model, calib_ds, device, epochs=args.epochs,
                      lr=args.lr, batch_size=args.batch)


def main():
    args = parse_args()
    if args.backbone != "tslanet":
        raise ValueError("ORTCL chain baseline supports backbone=tslanet only")
    set_seed(args.seed)
    device = get_device(args.device)
    spec = get_spec(args.dataset)

    if args.domain_order == "adatime":
        order, st = adatime_chain(args.dataset)
        print(f"AdaTime chain: {st['n_domains']} domains, "
              f"{st['forward']}/{st['n_pairs']} pairs forward")
    elif args.domain_order:
        order = [int(x) for x in args.domain_order.split(",")]
    else:
        order = list(range(spec.n_domains))
    labels = [spec.domain_label(d) for d in order]
    T = len(order)

    train_or_load_source(args.backbone, args.dataset, order[0], CKPT_ROOT,
                         DATA_ROOT, device)
    model = load_backbone(args.backbone, spec, CKPT_ROOT, args.dataset,
                          order[0], device, freeze=False)

    A = np.full((T, T), np.nan)
    A[0, 0] = eval_f1(model, spec, order[0], device, args.batch)
    print(f"[{args.method}] Step 0 — {labels[0]:<10} F1={A[0, 0]:.4f}")

    for t in range(1, T):
        print(f"[{args.method}] Step {t} — adapting to {labels[t]} …")
        adapt_domain(model, spec, order, t, device, args)
        for j in range(t + 1):
            A[t, j] = eval_f1(model, spec, order[j], device, args.batch)
        aa, bwt, f, rf, prf = compute_cl_metrics(A, t)
        print(f"[{args.method}] Step {t} — {labels[t]:<10} AA={aa:.4f} "
              f"BWT={bwt:+.4f} F={f:.4f} RF={rf:.4f} PRF={prf:.4f}")

    tag = (f"{args.dataset}_{args.backbone}_{args.domain_order}_"
           f"{args.method}_c{int(args.calib_frac * 100)}")
    out_dir = RESULTS_ROOT / "cl_baselines" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(A, index=labels, columns=labels).to_csv(out_dir / "f1_matrix.csv")
    aa, bwt, f, rf, prf = compute_cl_metrics(A, T - 1)
    pd.DataFrame([dict(method=METHOD_LABEL[args.method], AA=aa, BWT=bwt,
                       F=f, RF=rf, PRF=prf)]).to_csv(out_dir / "summary.csv",
                                                       index=False)
    print(f"\n[{args.method}] Final: AA={aa:.4f} BWT={bwt:+.4f} F={f:.4f} "
          f"RF={rf:.4f} PRF={prf:.4f}\nResults -> {out_dir}")


if __name__ == "__main__":
    main()
