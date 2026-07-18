"""Nearest Class Mean (NCM) on frozen embeddings — trivial baseline (B5).

At each domain step, class prototypes are the mean calibration embedding per class
from all domains seen so far (same 5% budget as PACT). No transport, no gate.

Example:
  python scripts/run_ncm_baseline.py --dataset fd --backbone tslanet
  python scripts/run_ncm_baseline.py --dataset hhar --tsfm moment
"""

import argparse
import copy

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

import _bootstrap  # noqa: F401
from _bootstrap import CKPT_ROOT, DATA_ROOT, RESULTS_ROOT

from src.data import get_spec
from src.data.calib import calib_indices
from src.data.datasets import adatime_chain
from src.method import compute_cl_metrics
from src.method.pact import extract_embs
from src.models.factory import BACKBONES, load_backbone
from src.models.tsfm import TSFM_MODELS, make_tsfm
from src.training import train_or_load_source
from src.utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["fd", "hhar", "sleep", "wisdm", "ucihar"], default="fd")
    p.add_argument("--backbone", choices=BACKBONES, default="tslanet")
    p.add_argument("--tsfm", choices=list(TSFM_MODELS), default=None)
    p.add_argument("--domain_order", default="adatime")
    p.add_argument("--calib_frac", type=float, default=0.05)
    p.add_argument("--calib_n", type=int, default=None,
                   help="Fixed calibration label budget (overrides calib_frac)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--device", default=None)
    return p.parse_args()


def fit_prototypes(embs, lbls, n_classes):
    protos = np.zeros((n_classes, embs.shape[1]), np.float32)
    for c in range(n_classes):
        m = lbls == c
        if m.any():
            protos[c] = embs[m].mean(0)
        else:
            protos[c] = np.nan
    return protos


@torch.no_grad()
def ncm_predict(embs, protos):
    valid = ~np.isnan(protos).any(axis=1)
    if not valid.any():
        return np.zeros(len(embs), dtype=np.int64)
    P = protos[valid]
    idx_map = np.where(valid)[0]
    d = np.linalg.norm(embs[:, None] - P[None], axis=2)
    return idx_map[d.argmin(1)]


def eval_ncm(model, spec, did, protos, device, batch):
    ld = torch.utils.data.DataLoader(
        spec.get_dataset(did, "test", DATA_ROOT), batch_size=batch)
    embs, lbls = extract_embs(model, ld, device)
    preds = ncm_predict(embs, protos)
    return f1_score(lbls, preds, average="macro", zero_division=0)


def load_model(args, spec, src_id, device):
    if args.tsfm:
        from run_sequential_tsfm import fit_standardizer_and_head
        model = make_tsfm(args.tsfm, spec.num_classes, device=str(device)).to(device)
        fit_standardizer_and_head(model, spec, src_id, device,
                                  150, 1e-3, min(args.batch, 64), args.seed)
        return model
    train_or_load_source(args.backbone, args.dataset, src_id, CKPT_ROOT,
                         DATA_ROOT, device)
    return load_backbone(args.backbone, spec, CKPT_ROOT, args.dataset, src_id,
                         device, freeze=True)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    spec = get_spec(args.dataset)
    if args.domain_order == "adatime":
        order, _ = adatime_chain(args.dataset)
    else:
        order = [int(x) for x in args.domain_order.split(",")]
    labels = [spec.domain_label(d) for d in order]
    T = len(order)
    model = load_model(args, spec, order[0], device)

    A = np.full((T, T), np.nan)
    acc_embs, acc_lbls = [], []

    # Source train prototypes (full train, not 5% — NCM needs coverage)
    src_ld = torch.utils.data.DataLoader(
        spec.get_dataset(order[0], "train", DATA_ROOT), batch_size=args.batch)
    src_embs, src_lbls = extract_embs(model, src_ld, device)
    acc_embs.append(src_embs)
    acc_lbls.append(src_lbls)
    protos = fit_prototypes(np.concatenate(acc_embs), np.concatenate(acc_lbls),
                            spec.num_classes)
    A[0, 0] = eval_ncm(model, spec, order[0], protos, device, args.batch)

    for t in range(1, T):
        ds = spec.get_dataset(order[t], "train", DATA_ROOT)
        y_all = ds.y.numpy()
        idx = calib_indices(y_all, args.calib_frac, args.seed, calib_n=args.calib_n)
        calib_ld = torch.utils.data.DataLoader(
            torch.utils.data.Subset(ds, idx), batch_size=args.batch)
        ce, cl = extract_embs(model, calib_ld, device)
        acc_embs.append(ce)
        acc_lbls.append(cl)
        protos = fit_prototypes(np.concatenate(acc_embs), np.concatenate(acc_lbls),
                                spec.num_classes)
        for j in range(t + 1):
            A[t, j] = eval_ncm(model, spec, order[j], protos, device, args.batch)
        aa, bwt, f, rf, prf = compute_cl_metrics(A, t)
        print(f"Step {t} — {labels[t]:<10} AA={aa:.4f} F={f:.4f}")

    aa, bwt, f, rf, prf = compute_cl_metrics(A, T - 1)
    bb = args.tsfm or args.backbone
    calib_tag = f"n{args.calib_n}" if args.calib_n else f"c{int(args.calib_frac*100)}"
    tag = f"{args.dataset}_{bb}_{args.domain_order}_ncm_{calib_tag}_s{args.seed}"
    out_dir = RESULTS_ROOT / "ncm_baselines" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(A, index=labels, columns=labels).to_csv(out_dir / "f1_matrix.csv")
    pd.DataFrame([dict(method="ncm", AA=aa, BWT=bwt, F=f, RF=rf, PRF=prf,
                       seed=args.seed)]).to_csv(out_dir / "summary.csv", index=False)
    print(f"Final NCM: AA={aa:.4f} F={f:.4f} -> {out_dir}")


if __name__ == "__main__":
    main()
