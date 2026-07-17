"""Chain-compatible external baselines on frozen TSFM embeddings.

Methods (see src/method/chain_baselines.py):
  sprompts    — B3 S-Prompts-style domain heads + KNN routing
  dualprompt  — B12 DualPrompt / L2P-style prompts on embeddings
  eata        — B10 EATA-style entropy minimization
  cotta       — B10 CoTTA-style teacher consistency

Example:
  python scripts/run_chain_baselines.py --dataset fd --method sprompts --tsfm moment
"""

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

import _bootstrap  # noqa: F401
from _bootstrap import DATA_ROOT, RESULTS_ROOT

from run_sequential_tsfm import fit_standardizer_and_head
from src.data import DATASET_NAMES, get_spec
from src.data.calib import calib_indices
from src.data.datasets import adatime_chain
from src.method import compute_cl_metrics
from src.method.chain_baselines import forward_logits, make_head, register_domain
from src.method.pact import extract_embs
from src.models.tsfm import TSFM_MODELS, make_tsfm
from src.utils import get_device, set_seed

METHODS = ["sprompts", "dualprompt", "eata", "cotta"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASET_NAMES, default="fd")
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--tsfm", choices=list(TSFM_MODELS), default="moment")
    p.add_argument("--domain_order", default="adatime")
    p.add_argument("--calib_frac", type=float, default=0.05)
    p.add_argument("--calib_n", type=int, default=None)
    p.add_argument("--head_ep", type=int, default=80)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default=None)
    return p.parse_args()


def calib_subset(spec, did, calib_frac, seed, calib_n=None):
    ds = spec.get_dataset(did, "train", DATA_ROOT)
    if calib_frac >= 1.0 and calib_n is None:
        return ds
    idx = calib_indices(ds.y.numpy(), calib_frac, seed, calib_n=calib_n)
    return torch.utils.data.Subset(ds, idx)


@torch.no_grad()
def eval_f1(model, head, spec, did, device, batch, method):
    model.eval()
    head.eval()
    ld = torch.utils.data.DataLoader(
        spec.get_dataset(did, "test", DATA_ROOT), batch_size=batch)
    ys, ps = [], []
    for xb, yb in ld:
        z = model.extract(xb.to(device))
        logits = forward_logits(head, z, method)
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(yb.numpy())
    return f1_score(np.concatenate(ys), np.concatenate(ps),
                    average="macro", zero_division=0)


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
    calib_tag = f"n{args.calib_n}" if args.calib_n else f"c{int(args.calib_frac * 100)}"

    print(f"Chain baseline={args.method}  tsfm={args.tsfm}  dataset={args.dataset}  "
          f"T={T}  calib={calib_tag}  seed={args.seed}")

    model = make_tsfm(args.tsfm, spec.num_classes, device=str(device)).to(device)
    fit_standardizer_and_head(model, spec, order[0], device,
                              150, 1e-3, args.batch, args.seed)

    head = make_head(args.method, model.emb_dim, spec.num_classes, max_domains=T).to(device)

    A = np.full((T, T), np.nan)

    for t in range(T):
        calib_ds = calib_subset(spec, order[t], args.calib_frac, args.seed, args.calib_n)
        calib_ld = torch.utils.data.DataLoader(calib_ds, batch_size=args.batch)
        embs, lbls = extract_embs(model, calib_ld, device)
        z = torch.from_numpy(embs).to(device)
        y = torch.from_numpy(lbls).long().to(device)
        register_domain(head, z, y, device, args.method,
                        epochs=args.head_ep, lr=args.head_lr, domain_idx=t)
        for j in range(t + 1):
            A[t, j] = eval_f1(model, head, spec, order[j], device, args.batch, args.method)
        aa, bwt, f, rf, prf = compute_cl_metrics(A, t)
        print(f"[{args.method}] Step {t} — {labels[t]:<10} AA={aa:.4f} "
              f"BWT={bwt:+.4f} F={f:.4f} RF={rf:.4f}")

    tag = f"{args.dataset}_{args.tsfm}_{args.domain_order}_{args.method}_{calib_tag}_s{args.seed}"
    out_dir = RESULTS_ROOT / "chain_baselines" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(A, index=labels, columns=labels).to_csv(out_dir / "f1_matrix.csv")
    aa, bwt, f, rf, prf = compute_cl_metrics(A, T - 1)
    pd.DataFrame([dict(method=args.method, AA=aa, BWT=bwt, F=f, RF=rf, PRF=prf,
                       seed=args.seed)]).to_csv(out_dir / "summary.csv", index=False)
    print(f"Final: AA={aa:.4f}  F={f:.4f}  -> {out_dir}")


if __name__ == "__main__":
    main()
