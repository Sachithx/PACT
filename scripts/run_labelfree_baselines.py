"""Genuinely label-free baselines on frozen TSFM embeddings: TENT and
source-free MAPU. Unlike scripts/run_mapu_baseline.py (which is a
label-supervised variant, kept in the paper with a dagger footnote), these
methods see NO calibration labels on target domains -- only the unlabeled
inputs of the same calibration-sized subset every other baseline uses (same
data budget, no label signal).

  tent      -- Wang et al. 2021: adapt the head by minimizing prediction
               entropy on unlabeled target batches. No pseudo-labels, no
               prototypes -- the purest test-time-adaptation baseline.
  mapu-free -- source-free MAPU: entropy minimization + self-training
               prototype alignment using the model's OWN pseudo-labels
               (argmax predictions), not ground truth. Faithful to the
               source-free premise of MAPU_SFDA_TS (labeled source, unlabeled
               target stream).

The source domain (step 0) is always labeled -- source-free DA assumes a
labeled source and only withholds labels on the target stream. Step 0 reuses
the standard source-fitted head (run_sequential_tsfm.fit_standardizer_and_head
on the FULL source train set), exactly like every other baseline in this repo.

Example:
  python scripts/run_labelfree_baselines.py --dataset fd --method tent --tsfm sundial
  python scripts/run_labelfree_baselines.py --dataset fd --method mapu-free --tsfm sundial
"""

import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

import _bootstrap  # noqa: F401
from _bootstrap import DATA_ROOT, RESULTS_ROOT

from run_sequential_tsfm import fit_standardizer_and_head
from src.data import DATASET_NAMES, get_spec
from src.data.calib import calib_indices
from src.data.datasets import adatime_chain
from src.method import compute_cl_metrics
from src.method.pact import extract_embs
from src.models.tsfm import TSFM_MODELS, make_tsfm
from src.utils import get_device, set_seed

METHODS = ["tent", "mapu-free"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASET_NAMES, default="fd")
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--tsfm", choices=list(TSFM_MODELS), default="moment")
    p.add_argument("--domain_order", default="adatime")
    p.add_argument("--calib_frac", type=float, default=0.05)
    p.add_argument("--calib_n", type=int, default=None,
                   help="Fixed unlabeled adaptation budget (overrides calib_frac); "
                        "same instance count as the labeled baselines, no labels used")
    p.add_argument("--head_ep", type=int, default=80)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--ent_weight", type=float, default=0.1)
    p.add_argument("--proto_weight", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default=None)
    return p.parse_args()


def unlabeled_calib_subset(spec, did, calib_frac, seed, calib_n=None):
    """Same index-selection as the labeled baselines (stratified by the
    hidden true labels, matching everyone else's instance budget) -- but the
    labels themselves are never passed to the adaptation loss below."""
    ds = spec.get_dataset(did, "train", DATA_ROOT)
    idx = calib_indices(ds.y.numpy(), calib_frac, seed, calib_n=calib_n)
    return torch.utils.data.Subset(ds, idx)


def entropy(logits):
    p = F.softmax(logits, 1)
    return -(p * F.log_softmax(logits, 1)).sum(1).mean()


def prototype_align_loss(head, z, pseudo_y, num_classes):
    """Pull each class's classifier weight row toward the mean embedding of
    samples currently pseudo-labeled that class (self-training / nearest-
    class-mean calibration). Embedding-space prototype loss can't move
    anything under a frozen encoder, so this is the gradient-effective
    analogue: it targets the one trainable component (the head) instead."""
    loss = z.new_zeros(())
    for c in range(num_classes):
        m = pseudo_y == c
        if m.any():
            proto = z[m].mean(0).detach()
            loss = loss + F.mse_loss(head.weight[c], proto)
    return loss


def adapt_head_labelfree(head, z, num_classes, method, epochs, lr, ent_weight, proto_weight):
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(z), batch_size=256, shuffle=True)
    head.train()
    for _ in range(epochs):
        for (zb,) in dl:
            opt.zero_grad()
            logits = head(zb)
            loss = ent_weight * entropy(logits)
            if method == "mapu-free":
                pseudo_y = logits.argmax(1).detach()
                loss = loss + proto_weight * prototype_align_loss(head, zb, pseudo_y, num_classes)
            loss.backward()
            opt.step()
    head.eval()


@torch.no_grad()
def eval_f1(model, head, spec, did, device, batch):
    model.eval()
    head.eval()
    ld = torch.utils.data.DataLoader(
        spec.get_dataset(did, "test", DATA_ROOT), batch_size=batch)
    ys, ps = [], []
    for xb, yb in ld:
        z = model.extract(xb.to(device))
        ps.append(head(z).argmax(1).cpu().numpy())
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

    print(f"Label-free baseline={args.method}  tsfm={args.tsfm}  dataset={args.dataset}  "
          f"T={T}  adapt_budget={calib_tag}  seed={args.seed}")

    model = make_tsfm(args.tsfm, spec.num_classes, device=str(device)).to(device)
    fit_standardizer_and_head(model, spec, order[0], device, 150, 1e-3, args.batch, args.seed)
    head = copy.deepcopy(model.head).to(device)  # labeled source head, step 0 reuses this as-is
    for p in head.parameters():
        p.requires_grad_(True)  # fit_standardizer_and_head froze the source model.head

    A = np.full((T, T), np.nan)
    A[0, 0] = eval_f1(model, head, spec, order[0], device, args.batch)
    print(f"[{args.method}] Step 0 — {labels[0]:<10} F1={A[0,0]:.4f}  (labeled source)")

    for t in range(1, T):
        calib_ds = unlabeled_calib_subset(spec, order[t], args.calib_frac, args.seed, args.calib_n)
        calib_ld = torch.utils.data.DataLoader(calib_ds, batch_size=args.batch)
        embs, _ = extract_embs(model, calib_ld, device)  # labels discarded, unlabeled target
        z = torch.from_numpy(embs).to(device)
        adapt_head_labelfree(head, z, spec.num_classes, args.method,
                             args.head_ep, args.head_lr, args.ent_weight, args.proto_weight)
        for j in range(t + 1):
            A[t, j] = eval_f1(model, head, spec, order[j], device, args.batch)
        aa, bwt, f, rf, prf = compute_cl_metrics(A, t)
        print(f"[{args.method}] Step {t} — {labels[t]:<10} AA={aa:.4f} "
              f"BWT={bwt:+.4f} F={f:.4f} RF={rf:.4f}")

    tag = f"{args.dataset}_{args.tsfm}_{args.domain_order}_{args.method}_{calib_tag}_s{args.seed}"
    out_dir = RESULTS_ROOT / "labelfree_baselines" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(A, index=labels, columns=labels).to_csv(out_dir / "f1_matrix.csv")
    aa, bwt, f, rf, prf = compute_cl_metrics(A, T - 1)
    pd.DataFrame([dict(method=args.method, AA=aa, BWT=bwt, F=f, RF=rf, PRF=prf,
                       seed=args.seed)]).to_csv(out_dir / "summary.csv", index=False)
    print(f"\n[{args.method}] Final: AA={aa:.4f}  F={f:.4f}  RF={rf:.4f}\nResults -> {out_dir}")


if __name__ == "__main__":
    main()
