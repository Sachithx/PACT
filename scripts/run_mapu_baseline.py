"""MAPU-style source-free head adaptation on frozen TSFM embeddings (B4).

Uses calibration embeddings only: CE + entropy regularization + prototype alignment
inspired by MAPU_SFDA_TS (mohamedr002/MAPU_SFDA_TS), without the full ADATIME trainer.
"""

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASET_NAMES, default="fd")
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
    idx = calib_indices(ds.y.numpy(), calib_frac, seed, calib_n=calib_n)
    return torch.utils.data.Subset(ds, idx)


def prototype_loss(z, y, num_classes):
    loss = z.new_zeros(())
    for c in range(num_classes):
        m = y == c
        if m.any():
            proto = z[m].mean(0)
            loss = loss + F.mse_loss(z[m], proto.expand_as(z[m]))
    return loss


def train_mapu_head(head, z, y, num_classes, epochs=80, lr=1e-3):
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(z, y), batch_size=256, shuffle=True)
    head.train()
    for _ in range(epochs):
        for zb, yb in dl:
            opt.zero_grad()
            logits = head(zb)
            ent = -(F.softmax(logits, 1) * F.log_softmax(logits, 1)).sum(1).mean()
            loss = ce(logits, yb) + 0.1 * ent + 0.05 * prototype_loss(zb, yb, num_classes)
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
    order, _ = adatime_chain(args.dataset) if args.domain_order == "adatime" else (
        [int(x) for x in args.domain_order.split(",")], {})
    labels = [spec.domain_label(d) for d in order]
    T = len(order)
    calib_tag = f"n{args.calib_n}" if args.calib_n else f"c{int(args.calib_frac * 100)}"

    model = make_tsfm(args.tsfm, spec.num_classes, device=str(device)).to(device)
    fit_standardizer_and_head(model, spec, order[0], device,
                              150, 1e-3, args.batch, args.seed)
    head = nn.Linear(model.emb_dim, spec.num_classes).to(device)

    A = np.full((T, T), np.nan)
    for t in range(T):
        calib_ld = torch.utils.data.DataLoader(
            calib_subset(spec, order[t], args.calib_frac, args.seed, args.calib_n),
            batch_size=args.batch)
        embs, lbls = extract_embs(model, calib_ld, device)
        z = torch.from_numpy(embs).to(device)
        y = torch.from_numpy(lbls).long().to(device)
        train_mapu_head(head, z, y, spec.num_classes, args.head_ep, args.head_lr)
        for j in range(t + 1):
            A[t, j] = eval_f1(model, head, spec, order[j], device, args.batch)
        aa, bwt, f, rf, prf = compute_cl_metrics(A, t)
        print(f"[mapu] Step {t} — {labels[t]:<10} AA={aa:.4f} F={f:.4f}")

    tag = f"{args.dataset}_{args.tsfm}_{args.domain_order}_mapu_{calib_tag}_s{args.seed}"
    out_dir = RESULTS_ROOT / "mapu_baselines" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(A, index=labels, columns=labels).to_csv(out_dir / "f1_matrix.csv")
    aa, bwt, f, rf, prf = compute_cl_metrics(A, T - 1)
    pd.DataFrame([dict(method="mapu", AA=aa, BWT=bwt, F=f, RF=rf, PRF=prf,
                       seed=args.seed)]).to_csv(out_dir / "summary.csv", index=False)
    print(f"Final MAPU: AA={aa:.4f}  -> {out_dir}")


if __name__ == "__main__":
    main()
