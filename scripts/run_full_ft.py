"""Full fine-tune-per-domain control (catastrophic-forgetting baseline).

Sequentially fine-tunes the ENTIRE backbone (backbone + head) on each new
domain, starting from the source-trained weights. Unlike PACT, nothing is
frozen, so adapting to a new domain overwrites the shared weights and erases
earlier-domain performance -- the textbook catastrophic-forgetting curve.
Reports the same AA/BWT/F/RF/PRF and the F1 retention matrix for a direct,
apples-to-apples comparison with PACT.

Example:
  python scripts/run_full_ft.py --dataset fd --backbone tslanet
"""

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import _bootstrap  # noqa: F401
from _bootstrap import CKPT_ROOT, DATA_ROOT, RESULTS_ROOT

from src.data import get_spec
from src.data.datasets import adatime_chain
from src.method import compute_cl_metrics
from src.models.factory import BACKBONES, load_backbone
from src.training import train_or_load_source
from src.utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["fd", "hhar", "sleep"], default="fd")
    p.add_argument("--backbone", choices=BACKBONES, default="tslanet")
    p.add_argument("--domain_order", default="adatime")
    p.add_argument("--epochs", type=int, default=40)
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


def finetune(model, spec, did, device, epochs, lr, batch):
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    ld = torch.utils.data.DataLoader(
        spec.get_dataset(did, "train", DATA_ROOT), batch_size=batch, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in ld:
            opt.zero_grad()
            crit(model(xb.to(device)), yb.to(device)).backward()
            opt.step()


def main():
    args = parse_args()
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
    print(f"Step 0 — {labels[0]:<10} F1={A[0,0]:.4f}")

    for t in range(1, T):
        finetune(model, spec, order[t], device, args.epochs, args.lr, args.batch)
        for j in range(t + 1):
            A[t, j] = eval_f1(model, spec, order[j], device, args.batch)
        aa, bwt, f, rf, prf = compute_cl_metrics(A, t)
        print(f"Step {t} — {labels[t]:<10} AA={aa:.4f} BWT={bwt:+.4f} "
              f"F={f:.4f} RF={rf:.4f} PRF={prf:.4f}")

    tag = f"{args.dataset}_{args.backbone}_{args.domain_order}_fullft"
    out_dir = RESULTS_ROOT / "controls" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(A, index=labels, columns=labels).to_csv(out_dir / "fullft_f1_matrix.csv")
    aa, bwt, f, rf, prf = compute_cl_metrics(A, T - 1)
    pd.DataFrame([dict(control="full-ft", AA=aa, BWT=bwt, F=f, RF=rf, PRF=prf)]).to_csv(
        out_dir / "summary.csv", index=False)
    print(f"\nFinal full-FT: AA={aa:.4f} BWT={bwt:+.4f} F={f:.4f} RF={rf:.4f} PRF={prf:.4f}")
    print(f"Results -> {out_dir}")


if __name__ == "__main__":
    main()
