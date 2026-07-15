"""Continual-learning baselines on the SAME domain-incremental chain as PACT.

Baselines (all fine-tune the shared backbone as domains arrive, so they must
trade off plasticity vs. forgetting -- unlike PACT which freezes everything):
  full-ft    : fine-tune the whole backbone+head each domain (no anti-forgetting)
  head-ft    : freeze backbone, fine-tune only the linear head each domain
  ewc        : full-ft + Elastic Weight Consolidation (Fisher-weighted L2 to
               each past task's parameters)  [Kirkpatrick+ 2017]
  mas        : full-ft + Memory Aware Synapses (unsupervised importance)  [Aljundi+ 2018]
  lwf        : full-ft + Learning without Forgetting (distill the PRE-domain
               model on the new data)  [Li & Hoiem 2017]
  srcdistill : full-ft + distillation anchored to the frozen SOURCE model

Same retention-matrix / AA-BWT-F-RF-PRF reporting as run_sequential.py, so the
numbers drop straight into the chain comparison table. Baselines get the FULL
labelled train set of each domain (the generous setting); PACT uses only ~5%.

Example:
  python scripts/run_cl_baselines.py --dataset fd --method ewc
"""

import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

import _bootstrap  # noqa: F401
from _bootstrap import CKPT_ROOT, DATA_ROOT, RESULTS_ROOT

from src.data import get_spec
from src.data.datasets import adatime_chain
from src.method import compute_cl_metrics
from src.models.factory import load_backbone
from src.training import train_or_load_source
from src.utils import get_device, set_seed

METHODS = ["full-ft", "head-ft", "ewc", "mas", "lwf", "srcdistill"]
DEFAULT_LAMBDA = {"ewc": 1000.0, "mas": 1.0, "lwf": 1.0, "srcdistill": 1.0}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["fd", "hhar", "sleep"], default="fd")
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--backbone", default="tslanet")
    p.add_argument("--domain_order", default="adatime")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--calib_frac", type=float, default=0.05,
                   help="labelled fraction per domain (matched to PACT)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lam", type=float, default=None, help="reg / distill weight")
    p.add_argument("--fisher_n", type=int, default=512)
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


def kd_loss(old_logits, new_logits, T=2.0):
    return F.kl_div(F.log_softmax(new_logits / T, 1),
                    F.softmax(old_logits / T, 1),
                    reduction="batchmean") * (T * T)


def trainable(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def snapshot_params(model):
    return {n: p.detach().clone() for n, p in trainable(model)}


def compute_fisher(model, loader, device, n_max):
    fisher = {n: torch.zeros_like(p) for n, p in trainable(model)}
    model.eval()
    count = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        for i in range(xb.size(0)):
            model.zero_grad()
            loss = F.cross_entropy(model(xb[i:i + 1]), yb[i:i + 1])
            loss.backward()
            for n, p in trainable(model):
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            count += 1
            if count >= n_max:
                break
        if count >= n_max:
            break
    for n in fisher:
        fisher[n] /= max(count, 1)
    return _normalize_importance(fisher)


def compute_mas(model, loader, device, n_max):
    omega = {n: torch.zeros_like(p) for n, p in trainable(model)}
    model.eval()
    count = 0
    for xb, _ in loader:
        xb = xb.to(device)
        for i in range(xb.size(0)):
            model.zero_grad()
            out = model(xb[i:i + 1])
            (out.pow(2).sum()).backward()      # squared L2 of output
            for n, p in trainable(model):
                if p.grad is not None:
                    omega[n] += p.grad.detach().abs()
            count += 1
            if count >= n_max:
                break
        if count >= n_max:
            break
    for n in omega:
        omega[n] /= max(count, 1)
    return _normalize_importance(omega)


def _normalize_importance(imp):
    """Scale importance to unit global mean so lambda transfers across
    tasks/datasets (raw Fisher/MAS magnitudes vary by orders of magnitude)."""
    tot = sum(v.sum().item() for v in imp.values())
    n = sum(v.numel() for v in imp.values())
    m = tot / max(n, 1)
    if m > 0:
        for k in imp:
            imp[k] = imp[k] / m
    return imp


def reg_penalty(model, anchors, device):
    """anchors: list of (ref_params, importance) dicts."""
    loss = torch.zeros((), device=device)
    pdict = dict(trainable(model))
    for ref, imp in anchors:
        for n, p in pdict.items():
            if n in imp:
                loss = loss + (imp[n] * (p - ref[n]) ** 2).sum()
    return loss


def calib_subset(spec, did, calib_frac, seed):
    ds = spec.get_dataset(did, "train", DATA_ROOT)
    if calib_frac >= 1.0:
        return ds
    y = ds.y.numpy()
    idx, _ = next(StratifiedShuffleSplit(
        n_splits=1, test_size=1 - calib_frac, random_state=seed
    ).split(np.zeros(len(y)), y))
    return torch.utils.data.Subset(ds, idx)


def finetune_domain(model, calib_ds, device, args, method, lam,
                    anchors, source_model):
    if method == "head-ft":
        for n, p in model.named_parameters():
            p.requires_grad_(n.startswith("head"))
    else:
        for p in model.parameters():
            p.requires_grad_(True)

    prev_model = None
    if method == "lwf":
        prev_model = copy.deepcopy(model).eval()

    ld = torch.utils.data.DataLoader(calib_ds, batch_size=args.batch, shuffle=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    model.train()
    for _ in range(args.epochs):
        for xb, yb in ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = ce(logits, yb)
            if method == "lwf" and prev_model is not None:
                with torch.no_grad():
                    old = prev_model(xb)
                loss = loss + lam * kd_loss(old, logits)
            elif method == "srcdistill":
                with torch.no_grad():
                    src = source_model(xb)
                loss = loss + lam * kd_loss(src, logits)
            elif method in ("ewc", "mas") and anchors:
                loss = loss + (lam / 2.0) * reg_penalty(model, anchors, device)
            loss.backward()
            opt.step()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    spec = get_spec(args.dataset)
    lam = args.lam if args.lam is not None else DEFAULT_LAMBDA.get(args.method, 0.0)

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
    source_model = copy.deepcopy(model).eval() if args.method == "srcdistill" else None

    A = np.full((T, T), np.nan)
    A[0, 0] = eval_f1(model, spec, order[0], device, args.batch)
    print(f"[{args.method}] Step 0 — {labels[0]:<10} F1={A[0,0]:.4f}  (lam={lam})")

    # EWC/MAS need an importance anchor on the SOURCE before the first update
    # (the source->domain-1 step is what most damages source knowledge).
    imp_fn = compute_fisher if args.method == "ewc" else compute_mas
    anchors = []
    if args.method in ("ewc", "mas"):
        ld0 = torch.utils.data.DataLoader(
            calib_subset(spec, order[0], args.calib_frac, args.seed), batch_size=args.batch)
        anchors.append((snapshot_params(model), imp_fn(model, ld0, device, args.fisher_n)))

    for t in range(1, T):
        calib_ds = calib_subset(spec, order[t], args.calib_frac, args.seed)
        finetune_domain(model, calib_ds, device, args, args.method, lam,
                        anchors, source_model)
        if args.method in ("ewc", "mas"):
            ld = torch.utils.data.DataLoader(calib_ds, batch_size=args.batch)
            anchors.append((snapshot_params(model), imp_fn(model, ld, device, args.fisher_n)))

        for j in range(t + 1):
            A[t, j] = eval_f1(model, spec, order[j], device, args.batch)
        aa, bwt, f, rf, prf = compute_cl_metrics(A, t)
        print(f"[{args.method}] Step {t} — {labels[t]:<10} AA={aa:.4f} "
              f"BWT={bwt:+.4f} F={f:.4f} RF={rf:.4f} PRF={prf:.4f}")

    tag = f"{args.dataset}_{args.backbone}_{args.domain_order}_{args.method}_c{int(args.calib_frac*100)}"
    out_dir = RESULTS_ROOT / "cl_baselines" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(A, index=labels, columns=labels).to_csv(out_dir / "f1_matrix.csv")
    aa, bwt, f, rf, prf = compute_cl_metrics(A, T - 1)
    pd.DataFrame([dict(method=args.method, AA=aa, BWT=bwt, F=f, RF=rf, PRF=prf)]).to_csv(
        out_dir / "summary.csv", index=False)
    print(f"\n[{args.method}] Final: AA={aa:.4f} BWT={bwt:+.4f} F={f:.4f} "
          f"RF={rf:.4f} PRF={prf:.4f}\nResults -> {out_dir}")


if __name__ == "__main__":
    main()
