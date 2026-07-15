"""Continual head-level baselines on a FROZEN TSFM (Table 2B).

Same domain-incremental chain and 5% calibration budget as PACT-on-MOMENT.
The encoder stays frozen; only the linear head is updated each domain:
  head-ft    : CE on new-domain calibration embeddings
  lwf        : CE + distillation from the PREVIOUS head
  srcdistill : CE + distillation from the frozen SOURCE head
  ewc        : CE + Fisher-weighted L2 to each past domain's head [Kirkpatrick+ 2017]
  mas        : CE + Memory-Aware-Synapses importance-weighted L2 [Aljundi+ 2018]

Full backbone fine-tuning is omitted (prohibitively expensive for a 341M-param
frozen encoder and contradicts the frozen-TSFM story); EWC/MAS here regularize
only the linear head's parameters (the only trainable part), with importance
computed on that same domain's calibration embeddings -- no extra encoder
forward passes beyond what head-ft/lwf/srcdistill already do.

Example:
  python scripts/run_cl_baselines_tsfm.py --dataset fd --method ewc
"""

import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
import _bootstrap  # noqa: F401
from _bootstrap import DATA_ROOT, RESULTS_ROOT

from src.data import DATASET_NAMES, get_spec
from src.data.calib import calib_indices
from src.data.datasets import adatime_chain
from src.method import compute_cl_metrics
from src.method.pact import extract_embs
from src.models.tsfm import TSFM_MODELS, make_tsfm
from src.utils import get_device, set_seed

METHODS = ["head-ft", "lwf", "srcdistill", "ewc", "mas"]
DEFAULT_LAMBDA = {"lwf": 1.0, "srcdistill": 1.0, "ewc": 1000.0, "mas": 1.0}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASET_NAMES, default="fd")
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--tsfm", choices=list(TSFM_MODELS), default="moment")
    p.add_argument("--domain_order", default="adatime")
    p.add_argument("--calib_frac", type=float, default=0.05)
    p.add_argument("--calib_n", type=int, default=None,
                   help="Fixed calibration label budget (overrides calib_frac)")
    p.add_argument("--head_ep", type=int, default=80)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--lam", type=float, default=None)
    p.add_argument("--fisher_n", type=int, default=512,
                   help="max calibration samples used for Fisher/MAS importance (ewc/mas)")
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


def trainable_head(head):
    return [(n, p) for n, p in head.named_parameters() if p.requires_grad]


def snapshot_params(head):
    return {n: p.detach().clone() for n, p in trainable_head(head)}


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


def compute_fisher_head(head, Xt, yt, n_max):
    """Diagonal Fisher of the head's CE loss w.r.t. its own params, estimated
    on (already-extracted, frozen-encoder) calibration embeddings."""
    fisher = {n: torch.zeros_like(p) for n, p in trainable_head(head)}
    head.eval()
    n = min(n_max, Xt.size(0))
    for i in range(n):
        head.zero_grad()
        loss = F.cross_entropy(head(Xt[i:i + 1]), yt[i:i + 1])
        loss.backward()
        for name, p in trainable_head(head):
            if p.grad is not None:
                fisher[name] += p.grad.detach() ** 2
    for k in fisher:
        fisher[k] /= max(n, 1)
    return _normalize_importance(fisher)


def compute_mas_head(head, Xt, n_max):
    """MAS importance: gradient magnitude of the squared-L2 output norm
    w.r.t. head params (unsupervised, no labels needed)."""
    omega = {n: torch.zeros_like(p) for n, p in trainable_head(head)}
    head.eval()
    n = min(n_max, Xt.size(0))
    for i in range(n):
        head.zero_grad()
        out = head(Xt[i:i + 1])
        (out.pow(2).sum()).backward()
        for name, p in trainable_head(head):
            if p.grad is not None:
                omega[name] += p.grad.detach().abs()
    for k in omega:
        omega[k] /= max(n, 1)
    return _normalize_importance(omega)


def reg_penalty(head, anchors, device):
    """anchors: list of (ref_params, importance) dicts, one per past domain."""
    loss = torch.zeros((), device=device)
    pdict = dict(trainable_head(head))
    for ref, imp in anchors:
        for n, p in pdict.items():
            if n in imp:
                loss = loss + (imp[n] * (p - ref[n]) ** 2).sum()
    return loss


def fit_source(model, spec, src_id, device, batch, head_ep, head_lr):
    ld = torch.utils.data.DataLoader(
        spec.get_dataset(src_id, "train", DATA_ROOT), batch_size=batch)
    X, y = extract_embs(model, ld, device)
    mean, std = X.mean(0), X.std(0)
    model.set_standardizer(mean, std)
    Xs = (X - mean) / np.clip(std, 1e-6, None)
    Xt = torch.from_numpy(Xs).to(device)
    yt = torch.from_numpy(y).long().to(device)
    opt = torch.optim.AdamW(model.head.parameters(), lr=head_lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, yt), batch_size=256, shuffle=True)
    model.head.train()
    for _ in range(head_ep):
        for eb, yb in dl:
            opt.zero_grad(); crit(model.head(eb), yb).backward(); opt.step()
    return copy.deepcopy(model.head).eval(), Xt, yt


def train_head_step(model, calib_ds, device, batch, head_ep, head_lr, method, lam,
                    prev_head, source_head, anchors=None):
    ld = torch.utils.data.DataLoader(calib_ds, batch_size=batch)
    embs, lbls = extract_embs(model, ld, device)
    Xt = torch.from_numpy(embs).to(device)
    yt = torch.from_numpy(lbls).long().to(device)
    for p in model.head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.head.parameters(), lr=head_lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, yt), batch_size=256, shuffle=True)
    model.head.train()
    for _ in range(head_ep):
        for eb, yb in dl:
            opt.zero_grad()
            logits = model.head(eb)
            loss = ce(logits, yb)
            if method == "lwf" and prev_head is not None:
                with torch.no_grad():
                    old = prev_head(eb)
                loss = loss + lam * kd_loss(old, logits)
            elif method == "srcdistill" and source_head is not None:
                with torch.no_grad():
                    src = source_head(eb)
                loss = loss + lam * kd_loss(src, logits)
            elif method in ("ewc", "mas") and anchors:
                loss = loss + (lam / 2.0) * reg_penalty(model.head, anchors, device)
            loss.backward(); opt.step()
    return Xt, yt


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    spec = get_spec(args.dataset)
    lam = args.lam if args.lam is not None else DEFAULT_LAMBDA.get(args.method, 0.0)

    if args.domain_order == "adatime":
        order, st = adatime_chain(args.dataset)
        print(f"AdaTime chain: {st['n_domains']} domains")
    elif args.domain_order:
        order = [int(x) for x in args.domain_order.split(",")]
    else:
        order = list(range(spec.n_domains))
    labels = [spec.domain_label(d) for d in order]
    T = len(order)

    calib_tag = f"n{args.calib_n}" if args.calib_n else f"c{int(args.calib_frac*100)}"
    print(f"Device={device}  tsfm={args.tsfm}(frozen)  method={args.method}  "
          f"calib={calib_tag}  lam={lam}")
    model = make_tsfm(args.tsfm, spec.num_classes, device=str(device)).to(device)
    source_head, Xt0, yt0 = fit_source(model, spec, order[0], device, args.batch,
                                       args.head_ep, args.head_lr)
    anchor_head = source_head if args.method == "srcdistill" else None

    # EWC/MAS need an importance anchor on the SOURCE before the first update
    # (the source->domain-1 step is what most damages source knowledge).
    imp_fn = compute_fisher_head if args.method == "ewc" else compute_mas_head
    anchors = []
    if args.method == "ewc":
        anchors.append((snapshot_params(model.head), imp_fn(model.head, Xt0, yt0, args.fisher_n)))
    elif args.method == "mas":
        anchors.append((snapshot_params(model.head), imp_fn(model.head, Xt0, args.fisher_n)))

    A = np.full((T, T), np.nan)
    A[0, 0] = eval_f1(model, spec, order[0], device, args.batch)
    print(f"[{args.method}] Step 0 — {labels[0]:<10} F1={A[0,0]:.4f}")

    for t in range(1, T):
        prev_head = copy.deepcopy(model.head).eval() if args.method == "lwf" else None
        calib_ds = calib_subset(spec, order[t], args.calib_frac, args.seed, args.calib_n)
        Xt, yt = train_head_step(model, calib_ds, device, args.batch, args.head_ep,
                                 args.head_lr, args.method, lam, prev_head, anchor_head,
                                 anchors)
        if args.method == "ewc":
            anchors.append((snapshot_params(model.head), imp_fn(model.head, Xt, yt, args.fisher_n)))
        elif args.method == "mas":
            anchors.append((snapshot_params(model.head), imp_fn(model.head, Xt, args.fisher_n)))
        for j in range(t + 1):
            A[t, j] = eval_f1(model, spec, order[j], device, args.batch)
        aa, bwt, f, rf, prf = compute_cl_metrics(A, t)
        print(f"[{args.method}] Step {t} — {labels[t]:<10} AA={aa:.4f} "
              f"BWT={bwt:+.4f} F={f:.4f} RF={rf:.4f}")

    tag = (f"{args.dataset}_{args.tsfm}_{args.domain_order}_{args.method}"
           f"_{calib_tag}")
    out_dir = RESULTS_ROOT / "cl_baselines_tsfm" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(A, index=labels, columns=labels).to_csv(out_dir / "f1_matrix.csv")
    aa, bwt, f, rf, prf = compute_cl_metrics(A, T - 1)
    pd.DataFrame([dict(method=args.method, AA=aa, BWT=bwt, F=f, RF=rf, PRF=prf)]).to_csv(
        out_dir / "summary.csv", index=False)
    print(f"\n[{args.method}] Final: AA={aa:.4f} F={f:.4f} RF={rf:.4f}\nResults -> {out_dir}")


if __name__ == "__main__":
    main()
