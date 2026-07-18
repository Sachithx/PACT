"""DER++ (Mammoth) on PACT domain-incremental chains, byte-matched buffer.

Faithful to baselines/mammoth/models/derpp.py:
  loss = CE(stream) + alpha * MSE(buf logits) + beta * CE(buf labels)

Protocol matches other CL baselines here: same chain + same calib budget.
Buffer size defaults to byte-matching PACT's embedding-key buffer on that chain.

Examples:
  python scripts/run_derpp_baseline.py --dataset fd --backbone tslanet \\
      --domain_order 2,1,0,3
  python scripts/run_derpp_baseline.py --dataset hhar --tsfm sundial \\
      --domain_order 6,0,8,2,5,1,3,4,7 --freeze_backbone
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import _bootstrap  # noqa: F401
from _bootstrap import CKPT_ROOT, DATA_ROOT, RESULTS_ROOT

from src.data import get_spec
from src.data.calib import calib_indices
from src.data.datasets import adatime_chain
from src.method import compute_cl_metrics
from src.method.derpp import (ReservoirBuffer, derpp_observe, forward_logits,
                              matched_buffer_size)
from src.models.factory import BACKBONES, load_backbone
from src.models.tsfm import TSFM_MODELS, make_tsfm
from src.training import train_or_load_source
from src.utils import get_device, set_seed

EMB_DIM = {"tslanet": 128, "moment": 1024, "sundial": 768}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["fd", "hhar", "sleep", "wisdm", "ucihar"], default="fd")
    p.add_argument("--backbone", choices=BACKBONES, default="tslanet")
    p.add_argument("--tsfm", choices=list(TSFM_MODELS), default=None)
    p.add_argument("--domain_order", default="adatime")
    p.add_argument("--calib_frac", type=float, default=0.05)
    p.add_argument("--calib_n", type=int, default=None)
    p.add_argument("--buffer_size", type=int, default=None,
                   help="If unset, byte-match PACT embedding buffer on this chain")
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head_ep", type=int, default=80)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--minibatch_size", type=int, default=32)
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Head-only DER++ (natural for frozen TSFMs)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=128)
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
        logits = forward_logits(model, xb.to(device))
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(yb.numpy())
    return f1_score(np.concatenate(ys), np.concatenate(ps),
                    average="macro", zero_division=0)


def estimate_buffer_size(spec, order, emb_dim, calib_frac, calib_n, seed):
    n_src = len(spec.get_dataset(order[0], "train", DATA_ROOT))
    n_calib = 0
    for did in order[1:]:
        ds = spec.get_dataset(did, "train", DATA_ROOT)
        n_calib += len(calib_indices(ds.y.numpy(), calib_frac, seed, calib_n=calib_n))
    return matched_buffer_size(emb_dim, n_src, n_calib,
                               spec.num_channels, spec.seq_len, spec.num_classes)


def load_model(args, spec, src_id, device):
    freeze = bool(args.tsfm) or args.freeze_backbone
    if args.tsfm:
        from run_sequential_tsfm import fit_standardizer_and_head
        model = make_tsfm(args.tsfm, spec.num_classes, device=str(device)).to(device)
        fit_standardizer_and_head(model, spec, src_id, device,
                                  150, 1e-3, min(args.batch, 64), args.seed)
        for n, p in model.named_parameters():
            p.requires_grad_(n.startswith("head"))
        return model, True
    train_or_load_source(args.backbone, args.dataset, src_id, CKPT_ROOT,
                         DATA_ROOT, device)
    model = load_backbone(args.backbone, spec, CKPT_ROOT, args.dataset, src_id,
                          device, freeze=freeze)
    if freeze:
        for n, p in model.named_parameters():
            p.requires_grad_(n.startswith("head"))
    else:
        for p in model.parameters():
            p.requires_grad_(True)
    return model, freeze


def train_domain_derpp(model, buffer, calib_ds, device, args, freeze_backbone):
    epochs = args.head_ep if (args.tsfm or freeze_backbone) else args.epochs
    lr = args.head_lr if (args.tsfm or freeze_backbone) else args.lr
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    ld = torch.utils.data.DataLoader(calib_ds, batch_size=args.batch, shuffle=True)
    for _ in range(epochs):
        for xb, yb in ld:
            xb, yb = xb.to(device), yb.to(device)
            derpp_observe(model, opt, ce, buffer, xb, yb,
                          args.alpha, args.beta, args.minibatch_size,
                          freeze_backbone=freeze_backbone)


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

    bb_key = args.tsfm or args.backbone
    emb_dim = EMB_DIM.get(bb_key, getattr(spec, "emb_dim", 128))
    if args.buffer_size is None:
        # refine emb_dim after model load for TSFMs
        pass

    model, freeze = load_model(args, spec, order[0], device)
    emb_dim = int(getattr(model, "emb_dim", emb_dim))
    buf_size = args.buffer_size or estimate_buffer_size(
        spec, order, emb_dim, args.calib_frac, args.calib_n, args.seed)

    example_shape = (spec.num_channels, spec.seq_len)
    buffer = ReservoirBuffer(buf_size, example_shape, spec.num_classes, device)

    calib_tag = f"n{args.calib_n}" if args.calib_n else f"c{int(args.calib_frac * 100)}"
    print(f"DER++  dataset={args.dataset}  model={bb_key}  T={T}  "
          f"buffer_size={buf_size}  alpha={args.alpha}  beta={args.beta}  "
          f"freeze={freeze}  calib={calib_tag}")

    # Seed buffer + source performance from source train (full) then evaluate.
    # Source step: fit/eval like other baselines (step 0 = source only).
    A = np.full((T, T), np.nan)
    A[0, 0] = eval_f1(model, spec, order[0], device, args.batch)
    print(f"[derpp] Step 0 — {labels[0]:<10} F1={A[0,0]:.4f}")

    # Populate buffer from a source calib slice so domain-1 already has dark knowledge.
    src_calib = calib_subset(spec, order[0], args.calib_frac, args.seed, args.calib_n)
    src_ld = torch.utils.data.DataLoader(src_calib, batch_size=args.batch, shuffle=True)
    model.eval()
    with torch.no_grad():
        for xb, yb in src_ld:
            xb, yb = xb.to(device), yb.to(device)
            logits = forward_logits(model, xb)
            buffer.add_data(xb, yb, logits.data)

    for t in range(1, T):
        calib_ds = calib_subset(spec, order[t], args.calib_frac, args.seed, args.calib_n)
        train_domain_derpp(model, buffer, calib_ds, device, args, freeze)
        for j in range(t + 1):
            A[t, j] = eval_f1(model, spec, order[j], device, args.batch)
        aa, bwt, f, rf, prf = compute_cl_metrics(A, t)
        print(f"[derpp] Step {t} — {labels[t]:<10} AA={aa:.4f} BWT={bwt:+.4f} "
              f"F={f:.4f} RF={rf:.4f}  |buf|={len(buffer)}")

    tag = (f"{args.dataset}_{bb_key}_{args.domain_order}_derpp"
           f"_{calib_tag}_b{buf_size}_s{args.seed}")
    out_dir = RESULTS_ROOT / "derpp_baselines" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(A, index=labels, columns=labels).to_csv(out_dir / "f1_matrix.csv")
    aa, bwt, f, rf, prf = compute_cl_metrics(A, T - 1)
    pd.DataFrame([dict(method="derpp", AA=aa, BWT=bwt, F=f, RF=rf, PRF=prf,
                       buffer_size=buf_size, buffer_nbytes=buffer.nbytes(),
                       alpha=args.alpha, beta=args.beta,
                       freeze_backbone=freeze)]).to_csv(
        out_dir / "summary.csv", index=False)
    meta = dict(buffer_size=buf_size, emb_dim=emb_dim, order=order,
                alpha=args.alpha, beta=args.beta)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[derpp] Final: AA={aa:.4f} F={f:.4f} RF={rf:.4f}\nResults -> {out_dir}")


if __name__ == "__main__":
    main()
