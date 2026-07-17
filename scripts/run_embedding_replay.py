"""Memory-matched embedding-replay baselines on the domain-incremental chain.

These baselines receive exactly the same retained source-train and calibration
embeddings that PACT uses for gate refitting, but store no raw time series.

Methods:
  emb-replay-head  : refit the linear classifier head on the accumulated buffer
  global-adapter   : K=1 residual transport per domain + the same learned gate
                     (no chart decomposition; memory-matched gate buffer)

Example:
  python scripts/run_embedding_replay.py --dataset fd --method emb-replay-head
  python scripts/run_embedding_replay.py --dataset fd --method global-adapter --tsfm moment
"""

import argparse
import copy
import time

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
from src.method import PactConfig, compute_cl_metrics, run_sequential
from src.method.pact import extract_embs
from src.models.factory import BACKBONES, load_backbone
from src.models.tsfm import TSFM_MODELS, make_tsfm
from src.training import train_or_load_source
from src.utils import get_device, set_seed

METHODS = ["emb-replay-head", "global-adapter"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["fd", "hhar", "sleep"], default="fd")
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--backbone", choices=BACKBONES, default="tslanet")
    p.add_argument("--tsfm", choices=list(TSFM_MODELS), default=None,
                   help="If set, use a frozen TSFM instead of --backbone")
    p.add_argument("--domain_order", default="adatime")
    p.add_argument("--calib_frac", type=float, default=0.05)
    p.add_argument("--head_ep", type=int, default=200)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--device", default=None)
    return p.parse_args()


@torch.no_grad()
def eval_head_f1(head, embs_np, labels_np, device):
    logits = head(torch.from_numpy(embs_np).to(device))
    preds = logits.argmax(1).cpu().numpy()
    return f1_score(labels_np, preds, average="macro", zero_division=0)


def run_emb_replay_head(model, spec, order, device, args):
    frozen_head = model.head
    labels = [spec.domain_label(d) for d in order]
    T = len(order)

    src_ld = torch.utils.data.DataLoader(
        spec.get_dataset(order[0], "train", DATA_ROOT), batch_size=args.batch)
    src_embs, src_lbls = extract_embs(model, src_ld, device)

    test_embs, test_lbls = {}, {}
    for did, lbl in zip(order, labels):
        ld = torch.utils.data.DataLoader(
            spec.get_dataset(did, "test", DATA_ROOT), batch_size=args.batch)
        test_embs[lbl], test_lbls[lbl] = extract_embs(model, ld, device)

    A = np.full((T, T), np.nan)
    A[0, 0] = eval_head_f1(frozen_head, test_embs[labels[0]], test_lbls[labels[0]], device)

    accumulated = [(src_embs, src_lbls)]
    head = copy.deepcopy(frozen_head)
    crit = nn.CrossEntropyLoss()

    for t in range(1, T):
        ds = spec.get_dataset(order[t], "train", DATA_ROOT)
        y_all = ds.y.numpy()
        idx = calib_indices(y_all, args.calib_frac, args.seed)
        calib_ld = torch.utils.data.DataLoader(
            torch.utils.data.Subset(ds, idx), batch_size=args.batch)
        calib_embs, calib_lbls = extract_embs(model, calib_ld, device)
        accumulated.append((calib_embs, calib_lbls))

        X = np.concatenate([e for e, _ in accumulated]).astype(np.float32)
        y = np.concatenate([l for _, l in accumulated]).astype(np.int64)
        Xt = torch.from_numpy(X).to(device)
        yt = torch.from_numpy(y).long().to(device)
        for p in head.parameters():
            p.requires_grad_(True)
        opt = torch.optim.AdamW(head.parameters(), lr=args.head_lr, weight_decay=1e-4)
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xt, yt),
            batch_size=min(args.batch, len(X)), shuffle=True)
        head.train()
        for _ in range(args.head_ep):
            for xb, yb in dl:
                opt.zero_grad()
                crit(head(xb), yb).backward()
                opt.step()
        head.eval()
        for p in head.parameters():
            p.requires_grad_(False)

        for j in range(t + 1):
            A[t, j] = eval_head_f1(head, test_embs[labels[j]], test_lbls[labels[j]], device)

    return A, labels


def run_global_adapter(model, spec, order, device, args):
    cfg = PactConfig(k=1, emb_dim=model.emb_dim, calib_frac=args.calib_frac,
                     seed=args.seed, batch=args.batch, gate_kind="learned")
    res = run_sequential(model, spec, order, DATA_ROOT, device, cfg,
                         temperature=args.temperature, verbose=False)
    return res["A_soft"], res["labels"]


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

    print(f"Embedding-replay baseline: {args.method}  dataset={args.dataset}  "
          f"seed={args.seed}  calib={args.calib_frac*100:.0f}%")

    t0 = time.perf_counter()
    model = load_model(args, spec, order[0], device)
    if args.method == "emb-replay-head":
        A, labels = run_emb_replay_head(model, spec, order, device, args)
    else:
        A, labels = run_global_adapter(model, spec, order, device, args)
    elapsed = time.perf_counter() - t0

    T = len(order)
    aa, bwt, f, rf, prf = compute_cl_metrics(A, T - 1)
    tag = (f"{args.dataset}_{args.tsfm or args.backbone}_{args.method}"
           f"_c{int(args.calib_frac*100)}_s{args.seed}")
    out_dir = RESULTS_ROOT / "embedding_replay" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(A, index=labels, columns=labels).to_csv(out_dir / "soft_f1_matrix.csv")
    pd.DataFrame([dict(method=args.method, AA=aa, BWT=bwt, F=f, RF=rf, PRF=prf,
                       seconds=elapsed, seed=args.seed)]).to_csv(
        out_dir / "metrics.csv", index=False)
    print(f"Final: AA={aa:.4f}  BWT={bwt:+.4f}  F={f:.4f}  RF={rf:.4f}  "
          f"({elapsed:.1f}s)")
    print(f"Results -> {out_dir}")


if __name__ == "__main__":
    main()
