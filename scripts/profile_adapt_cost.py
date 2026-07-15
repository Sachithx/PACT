"""Profile adaptation cost (wall-clock + trainable params) on FD / MOMENT.

Measures the continual-learning loop only (after TSFM load + source-head fit),
so methods are compared on equal footing. Saves:
  results/cost_profile/fd_moment_c5.csv

Example:
  python scripts/profile_adapt_cost.py --device cuda
"""

import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

import _bootstrap  # noqa: F401
from _bootstrap import DATA_ROOT, RESULTS_ROOT

from src.data import get_spec
from src.data.calib import calib_indices
from src.data.datasets import adatime_chain
from src.method import PactConfig, compute_cl_metrics, run_sequential
from src.method.pact import (eval_uniform_mixture, extract_embs, no_adapt_f1,
                             train_adapters)
from src.models.adapters import make_adapter_bank
from src.models.concepts import assign_concepts, kmeans_numpy
from src.models.gate import SoftGateMLP
from src.models.tsfm import make_tsfm
from src.utils import get_device, set_seed


def count_params(modules):
    return sum(p.numel() for m in modules for p in m.parameters() if p.requires_grad)


def fit_source(model, spec, src_id, device, batch):
    ld = torch.utils.data.DataLoader(
        spec.get_dataset(src_id, "train", DATA_ROOT), batch_size=batch)
    X, y = extract_embs(model, ld, device)
    mean, std = X.mean(0), X.std(0)
    model.set_standardizer(mean, std)
    Xs = (X - mean) / np.clip(std, 1e-6, None)
    Xt = torch.from_numpy(Xs).to(device)
    yt = torch.from_numpy(y).long().to(device)
    opt = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, yt), batch_size=256, shuffle=True)
    model.head.train()
    for _ in range(150):
        for eb, yb in dl:
            opt.zero_grad(); crit(model.head(eb), yb).backward(); opt.step()
    model.freeze_head()
    return X, y


def kd_loss(old_logits, new_logits, T=2.0):
    return F.kl_div(F.log_softmax(new_logits / T, 1),
                    F.softmax(old_logits / T, 1),
                    reduction="batchmean") * (T * T)


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


def profile_head(method, model, spec, order, device, batch, calib_frac, seed,
                 head_ep=80, head_lr=1e-3, lam=1.0):
    T = len(order)
    source_head = copy.deepcopy(model.head).eval()
    A = np.full((T, T), np.nan)
    train_params = 0
    t0 = time.perf_counter()
    A[0, 0] = eval_f1(model, spec, order[0], device, batch)
    for t in range(1, T):
        for p in model.head.parameters():
            p.requires_grad_(True)
        train_params = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
        prev = copy.deepcopy(model.head).eval() if method == "lwf" else None
        ds = spec.get_dataset(order[t], "train", DATA_ROOT)
        y_all = ds.y.numpy()
        idx = calib_indices(y_all, calib_frac, seed)
        calib_ld = torch.utils.data.DataLoader(
            torch.utils.data.Subset(ds, idx), batch_size=batch)
        embs, lbls = extract_embs(model, calib_ld, device)
        Xt = torch.from_numpy(embs).to(device)
        yt = torch.from_numpy(lbls).long().to(device)
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
                if method == "lwf" and prev is not None:
                    with torch.no_grad():
                        old = prev(eb)
                    loss = loss + lam * kd_loss(old, logits)
                elif method == "srcdistill":
                    with torch.no_grad():
                        src = source_head(eb)
                    loss = loss + lam * kd_loss(src, logits)
                loss.backward(); opt.step()
        model.freeze_head()
        for j in range(t + 1):
            A[t, j] = eval_f1(model, spec, order[j], device, batch)
    elapsed = time.perf_counter() - t0
    aa, _, f, _, _ = compute_cl_metrics(A, T - 1)
    return dict(elapsed_s=elapsed, train_params=train_params, AA=aa, F=f)


def profile_noadapt_nogate(kind, model, spec, order, device, batch, cfg):
    frozen_head = model.head
    labels = [spec.domain_label(d) for d in order]
    T = len(order)
    src_ld = torch.utils.data.DataLoader(
        spec.get_dataset(order[0], "train", DATA_ROOT), batch_size=batch)
    src_embs, _ = extract_embs(model, src_ld, device)
    centroids, _ = kmeans_numpy(src_embs, cfg.k, seed=cfg.seed)
    test_embs, test_lbls = {}, {}
    for did, lbl in zip(order, labels):
        ld = torch.utils.data.DataLoader(
            spec.get_dataset(did, "test", DATA_ROOT), batch_size=batch)
        test_embs[lbl], test_lbls[lbl] = extract_embs(model, ld, device)
    noadapt_f1 = {lbl: no_adapt_f1(test_embs[lbl], test_lbls[lbl], frozen_head, device)
                  for lbl in labels}
    A_na = np.full((T, T), np.nan)
    A_ng = np.full((T, T), np.nan)
    A_na[0, 0] = A_ng[0, 0] = noadapt_f1[labels[0]]
    adapters_list = []
    train_params = 0
    t0 = time.perf_counter()
    for t in range(1, T):
        ds = spec.get_dataset(order[t], "train", DATA_ROOT)
        y_all = ds.y.numpy()
        idx = calib_indices(y_all, cfg.calib_frac, cfg.seed, calib_n=cfg.calib_n)
        calib_ld = torch.utils.data.DataLoader(
            torch.utils.data.Subset(ds, idx), batch_size=batch)
        calib_embs, calib_lbls = extract_embs(model, calib_ld, device)
        calib_lbls = y_all[idx]
        bank = train_adapters(calib_embs, calib_lbls, centroids, frozen_head, device, cfg)
        adapters_list.append(bank)
        if kind == "no-gate":
            train_params = count_params(adapters_list[-1])
        for j in range(t + 1):
            lbl_j = labels[j]
            A_na[t, j] = noadapt_f1[lbl_j]
            if kind == "no-gate":
                A_ng[t, j] = eval_uniform_mixture(
                    test_embs[lbl_j], test_lbls[lbl_j], centroids, adapters_list,
                    frozen_head, device, cfg.k)
    elapsed = time.perf_counter() - t0
    A = A_na if kind == "no-adapt" else A_ng
    aa, _, f, _, _ = compute_cl_metrics(A, T - 1)
    return dict(elapsed_s=elapsed, train_params=train_params, AA=aa, F=f)


def profile_pact(model, spec, order, device, cfg):
    t0 = time.perf_counter()
    res = run_sequential(model, spec, order, DATA_ROOT, device, cfg,
                         temperature=0.5, verbose=False)
    elapsed = time.perf_counter() - t0
    final = res["metrics"][-1]
    # per-step trainable: K adapters + K gates for current domain count
    T = len(order)
    dim, k = cfg.emb_dim, cfg.k
    adapter_p = dim * dim + dim
    gate_p = sum(SoftGateMLP(dim, 64, n_domains=t).net[-1].out_features
                 for t in range(1, T))  # rough upper bound on last step
    train_params = k * (adapter_p + (1024 * 64 + 64 + 64 * (T) + T))
    return dict(elapsed_s=elapsed, train_params=train_params,
                AA=final["AA"], F=final["F_all"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    spec = get_spec("fd")
    order, _ = adatime_chain("fd")
    batch = 64
    calib_frac = 0.05
    k = 16

    rows = []
    for method in ["head-ft", "lwf", "srcdistill", "no-adapt", "no-gate", "pact"]:
        print(f"\n=== Profiling {method} ===")
        set_seed(args.seed)
        model = make_tsfm("moment", spec.num_classes, device=str(device)).to(device)
        fit_source(model, spec, order[0], device, batch)
        if method in ("head-ft", "lwf", "srcdistill"):
            r = profile_head(method, model, spec, order, device, batch, calib_frac, args.seed)
        elif method in ("no-adapt", "no-gate"):
            cfg = PactConfig(k=k, emb_dim=model.emb_dim, calib_frac=calib_frac,
                             seed=args.seed, batch=batch)
            r = profile_noadapt_nogate(method, model, spec, order, device, batch, cfg)
        else:
            cfg = PactConfig(k=k, emb_dim=model.emb_dim, calib_frac=calib_frac,
                             seed=args.seed, batch=batch)
            r = profile_pact(model, spec, order, device, cfg)
        rows.append(dict(method=method, dataset="fd", tsfm="moment", **r))
        print(f"  time={r['elapsed_s']:.1f}s  params={r['train_params']:,}  "
              f"AA={r['AA']:.4f}  F={r['F']:.4f}")
        del model
        torch.cuda.empty_cache()

    out_dir = RESULTS_ROOT / "cost_profile"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "fd_moment_c5.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
