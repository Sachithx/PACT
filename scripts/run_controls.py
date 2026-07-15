"""Control baselines that share PACT's frozen backbone + identical adapters, so
each isolates one design choice:

  * no-adapt     : frozen backbone + source linear head, NO adaptation at all
                   ("is the (foundation) model already domain-robust?"). Head is
                   frozen, so this is constant across the stream; low target
                   accuracy here = adaptation is necessary.
  * no-gate      : PACT's per-domain adapters but the gate is replaced by
                   UNIFORM mixture weights (no learned routing). Any drop vs the
                   full soft gate is exactly the gate's contribution
                   (interference avoidance / forgetting prevention).

Works for both small backbones (tslanet/patchtst/timesnet) and frozen TSFMs
(moment/chronos/chronos-bolt). Writes matrices + metrics to results/controls/.

Examples:
  python scripts/run_controls.py --dataset fd --backbone tslanet
  python scripts/run_controls.py --dataset fd --backbone moment
"""

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import _bootstrap  # noqa: F401
from _bootstrap import CKPT_ROOT, DATA_ROOT, RESULTS_ROOT

from src.data import DATASET_NAMES, get_spec
from src.data.calib import calib_indices
from src.data.datasets import adatime_chain
from src.method import PactConfig, compute_cl_metrics
from src.method.pact import (eval_uniform_mixture, extract_embs, no_adapt_f1,
                             train_adapters)
from src.models.concepts import assign_concepts, kmeans_numpy
from src.models.factory import BACKBONES
from src.models.tsfm import TSFM_MODELS, make_tsfm
from src.utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASET_NAMES, default="fd")
    p.add_argument("--backbone", default="tslanet",
                   help="small backbone (%s) or TSFM (%s)"
                        % (BACKBONES, list(TSFM_MODELS)))
    p.add_argument("--domain_order", default="adatime")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--calib_frac", type=float, default=0.05)
    p.add_argument("--calib_n", type=int, default=None,
                   help="Fixed calibration label budget (overrides calib_frac)")
    p.add_argument("--adapter_kind", choices=["identity", "lora"], default="identity")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--device", default=None)
    return p.parse_args()


def build_order(dataset, spec, spec_order):
    if spec_order == "adatime":
        order, st = adatime_chain(dataset)
        print(f"AdaTime chain: {st['n_domains']} domains, "
              f"{st['forward']}/{st['n_pairs']} pairs forward")
        return order
    if spec_order:
        return [int(x) for x in spec_order.split(",")]
    return list(range(spec.n_domains))


def fit_tsfm_head(model, spec, src_id, device, batch):
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


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    spec = get_spec(args.dataset)
    order = build_order(args.dataset, spec, args.domain_order)
    is_tsfm = args.backbone in TSFM_MODELS

    if is_tsfm:
        model = make_tsfm(args.backbone, spec.num_classes, device=str(device)).to(device)
        fit_tsfm_head(model, spec, order[0], device, 64 if is_tsfm else args.batch)
    else:
        from src.training import train_or_load_source
        from src.models.factory import load_backbone
        train_or_load_source(args.backbone, args.dataset, order[0], CKPT_ROOT,
                             DATA_ROOT, device)
        model = load_backbone(args.backbone, spec, CKPT_ROOT, args.dataset,
                              order[0], device, freeze=True)

    frozen_head = model.head
    labels = [spec.domain_label(d) for d in order]
    T = len(order)
    batch = 64 if is_tsfm else args.batch

    src_ld = torch.utils.data.DataLoader(
        spec.get_dataset(order[0], "train", DATA_ROOT), batch_size=batch)
    src_embs, _ = extract_embs(model, src_ld, device)
    centroids, _ = kmeans_numpy(src_embs, args.k, seed=args.seed)

    test_embs, test_lbls = {}, {}
    for did, lbl in zip(order, labels):
        ld = torch.utils.data.DataLoader(
            spec.get_dataset(did, "test", DATA_ROOT), batch_size=batch)
        test_embs[lbl], test_lbls[lbl] = extract_embs(model, ld, device)

    cfg = PactConfig(k=args.k, emb_dim=model.emb_dim, calib_frac=args.calib_frac,
                     calib_n=args.calib_n, seed=args.seed, batch=args.batch,
                     adapter_kind=args.adapter_kind, lora_rank=args.lora_rank)

    A_noadapt = np.full((T, T), np.nan)
    A_nogate = np.full((T, T), np.nan)

    # no-adapt is constant across steps -> precompute per-domain frozen-head F1.
    noadapt_f1 = {lbl: no_adapt_f1(test_embs[lbl], test_lbls[lbl], frozen_head, device)
                  for lbl in labels}
    A_noadapt[0, 0] = A_nogate[0, 0] = noadapt_f1[labels[0]]
    print(f"Step 0 — {labels[0]:<10} noadapt/nogate F1={A_noadapt[0,0]:.4f}")

    adapters_list = []
    for t in range(1, T):
        did_t = order[t]
        ds = spec.get_dataset(did_t, "train", DATA_ROOT)
        y_all = ds.y.numpy()
        idx = calib_indices(y_all, cfg.calib_frac, cfg.seed, calib_n=cfg.calib_n)
        calib_ld = torch.utils.data.DataLoader(
            torch.utils.data.Subset(ds, idx), batch_size=batch)
        calib_embs, _ = extract_embs(model, calib_ld, device)
        calib_lbls = y_all[idx]
        adapters_list.append(
            train_adapters(calib_embs, calib_lbls, centroids, frozen_head, device, cfg))

        for j in range(t + 1):
            lbl_j = labels[j]
            A_noadapt[t, j] = noadapt_f1[lbl_j]
            A_nogate[t, j] = eval_uniform_mixture(
                test_embs[lbl_j], test_lbls[lbl_j], centroids, adapters_list,
                frozen_head, device, cfg.k)
        aa_na, _, f_na, rf_na, _ = compute_cl_metrics(A_noadapt, t)
        aa_ng, _, f_ng, rf_ng, _ = compute_cl_metrics(A_nogate, t)
        print(f"Step {t} — {labels[t]:<10} "
              f"no-adapt AA={aa_na:.4f} F={f_na:.4f} | "
              f"no-gate AA={aa_ng:.4f} F={f_ng:.4f} RF={rf_ng:.4f}")

    calib_tag = f"n{args.calib_n}" if args.calib_n else f"c{int(args.calib_frac*100)}"
    tag = f"{args.dataset}_{args.backbone}_{args.domain_order}_K{args.k}_{calib_tag}"
    out_dir = RESULTS_ROOT / "controls" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(A_noadapt, index=labels, columns=labels).to_csv(out_dir / "noadapt_f1_matrix.csv")
    pd.DataFrame(A_nogate, index=labels, columns=labels).to_csv(out_dir / "nogate_f1_matrix.csv")

    rows = []
    for name, A in [("no-adapt", A_noadapt), ("no-gate", A_nogate)]:
        aa, bwt, f, rf, prf = compute_cl_metrics(A, T - 1)
        rows.append(dict(control=name, AA=aa, BWT=bwt, F=f, RF=rf, PRF=prf))
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
    print("\n" + pd.DataFrame(rows).to_string(index=False))
    print(f"Results -> {out_dir}")


if __name__ == "__main__":
    main()
