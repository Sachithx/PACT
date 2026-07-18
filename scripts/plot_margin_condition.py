"""Empirical check of Proposition 1 margin condition (Eq. 15) on source test embeddings.

Computes per-sample m(z), pi_0(z), B(z)=max_d ||Delta^d_{kappa(z)}(z)||_2, and reports
the violation rate of (1-pi_0(z)) B(z) < m(z)/(2L).

Example:
  python scripts/plot_margin_condition.py --dataset fd --tsfm moment --device cuda
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import _bootstrap  # noqa: F401
from _bootstrap import DATA_ROOT, RESULTS_ROOT

from src.data import get_spec
from src.data.datasets import adatime_chain
from src.method import PactConfig
from src.method.pact import extract_embs, train_adapters
from src.method.sequential import fit_gates
from src.models.concepts import assign_concepts, kmeans_numpy
from src.models.tsfm import make_tsfm
from src.utils import get_device, set_seed
from run_sequential_tsfm import fit_standardizer_and_head
from src.data.calib import calib_indices


@torch.no_grad()
def head_margin(head, z_t, y_true):
    logits = head(z_t)
    y_star = int(y_true)
    m = (logits[0, y_star] - logits[0, torch.arange(logits.shape[1]) != y_star].max()).item()
    return m


@torch.no_grad()
def compute_sample_stats(embs_np, labels_np, centroids, soft_gates, adapters_list,
                         frozen_head, device, k, temperature=0.5):
    W = frozen_head.weight.detach()
    L = float(W.norm(dim=1).max().cpu())
    rows = []
    asgn = assign_concepts(embs_np, centroids)
    for i in range(len(embs_np)):
        z_np = embs_np[i:i + 1]
        y = int(labels_np[i])
        z = torch.from_numpy(z_np).to(device)
        c = int(asgn[i])
        m = head_margin(frozen_head, z, y)
        if m <= 0:
            continue
        gate = soft_gates[c]
        if gate is None:
            pi0 = 1.0
            B = 0.0
        else:
            gate.eval()
            logits_g = gate(z)
            w = torch.softmax(logits_g / temperature, dim=1)[0]
            pi0 = float(w[0].cpu())
            deltas = []
            for j, bank in enumerate(adapters_list):
                bank[c].eval()
                z_corr = bank[c](z)
                deltas.append(float((z_corr - z).norm().cpu()))
            B = max(deltas) if deltas else 0.0
        lhs = (1.0 - pi0) * B
        rhs = m / (2.0 * L)
        rows.append(dict(margin=m, pi0=pi0, B=B, lhs=lhs, rhs=rhs,
                         satisfies=int(lhs < rhs), chart=c))
    return pd.DataFrame(rows), L


def run_pact_to_final(model, spec, order, device, cfg, temperature=0.5):
    from src.data.calib import calib_indices as ci
    frozen_head = model.head
    src_ld = torch.utils.data.DataLoader(
        spec.get_dataset(order[0], "train", DATA_ROOT), batch_size=cfg.batch)
    src_embs, src_lbls = extract_embs(model, src_ld, device)
    centroids, src_asgn = kmeans_numpy(src_embs, cfg.k, seed=cfg.seed,
                                       n_init=cfg.kmeans_n_init)
    accumulated, adapters_list = [], []
    soft_gates = None
    for t in range(1, len(order)):
        ds = spec.get_dataset(order[t], "train", DATA_ROOT)
        y_all = ds.y.numpy()
        idx = ci(y_all, cfg.calib_frac, cfg.seed, calib_n=cfg.calib_n)
        calib_ld = torch.utils.data.DataLoader(
            torch.utils.data.Subset(ds, idx), batch_size=cfg.batch)
        calib_embs, calib_lbls = extract_embs(model, calib_ld, device)
        calib_asgn = assign_concepts(calib_embs, centroids)
        accumulated.append((calib_embs, calib_asgn))
        adapters_list.append(train_adapters(calib_embs, calib_lbls, centroids,
                                           frozen_head, device, cfg))
        soft_gates = fit_gates(src_embs, src_asgn, accumulated, device, cfg)
    return centroids, soft_gates, adapters_list, frozen_head


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="fd")
    p.add_argument("--tsfm", default="moment")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--temperature", type=float, default=0.5)
    args = p.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    spec = get_spec(args.dataset)
    order, _ = adatime_chain(args.dataset)

    model = make_tsfm(args.tsfm, spec.num_classes, device=str(device)).to(device)
    fit_standardizer_and_head(model, spec, order[0], device,
                              150, 1e-3, 64, args.seed)
    cfg = PactConfig(k=16, emb_dim=model.emb_dim, calib_frac=0.05, seed=args.seed,
                     batch=64, adapter_kind="lora")
    print("Running PACT chain to final step...")
    centroids, soft_gates, adapters_list, head = run_pact_to_final(
        model, spec, order, device, cfg, args.temperature)

    test_ld = torch.utils.data.DataLoader(
        spec.get_dataset(order[0], "test", DATA_ROOT), batch_size=cfg.batch)
    test_embs, test_lbls = extract_embs(model, test_ld, device)
    df, L = compute_sample_stats(test_embs, test_lbls, centroids, soft_gates,
                                 adapters_list, head, device, cfg.k, args.temperature)
    n = len(df)
    sat = df["satisfies"].mean()
    print(f"L={L:.4f}  correctly-classified source samples={n}")
    print(f"Eq. (15) satisfied: {sat*100:.1f}%  violated: {(1-sat)*100:.1f}%")

    out_dir = RESULTS_ROOT / "margin_condition"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset}_{args.tsfm}_s{args.seed}"
    df.to_csv(out_dir / f"{tag}.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.scatter(df["rhs"], df["lhs"], s=8, alpha=0.35, c=df["satisfies"].map({1: "#0072B2", 0: "#D55E00"}))
    lim = max(df["rhs"].max(), df["lhs"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="boundary")
    ax.set_xlabel(r"$m(z)/(2L)$")
    ax.set_ylabel(r"$(1-\pi_0(z))B(z)$")
    ax.set_title(f"Margin condition (source test, {tag})\n"
                 f"{sat*100:.1f}% satisfy Eq.~(15)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(RESULTS_ROOT / "figures" / f"fig_margin_condition_{tag}.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot -> results/figures/fig_margin_condition_{tag}.pdf")


if __name__ == "__main__":
    main()
