"""Sequential domain-incremental continual learning with PACT (main experiment).

Domains arrive one at a time; PACT adapts to each without touching the frozen
backbone/head, so past-domain performance is preserved. Reports AA/BWT/F/RF/PRF.

Examples:
  python scripts/run_sequential.py --dataset fd
  python scripts/run_sequential.py --dataset hhar --backbone patchtst
  python scripts/run_sequential.py --dataset fd --domain_order 3,2,1,0 --calib_frac 0.05
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _bootstrap  # noqa: F401
from _bootstrap import CKPT_ROOT, DATA_ROOT, RESULTS_ROOT

from src.data import get_spec
from src.method import PactConfig, run_sequential
from src.models.factory import BACKBONES, load_backbone
from src.training import train_or_load_source
from src.utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["fd", "hhar", "sleep", "wisdm"], default="fd")
    p.add_argument("--backbone", choices=BACKBONES, default="tslanet")
    p.add_argument("--domain_order", default=None,
                   help="Comma-separated domain ids (first = source), or "
                        "'adatime' to derive one chain from the AdaTime pair set.")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--calib_frac", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--adapter_kind", choices=["identity", "lora"], default="lora")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--gate_kind", choices=["learned", "prototype"], default="learned",
                   help="learned = MLP re-fit on accumulated calibration (rehearsal); "
                        "prototype = parameter-free per-concept Gaussian, O(1) per new domain")
    p.add_argument("--adapter_ep", type=int, default=200)
    p.add_argument("--gate_ep", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--device", default=None)
    return p.parse_args()


def save_heatmap(A, labels, title, path):
    T = len(labels)
    fig, ax = plt.subplots(figsize=(1.3 * T + 2, 1.1 * T + 1))
    im = ax.imshow(np.where(np.isnan(A), 0, A), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(T)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(T)); ax.set_yticklabels([f"after {l}" for l in labels], fontsize=8)
    for i in range(T):
        for j in range(T):
            if not np.isnan(A[i, j]):
                ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center",
                        color="white" if A[i, j] < 0.6 else "black", fontsize=7)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    spec = get_spec(args.dataset)

    if args.domain_order == "adatime":
        from src.data.datasets import adatime_chain
        order, chain_stats = adatime_chain(args.dataset)
        print(f"AdaTime-derived chain: {chain_stats['n_domains']} domains, "
              f"{chain_stats['forward']}/{chain_stats['n_pairs']} AdaTime pairs "
              f"captured forward ({chain_stats['backward']} backward due to cycles)")
    elif args.domain_order:
        order = [int(x) for x in args.domain_order.split(",")]
    else:
        order = list(range(spec.n_domains))

    print(f"Device={device}  dataset={args.dataset}  backbone={args.backbone}  "
          f"T={len(order)}  K={args.k}  calib={args.calib_frac*100:.0f}%  "
          f"temp={args.temperature}  adapter={args.adapter_kind}")
    print(f"Domain order: {[spec.domain_label(d) for d in order]}")

    # Ensure source backbone exists (train if missing), then load frozen.
    train_or_load_source(args.backbone, args.dataset, order[0], CKPT_ROOT,
                         DATA_ROOT, device)
    model = load_backbone(args.backbone, spec, CKPT_ROOT, args.dataset, order[0],
                          device, freeze=True)

    cfg = PactConfig(k=args.k, emb_dim=model.emb_dim, calib_frac=args.calib_frac,
                     adapter_ep=args.adapter_ep, gate_ep=args.gate_ep,
                     seed=args.seed, batch=args.batch,
                     adapter_kind=args.adapter_kind, lora_rank=args.lora_rank,
                     gate_kind=args.gate_kind)

    res = run_sequential(model, spec, order, DATA_ROOT, device, cfg,
                         temperature=args.temperature)

    order_tag = "adatime" if args.domain_order == "adatime" else "_".join(map(str, order))
    tag = (f"{args.dataset}_{args.backbone}_{order_tag}"
           f"_K{args.k}_c{int(args.calib_frac*100)}_t{args.temperature}"
           f"_{args.adapter_kind}_{args.gate_kind}_s{args.seed}")
    out_dir = RESULTS_ROOT / "sequential" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = res["labels"]
    pd.DataFrame(res["metrics"]).to_csv(out_dir / "metrics.csv", index=False)
    for name, mat in [("soft", res["A_soft"]), ("hard", res["A_hard"]),
                      ("oracle", res["A_orc"]), ("acc", res["ACC"])]:
        pd.DataFrame(mat, index=labels, columns=labels).to_csv(
            out_dir / f"{name}_f1_matrix.csv")
    save_heatmap(res["A_soft"], labels,
                 f"PACT soft-mixture F1 — {args.dataset.upper()} ({args.backbone})",
                 out_dir / "soft_f1_heatmap.png")

    final = res["metrics"][-1]
    print(f"\nFinal: AA={final['AA']:.4f}  BWT={final['BWT']:+.4f}  "
          f"F_all={final['F_all']:.4f}  F_tgt={final['F_tgt']:.4f}  "
          f"RF={final['RF']:.4f}  PRF={final['PRF']:.4f}")
    print(f"Results -> {out_dir}")


if __name__ == "__main__":
    main()
