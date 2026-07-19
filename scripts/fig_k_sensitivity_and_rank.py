"""Two publication-ready ablation figures, Sundial backbone, PACT rank=8
(main) K-sweep across all 5 datasets (FD, HHAR, Sleep, WISDM, UCIHAR):

  1. fig_k_sensitivity.{pdf,png}   -- AA vs K, one compact panel per dataset.
  2. fig_lora_vs_fullrank.{pdf,png} -- AA vs K, rank=8 (LoRA) vs rank=768
     (full-rank) overlaid per panel.

Large, clear fonts; small, space-saving panels (5-across single row).
Data: results/tables/{kchains_sweep,fdchains_sweep,ucihar_chains_sweep}.csv
      (+ _r768 variants for the rank comparison)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12.5,
    "font.family": "sans-serif", "axes.linewidth": 0.9,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

C = dict(gray="#8c8c8c", orange="#E69F00", sky="#56B4E9", teal="#009E73",
          blue="#0072B2", verm="#D55E00", purple="#CC79A7")

DATASETS = ["fd", "hhar", "sleep", "wisdm", "ucihar"]
TITLES = {"fd": "FD", "hhar": "HHAR", "sleep": "Sleep-EDF", "wisdm": "WISDM", "ucihar": "UCI-HAR"}
KS = [1, 4, 8, 16, 32]


def load_rank8():
    df = pd.concat([
        pd.read_csv(RESULTS_ROOT / "tables" / "fdchains_sweep.csv"),
        pd.read_csv(RESULTS_ROOT / "tables" / "kchains_sweep.csv"),
        pd.read_csv(RESULTS_ROOT / "tables" / "ucihar_chains_sweep.csv"),
    ], ignore_index=True)
    return df


def load_r768():
    df = pd.concat([
        pd.read_csv(RESULTS_ROOT / "tables" / "fdchains_sweep_r768.csv"),
        pd.read_csv(RESULTS_ROOT / "tables" / "kchains_sweep_r768.csv"),
        pd.read_csv(RESULTS_ROOT / "tables" / "ucihar_chains_sweep_r768.csv"),
    ], ignore_index=True)
    return df


def style_panel(ax, ds, first):
    ax.set_xscale("log", base=2)
    ax.set_xticks(KS)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlim(0.85, 37)
    ax.set_title(TITLES[ds], fontsize=14, fontweight="bold", pad=6)
    ax.grid(True, linewidth=0.5, color="#e6e6e0", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if first:
        ax.set_ylabel(r"$AMF_1$", fontsize=14)
    ax.tick_params(axis="both", length=3)


def fig_k_sensitivity():
    df = load_rank8()
    fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.15), sharey=False)
    for i, ds in enumerate(DATASETS):
        ax = axes[i]
        sub = df[df["dataset"] == ds].sort_values("K")
        ax.plot(sub["K"], sub["mean_AA"], "-", color=C["teal"], lw=2.6,
                 marker="*", markersize=13, markeredgecolor="black",
                 markeredgewidth=0.7, zorder=5)
        best = sub.loc[sub["mean_AA"].idxmax()]
        ax.scatter([best["K"]], [best["mean_AA"]], s=210, facecolor="none",
                    edgecolor=C["verm"], linewidth=2.2, zorder=6)
        lo = (sub["mean_AA"] - sub.get("std_AA", 0)).min()
        hi = (sub["mean_AA"] + sub.get("std_AA", 0)).max()
        pad = max(0.03, (hi - lo) * 0.25)
        ax.set_ylim(lo - pad, hi + pad)
        style_panel(ax, ds, i == 0)
    fig.text(0.5, -0.02, r"Number of charts $K$", ha="center", fontsize=14)
    fig.suptitle("PACT: accuracy vs. number of charts $K$ (Sundial, rank=8 LoRA)",
                 fontsize=14.5, y=1.06)
    fig.tight_layout(rect=[0, 0, 1, 1])
    out = RESULTS_ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_k_sensitivity.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}/fig_k_sensitivity.pdf (+ .png)")


def fig_lora_vs_fullrank():
    r8 = load_rank8()
    r768 = load_r768()
    fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.3), sharey=False)
    for i, ds in enumerate(DATASETS):
        ax = axes[i]
        s8 = r8[r8["dataset"] == ds].sort_values("K")
        s768 = r768[r768["dataset"] == ds].sort_values("K")
        ax.plot(s8["K"], s8["mean_AA"], "-", color=C["teal"], lw=2.4,
                 marker="*", markersize=11, markeredgecolor="black",
                 markeredgewidth=0.6, zorder=5, label="LoRA rank=8 (ours)")
        ax.plot(s768["K"], s768["mean_AA"], "--", color=C["verm"], lw=2.0,
                 marker="o", markersize=7, markeredgecolor="white",
                 markeredgewidth=0.5, zorder=4, alpha=0.95, label="Full rank=768")
        allvals = pd.concat([s8["mean_AA"], s768["mean_AA"]])
        lo, hi = allvals.min(), allvals.max()
        pad = max(0.03, (hi - lo) * 0.3)
        ax.set_ylim(lo - pad, hi + pad)
        style_panel(ax, ds, i == 0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=13, bbox_to_anchor=(0.5, 1.14), handletextpad=0.5,
               columnspacing=1.6)
    fig.text(0.5, -0.02, r"Number of charts $K$", ha="center", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 1])
    out = RESULTS_ROOT / "figures"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_lora_vs_fullrank.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}/fig_lora_vs_fullrank.pdf (+ .png)")


def main():
    fig_k_sensitivity()
    fig_lora_vs_fullrank()


if __name__ == "__main__":
    main()
