"""Single-figure backbone-agnosticism summary: a backbone x dataset heatmap
of PACT's best-K mean AMF1 (average macro-F1), annotated with exact values.

Data: results/tables/backbone_agnostic_ablation.csv
Outputs: results/figures/fig_backbone_heatmap.{pdf,png}
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42,
})

BACKBONE_ORDER = ["mantis", "moment", "chronos-bolt", "moirai", "tslanet", "patchtst"]
BACKBONE_LABEL = {"mantis": "Mantis", "moment": "MOMENT", "chronos-bolt": "Chronos-Bolt",
                  "moirai": "Moirai", "tslanet": "TSLANet", "patchtst": "PatchTST"}
DATASET_ORDER = ["fd", "hhar", "sleep", "wisdm", "ucihar"]
DATASET_LABEL = {"fd": "FD", "hhar": "HHAR", "sleep": "Sleep-EDF",
                 "wisdm": "WISDM", "ucihar": "UCI-HAR"}

# Okabe-Ito inspired sequential ramp: pale -> teal (matches PACT's "ours" colour elsewhere)
CMAP = LinearSegmentedColormap.from_list("teal_seq", ["#fdf7ec", "#a6d9c8", "#009E73", "#00563f"])


def main():
    df = pd.read_csv(RESULTS_ROOT / "tables" / "backbone_agnostic_ablation.csv")
    mat = df.pivot(index="backbone", columns="dataset", values="mean_AA")
    mat = mat.loc[BACKBONE_ORDER, DATASET_ORDER]

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    im = ax.imshow(mat.values, cmap=CMAP, vmin=0.45, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(DATASET_ORDER)))
    ax.set_xticklabels([DATASET_LABEL[d] for d in DATASET_ORDER])
    ax.set_yticks(range(len(BACKBONE_ORDER)))
    ax.set_yticklabels([BACKBONE_LABEL[b] for b in BACKBONE_ORDER])
    ax.tick_params(length=0)

    for i, bb in enumerate(BACKBONE_ORDER):
        for j, ds in enumerate(DATASET_ORDER):
            v = mat.loc[bb, ds]
            txt_color = "white" if v > 0.80 else "#1a1a1a"
            weight = "bold" if bb == "mantis" else "normal"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                     fontsize=13, color=txt_color, fontweight=weight)

    ax.set_xticks(np.arange(-0.5, len(DATASET_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(BACKBONE_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(r"$AMF_1$ at best $K$", fontsize=12.5)
    cbar.ax.tick_params(labelsize=11)

    ax.set_title("PACT is backbone-agnostic: best-$K$ $AMF_1$ across\n"
                 "6 backbones $\\times$ 5 domain-incremental datasets", fontsize=14.5, pad=12)

    fig.tight_layout()
    out = RESULTS_ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_backbone_heatmap.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}/fig_backbone_heatmap.pdf (+ .png)")


if __name__ == "__main__":
    main()
