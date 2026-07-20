"""Single-figure backbone-agnosticism summary: a dataset x backbone heatmap
of PACT's mean AMF1 (average macro-F1), annotated with exact values.
Transposed (datasets as rows, backbones as columns) to fit a single column
of a two-column paper -- wide and short rather than tall and narrow.

Data: results/tables/backbone_agnostic_ablation.csv, plus a manually-supplied
Sundial row (Sundial is PACT's main backbone throughout the paper, evaluated
separately from the 6-backbone ablation sweep).
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
    "font.size": 16, "axes.titlesize": 16, "axes.labelsize": 16,
    "xtick.labelsize": 16, "ytick.labelsize": 16,
    "font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42,
})

BACKBONE_ORDER = ["mantis", "unishape", "sundial", "moment", "chronos-bolt", "moirai", "tslanet", "patchtst"]
BACKBONE_LABEL = {"mantis": "Mantis", "unishape": "UniShape", "sundial": "Sundial", "moment": "MOMENT",
                  "chronos-bolt": "Chronos-Bolt", "moirai": "Moirai",
                  "tslanet": "TSLANet", "patchtst": "PatchTST"}
DATASET_ORDER = ["fd", "hhar", "sleep", "wisdm", "ucihar"]
DATASET_LABEL = {"fd": "FD", "hhar": "HHAR", "sleep": "Sleep-EDF",
                 "wisdm": "WISDM", "ucihar": "UCI-HAR"}

# Sundial (PACT's main backbone) is evaluated on its own random-chain K-sweep
# elsewhere in the paper, not part of the 6-backbone ablation CSV.
SUNDIAL_AMF1 = {"fd": 0.9575, "wisdm": 0.8774, "hhar": 0.8648, "ucihar": 0.8169, "sleep": 0.6508}

# Okabe-Ito inspired sequential ramp: pale -> teal (matches PACT's "ours" colour elsewhere)
CMAP = LinearSegmentedColormap.from_list("teal_seq", ["#fdf7ec", "#a6d9c8", "#009E73", "#00563f"])


def main():
    df = pd.read_csv(RESULTS_ROOT / "tables" / "backbone_agnostic_ablation.csv")
    mat = df.pivot(index="backbone", columns="dataset", values="mean_AA")
    mat.loc["sundial"] = pd.Series(SUNDIAL_AMF1)
    mat = mat.loc[BACKBONE_ORDER, DATASET_ORDER].T  # transpose: rows=datasets, cols=backbones

    fig, ax = plt.subplots(figsize=(10.2, 3.7))
    im = ax.imshow(mat.values, cmap=CMAP, vmin=0.45, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(BACKBONE_ORDER)))
    ax.set_xticklabels([BACKBONE_LABEL[b] for b in BACKBONE_ORDER], rotation=30, ha="right")
    ax.set_yticks(range(len(DATASET_ORDER)))
    ax.set_yticklabels([DATASET_LABEL[d] for d in DATASET_ORDER])
    ax.tick_params(length=0)

    for i, ds in enumerate(DATASET_ORDER):
        for j, bb in enumerate(BACKBONE_ORDER):
            v = mat.loc[ds, bb]
            txt_color = "white" if v > 0.80 else "#1a1a1a"
            weight = "bold" if bb in ("mantis", "unishape", "sundial") else "normal"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                     fontsize=12.5, color=txt_color, fontweight=weight)

    ax.set_xticks(np.arange(-0.5, len(BACKBONE_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DATASET_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
    cbar.set_label(r"$AMF_1$", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    fig.tight_layout(pad=0.3)
    out = RESULTS_ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_backbone_heatmap.{ext}", dpi=300, bbox_inches="tight",
                   pad_inches=0.03)
    plt.close(fig)
    print(f"Saved -> {out}/fig_backbone_heatmap.pdf (+ .png)")


if __name__ == "__main__":
    main()
