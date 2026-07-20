"""Two ablations in one compact figure, grouped bar charts, 5 datasets:

  left:  LoRA rank=8 (ours) vs. full-rank=768, at each dataset's best K
         (isolates the rank effect -- same K used for both bars).
  right: gate ablation -- soft (default, learned mixture), hard (argmax
         single-adapter routing), oracle (ground-truth domain routing,
         upper bound).

Data: results/tables/lora_vs_fullrank_matched_k.csv, results/tables/gate_ablation.csv
Outputs: results/figures/fig_lora_and_gate_ablation.{pdf,png}
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT

plt.rcParams.update({
    "font.size": 15, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12.5,
    "font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42,
})

C = dict(teal="#009E73", gray="#8c8c8c", gold="#E69F00", verm="#D55E00")
DS_ORDER = ["fd", "hhar", "sleep", "wisdm", "ucihar"]
DS_LABEL = {"fd": "FD", "hhar": "HHAR", "sleep": "Sleep-EDF", "wisdm": "WISDM", "ucihar": "UCI-HAR"}


def grouped_bars(ax, df, group_col, group_order, group_label, group_color, value_col="mean_AA"):
    n_groups = len(group_order)
    width = 0.8 / n_groups
    x = np.arange(len(DS_ORDER))
    for i, g in enumerate(group_order):
        sub = df[df[group_col] == g].set_index("dataset").reindex(DS_ORDER)
        offset = (i - (n_groups - 1) / 2) * width
        ax.bar(x + offset, sub[value_col], width=width * 0.92,
              color=group_color[g], label=group_label[g],
              edgecolor="black", linewidth=0.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABEL[d] for d in DS_ORDER])
    ax.grid(True, axis="y", linewidth=0.5, color="#e6e6e0", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    lora_df = pd.read_csv(RESULTS_ROOT / "tables" / "lora_vs_fullrank_matched_k.csv")
    gate_df = pd.read_csv(RESULTS_ROOT / "tables" / "gate_ablation.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0))

    grouped_bars(ax1, lora_df, "rank", ["lora8", "full768"],
                {"lora8": "LoRA r=8 (ours)", "full768": "Full rank=768"},
                {"lora8": C["teal"], "full768": C["gray"]})
    ax1.set_ylabel(r"$AMF_1$")
    ax1.set_ylim(0.55, 1.0)
    ax1.legend(frameon=False, loc="upper right", handletextpad=0.4, labelspacing=0.25,
              fontsize=11.5, ncol=1)

    grouped_bars(ax2, gate_df, "gate", ["soft", "hard", "oracle"],
                {"soft": "Soft (ours)", "hard": "Hard", "oracle": "Oracle"},
                {"soft": C["teal"], "hard": C["gray"], "oracle": C["gold"]})
    ax2.set_ylim(0.55, 1.0)
    ax2.legend(frameon=False, loc="upper right", handletextpad=0.4, labelspacing=0.25,
              fontsize=11.5, ncol=1)

    fig.tight_layout(pad=0.4, w_pad=1.4)
    out = RESULTS_ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_lora_and_gate_ablation.{ext}", dpi=300,
                   bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved -> {out}/fig_lora_and_gate_ablation.pdf (+ .png)")


if __name__ == "__main__":
    main()
