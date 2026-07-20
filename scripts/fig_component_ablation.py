"""Model-component ablation: how much does each part of PACT contribute?
Progressively strips components from the full model, same K/chain/backbone
throughout so only one thing changes at a time:

  w/o Adapters   -- frozen backbone, no transport at all (source-only floor)
  Head fine-tune -- fine-tune the head online, no charts/gate/adapters
  w/o Gate       -- uniform mixture instead of the learned soft gate (K held at best-K)
  PACT           -- full model: K charts (best-K), LoRA adapters, learned soft gate

Two panels, same component colours: left = AMF1 (higher is better), right =
PRF (lower is better). w/o Adapters has PRF=0 by construction (a frozen head
never updates, so it can't "forget").

Data: results/tables/component_ablation.csv
Outputs: results/figures/fig_component_ablation.{pdf,png}
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT

plt.rcParams.update({
    "font.size": 17, "axes.labelsize": 17,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 16,
    "font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42,
})

DS_ORDER = ["fd", "hhar", "sleep", "wisdm", "ucihar"]
DS_LABEL = {"fd": "FD", "hhar": "HHAR", "sleep": "Sleep-EDF", "wisdm": "WISDM", "ucihar": "UCI-HAR"}

COMPONENT_ORDER = ["no_adapters", "head_ft", "no_gate", "pact"]
COMPONENT_LABEL = {"no_adapters": "w/o Adapters", "head_ft": "Head fine-tune",
                   "no_gate": "w/o Gate", "pact": "PACT"}
COMPONENT_COLOR = {"no_adapters": "#8c8c8c", "head_ft": "#CC79A7",
                   "no_gate": "#E69F00", "pact": "#009E73"}


def grouped_bars(ax, df, value_col):
    n_comp = len(COMPONENT_ORDER)
    width = 0.8 / n_comp
    x = np.arange(len(DS_ORDER))
    for i, comp in enumerate(COMPONENT_ORDER):
        sub = df[df["component"] == comp].set_index("dataset").reindex(DS_ORDER)
        offset = (i - (n_comp - 1) / 2) * width
        ax.bar(x + offset, sub[value_col], width=width * 0.92,
              color=COMPONENT_COLOR[comp], label=COMPONENT_LABEL[comp],
              edgecolor="black", linewidth=0.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABEL[d] for d in DS_ORDER])
    ax.grid(True, axis="y", linewidth=0.5, color="#e6e6e0", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    df = pd.read_csv(RESULTS_ROOT / "tables" / "component_ablation.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 3.5))

    grouped_bars(ax1, df, "mean_AA")
    ax1.set_ylabel(r"$AMF_1$  ($\uparrow$)")
    ax1.set_ylim(0.45, 1.0)

    grouped_bars(ax2, df, "mean_PRF")
    ax2.set_ylabel(r"$PRF$  ($\downarrow$)")
    ax2.set_ylim(0, 0.12)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.1),
              ncol=4, frameon=False, handletextpad=0.35, columnspacing=1.1)

    fig.tight_layout(pad=0.2, w_pad=1.0)
    out = RESULTS_ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_component_ablation.{ext}", dpi=300,
                   bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved -> {out}/fig_component_ablation.pdf (+ .png)")


if __name__ == "__main__":
    main()
