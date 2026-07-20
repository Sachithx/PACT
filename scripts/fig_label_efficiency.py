"""Label-efficiency ablation: how PACT's AA scales with the calibration
label budget, Mantis backbone, K=4. Two panels in one compact figure:

  left:  FD / HHAR / Sleep-EDF, budget as % of each domain's train set
         (1, 5, 10, 20, 50, 100)
  right: WISDM / UCI-HAR, fixed label count per domain
         (20, 50, 100, 200, 400, max)

Data: results/tables/label_efficiency.csv
Outputs: results/figures/fig_label_efficiency.{pdf,png}
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT

plt.rcParams.update({
    "font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 14,
    "font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42,
})

C = dict(gray="#8c8c8c", orange="#E69F00", sky="#56B4E9", teal="#009E73",
          blue="#0072B2", verm="#D55E00", purple="#CC79A7")

FRAC_DATASETS = [("fd", "FD", C["blue"], "o"), ("hhar", "HHAR", C["verm"], "^"),
                 ("sleep", "Sleep-EDF", C["teal"], "s")]
N_DATASETS = [("wisdm", "WISDM", C["orange"], "o"), ("ucihar", "UCI-HAR", C["purple"], "^")]
FRAC_TICKS = [1, 5, 10, 20, 50, 100]
N_TICKS = [20, 50, 100, 200, 400, 10000]


def main():
    df = pd.read_csv(RESULTS_ROOT / "tables" / "label_efficiency.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.0))

    for ds, label, col, mk in FRAC_DATASETS:
        sub = df[(df["dataset"] == ds) & (df["budget_type"] == "frac")].sort_values("budget_value")
        ax1.plot(sub["budget_value"], sub["mean_AA"], "-", color=col, lw=2.2,
                 marker=mk, markersize=12, markeredgecolor="white", markeredgewidth=0.6,
                 label=label, zorder=3)
    ax1.set_xscale("log")
    ax1.set_xticks(FRAC_TICKS)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax1.set_xlabel("Calibration labels (% of domain)")
    ax1.set_ylabel(r"$AMF_1$")
    ax1.legend(frameon=False, loc="lower right", handletextpad=0.4, labelspacing=0.3)

    for ds, label, col, mk in N_DATASETS:
        sub = df[(df["dataset"] == ds) & (df["budget_type"] == "n")].sort_values("budget_value")
        ax2.plot(sub["budget_value"], sub["mean_AA"], "-", color=col, lw=2.2,
                 marker=mk, markersize=12, markeredgecolor="white", markeredgewidth=0.6,
                 label=label, zorder=3)
    ax2.set_xscale("log")
    ax2.set_xticks(N_TICKS)
    ax2.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: "max" if v >= 10000 else f"{int(v)}"))
    ax2.set_xlabel("Calibration labels (count)")
    ax2.legend(frameon=False, loc="lower right", handletextpad=0.4, labelspacing=0.3)

    for ax in (ax1, ax2):
        ax.grid(True, which="both", linewidth=0.5, color="#e6e6e0", zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.tight_layout(pad=0.4, w_pad=1.6)
    out = RESULTS_ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_label_efficiency.{ext}", dpi=300, bbox_inches="tight",
                   pad_inches=0.03)
    plt.close(fig)
    print(f"Saved -> {out}/fig_label_efficiency.pdf (+ .png)")


if __name__ == "__main__":
    main()
