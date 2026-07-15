"""Publication-ready locality figure from saved CSV measurements.

This script is intentionally lightweight: it does not rerun the model or
adapter training. It consumes the CSVs emitted by ``fig_shift_locality.py`` and
produces the three-panel figure used in the main paper.

Outputs:
  results/figures/fig_locality.{pdf,png}
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT


plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "legend.fontsize": 7.2,
        "font.family": "sans-serif",
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.edgecolor": "#333333",
        "text.color": "#1a1a1a",
        "axes.labelcolor": "#1a1a1a",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
    }
)

C = {
    "blue": "#0072B2",
    "teal": "#009E73",
    "verm": "#D55E00",
    "gray": "#777777",
    "grid": "#e1e0d9",
}


def _spines_off(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def fig_locality():
    root = RESULTS_ROOT / "shift_locality"
    panel_a = pd.read_csv(root / "panel_a_per_cluster_drop.csv")
    panel_b = pd.read_csv(root / "panel_b_residual_vs_k.csv")
    panel_c = pd.read_csv(root / "panel_c_collateral_on_source.csv")

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.45), gridspec_kw={"wspace": 0.42})
    ax_a, ax_b, ax_c = axes

    # (a) Heterogeneous local damage.
    drop = (
        panel_a.groupby("cluster", as_index=False)["drop"]
        .mean()
        .dropna()
        .sort_values("drop", ascending=True)
    )
    ax_a.barh(
        np.arange(len(drop)),
        drop["drop"],
        color=C["blue"],
        edgecolor="black",
        linewidth=0.35,
        zorder=3,
    )
    ax_a.axvline(0, color="#333333", linewidth=0.7)
    ax_a.set_yticks(np.arange(len(drop)))
    ax_a.set_yticklabels([f"c{int(c)}" for c in drop["cluster"]])
    ax_a.set_xlabel(r"Chart-wise macro-$F_1$ drop")
    ax_a.set_ylabel("Chart")
    ax_a.set_title("(a) Shift is local")
    ax_a.grid(True, axis="x", color=C["grid"], linewidth=0.45, zorder=0)
    cv = np.nanstd(drop["drop"]) / max(np.nanmean(drop["drop"]), 1e-9)
    ax_a.text(
        0.98,
        0.05,
        f"CV={cv:.2f}",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#cccccc", lw=0.6),
    )
    _spines_off(ax_a)

    # (b) Local maps fit the shift far better than a single global map.
    ax_b.plot(
        panel_b["K"],
        panel_b["residual"],
        "-o",
        color=C["teal"],
        lw=1.7,
        ms=4.6,
        markeredgecolor="white",
        markeredgewidth=0.4,
        zorder=3,
    )
    ax_b.axhline(
        float(panel_b.loc[panel_b["K"] == 1, "residual"].iloc[0]),
        color=C["gray"],
        lw=0.8,
        ls="--",
    )
    ax_b.axvspan(2, 32, color=C["teal"], alpha=0.10, zorder=0)
    for _, row in panel_b.iterrows():
        if row["K"] in (1, 2, 16, 64):
            ax_b.annotate(
                f"{row['residual']:.3f}",
                (row["K"], row["residual"]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=6.8,
            )
    ax_b.set_xscale("log", base=2)
    ax_b.set_xticks(panel_b["K"])
    ax_b.set_xticklabels([str(int(k)) for k in panel_b["K"]])
    ax_b.set_xlabel("Local charts $K$")
    ax_b.set_ylabel(r"Held-out test residual $1-F_1$")
    ax_b.set_title("(b) Local maps fit")
    ax_b.grid(True, color=C["grid"], linewidth=0.45, zorder=0)
    _spines_off(ax_b)

    # (c) The global correction changes target and source margins differently.
    c = panel_c.dropna(subset=["delta_margin_t", "delta_margin_src"]).copy()
    c = c.sort_values("delta_margin_t", ascending=False)
    labels = [f"c{int(v)}" for v in c["cluster"]]
    x = np.arange(len(c))
    width = 0.38
    ax_c.bar(
        x - width / 2,
        c["delta_margin_t"],
        width,
        color=C["blue"],
        edgecolor="black",
        linewidth=0.35,
        label="Fit target",
        zorder=3,
    )
    ax_c.bar(
        x + width / 2,
        c["delta_margin_src"],
        width,
        color=C["gray"],
        edgecolor="black",
        linewidth=0.35,
        label="Source",
        zorder=3,
    )
    ax_c.axhline(0, color="#333333", linewidth=0.75)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(labels, rotation=35, ha="right")
    ax_c.set_xlabel("Chart, sorted by target gain")
    ax_c.set_ylabel(r"$\Delta$ margin")
    ax_c.set_title("(c) Global map changes margins")
    ax_c.legend(frameon=False, loc="upper right", handlelength=1.2)
    ax_c.grid(True, axis="y", color=C["grid"], linewidth=0.45, zorder=0)
    ax_c.text(
        0.03,
        0.05,
        r"source $F_1$: 0.998 $\rightarrow$ 0.426",
        transform=ax_c.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.8,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#cccccc", lw=0.6),
    )
    _spines_off(ax_c)

    fig.suptitle(
        "Operating-condition shift is concentrated in local embedding regions",
        fontsize=9.8,
        fontweight="bold",
        y=1.04,
    )
    out = RESULTS_ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_locality.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}/fig_locality.pdf (+ .png)")


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
    fig_locality()
