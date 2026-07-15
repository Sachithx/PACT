"""Results-section figures for the paper (replace Tables 2-7 in the main text).

Fig 1 (tradeoff)    -- accuracy-forgetting scatter, domain-incremental chain,
                       TSLANet backbone (Table 2A). The headline result.
Fig 2 (backbones)   -- AA / F across five backbones, chain (Table 3).
Fig 3 (ablation)    -- (a) component ablation FD chain (Table 4);
                       (b) gate routing FD/HHAR/Sleep (Table 5, TSLANet).
Fig 4 (robustness)  -- (a) label-budget scaling (Table 6+7); (b) concept
                       count K sweep; (c) gate temperature tau sweep (Table 6).

All numbers hardcoded from results/tables/{ALL_TABLES.md,paper_tables.tex,
main.csv,table2_chain.csv} -- verified against those files, not re-derived.

Outputs: results/figures/fig_{tradeoff,backbones,ablation,robustness}.{pdf,png}
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "font.family": "sans-serif",
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# Okabe-Ito colour-blind-safe palette (matches fig_motivation.py). PACT stays
# teal and Full fine-tune stays vermillion everywhere in the paper; TSLANet
# and MOMENT stay blue/purple everywhere they recur (Figs 2, 3a, 4b-c).
C = dict(black="#2b2b2b", gray="#8c8c8c", blue="#0072B2", sky="#56B4E9", teal="#009E73",
          verm="#D55E00", orange="#E69F00", purple="#CC79A7")
DATASETS = ["FD", "HHAR", "Sleep-EDF"]
FIGSIZE_W = 7.6  # matches fig_motivation.py -> same on-page scale at \textwidth


def _spines_off(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _grouped_bars(ax, groups, series, colors, ylim=None, ylabel=""):
    x = np.arange(len(groups))
    w = 0.8 / len(series)
    for i, (name, vals) in enumerate(series.items()):
        xp = x - 0.4 + w * (i + 0.5)
        heights = [0 if v is None else v for v in vals]
        ax.bar(xp, heights, width=w * 0.85, color=colors[i], label=name,
               edgecolor="black", linewidth=0.5, zorder=3)
        for xi, v in zip(xp, vals):
            if v is None:
                y0 = ylim[0] if ylim else 0
                ax.text(xi, y0 + 0.015, "n/a", ha="center", va="bottom",
                        fontsize=6, rotation=90, color="#595959")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linewidth=0.5, color="#e1e0d9", zorder=0)
    _spines_off(ax)


# --------------------------------------------------------------------------
# Fig 1 -- accuracy/forgetting trade-off (Table 2A, TSLANet chain)
# --------------------------------------------------------------------------
def fig_tradeoff():
    # name -> (color, marker, {dataset: (F, AA)})
    M = {
        "Frozen / no adaptation": (C["black"], "X", {"FD": (0.000, 0.574), "HHAR": (0.000, 0.349), "Sleep-EDF": (0.000, 0.622)}),
        "Full fine-tune": (C["verm"], "o", {"FD": (0.283, 0.640), "HHAR": (0.345, 0.654), "Sleep-EDF": (0.116, 0.584)}),
        "Head fine-tune": (C["sky"], "s", {"FD": (0.012, 0.586), "HHAR": (0.133, 0.576), "Sleep-EDF": (0.056, 0.605)}),
        "EWC":            (C["orange"], "^", {"FD": (0.001, 0.590), "HHAR": (0.093, 0.526), "Sleep-EDF": (0.004, 0.620)}),
        "MAS":            (C["purple"], "D", {"FD": (0.057, 0.601), "HHAR": (0.280, 0.653), "Sleep-EDF": (0.052, 0.605)}),
        "LwF":            (C["blue"], "v", {"FD": (0.256, 0.549), "HHAR": (0.155, 0.723), "Sleep-EDF": (0.026, 0.622)}),
        "Source-distill": (C["gray"], "P", {"FD": (0.256, 0.573), "HHAR": (0.083, 0.482), "Sleep-EDF": (0.026, 0.622)}),
        "PACT (ours)":    (C["teal"], "*", {"FD": (0.002, 0.978), "HHAR": (0.007, 0.886), "Sleep-EDF": (0.013, 0.663)}),
    }
    fig, axes = plt.subplots(1, 3, figsize=(FIGSIZE_W, 2.95), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        for name, (col, mk, d) in M.items():
            f, aa = d[ds]
            if name == "PACT (ours)":
                ax.scatter(f, aa, s=170, marker=mk, color=col, edgecolor="black",
                           linewidth=0.7, zorder=5)
                ax.annotate("PACT", (f, aa), textcoords="offset points", xytext=(9, -3),
                            fontsize=8, fontweight="bold", color=col)
            else:
                ax.scatter(f, aa, s=42, marker=mk, color=col, edgecolor="white",
                           linewidth=0.5, zorder=3)
        ax.set_title(ds, fontsize=9.5)
        ax.set_xlabel(r"Forgetting $F$  ($\rightarrow$ lower is better)", fontsize=8)
        ax.set_xlim(-0.015, 0.37)
        ax.set_ylim(0.30, 1.02)
        ax.grid(True, linewidth=0.5, color="#e1e0d9", zorder=0)
        _spines_off(ax)
    axes[0].set_ylabel(r"Average macro-$F_1$, $AA$  ($\uparrow$)")
    axes[0].annotate("better", xy=(0.02, 0.965), xytext=(0.16, 0.78),
                      arrowprops=dict(arrowstyle="->", lw=0.9, color="black"),
                      fontsize=7.5, ha="center")

    handles = [Line2D([0], [0], marker=mk, color="none", markerfacecolor=col,
                       markeredgecolor="black" if n == "PACT (ours)" else "white",
                       markeredgewidth=0.7 if n == "PACT (ours)" else 0.5,
                       markersize=11 if n == "PACT (ours)" else 6.5, label=n)
               for n, (col, mk, _) in M.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.11), columnspacing=1.1, handletextpad=0.35,
               fontsize=7.6)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    _save(fig, "fig_tradeoff")


# --------------------------------------------------------------------------
# Fig 2 -- PACT across backbones (Table 3, chain)
# --------------------------------------------------------------------------
def fig_backbones():
    colors = [C["blue"], C["sky"], C["purple"], C["gray"], C["teal"], C["verm"], C["orange"]]
    AA = {
        "TSLANet": [0.978, 0.886, 0.663], "PatchTST": [0.889, 0.767, 0.653],
        "MOMENT (frozen)": [0.984, 0.627, 0.638], "Chronos-Bolt (frozen)": [0.977, 0.630, 0.620],
        "Sundial (frozen)": [0.982, 0.806, 0.608], "Time-MoE (frozen)": [0.943, 0.757, 0.519],
        "Moirai (frozen)": [0.972, 0.666, 0.657],
    }
    F = {
        "TSLANet": [0.002, 0.007, 0.013], "PatchTST": [0.003, 0.022, 0.010],
        "MOMENT (frozen)": [0.001, 0.017, 0.011], "Chronos-Bolt (frozen)": [0.002, 0.013, 0.008],
        "Sundial (frozen)": [0.000, 0.013, 0.006], "Time-MoE (frozen)": [0.006, 0.017, 0.015],
        "Moirai (frozen)": [0.006, 0.010, 0.014],
    }
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(FIGSIZE_W, 4.3), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.3]})
    _grouped_bars(a1, DATASETS, AA, colors, ylim=(0.45, 1.02), ylabel=r"AA  ($\uparrow$)")
    a1.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.38),
              columnspacing=1.1, handletextpad=0.35, fontsize=7.8)
    _grouped_bars(a2, DATASETS, F, colors, ylim=(0, 0.05), ylabel=r"$F$  ($\downarrow$)")
    a2.axhline(0, color="black", linewidth=0.6, zorder=2)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    _save(fig, "fig_backbones")


# --------------------------------------------------------------------------
# Fig 3 -- ablations: (a) components FD chain (Table 4); (b) gate routing (Table 5)
# --------------------------------------------------------------------------
def fig_ablation():
    fig, (a, b) = plt.subplots(1, 2, figsize=(FIGSIZE_W, 3.25))

    variants = ["Frozen\n(no adapt)", "Full\nfine-tune", "Adapters,\nno gate", "PACT\n(ours)"]
    x = np.arange(4)
    w = 0.36
    tsl_aa = [0.574, 0.640, 0.748, 0.978]
    tsl_f = [0.000, 0.283, 0.032, 0.002]
    mom_aa = [0.535, None, 0.879, 0.984]
    mom_f = [0.000, None, 0.113, 0.001]

    def bars(vals, f, col, lab, off):
        heights = [0 if v is None else v for v in vals]
        a.bar(x + off, heights, w, color=col, label=lab, edgecolor="black",
              linewidth=0.5, zorder=3)
        for xi, (v, ff) in enumerate(zip(vals, f)):
            if v is None:
                a.text(xi + off, 0.465, "n/a", ha="center", va="bottom",
                       fontsize=6, rotation=90, color="#595959")
            else:
                a.text(xi + off, v + 0.012, f"F={ff:.3f}", ha="center", va="bottom",
                       fontsize=5.8, rotation=90, color="black")

    bars(tsl_aa, tsl_f, C["blue"], "TSLANet", -w / 2)
    bars(mom_aa, mom_f, C["purple"], "MOMENT (frozen)", w / 2)
    a.set_xticks(x)
    a.set_xticklabels(variants, fontsize=7.8)
    a.set_ylim(0.45, 1.14)
    a.set_ylabel(r"AA  ($\uparrow$, FD chain)")
    a.grid(True, axis="y", linewidth=0.5, color="#e1e0d9", zorder=0)
    a.legend(loc="upper left", frameon=False, fontsize=7.6)
    a.set_title("(a) Component ablation", fontsize=9)
    _spines_off(a)

    routing = {"Soft (ours)": [0.978, 0.886, 0.663], "Hard": [0.979, 0.883, 0.658],
               "Oracle": [0.982, 0.919, 0.664]}
    _grouped_bars(b, DATASETS, routing, [C["teal"], C["orange"], C["gray"]],
                  ylim=(0.45, 1.14), ylabel=r"Final-step AA  ($\uparrow$)")
    b.legend(loc="upper center", ncol=3, frameon=False, fontsize=7.8,
             columnspacing=1.1, handletextpad=0.35)
    b.set_title("(b) Gate routing (TSLANet)", fontsize=9)

    fig.tight_layout()
    _save(fig, "fig_ablation")


# --------------------------------------------------------------------------
# Fig 4 -- robustness / label efficiency (Table 6 + 7)
# --------------------------------------------------------------------------
def fig_robustness():
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(FIGSIZE_W, 2.85))

    xb = [1, 2, 5, 10, 20]
    a.plot(xb, [0.955, 0.970, 0.984, 0.987, 0.993], "-o", color=C["verm"],
           ms=5, lw=1.5, label="FD (MOMENT)")
    a.plot(xb, [0.564, 0.597, 0.627, 0.662, 0.689], "-s", color=C["orange"],
           ms=5, lw=1.5, label="HHAR (MOMENT)")
    a.plot(xb, [0.630, 0.610, 0.638, 0.658, 0.679], "-^", color=C["gray"],
           ms=5, lw=1.5, label="Sleep-EDF (MOMENT)")
    a.plot(xb, [0.959, 0.968, 0.978, 0.986, 0.990], "--o", color=C["verm"],
           ms=5, lw=1.3, markerfacecolor="white", label="FD (TSLANet)")
    a.set_xlabel("Calibration labels (%)")
    a.set_ylabel(r"AA  ($\uparrow$)")
    a.set_xticks(xb)
    a.set_ylim(0.5, 1.02)
    a.set_title("(a) Label efficiency", fontsize=9)
    a.grid(True, linewidth=0.5, color="#e1e0d9", zorder=0)
    a.legend(frameon=False, fontsize=6.6, loc="center", bbox_to_anchor=(0.56, 0.565),
              handletextpad=0.35)
    _spines_off(a)

    b.plot([4, 8, 16, 32, 64], [0.968, 0.971, 0.978, 0.976, 0.975], "-o",
           color=C["blue"], ms=5, lw=1.5, label="TSLANet")
    b.plot([8, 16, 32], [0.982, 0.984, 0.980], "-s", color=C["purple"],
           ms=5, lw=1.5, label="MOMENT")
    b.set_xscale("log", base=2)
    b.set_xticks([4, 8, 16, 32, 64])
    b.set_xticklabels(["4", "8", "16", "32", "64"])
    b.set_xlabel(r"Concepts $K$")
    b.set_ylabel(r"AA  ($\uparrow$, FD chain)")
    b.set_ylim(0.955, 0.995)
    b.set_title("(b) Concept count", fontsize=9)
    b.grid(True, linewidth=0.5, color="#e1e0d9", zorder=0)
    b.legend(frameon=False, loc="lower center", fontsize=7.5)
    _spines_off(b)

    c.plot([0.1, 0.3, 0.5, 1.0, 2.0], [0.979, 0.979, 0.978, 0.978, 0.971], "-o",
           color=C["blue"], ms=5, lw=1.5)
    c.set_xlabel(r"Gate temperature $\tau$")
    c.set_ylabel(r"AA  ($\uparrow$, FD chain, TSLANet)")
    c.set_ylim(0.965, 0.985)
    c.set_title(r"(c) Gate temperature", fontsize=9)
    c.grid(True, linewidth=0.5, color="#e1e0d9", zorder=0)
    _spines_off(c)

    fig.tight_layout()
    _save(fig, "fig_robustness")


def _save(fig, stem):
    out = RESULTS_ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}/{stem}.pdf (+ .png)")


def main():
    fig_tradeoff()
    fig_backbones()
    fig_ablation()
    fig_robustness()


if __name__ == "__main__":
    main()
