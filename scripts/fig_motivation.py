"""Motivation figure (TSFM numbers) for the paper.

Panel (a): a frozen time-series foundation model (MOMENT) + linear probe is NOT
           domain-robust -- source F1 stays ~0.92 but shifted-target F1 collapses
           (per AdaTime FD shift). Motivates the need for adaptation.
Panel (b): accuracy-vs-forgetting trade-off. Frozen (no-adapt) has no forgetting
           but low accuracy; full fine-tune forgets catastrophically; adapters
           without the gate still interfere; PACT reaches the desirable
           high-accuracy / near-zero-forgetting corner.

Outputs: results/figures/motivation_tsfm.{pdf,png}
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "font.family": "sans-serif",
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# Okabe-Ito colour-blind-safe palette
C = dict(src="#8c8c8c", tgt="#0072B2", frozen="#8c8c8c", fullft="#D55E00",
         nogate="#E69F00", pact="#009E73")

FD_NAMES = ["A", "B", "C", "D"]
FD_PAIRS = [(0, 1), (0, 3), (1, 0), (1, 2), (1, 3),
            (2, 1), (2, 3), (3, 0), (3, 1), (3, 2)]


def load_linprobe():
    df = pd.read_csv(RESULTS_ROOT / "tsfm_motivation" / "fd_moment_linprobe.csv")
    labels = [f"{FD_NAMES[s]}\u2192{FD_NAMES[t]}" for s, t in FD_PAIRS][:len(df)]
    return df["src_f1"].values, df["tgt_f1"].values, labels


def panel_a(ax):
    src, tgt, labels = load_linprobe()
    order = np.argsort(tgt)
    src, tgt = src[order], tgt[order]
    labels = [labels[i] for i in order]
    y = np.arange(len(tgt))
    smean, tmean = src.mean(), tgt.mean()

    for i in range(len(y)):
        ax.plot([tgt[i], src[i]], [y[i], y[i]], color="#d9d9d9", lw=2.2, zorder=1)
    ax.scatter(src, y, color=C["src"], s=40, zorder=3)
    ax.scatter(tgt, y, color=C["tgt"], s=40, zorder=3)

    # mean lines spanning the dumbbell cloud only (stop below the annotation band)
    n = len(y)
    ax.axvline(smean, color=C["src"], ls="--", lw=1.0, alpha=0.8, zorder=0)
    ax.axvline(tmean, color=C["tgt"], ls="--", lw=1.0, alpha=0.8, zorder=0)

    # mean-drop bracket in the EMPTY band above every dumbbell
    yb = n + 0.35
    ax.annotate("", xy=(smean, yb), xytext=(tmean, yb),
                arrowprops=dict(arrowstyle="<|-|>", color="k", lw=1.0,
                                mutation_scale=10))
    ax.text((smean + tmean) / 2, yb + 0.28,
            f"mean gap \u2248 {100*(smean - tmean):.0f} F1 pts",
            ha="center", va="bottom", fontsize=7.8)

    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_ylim(-0.7, n + 1.6)
    ax.set_xlim(0.2, 1.0)
    ax.set_xlabel("macro-F1")
    ax.set_ylabel("Fault Diagnosis operating-condition shifts")
    ax.set_title("(a) TSFM (AutonLab/MOMENT-1-large) is not domain-robust",
                 pad=16, fontsize=8.8)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C["src"],
               markeredgecolor="none", markersize=7, label="source domain"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C["tgt"],
               markeredgecolor="none", markersize=7, label="shifted target")],
        loc="upper left", fontsize=7.8, frameon=False,
        handletextpad=0.2, borderpad=0.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_b(ax):
    # MOMENT-FD for frozen/no-gate/PACT; full fine-tune shown on a specialist
    # backbone (TSFM full-FT is prohibitive: frozen 341M encoder).
    # (name, F, AA, colour, marker, (dF,dAA) label offset, ha, va)
    # (name, F, AA, colour, marker, (dF,dAA) label offset, ha, va, bold)
    pts = [
        ("Frozen\n(no adapt)", 0.0000, 0.5348, C["frozen"], "o",
         (0.0, 0.085), "center", "bottom", False),
        ("Full fine-tune*",    0.2915, 0.7698, C["fullft"], "s",
         (0.0, 0.055), "center", "bottom", False),
        ("Adapters,\nno gate", 0.1130, 0.8787, C["nogate"], "^",
         (0.0, -0.055), "center", "top", False),
        ("ours",               0.0005, 0.9836, C["pact"], "*",
         (0.0, 0.052), "center", "bottom", True),
    ]
    for name, f, aa, col, mk, (df, daa), ha, va, bold in pts:
        ax.scatter(f, aa, s=(520 if mk == "*" else 150), color=col, marker=mk,
                   edgecolor="k", linewidth=0.6, zorder=3)
        ax.annotate(name, (f + df, aa + daa), fontsize=8.5, ha=ha, va=va,
                    fontweight=("bold" if bold else "normal"))

    ax.set_xlabel("Forgetting  $F$  (\u2192 lower is better)")
    ax.set_ylabel("Average accuracy  $AA$")
    ax.set_xlim(-0.03, 0.35); ax.set_ylim(0.45, 1.08)
    ax.invert_xaxis()   # low forgetting (good) on the right
    ax.set_title("(b) Accuracy\u2013forgetting trade-off", pad=16)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    out = RESULTS_ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    panel_a(axes[0]); panel_b(axes[1])
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.15, top=0.86, wspace=0.32)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"motivation_tsfm.{ext}", dpi=220, bbox_inches="tight")
    print(f"Saved -> {out}/motivation_tsfm.pdf (+ .png)")


if __name__ == "__main__":
    main()
