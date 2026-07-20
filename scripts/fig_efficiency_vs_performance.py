"""Efficiency vs. performance: wall-clock time (log scale) vs AMF1, all 16
methods on the same chain/dataset/backbone (FD, chain 2,1,0,3, Sundial,
PACT K=4) for a fair one-to-one comparison -- pairs the timing microbenchmark
with same-run accuracy, not the 4-chain-averaged numbers used elsewhere.

Data: results/tables/efficiency_vs_performance.csv
Outputs: results/figures/fig_efficiency_vs_performance.{pdf,png}
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT

plt.rcParams.update({
    "font.size": 14, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42,
})

LABEL = {"pact": "PACT (ours)", "head-ft": "Head FT", "lwf": "LwF",
        "srcdistill": "Src-distill", "ewc": "EWC", "mas": "MAS",
        "dualcd": "DualCD", "emb-replay-head": "Emb-replay", "sprompts": "S-Prompts",
        "eata": "EATA", "cotta": "CoTTA", "mapu": "MAPU", "ncm": "NCM",
        "derpp": "DER++", "tent": "TENT", "mapu-free": "MAPU-free"}

# Simple, well-separated points: (dx, dy) offset in points.
OFFSETS = {"pact": (10, 8), "head-ft": (9, -4), "dualcd": (9, 7),
          "emb-replay-head": (-15, 10), "sprompts": (9, 8),
          "derpp": (-42, 8), "mapu-free": (9, -4)}

# Tightly-clustered points (69-81s, AA 0.72-0.78): route to explicit data
# coordinates via a thin leader line instead, stacked so nothing overlaps.
LEADER_TARGETS = {
    "lwf":        (86, 0.805),
    "mas":        (86, 0.785),
    "srcdistill": (86, 0.765),
    "eata":       (86, 0.745),
    "ncm":        (86, 0.725),
    "cotta":      (86, 0.705),
    "mapu":       (86, 0.685),
    "ewc":        (86, 0.665),
    "tent":       (86, 0.645),
}

C_PACT = "#009E73"
C_OTHER = "#6b6b6b"


def main():
    df = pd.read_csv(RESULTS_ROOT / "tables" / "efficiency_vs_performance.csv")
    df = df[df["method"] != "emb-replay-head"]

    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    for _, r in df.iterrows():
        method = r["method"]
        is_pact = method == "pact"
        ax.scatter(r["seconds"], r["AA"], s=210 if is_pact else 70,
                  marker="*" if is_pact else "o",
                  c=C_PACT if is_pact else C_OTHER,
                  edgecolor="black" if is_pact else "white",
                  linewidth=0.8 if is_pact else 0.6,
                  zorder=5 if is_pact else 3, alpha=1.0 if is_pact else 0.85)
        if method in LEADER_TARGETS:
            tx, ty = LEADER_TARGETS[method]
            ax.annotate(LABEL[method], xy=(r["seconds"], r["AA"]),
                       xytext=(tx, ty), textcoords="data", ha="left", va="center",
                       fontsize=11, color="#2a2a2a",
                       arrowprops=dict(arrowstyle="-", color="#b0b0b0", lw=0.7,
                                      shrinkA=3, shrinkB=3), zorder=6)
        else:
            dx, dy = OFFSETS[method]
            ax.annotate(LABEL[method], (r["seconds"], r["AA"]),
                       textcoords="offset points", xytext=(dx, dy),
                       fontsize=11.5, color=C_PACT if is_pact else "#2a2a2a",
                       fontweight="bold" if is_pact else "normal", zorder=6)

    ax.set_xscale("log")
    ax.set_xlabel("Wall-clock time (s, log scale) — FD chain, Sundial")
    ax.set_ylabel(r"$AMF_1$")
    ax.grid(True, which="both", linewidth=0.5, color="#e6e6e0", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.annotate("better", xy=(0.06, 0.94), xycoords="axes fraction",
               xytext=(0.22, 0.80), textcoords="axes fraction",
               fontsize=12, color="#999999", ha="center",
               arrowprops=dict(arrowstyle="-|>", color="#999999", lw=1.4))

    fig.tight_layout(pad=0.4)
    out = RESULTS_ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_efficiency_vs_performance.{ext}", dpi=300,
                   bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved -> {out}/fig_efficiency_vs_performance.pdf (+ .png)")


if __name__ == "__main__":
    main()
