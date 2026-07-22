"""Empirical verification figure for Proposition 2 (target recovery under
chart relatedness), FD/Sundial/chain 2,1,0,3, K in {8,16,64}.

(a) Pointwise check at the dataset's best-K (8): scatter of the recovery
    threshold gamma_c(z)/(2L) against the achieved
    eps_c(z) + 2(1-pi_own(z))B(z), log-log, colored by whether the deployed
    (gated-mixture) prediction is actually correct. The literal certificate
    never binds (all points sit above the diagonal) -- expected for a
    Lipschitz-margin bound in a high-dimensional embedding space -- but most
    points are still colored "correct", showing recovery happens even where
    the bound does not fire.
(b) Recovery accuracy vs K: isolated (own transport alone) and deployed
    (full gated mixture) recovery stay high across K=8/16/64 even though the
    raw gap does not shrink, i.e. the mechanism (chart-local affine
    relatedness) is doing real work independent of this particular
    raw-distance instantiation of epsilon_c.

Data: results/prop_verification/fd_sundial_2,1,0,3_K{8,16,64}_c5_s42_prop2.csv
Outputs: results/figures/fig_prop2_verification.{pdf,png}
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

BLUE, VERM, GRID = "#0072B2", "#D55E00", "#e1e0d9"
TEAL, GRAY = "#009E73", "#8c8c8c"
K_VALUES = [8, 16, 64]
TAG = "fd_sundial_2,1,0,3"


def load(k):
    return pd.read_csv(RESULTS_ROOT / "prop_verification" / f"{TAG}_K{k}_c5_s42_prop2.csv")


def main():
    df8 = load(8)
    d = df8[df8["has_anchor"]].copy()
    lhs = d["eps"].to_numpy() + 2.0 * (1.0 - d["pi_own"].to_numpy()) * d["B"].to_numpy()
    rhs = d["rhs"].to_numpy()
    deployed = d["deployed_correct"].to_numpy().astype(bool)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    colors = np.where(deployed, BLUE, VERM)
    ax1.scatter(rhs, lhs, s=11, c=colors, alpha=0.45, linewidths=0, rasterized=True)
    low = min(rhs.min(), lhs.min()) * 0.8
    high = max(rhs.max(), lhs.max()) * 1.2
    ax1.plot([low, high], [low, high], "--", color="#333333", lw=1.0, label="boundary")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(low, high)
    ax1.set_ylim(low, high)
    ax1.set_xlabel(r"Threshold $\gamma_c(z)/(2L)$")
    ax1.set_ylabel(r"$\varepsilon_c(z)+2(1{-}\pi_{c,d}(z))B(z)$")
    ax1.set_title("(a) Recovery condition, K=8", fontsize=14)
    ax1.grid(True, color=GRID, linewidth=0.5)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=BLUE, label="Deployed correct"),
              Patch(facecolor=VERM, label="Deployed wrong")]
    ax1.legend(handles=handles, loc="lower right", frameon=False, fontsize=11.5,
              handletextpad=0.4)

    ks = K_VALUES
    iso, dep, eps_med = [], [], []
    for k in ks:
        dfk = load(k)
        dk = dfk[dfk["has_anchor"]]
        iso.append(dk["isolated_correct"].mean() * 100)
        dep.append(dfk["deployed_correct"].mean() * 100)
        eps_med.append(dk["eps"].median())

    x = np.arange(len(ks))
    width = 0.32
    ax2.bar(x - width / 2, iso, width=width, color=TEAL, edgecolor="black",
           linewidth=0.5, label="Isolated recovery", zorder=3)
    ax2.bar(x + width / 2, dep, width=width, color=GRAY, edgecolor="black",
           linewidth=0.5, label="Deployed recovery", zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"K={k}" for k in ks])
    ax2.set_ylabel("Target recovery accuracy (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title("(b) Recovery stays high across K", fontsize=14)
    ax2.grid(True, axis="y", color=GRID, linewidth=0.5, zorder=0)
    ax2.legend(loc="lower center", frameon=False, fontsize=11.5, ncol=1)

    ax2b = ax2.twinx()
    ax2b.plot(x, eps_med, "o--", color="#333333", markersize=6, linewidth=1.2,
             label=r"median $\varepsilon_c$")
    ax2b.set_ylabel(r"median $\varepsilon_c(z)$", fontsize=13)
    ax2b.set_ylim(0, max(eps_med) * 1.4)
    ax2b.legend(loc="upper right", frameon=False, fontsize=11.5)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax2b.spines["top"].set_visible(False)

    fig.tight_layout(pad=0.4, w_pad=1.6)
    out = RESULTS_ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_prop2_verification.{ext}", dpi=300,
                   bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved -> {out}/fig_prop2_verification.pdf (+ .png)")


if __name__ == "__main__":
    main()
