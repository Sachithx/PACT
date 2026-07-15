"""Plot the empirical check of Proposition 1 from a sequential-run artifact."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to margin_condition_source.csv")
    parser.add_argument("--stem", default="fig_margin_condition")
    return parser.parse_args()


def main():
    args = parse_args()
    data = pd.read_csv(args.input)
    data = data[data["correct"].astype(bool)].copy()
    if data.empty:
        raise ValueError("No correctly classified source samples in diagnostic artifact")

    threshold = data["margin_threshold"].to_numpy()
    displacement = data["displacement_bound"].to_numpy()
    positive = np.concatenate([threshold[threshold > 0], displacement[displacement > 0]])
    floor = max(float(positive.min()) / 2.0, 1e-12) if len(positive) else 1e-12
    x = np.maximum(threshold, floor)
    y = np.maximum(displacement, floor)
    ratio = displacement / np.maximum(threshold, floor)
    violation_rate = float((displacement >= threshold).mean())

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8.5,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    blue, vermillion, grid = "#0072B2", "#D55E00", "#e1e0d9"
    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(7.2, 2.55))

    colors = np.where(displacement >= threshold, vermillion, blue)
    ax_scatter.scatter(x, y, s=9, c=colors, alpha=0.48, linewidths=0, rasterized=True)
    low = min(x.min(), y.min())
    high = max(x.max(), y.max())
    ax_scatter.plot([low, high], [low, high], "--", color="#333333", lw=0.9)
    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlim(low, high)
    ax_scatter.set_ylim(low, high)
    ax_scatter.set_xlabel(r"Margin threshold $m(z)/(2L)$")
    ax_scatter.set_ylabel(r"Routing bound $(1-\pi_0(z))B(z)$")
    ax_scatter.set_title("(a) Pointwise preservation check")
    ax_scatter.grid(True, color=grid, linewidth=0.45)

    finite_ratio = ratio[np.isfinite(ratio)]
    lo = min(float(finite_ratio.min()), 1.0)
    hi = max(float(finite_ratio.max()), 1.0)
    bins = np.logspace(np.log10(max(lo, 1e-6)), np.log10(max(hi, 1e-6)), 30)
    if np.isclose(bins[0], bins[-1]):
        bins = np.logspace(-2, 2, 30)
    ax_hist.hist(np.maximum(finite_ratio, 1e-6), bins=bins, color=blue,
                 edgecolor="white", linewidth=0.35)
    ax_hist.axvline(1.0, color=vermillion, linestyle="--", linewidth=1.1)
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel(r"Condition ratio $(1-\pi_0)B\,/\,[m/(2L)]$")
    ax_hist.set_ylabel("Source test samples")
    ax_hist.set_title("(b) Safety-ratio distribution")
    ax_hist.grid(True, axis="y", color=grid, linewidth=0.45)
    ax_hist.text(0.97, 0.94, f"violation rate = {violation_rate:.2%}",
                 transform=ax_hist.transAxes, ha="right", va="top", fontsize=7.5)

    for ax in (ax_scatter, ax_hist):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    out = RESULTS_ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(out / f"{args.stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}/{args.stem}.pdf (+ .png)")


if __name__ == "__main__":
    main()
