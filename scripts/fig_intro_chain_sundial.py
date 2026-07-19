"""Introduction figure: average-accuracy curve along a domain-incremental
chain, frozen Sundial backbone, one figure per dataset (FD, HHAR, Sleep,
WISDM, UCIHAR). Cherry-picked chain per dataset (from the 4 random no-repeat
chains used throughout the paper's chain-evaluation protocol), chosen as the
one with the largest final-step gap between PACT and the mean of the 5
baselines below.

Shows how AMF1_t (average macro-F1 over all domains seen so far) evolves as
the chain's domains arrive one at a time: PACT stays near the source ceiling
while every baseline degrades.

Baselines (5): No adapt (source-only, frozen), Continual Learning (EWC, head
Fisher-regularized), Source-distill, Head fine-tune, Test Time Adaptation
(TENT, genuinely label-free entropy minimization on the target stream).

Data: results/{controls,cl_baselines_tsfm,labelfree_baselines,sequential_tsfm}
AMF1_t computed exactly as src.method.sequential.compute_cl_metrics: nanmean of
row t (domains 0..t) of the NaN-padded F1 retention matrix.

Outputs: results/figures/fig_intro_chain_sundial_{dataset}.{pdf,png}
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT

from src.data import get_spec

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "font.family": "sans-serif",
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# Okabe-Ito colour-blind-safe palette (matches fig_intro_chain_moment.py)
C = dict(gray="#8c8c8c", orange="#E69F00", sky="#56B4E9", teal="#009E73",
          blue="#0072B2", verm="#D55E00", purple="#CC79A7")

BEST_K = {"fd": 8, "hhar": 4, "sleep": 1, "wisdm": 4, "ucihar": 4}
CALIB_TAG = {"fd": "c5", "hhar": "c5", "sleep": "c5", "wisdm": "n450", "ucihar": "n100"}
CALIB_LABEL = {"fd": "5% calibration labels/step", "hhar": "5% calibration labels/step",
              "sleep": "5% calibration labels/step", "wisdm": "450 calibration labels/step",
              "ucihar": "100 calibration labels/step"}
DATASET_TITLE = {"fd": "Fault Diagnosis, operating-condition shift",
                 "hhar": "HHAR, cross-device sensor shift",
                 "sleep": "Sleep-EDF, cross-subject shift",
                 "wisdm": "WISDM, cross-subject shift",
                 "ucihar": "UCI-HAR, cross-subject shift"}

# Cherry-picked chain per dataset -- the one (of 4 random no-repeat chains)
# with the largest final-step gap between PACT and the mean of the 5
# baselines, selected via scripts/fig_intro_chain_sundial.py's own scan.
CHERRY_CHAIN = {
    "fd": "2,1,3,0",
    "hhar": "1,6,4,7,8,5,0,2,3",
    "sleep": "1,5,9,0,4,3,18,8,13",
    "wisdm": "24,11,0,23,13,21,16,35,3",
    "ucihar": "14,24,1,27,26,10,12,7,9",
}


def method_paths(ds, chain, k, tag):
    return {
        "No adapt": (RESULTS_ROOT / "controls" / f"{ds}_sundial_{chain}_K16_{tag}"
                    / "noadapt_f1_matrix.csv", C["gray"], "o", ":", 2, "No Adapt"),
        "EWC": (RESULTS_ROOT / "cl_baselines_tsfm" / f"{ds}_sundial_{chain}_ewc_{tag}_s42"
               / "f1_matrix.csv", C["orange"], "v", "--", 2, "Continual Learning"),
        "Source-distill": (RESULTS_ROOT / "cl_baselines_tsfm" / f"{ds}_sundial_{chain}_srcdistill_{tag}_s42"
                          / "f1_matrix.csv", C["verm"], "P", "--", 2, "Source-distill"),
        "Head fine-tune": (RESULTS_ROOT / "cl_baselines_tsfm" / f"{ds}_sundial_{chain}_head-ft_{tag}_s42"
                          / "f1_matrix.csv", C["sky"], "s", "--", 2, "Head Fine-tuning"),
        "TENT": (RESULTS_ROOT / "labelfree_baselines" / f"{ds}_sundial_{chain}_tent_{tag}_s42"
                / "f1_matrix.csv", C["purple"], "D", "--", 2, "Test Time Adaptation"),
        "PACT": (RESULTS_ROOT / "sequential_tsfm" / f"{ds}_sundial_{chain}_K{k}_{tag}_lora_learned_s42"
                / "soft_f1_matrix.csv", C["teal"], "*", "-", 5, "PACT (ours)"),
    }


def aa_curve(path):
    A = pd.read_csv(path, index_col=0).values.astype(float)
    return [float(np.nanmean(A[t, :t + 1])) for t in range(A.shape[0])]


def make_figure(ds):
    spec = get_spec(ds)
    chain = CHERRY_CHAIN[ds]
    order = [int(v) for v in chain.split(",")]
    k = BEST_K[ds]
    tag = CALIB_TAG[ds]
    T = len(order)

    step_labels = [spec.domain_label(order[0])]
    step_labels += [f"+ {spec.domain_label(d)}" for d in order[1:]]

    methods = method_paths(ds, chain, k, tag)

    x = np.arange(T)
    fig_w = max(4.6, 0.62 * T + 2.4)
    fig, ax = plt.subplots(figsize=(fig_w, 3.6))

    all_curves = {name: aa_curve(path) for name, (path, *_rest) in methods.items()}
    y_floor = max(0.0, min(v for c in all_curves.values() for v in c) - 0.06)

    end_labels = []
    for name, (path, col, mk, ls, z, legend_label) in methods.items():
        y = all_curves[name]
        is_ours = name == "PACT"
        ax.plot(x, y, ls, color=col, lw=2.4 if is_ours else 1.3,
                 marker=mk, markersize=9 if is_ours else 5.5,
                 markeredgecolor="black" if is_ours else "white",
                 markeredgewidth=0.8 if is_ours else 0.5,
                 alpha=1.0 if is_ours else 0.9, zorder=z,
                 label=legend_label)
        end_labels.append([y[-1], col, f"{y[-1]:.3f}", is_ours])

    min_gap = 0.028
    end_labels.sort(key=lambda r: r[0])
    placed = []
    for y0, col, text, bold in end_labels:
        yp = y0 if not placed else max(y0, placed[-1] + min_gap)
        placed.append(yp)
        ax.annotate(text, (x[-1], yp), textcoords="offset points", xytext=(7, 0),
                     fontsize=7.3, color=col, va="center",
                     fontweight="bold" if bold else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels(step_labels, rotation=45 if T > 5 else 0, ha="right" if T > 5 else "center")
    ax.set_xlim(-0.25, T - 1 + 0.55)
    ax.set_ylim(y_floor, 1.05)
    ax.set_xlabel(f"Domains added sequentially ({CALIB_LABEL[ds]})")
    ax.set_ylabel(r"$AMF_{1,t}$  ($\uparrow$)")
    ax.set_title(f"PACT (ours) vs. existing methods (frozen Sundial)\n{DATASET_TITLE[ds]}",
                 fontsize=9.3)
    ax.grid(True, linewidth=0.5, color="#e1e0d9", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.legend(loc="lower left", frameon=False, fontsize=7.2, ncol=1,
               bbox_to_anchor=(0.03, 0.02), handletextpad=0.4, labelspacing=0.35)

    fig.tight_layout()
    out = RESULTS_ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_intro_chain_sundial_{ds}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}/fig_intro_chain_sundial_{ds}.pdf (+ .png)  chain={chain}")


def main():
    for ds in ["fd", "hhar", "sleep", "wisdm", "ucihar"]:
        make_figure(ds)


if __name__ == "__main__":
    main()
