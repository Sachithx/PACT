"""Introduction figure: average-accuracy curve along the FD domain-incremental
chain (0->1->2->3), frozen MOMENT-1-large backbone, 5% calibration per step.

Shows how AA_t = mean F1 over all domains seen so far evolves as FD-A -> B ->
C -> D arrive one at a time: PACT stays essentially flat and near the
source ceiling while every CL baseline degrades, and the frozen (no-adapt)
backbone collapses fastest. Full fine-tuning of the 341M-parameter frozen
MOMENT encoder is not run (prohibitive / not scalable -- same call as
fig_motivation.py / fig_ablation.py) and is annotated rather than plotted.

Data: results/{controls,cl_baselines_tsfm,sequential_tsfm}/fd_moment_adatime_*_c5
AA_t computed exactly as src.method.sequential.compute_cl_metrics: nanmean of
row t (domains 0..t) of the NaN-padded F1 retention matrix.

Outputs: results/figures/fig_intro_chain_moment.{pdf,png}
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "font.family": "sans-serif",
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# Okabe-Ito colour-blind-safe palette (matches fig_motivation.py / fig_experiments.py)
C = dict(gray="#8c8c8c", orange="#E69F00", sky="#56B4E9", teal="#009E73",
          blue="#0072B2", verm="#D55E00", purple="#CC79A7")

STEP_LABELS = ["FD-A", "+ FD-B", "+ FD-C", "+ FD-D"]

# (display name, results subdir, matrix filename, colour, marker, linestyle, zorder)
# EWC/MAS regularize only the frozen-TSFM's linear head (Fisher / MAS-importance
# weighted L2 to each past domain's head), added to run_cl_baselines_tsfm.py and
# re-run 2026-07-13 (~20min/run on MOMENT-1-large) after the previously present
# fd_moment_adatime_{ewc,mas}_c5 folders were found to be stale 2-second
# placeholder artifacts not reproducible with any code in this repo.
METHODS = [
    ("No adapt (frozen)", "controls/fd_moment_adatime_K16_c5", "noadapt_f1_matrix.csv",
     C["gray"], "o", ":", 2),
    ("EWC (head)", "cl_baselines_tsfm/fd_moment_adatime_ewc_c5", "f1_matrix.csv",
     C["orange"], "v", "--", 2),
    ("Source-distill", "cl_baselines_tsfm/fd_moment_adatime_srcdistill_c5", "f1_matrix.csv",
     C["verm"], "P", "--", 2),
    ("LwF", "cl_baselines_tsfm/fd_moment_adatime_lwf_c5", "f1_matrix.csv",
     C["blue"], "^", "--", 2),
    ("Head fine-tune", "cl_baselines_tsfm/fd_moment_adatime_head-ft_c5", "f1_matrix.csv",
     C["sky"], "s", "--", 2),
    ("MAS (head)", "cl_baselines_tsfm/fd_moment_adatime_mas_c5", "f1_matrix.csv",
     C["purple"], "D", "--", 2),
    ("PACT (ours)", "sequential_tsfm/fd_moment_adatime_K16_c5", "soft_f1_matrix.csv",
     C["teal"], "*", "-", 5),
]


def aa_curve(path):
    A = pd.read_csv(path, index_col=0).values.astype(float)
    return [float(np.nanmean(A[t, :t + 1])) for t in range(A.shape[0])]


def main():
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))

    end_labels = []  # (y_final, color, text, bold)
    for name, subdir, fname, col, mk, ls, z in METHODS:
        y = aa_curve(RESULTS_ROOT / subdir / fname)
        is_ours = name.startswith("PACT")
        ax.plot(x, y, ls, color=col, lw=2.4 if is_ours else 1.3,
                 marker=mk, markersize=9 if is_ours else 5.5,
                 markeredgecolor="black" if is_ours else "white",
                 markeredgewidth=0.8 if is_ours else 0.5,
                 alpha=1.0 if is_ours else 0.9, zorder=z,
                 label=name)
        end_labels.append([y[-1], col, f"{y[-1]:.3f}", is_ours])

    # Greedily separate right-edge value labels that would otherwise overlap
    # (axis spans ~0.75 data units over ~300 px of plot height at this figsize).
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
    ax.set_xticklabels(STEP_LABELS)
    ax.set_xlim(-0.25, 3.55)
    ax.set_ylim(0.30, 1.05)
    ax.set_xlabel("Domains added sequentially (5% calibration labels/step)")
    ax.set_ylabel(r"Average macro-$F_1$, $AA_t$  ($\uparrow$)")
    ax.set_title("PACT (ours) vs. existing methods (frozen MOMENT-1-large)\n"
                  "Fault Diagnosis, operating-condition shift",
                  fontsize=9.3)
    ax.grid(True, linewidth=0.5, color="#e1e0d9", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.legend(loc="lower left", frameon=False, fontsize=7.2, ncol=1,
               bbox_to_anchor=(0.03, 0.10), handletextpad=0.4, labelspacing=0.35)

    fig.tight_layout()
    out = RESULTS_ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_intro_chain_moment.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved -> {out}/fig_intro_chain_moment.pdf (+ .png)")


if __name__ == "__main__":
    main()
