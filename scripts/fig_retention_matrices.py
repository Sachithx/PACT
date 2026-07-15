"""Publication figure for the full PACT per-domain retention matrices."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT


plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "font.family": "sans-serif",
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


RUNS = {
    "FD": "fd_tslanet_adatime_K16_c5_t0.5_identity",
    "HHAR": "hhar_tslanet_adatime_K16_c5_t0.5_identity",
    "Sleep-EDF": "sleep_tslanet_adatime_K16_c5_t0.5_identity",
}


def _short_labels(dataset, labels):
    if dataset == "FD":
        return [label.replace("FD-", "") for label in labels]
    if dataset == "HHAR":
        return [label.replace("User-", "U") for label in labels]
    return [label.replace("Subj-", "S") for label in labels]


def _draw_matrix(ax, dataset, frame, annotate):
    values = frame.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    image = ax.imshow(
        masked,
        cmap=cmap,
        vmin=0.35,
        vmax=1.0,
        interpolation="none",
        aspect="equal",
    )

    labels = _short_labels(dataset, list(frame.columns))
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Held-out test domain")
    ax.set_ylabel("After adapting through")
    ax.set_title(dataset, pad=5, fontweight="bold")

    # Thin cell boundaries keep the long Sleep-EDF stream legible in print.
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.35, alpha=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=2)

    if annotate:
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                if np.isnan(value):
                    continue
                color = "white" if value < 0.72 else "#111111"
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.1 if len(labels) <= 4 else 4.1,
                    color=color,
                )
    return image


def main():
    matrices = {}
    for dataset, run in RUNS.items():
        path = RESULTS_ROOT / "sequential" / run / "soft_f1_matrix.csv"
        matrices[dataset] = pd.read_csv(path, index_col=0)

    fig = plt.figure(figsize=(7.6, 6.25))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(0.82, 1.72),
        left=0.08,
        right=0.88,
        bottom=0.09,
        top=0.97,
        hspace=0.48,
        wspace=0.42,
    )
    axes = {
        "FD": fig.add_subplot(grid[0, 0]),
        "HHAR": fig.add_subplot(grid[0, 1]),
        "Sleep-EDF": fig.add_subplot(grid[1, :]),
    }

    image = _draw_matrix(axes["FD"], "FD", matrices["FD"], annotate=True)
    _draw_matrix(axes["HHAR"], "HHAR", matrices["HHAR"], annotate=True)
    _draw_matrix(
        axes["Sleep-EDF"],
        "Sleep-EDF",
        matrices["Sleep-EDF"],
        annotate=False,
    )

    colorbar_ax = fig.add_axes((0.915, 0.18, 0.018, 0.66))
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label(r"Macro-$F_1$, $A_{t,j}$", labelpad=6)
    colorbar.set_ticks([0.4, 0.6, 0.8, 1.0])

    out = RESULTS_ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_retention_matrices.pdf", bbox_inches="tight")
    fig.savefig(
        out / "fig_retention_matrices.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
