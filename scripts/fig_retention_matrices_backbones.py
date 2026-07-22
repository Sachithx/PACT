"""Per-domain retention matrices (A_{t,j}) for every backbone, one figure per
dataset, extending the original TSLANet-only/3-dataset figure
(fig_retention_matrices.py) to all 5 datasets x all 8 backbones with
complete established-chain coverage.

Uses each dataset's FIRST established random chain and the same per-backbone
K used in the backbones comparison table (results/tables/chain_system_backbones.csv):
K=16 for TSLANet/PatchTST/Sundial/Mantis/Moirai/UniShape/Chronos-Bolt, and
each dataset's tuned best-K for MOMENT (only available at that K).

Outputs results/figures/fig_retention_matrices_{dataset}.{pdf,png}, one file
per dataset, each with 8 backbone panels (2x4 grid) sharing one colorbar.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9.5, "axes.labelsize": 8,
    "xtick.labelsize": 6, "ytick.labelsize": 6,
    "font.family": "sans-serif", "axes.linewidth": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

with open("/tmp/random_chains_fd.json") as f:
    FD_CHAINS = json.load(f)["fd"]
with open("/tmp/random_chains.json") as f:
    OTHER_CHAINS = json.load(f)
with open("/tmp/random_chains_ucihar.json") as f:
    UCIHAR_CHAINS = json.load(f)["ucihar"]
CHAINS = {"fd": FD_CHAINS, "ucihar": UCIHAR_CHAINS, **OTHER_CHAINS}

BEST_K = {"fd": 8, "hhar": 4, "sleep": 1, "wisdm": 4, "ucihar": 4}
CALIB_TAG = {"fd": "c5", "hhar": "c5", "sleep": "c5", "wisdm": "n450", "ucihar": "n100"}
DS_ORDER = ["fd", "hhar", "sleep", "wisdm", "ucihar"]
DS_LABEL = {"fd": "FD", "hhar": "HHAR", "sleep": "Sleep-EDF", "wisdm": "WISDM", "ucihar": "UCI-HAR"}

# (K or None-for-best-K, path suffix, results root)
BACKBONE_SPEC = {
    "TSLANet": (16, "t0.5_lora_learned_s42", "sequential"),
    "PatchTST": (16, "t0.5_lora_learned_s42", "sequential"),
    "MOMENT (frozen)": (None, "lora_learned_s42", "sequential_tsfm"),
    "Chronos-Bolt (frozen)": (16, "lora_learned_s42", "sequential_tsfm"),
    "Sundial (frozen, ours)": (16, "lora_learned_s42", "sequential_tsfm"),
    "Mantis (frozen)": (16, "lora_learned_s42", "sequential_tsfm"),
    "Moirai (frozen)": (16, "lora_learned_s42", "sequential_tsfm"),
    "UniShape (frozen)": (16, "lora_learned_s42", "sequential_tsfm"),
}
BACKBONE_DIR_TOKEN = {
    "TSLANet": "tslanet", "PatchTST": "patchtst", "MOMENT (frozen)": "moment",
    "Chronos-Bolt (frozen)": "chronos-bolt", "Sundial (frozen, ours)": "sundial",
    "Mantis (frozen)": "mantis", "Moirai (frozen)": "moirai", "UniShape (frozen)": "unishape",
}


def cstr(chain):
    return ",".join(map(str, chain))


def short_labels(dataset, labels):
    if dataset == "fd":
        return [l.replace("FD-", "") for l in labels]
    if dataset == "hhar":
        return [l.replace("User-", "U") for l in labels]
    return [l.replace("Subj-", "S") for l in labels]


def draw_matrix(ax, dataset, frame, title):
    values = frame.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    image = ax.imshow(masked, cmap=cmap, vmin=0.35, vmax=1.0, interpolation="none", aspect="equal")

    labels = short_labels(dataset, list(frame.columns))
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels)
    ax.set_title(title, pad=4, fontweight="bold", fontsize=8.5)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.3, alpha=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=2)
    return image


def main():
    out_dir = RESULTS_ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in DS_ORDER:
        tag = CALIB_TAG[ds]
        chain = CHAINS[ds][0]
        cs = cstr(chain)

        matrices, missing = {}, []
        for bb_label, (k, suffix, root) in BACKBONE_SPEC.items():
            k_use = k if k is not None else BEST_K[ds]
            token = BACKBONE_DIR_TOKEN[bb_label]
            path = (RESULTS_ROOT / root /
                   f"{ds}_{token}_{cs}_K{k_use}_{tag}_{suffix}" / "soft_f1_matrix.csv")
            if path.exists():
                matrices[bb_label] = pd.read_csv(path, index_col=0)
            else:
                missing.append((bb_label, str(path)))
        if missing:
            print(f"[{ds}] missing {len(missing)} backbone matrices:")
            for lbl, p in missing:
                print(f"    {lbl}: {p}")

        n_domains = len(next(iter(matrices.values())).columns)
        fig_w = 13.5
        fig_h = 3.3 * 2 if n_domains <= 4 else 3.6 * 2
        fig, axes = plt.subplots(2, 4, figsize=(fig_w, fig_h))
        image = None
        for ax, (bb_label, frame) in zip(axes.flat, matrices.items()):
            image = draw_matrix(ax, ds, frame, bb_label)
        for ax in axes.flat[len(matrices):]:
            ax.axis("off")

        if image is not None:
            fig.subplots_adjust(left=0.05, right=0.90, bottom=0.08, top=0.93,
                               hspace=0.65 if n_domains <= 4 else 0.9, wspace=0.35)
            cax = fig.add_axes((0.925, 0.15, 0.012, 0.7))
            cbar = fig.colorbar(image, cax=cax)
            cbar.set_label(r"Macro-$F_1$, $A_{t,j}$", labelpad=6)
            cbar.set_ticks([0.4, 0.6, 0.8, 1.0])
            fig.suptitle(f"{DS_LABEL[ds]} — per-domain retention across backbones "
                        f"(chain {cs})", fontsize=10, y=0.985)

        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"fig_retention_matrices_{ds}.{ext}", dpi=250,
                       bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[{ds}] saved -> {out_dir}/fig_retention_matrices_{ds}.pdf (+ .png)  "
              f"({len(matrices)}/8 backbones)")


if __name__ == "__main__":
    main()
