"""Introduction chain figures: AA_t curves along the domain-incremental chain.

Supports frozen TSFMs (MOMENT, Sundial) and the TSLANet specialist backbone.
Generates results/figures/fig_intro_chain_{backbone}_{dataset}.{pdf,png}

Usage:
  python scripts/fig_intro_chain.py --backbone sundial,tslanet --dataset all
  python scripts/fig_intro_chain.py --backbone moment --dataset fd
"""

import argparse

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

C = dict(gray="#8c8c8c", orange="#E69F00", sky="#56B4E9", teal="#009E73",
         blue="#0072B2", verm="#D55E00", purple="#CC79A7")

DATASET_META = {
    "fd": dict(title="Fault Diagnosis, operating-condition shift", suffix="c5",
               calib_note="5% calibration labels/step"),
    "hhar": dict(title="HHAR, cross-user shift", suffix="c5",
                 calib_note="5% calibration labels/step"),
    "sleep": dict(title="Sleep-EDF, cross-subject shift", suffix="c5",
                  calib_note="5% calibration labels/step"),
    "wisdm": dict(title="WISDM, cross-subject shift", suffix="n450",
                  calib_note="$\\min(450, |\\mathrm{train}|)$ labels/step"),
}

BACKBONES = {
    "moment": dict(
        display="frozen MOMENT-1-large",
        baselines_dir="cl_baselines_tsfm",
        controls="controls/{ds}_moment_adatime_K16_{suf}",
        baseline="cl_baselines_tsfm/{ds}_moment_adatime_{method}_{suf}",
        pact={
            "fd": "sequential_tsfm/fd_moment_adatime_K16_c5",
            "hhar": "sequential_tsfm/hhar_moment_adatime_K16_c5",
            "sleep": "sequential_tsfm/sleep_moment_adatime_K16_c5",
            "wisdm": "sequential_tsfm/wisdm_moment_adatime_K16_n450_learned",
        },
        fd_extra=[
            ("EWC", "cl_baselines_tsfm/fd_moment_adatime_ewc_c5", "f1_matrix.csv",
             C["orange"], "v", "--", 2),
            ("MAS", "cl_baselines_tsfm/fd_moment_adatime_mas_c5", "f1_matrix.csv",
             C["purple"], "D", "--", 2),
        ],
    ),
    "sundial": dict(
        display="frozen Sundial",
        baselines_dir="cl_baselines_tsfm",
        controls="controls/{ds}_sundial_adatime_K16_{suf}",
        baseline="cl_baselines_tsfm/{ds}_sundial_adatime_{method}_{suf}",
        pact={
            "fd": "sequential_tsfm/fd_sundial_adatime_K16_c5",
            "hhar": "sequential_tsfm/hhar_sundial_adatime_K16_c5",
            "sleep": "sequential_tsfm/sleep_sundial_adatime_K16_c5",
            "wisdm": "sequential_tsfm/wisdm_sundial_adatime_K16_n450_learned",
        },
        fd_extra=[],
    ),
    "tslanet": dict(
        display="TSLANet",
        baselines_dir="cl_baselines",
        controls="controls/{ds}_tslanet_adatime_K16_c5",
        baseline="cl_baselines/{ds}_tslanet_adatime_{method}_c5",
        pact={
            "fd": "sequential/fd_tslanet_adatime_K16_c5_t0.5_identity",
            "hhar": "sequential/hhar_tslanet_adatime_K16_c5_t0.5_identity",
            "sleep": "sequential/sleep_tslanet_adatime_K16_c5_t0.5_identity",
        },
        fd_extra=[
            ("EWC", "cl_baselines/fd_tslanet_adatime_ewc_c5", "f1_matrix.csv",
             C["orange"], "v", "--", 2),
            ("MAS", "cl_baselines/fd_tslanet_adatime_mas_c5", "f1_matrix.csv",
             C["purple"], "D", "--", 2),
        ],
    ),
}

HEAD_METHODS = [
    ("srcdistill", "Source-distill", C["verm"], "P"),
    ("lwf", "LwF", C["blue"], "^"),
    ("head-ft", "Head fine-tune", C["sky"], "s"),
]


def aa_curve(path):
    A = pd.read_csv(path, index_col=0).values.astype(float)
    return [float(np.nanmean(A[t, :t + 1])) for t in range(A.shape[0])]


def step_labels(pact_metrics_path):
    df = pd.read_csv(pact_metrics_path)
    return [str(dom) if i == 0 else f"+ {dom}"
            for i, dom in enumerate(df["domain"])]


def build_methods(backbone, ds):
    bb = BACKBONES[backbone]
    meta = DATASET_META[ds]
    suf = meta["suffix"]
    pact_sub = bb["pact"].get(ds)
    if not pact_sub:
        return None, None

    methods = [
        ("No adapt (frozen)", bb["controls"].format(ds=ds, suf=suf),
         "noadapt_f1_matrix.csv", C["gray"], "o", ":", 2),
    ]
    for method_key, label, col, mk in HEAD_METHODS:
        sub = bb["baseline"].format(ds=ds, suf=suf, method=method_key)
        methods.append((label, sub, "f1_matrix.csv", col, mk, "--", 2))
    if ds == "fd" and bb.get("fd_extra"):
        methods.extend(bb["fd_extra"])
    methods.append(("PACT (ours)", pact_sub, "soft_f1_matrix.csv",
                    C["teal"], "*", "-", 5))
    return methods, pact_sub


def plot_dataset(backbone, ds, out_dir):
    bb = BACKBONES[backbone]
    dmeta = DATASET_META[ds]
    methods, pact_sub = build_methods(backbone, ds)
    if methods is None:
        print(f"  skip {ds}: no PACT run for {backbone}")
        return

    pact_metrics = RESULTS_ROOT / pact_sub / "metrics.csv"
    if not pact_metrics.exists():
        print(f"  skip {ds}: missing {pact_metrics}")
        return

    curves = []
    for name, subdir, fname, col, mk, ls, z in methods:
        path = RESULTS_ROOT / subdir / fname
        if not path.exists():
            print(f"  skip {name}: {path} not found")
            continue
        y = aa_curve(path)
        curves.append((name, y, col, mk, ls, z))

    if not curves:
        print(f"  skip {ds}: no curves loaded")
        return

    T = len(curves[0][1])
    x = np.arange(T)
    labels = step_labels(pact_metrics)

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    end_labels = []
    for name, y, col, mk, ls, z in curves:
        is_ours = name.startswith("PACT")
        ax.plot(x, y, ls, color=col, lw=2.4 if is_ours else 1.3,
                marker=mk, markersize=9 if is_ours else 5.5,
                markeredgecolor="black" if is_ours else "white",
                markeredgewidth=0.8 if is_ours else 0.5,
                alpha=1.0 if is_ours else 0.9, zorder=z, label=name)
        end_labels.append([y[-1], col, f"{y[-1]:.3f}", is_ours])

    min_gap = 0.028 if T <= 6 else 0.018
    end_labels.sort(key=lambda r: r[0])
    placed = []
    for y0, col, text, bold in end_labels:
        yp = y0 if not placed else max(y0, placed[-1] + min_gap)
        placed.append(yp)
        ax.annotate(text, (x[-1], yp), textcoords="offset points", xytext=(7, 0),
                    fontsize=7.3, color=col, va="center",
                    fontweight="bold" if bold else "normal")

    ax.set_xticks(x)
    if T <= 10:
        ax.set_xticklabels(labels, rotation=35 if T > 5 else 0, ha="right")
    else:
        tick_labels = [labels[i] if i % 2 == 0 or i == T - 1 else "" for i in range(T)]
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)

    ax.set_xlim(-0.35, T - 1 + 0.55)
    ylo = max(0.25, min(y[-1] for _, y, *_ in curves) - 0.12)
    ax.set_ylim(ylo, 1.05)
    ax.set_xlabel(f"Domains added sequentially ({dmeta['calib_note']})")
    ax.set_ylabel(r"Average accuracy $AA_t$  ($\uparrow$)")
    ax.set_title(f"PACT (ours) vs. existing methods ({bb['display']})\n"
                 + dmeta["title"], fontsize=9.3)
    ax.grid(True, linewidth=0.5, color="#e1e0d9", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.legend(loc="lower left", frameon=False, fontsize=7.0, ncol=1,
              bbox_to_anchor=(0.03, 0.08), handletextpad=0.4, labelspacing=0.32)

    fig.tight_layout()
    stem = f"fig_intro_chain_{backbone}_{ds}"
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    # backward-compat: original FD MOMENT figure name
    if backbone == "moment" and ds == "fd":
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"fig_intro_chain_moment.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_dir}/{stem}.pdf (+ .png)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="sundial,tslanet",
                   help="moment, sundial, tslanet (comma-separated or 'all')")
    p.add_argument("--dataset", default="all",
                   help="fd,hhar,sleep,wisdm or 'all'")
    args = p.parse_args()

    backbones = list(BACKBONES) if args.backbone == "all" else [b.strip() for b in args.backbone.split(",")]
    datasets = list(DATASET_META) if args.dataset == "all" else [d.strip() for d in args.dataset.split(",")]

    out_dir = RESULTS_ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for bb in backbones:
        if bb not in BACKBONES:
            raise ValueError(f"Unknown backbone {bb!r}")
        for ds in datasets:
            if ds not in DATASET_META:
                raise ValueError(f"Unknown dataset {ds!r}")
            print(f"=== {bb} / {ds} ===")
            plot_dataset(bb, ds, out_dir)


if __name__ == "__main__":
    main()
