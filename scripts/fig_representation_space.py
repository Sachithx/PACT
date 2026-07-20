"""Publication figure: how PACT reshapes the representation space across a
2-step domain-incremental chain (Mantis backbone, HHAR dataset: source
User-1 -> step1 User-7 -> step2 User-0, chosen for a large raw no-adapt drop).

Three-panel progressive story on a SOURCE-FITTED LDA projection (not t-SNE):
LDA finds the axes that actually separate classes for a linear classifier,
so a point landing in the "wrong" region is meaningful here in a way it
isn't under t-SNE (which preserves local neighbor structure, not linear
decision boundaries -- a small-but-classification-fixing PACT correction can
be invisible in t-SNE while still being real).

  1. Source only -- reference class clusters (LDA axes fit here).
  2. New domains WITHOUT PACT (faint source backdrop + step1/step2 raw
     embeddings, same LDA transform) -- shows raw points landing off the
     correct class region.
  3. New domains WITH PACT -- arrows from each point's raw position to its
     PACT-transported position, directly visualizing the correction.

Data is generated fresh here (not read from a results/ table) since it needs
raw per-sample embeddings, which the standard aggregate pipeline discards.

Outputs: results/figures/fig_representation_space.{pdf,png}
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import _bootstrap  # noqa: F401
from _bootstrap import DATA_ROOT, RESULTS_ROOT

from run_sequential_tsfm import fit_standardizer_and_head
from src.data import get_spec
from src.method import PactConfig, run_sequential
from src.method.pact import extract_embs, transport_embeddings
from src.models.tsfm import make_tsfm
from src.utils import get_device, set_seed

plt.rcParams.update({
    "font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18,
    "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18,
    "font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42,
})

DATASET = "hhar"
DOMAIN_ORDER = [1, 7, 0]            # User-1 (source) -> User-7 (step1) -> User-0 (step2): largest raw no-adapt drop
K = 1                               # Mantis's best K for HHAR
CALIB_FRAC = 0.05
SEED = 42
MAX_PER_DOMAIN = 60                 # fewer points so panel-3 arrows stay legible

CLASS_COLORS = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]  # Okabe-Ito, HHAR's 6 classes
STEP_MARKERS = {1: "^", 2: "s"}
STEP_LABELS = {1: "Step 1", 2: "Step 2"}


def subsample(embs, lbls, n, seed):
    if len(embs) <= n:
        return embs, lbls
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(embs), n, replace=False)
    return embs[idx], lbls[idx]


def main():
    device = get_device(None)
    set_seed(SEED)
    spec = get_spec(DATASET)
    class_names = spec.class_names

    model = make_tsfm("mantis", spec.num_classes, device=str(device)).to(device)
    fit_standardizer_and_head(model, spec, DOMAIN_ORDER[0], device, 150, 1e-3, 64, SEED)

    cfg = PactConfig(k=K, emb_dim=model.emb_dim, calib_frac=CALIB_FRAC, seed=SEED,
                     batch=64, adapter_kind="lora", lora_rank=8, gate_kind="learned")
    res = run_sequential(model, spec, DOMAIN_ORDER, DATA_ROOT, device, cfg,
                         temperature=0.5, verbose=True)
    centroids = res["centroids"]
    adapters_list = res["adapters_list"]
    soft_gates = res["soft_gates"]
    labels = res["labels"]

    def get_test(did):
        ld = torch.utils.data.DataLoader(
            spec.get_dataset(did, "test", DATA_ROOT), batch_size=64)
        embs, lbls_np = extract_embs(model, ld, device)
        return subsample(embs, lbls_np, MAX_PER_DOMAIN, SEED + did)

    src_raw, src_cls = get_test(DOMAIN_ORDER[0])
    s1_raw, s1_cls = get_test(DOMAIN_ORDER[1])
    s2_raw, s2_cls = get_test(DOMAIN_ORDER[2])
    s1_pact = transport_embeddings(s1_raw, centroids, soft_gates, adapters_list, device, K, 0.5)
    s2_pact = transport_embeddings(s2_raw, centroids, soft_gates, adapters_list, device, K, 0.5)
    print(f"source {labels[0]}: {len(src_cls)}  step1 {labels[1]}: {len(s1_cls)}  "
          f"step2 {labels[2]}: {len(s2_cls)}")

    # Fit LDA on the SOURCE only -- these axes are what the linear classifier
    # actually separates classes on. Every other group is a genuine
    # out-of-sample .transform() into that same fixed space (unlike t-SNE).
    lda = LinearDiscriminantAnalysis(n_components=2)
    src_2d = lda.fit_transform(src_raw, src_cls)
    s1_raw_2d = lda.transform(s1_raw)
    s2_raw_2d = lda.transform(s2_raw)
    s1_pact_2d = lda.transform(s1_pact)
    s2_pact_2d = lda.transform(s2_pact)

    def scatter_source(ax, alpha=1.0, size=46, edge="black"):
        for c in range(spec.num_classes):
            m = src_cls == c
            if not m.any():
                continue
            ax.scatter(src_2d[m, 0], src_2d[m, 1], s=size, c=CLASS_COLORS[c],
                      marker="o", edgecolor=edge, linewidth=0.5, alpha=alpha, zorder=2)

    def scatter_steps(ax, pts1, pts2):
        for step, (pts, cls) in [(1, (pts1, s1_cls)), (2, (pts2, s2_cls))]:
            for c in range(spec.num_classes):
                m = cls == c
                if not m.any():
                    continue
                ax.scatter(pts[m, 0], pts[m, 1], s=46, c=CLASS_COLORS[c],
                          marker=STEP_MARKERS[step], edgecolor="white",
                          linewidth=0.5, alpha=0.85, zorder=3)

    def arrow_steps(ax, raw1, pact1, raw2, pact2):
        for raw, pact, cls, step in [(raw1, pact1, s1_cls, 1), (raw2, pact2, s2_cls, 2)]:
            for i in range(len(raw)):
                c = CLASS_COLORS[cls[i]]
                ax.annotate("", xy=pact[i], xytext=raw[i],
                           arrowprops=dict(arrowstyle="-|>", color=c, alpha=0.55,
                                          linewidth=1.1, shrinkA=0, shrinkB=0,
                                          mutation_scale=9), zorder=3)
                ax.scatter(*pact[i], s=30, c=c, marker=STEP_MARKERS[step],
                          edgecolor="white", linewidth=0.4, alpha=0.95, zorder=4)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    fig.subplots_adjust(wspace=0.04)

    # Tight, shared bounds -- zoom to the actual data extent, not matplotlib's
    # default padded autoscale.
    all_x = np.concatenate([src_2d[:, 0], s1_raw_2d[:, 0], s2_raw_2d[:, 0],
                            s1_pact_2d[:, 0], s2_pact_2d[:, 0]])
    all_y = np.concatenate([src_2d[:, 1], s1_raw_2d[:, 1], s2_raw_2d[:, 1],
                            s1_pact_2d[:, 1], s2_pact_2d[:, 1]])
    pad_x = (all_x.max() - all_x.min()) * 0.04
    pad_y = (all_y.max() - all_y.min()) * 0.04
    xlim = (all_x.min() - pad_x, all_x.max() + pad_x)
    ylim = (all_y.min() - pad_y, all_y.max() + pad_y)

    ax = axes[0]
    scatter_source(ax)
    ax.set_title(f"1. Source ({labels[0]})", fontsize=13, pad=5)

    ax = axes[1]
    scatter_source(ax, alpha=0.18, size=30, edge="none")
    scatter_steps(ax, s1_raw_2d, s2_raw_2d)
    ax.set_title("2. Without PACT", fontsize=13, pad=5)

    ax = axes[2]
    scatter_source(ax, alpha=0.18, size=30, edge="none")
    arrow_steps(ax, s1_raw_2d, s1_pact_2d, s2_raw_2d, s2_pact_2d)
    ax.set_title("3. With PACT (transport)", fontsize=13, pad=5)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        for s in ax.spines.values():
            s.set_linewidth(0.8)

    class_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS[c],
                            markersize=8, label=class_names[c]) for c in range(spec.num_classes)]
    step_handles = [Line2D([0], [0], marker=STEP_MARKERS[s], color="w",
                           markerfacecolor="#888888", markeredgecolor="black",
                           markersize=8, label=f"{STEP_LABELS[s]} ({labels[s]})") for s in (1, 2)]
    fig.legend(handles=class_handles + step_handles, loc="upper center",
              bbox_to_anchor=(0.5, 1.06), ncol=len(class_handles) + len(step_handles),
              frameon=False, fontsize=18, handletextpad=0.3, columnspacing=1.0)

    fig.tight_layout(rect=[0, 0, 1, 0.94], pad=0.3, w_pad=0.2)
    out = RESULTS_ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_representation_space.{ext}", dpi=300, bbox_inches="tight",
                   pad_inches=0.05)
    plt.close(fig)
    print(f"Saved -> {out}/fig_representation_space.pdf (+ .png)")


if __name__ == "__main__":
    main()
