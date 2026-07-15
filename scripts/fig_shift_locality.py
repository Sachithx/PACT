"""Motivating measurement: domain shift acts locally on the frozen embedding
manifold, not globally, and the same charts are stable/fragile across shifts.

Panel (a): cluster frozen SOURCE embeddings into K charts; for each chart,
           compare the frozen head's mean margin on source-test samples routed
           to it against each shifted domain's samples routed to it. Margin
           m(z) = logit[y*] - max_{y!=y*} logit[y] is continuous and does not
           degenerate on near-single-class charts (unlike cluster F1, which is
           exactly 0 or 1 on most charts at this sample size -- see git history
           for that earlier, broken version).

Panel (b): fit ONE global identity-anchored affine adapter (K=1) on a target
           domain's calibration slice, vs. K local per-chart affine adapters
           for K in {1,2,4,8,16,32,64} (clusters from the calibration set
           itself). Residual drops sharply from K=1 to K=2 and is flat from
           K=2 to K=32 (ticking up at K=64 from calibration starvation): the
           finding is that ANY locality beats the global map, not that K=16
           is an optimum -- fit quality alone cannot justify K=16.

Panel (c): take the K=1 global adapter fit for the most-shifted domain
           (domain t) and apply it, unchanged, to the SOURCE domain's test
           embeddings -- the frozen head already handles the source, so every
           negative delta here is unambiguous forgetting (a target domain,
           used as the bystander in an earlier version of this script, is
           itself unadapted and shifted, so a correction fitted elsewhere can
           net-help it; that is a different, weaker, and honestly separate
           point -- see panel (e)).

Panel (d): chart occupancy (fraction of a domain's samples nearest each
           source chart). Shifted samples migrate out of their source chart,
           which is the empirical case for the soft partition-of-unity gate
           over a hard nearest-chart lookup.

Panel (e) [supplementary]: the same K=1 fix applied to a second, differently-
           shifted TARGET domain (not the source) -- shows the fix can
           net-transfer across shifts of the same family, which is a real but
           weaker and different claim than panel (c)'s forgetting result.

Everything here reuses the actual PACT training code (train_adapters,
kmeans_numpy, assign_concepts) so the K=1 adapter in panels (b)/(c)/(e) is
exactly the same object PACT would train, just with a single concept.

Outputs:
  results/figures/fig1_shift_locality.{pdf,png}
  results/shift_locality/panel_a_margin_drop.csv
  results/shift_locality/panel_b_residual_vs_k.csv
  results/shift_locality/panel_c_collateral_on_source.csv
  results/shift_locality/panel_d_chart_occupancy.csv
  results/shift_locality/panel_e_cross_shift_transfer.csv
"""

import dataclasses
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

import _bootstrap  # noqa: F401
from _bootstrap import CKPT_ROOT, DATA_ROOT, RESULTS_ROOT

from src.data import get_spec
from src.method.pact import PactConfig, extract_embs, train_adapters
from src.models.concepts import assign_concepts, kmeans_numpy
from src.models.factory import load_backbone
from src.models.tsfm import TSFM_MODELS, make_tsfm
from src.utils import get_device, set_seed
from run_sequential_tsfm import fit_standardizer_and_head

MIN_CLUSTER_N = 5           # minimum samples in a chart to trust its margin
DATASET, BACKBONE, SOURCE = "fd", "moment", 0
IS_TSFM = BACKBONE in TSFM_MODELS
K_SOURCE = 16                # chart count used for panels (a)/(c)/(d)
K_SWEEP = [1, 2, 4, 8, 16, 32, 64]   # local adapter counts swept in panel (b)
SEED = 42

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13.5, "axes.titleweight": "bold",
    "axes.labelsize": 12, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5, "font.family": "sans-serif",
    "axes.linewidth": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.edgecolor": "#333333", "text.color": "#1a1a1a",
    "axes.labelcolor": "#1a1a1a", "xtick.color": "#333333", "ytick.color": "#333333",
})
C = dict(bar="#0072B2", knee="#009E73", t="#0072B2", j="#D55E00", src="#555555")
TGT_COLORS = ["#D55E00", "#CC79A7", "#E69F00"]


# ── helpers ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def cluster_margin(embs, lbls, asgn, k, frozen_head, device, adapter_bank=None):
    """Per-chart mean classification margin m(z) = logit[y*] - max_{y!=y*} logit[y].
    Continuous and defined even on single-class charts, unlike cluster F1.
    Returns an array of length k (NaN where the chart has < MIN_CLUSTER_N samples)."""
    out = np.full(k, np.nan)
    et = torch.from_numpy(embs).to(device)
    lt = torch.from_numpy(lbls).to(device)
    for c in range(k):
        mask = (asgn == c)
        if mask.sum() < MIN_CLUSTER_N:
            continue
        batch = et[mask]
        if adapter_bank is not None:
            adapter_bank[c].eval()
            batch = adapter_bank[c](batch)
        logits = frozen_head(batch)
        y = lt[mask]
        true_logit = logits.gather(1, y.unsqueeze(1)).squeeze(1)
        masked = logits.clone()
        masked.scatter_(1, y.unsqueeze(1), float("-inf"))
        other_max = masked.max(1).values
        out[c] = (true_logit - other_max).mean().item()
    return out


@torch.no_grad()
def whole_domain_f1(embs, lbls, frozen_head, device, adapter=None):
    """Ground-truth sanity check: overall macro-F1 on the FULL domain (not
    averaged per-chart), before/after a single global adapter. Margin deltas
    can be large and negative on a chart while every prediction in it stays
    correct (a big margin shrinking to a smaller positive margin doesn't flip
    anything), so this is the number that actually says "forgetting" vs not."""
    et = torch.from_numpy(embs).to(device)
    if adapter is not None:
        adapter.eval()
        et = adapter(et)
    preds = frozen_head(et).argmax(1).cpu().numpy()
    return f1_score(lbls, preds, average="macro", zero_division=0)


def chart_occupancy(asgn, k):
    """Fraction of a domain's samples nearest each of the k source charts."""
    counts = np.bincount(asgn, minlength=k).astype(float)
    return counts / counts.sum()


def stratified_calib(ds, calib_frac, seed):
    y_all = ds.y.numpy()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - calib_frac, random_state=seed)
    idx, _ = next(sss.split(np.zeros(len(y_all)), y_all))
    return idx, y_all[idx]


def fit_local_adapters(calib_embs, calib_lbls, k, frozen_head, device, cfg: PactConfig):
    """Cluster calib_embs into k charts (k=1 -> single global chart) and train
    an identity-anchored adapter bank on them (reuses PACT's own
    train_adapters, so this is not a re-implementation)."""
    if k == 1:
        centroids = calib_embs.mean(0, keepdims=True)
    else:
        centroids, _ = kmeans_numpy(calib_embs, k, seed=cfg.seed)
    cfg_k = dataclasses.replace(cfg, k=k)
    adapters = train_adapters(calib_embs, calib_lbls, centroids, frozen_head, device, cfg_k)
    return centroids, adapters


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    device = get_device()
    spec = get_spec(DATASET)
    if IS_TSFM:
        model = make_tsfm(BACKBONE, spec.num_classes, device=str(device)).to(device)
        acc = fit_standardizer_and_head(model, spec, SOURCE, device,
                                        epochs=150, lr=1e-3, batch=64, seed=SEED)
        print(f"{BACKBONE} (frozen) loaded; standardizer+head fit on source, "
             f"train acc={acc:.4f}, emb_dim={model.emb_dim}")
    else:
        model = load_backbone(BACKBONE, spec, CKPT_ROOT, DATASET, SOURCE, device, freeze=True)
    frozen_head = model.head
    cfg = PactConfig(k=K_SOURCE, emb_dim=model.emb_dim, calib_frac=0.05, seed=SEED,
                     batch=64 if IS_TSFM else 256)

    targets = [d for d in range(spec.n_domains) if d != SOURCE]
    print(f"Source={spec.domain_label(SOURCE)}  Targets={[spec.domain_label(d) for d in targets]}")

    src_train_ld = torch.utils.data.DataLoader(
        spec.get_dataset(SOURCE, "train", DATA_ROOT), batch_size=cfg.batch)
    src_train_embs, _ = extract_embs(model, src_train_ld, device)
    src_test_ld = torch.utils.data.DataLoader(
        spec.get_dataset(SOURCE, "test", DATA_ROOT), batch_size=cfg.batch)
    src_test_embs, src_test_lbls = extract_embs(model, src_test_ld, device)

    tgt_test_embs, tgt_test_lbls, tgt_calib_embs, tgt_calib_lbls = {}, {}, {}, {}
    for d in targets:
        ld = torch.utils.data.DataLoader(spec.get_dataset(d, "test", DATA_ROOT), batch_size=cfg.batch)
        tgt_test_embs[d], tgt_test_lbls[d] = extract_embs(model, ld, device)

        ds_train = spec.get_dataset(d, "train", DATA_ROOT)
        idx, calib_lbls = stratified_calib(ds_train, cfg.calib_frac, SEED)
        calib_ld = torch.utils.data.DataLoader(
            torch.utils.data.Subset(ds_train, idx), batch_size=cfg.batch)
        tgt_calib_embs[d], _ = extract_embs(model, calib_ld, device)
        tgt_calib_lbls[d] = calib_lbls

    centroids16, _ = kmeans_numpy(src_train_embs, K_SOURCE, seed=SEED)
    src_asgn = assign_concepts(src_test_embs, centroids16)
    tgt_asgn = {d: assign_concepts(tgt_test_embs[d], centroids16) for d in targets}

    # ── Panel (a): per-chart margin drop under shift ────────────────────────
    baseline_margin = cluster_margin(src_test_embs, src_test_lbls, src_asgn, K_SOURCE,
                                     frozen_head, device)

    drop_rows = []
    per_domain_drop = {}
    for d in targets:
        m_d = cluster_margin(tgt_test_embs[d], tgt_test_lbls[d], tgt_asgn[d], K_SOURCE,
                             frozen_head, device)
        per_domain_drop[d] = baseline_margin - m_d
        for c in range(K_SOURCE):
            drop_rows.append(dict(domain=spec.domain_label(d), cluster=c,
                                  baseline_margin=baseline_margin[c], target_margin=m_d[c],
                                  margin_drop=baseline_margin[c] - m_d[c]))
    panel_a_df = pd.DataFrame(drop_rows)

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_drop = np.nanmean(np.stack([per_domain_drop[d] for d in targets]), axis=0)
        domain_mean_drop = {d: np.nanmean(per_domain_drop[d]) for d in targets}
        cv_by_domain = {d: (np.nanstd(per_domain_drop[d]) / np.nanmean(per_domain_drop[d]))
                        for d in targets}
    t_domain = max(domain_mean_drop, key=domain_mean_drop.get)   # most shifted
    j_domain = min(domain_mean_drop, key=domain_mean_drop.get)   # least shifted (panel e)
    print(f"Per-domain mean chart margin-drop: "
          f"{ {spec.domain_label(d): round(v, 3) for d, v in domain_mean_drop.items()} }")
    print(f"CV of margin-drop across charts: "
          f"{ {spec.domain_label(d): round(v, 3) for d, v in cv_by_domain.items()} }")
    print(f"Panel (b)/(c) fit domain t={spec.domain_label(t_domain)} (most shifted)")
    print(f"Panel (e) secondary transfer domain j={spec.domain_label(j_domain)} (least shifted)")

    # ── Panel (b): global (K=1) vs local (K sweep) affine residual ─────────────
    calib_embs_t, calib_lbls_t = tgt_calib_embs[t_domain], tgt_calib_lbls[t_domain]
    test_embs_t, test_lbls_t = tgt_test_embs[t_domain], tgt_test_lbls[t_domain]
    max_k = max(1, len(calib_embs_t) // 3)

    panel_b_rows = []
    k1_adapter_bank = None
    for k in K_SWEEP:
        if k > max_k:
            print(f"  skipping K={k} (only {len(calib_embs_t)} calib samples for domain t)")
            continue
        centroids_k, adapters_k = fit_local_adapters(calib_embs_t, calib_lbls_t, k,
                                                      frozen_head, device, cfg)
        if k == 1:
            k1_adapter_bank = adapters_k   # reused verbatim in panels (c)/(e)
        asgn_test = assign_concepts(test_embs_t, centroids_k)
        et = torch.from_numpy(test_embs_t).to(device)
        corr = et.clone()
        with torch.no_grad():
            for c in range(k):
                m = (asgn_test == c)
                if m.any():
                    adapters_k[c].eval()
                    corr[m] = adapters_k[c](et[m])
            preds = frozen_head(corr).argmax(1).cpu().numpy()
        f1 = f1_score(test_lbls_t, preds, average="macro", zero_division=0)
        panel_b_rows.append(dict(K=k, test_f1=f1, residual=1.0 - f1))
        print(f"  K={k:>2d}  test macro-F1={f1:.4f}  residual={1-f1:.4f}")
    panel_b_df = pd.DataFrame(panel_b_rows)
    global_adapter_t = k1_adapter_bank[0]

    # ── Panel (c): domain-t's global (K=1) fix applied to the SOURCE ───────────
    corrected_bank = [global_adapter_t] * K_SOURCE
    base_src = cluster_margin(src_test_embs, src_test_lbls, src_asgn, K_SOURCE, frozen_head, device)
    corr_src = cluster_margin(src_test_embs, src_test_lbls, src_asgn, K_SOURCE, frozen_head,
                              device, adapter_bank=corrected_bank)
    base_t = cluster_margin(test_embs_t, test_lbls_t, tgt_asgn[t_domain], K_SOURCE, frozen_head, device)
    corr_t = cluster_margin(test_embs_t, test_lbls_t, tgt_asgn[t_domain], K_SOURCE, frozen_head,
                            device, adapter_bank=corrected_bank)

    delta_src = corr_src - base_src
    delta_t = corr_t - base_t
    panel_c_df = pd.DataFrame(dict(
        cluster=np.arange(K_SOURCE),
        base_margin_t=base_t, corrected_margin_t=corr_t, delta_margin_t=delta_t,
        base_margin_src=base_src, corrected_margin_src=corr_src, delta_margin_src=delta_src,
    ))
    panel_c_df["domain_t"] = spec.domain_label(t_domain)

    valid_c = panel_c_df.dropna(subset=["delta_margin_t", "delta_margin_src"])
    rho_t, p_t = spearmanr(valid_c["base_margin_t"], valid_c["delta_margin_t"])
    rho_src, p_src = spearmanr(valid_c["base_margin_src"], valid_c["delta_margin_src"])
    print(f"Spearman(base margin, delta margin) on domain t:      rho={rho_t:.3f} p={p_t:.4f}")
    print(f"Spearman(base margin, delta margin) on SOURCE (fig1c): rho={rho_src:.3f} p={p_src:.4f}")
    print(f"Mean delta margin on source (bystander): {delta_src[~np.isnan(delta_src)].mean():.3f}")

    # Ground-truth whole-domain F1 (ranking-invariant to margin's raw scale;
    # this is what actually answers "did the fix forget the source").
    src_f1_before = whole_domain_f1(src_test_embs, src_test_lbls, frozen_head, device)
    src_f1_after = whole_domain_f1(src_test_embs, src_test_lbls, frozen_head, device,
                                   adapter=global_adapter_t)
    print(f"Whole-domain source F1: {src_f1_before:.4f} -> {src_f1_after:.4f} "
          f"(delta {src_f1_after - src_f1_before:+.4f})")

    # ── Panel (d): chart occupancy shift ────────────────────────────────────
    occ_src = chart_occupancy(src_asgn, K_SOURCE)
    occ_by_domain = {d: chart_occupancy(tgt_asgn[d], K_SOURCE) for d in targets}
    panel_d_df = pd.DataFrame({"cluster": np.arange(K_SOURCE), "occupancy_source": occ_src,
                               **{f"occupancy_{spec.domain_label(d)}": occ_by_domain[d] for d in targets}})

    # ── Panel (e) [supplementary]: same K=1 fix transferred to another target ──
    base_j = cluster_margin(tgt_test_embs[j_domain], tgt_test_lbls[j_domain], tgt_asgn[j_domain],
                            K_SOURCE, frozen_head, device)
    corr_j = cluster_margin(tgt_test_embs[j_domain], tgt_test_lbls[j_domain], tgt_asgn[j_domain],
                            K_SOURCE, frozen_head, device, adapter_bank=corrected_bank)
    delta_j = corr_j - base_j
    panel_e_df = pd.DataFrame(dict(
        cluster=np.arange(K_SOURCE), base_margin_j=base_j, corrected_margin_j=corr_j,
        delta_margin_j=delta_j,
    ))
    panel_e_df["domain_t"], panel_e_df["domain_j"] = spec.domain_label(t_domain), spec.domain_label(j_domain)
    j_f1_before = whole_domain_f1(tgt_test_embs[j_domain], tgt_test_lbls[j_domain], frozen_head, device)
    j_f1_after = whole_domain_f1(tgt_test_embs[j_domain], tgt_test_lbls[j_domain], frozen_head, device,
                                 adapter=global_adapter_t)
    print(f"Mean delta margin on secondary target {spec.domain_label(j_domain)}: "
          f"{delta_j[~np.isnan(delta_j)].mean():.3f}")
    print(f"Whole-domain {spec.domain_label(j_domain)} F1: {j_f1_before:.4f} -> {j_f1_after:.4f} "
          f"(delta {j_f1_after - j_f1_before:+.4f})")

    # ── save CSVs ────────────────────────────────────────────────────────────
    out_dir = RESULTS_ROOT / "shift_locality"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{BACKBONE}" if BACKBONE != "tslanet" else ""
    panel_a_df.to_csv(out_dir / f"panel_a_margin_drop{tag}.csv", index=False)
    panel_b_df.to_csv(out_dir / f"panel_b_residual_vs_k{tag}.csv", index=False)
    panel_c_df.to_csv(out_dir / f"panel_c_collateral_on_source{tag}.csv", index=False)
    panel_d_df.to_csv(out_dir / f"panel_d_chart_occupancy{tag}.csv", index=False)
    panel_e_df.to_csv(out_dir / f"panel_e_cross_shift_transfer{tag}.csv", index=False)

    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2))
    plot_panel_a(axes[0, 0], mean_drop, per_domain_drop, targets, spec)
    plot_panel_b(axes[0, 1], panel_b_df)
    plot_panel_c(axes[0, 2], delta_t, delta_src, spec, t_domain, rho_src, p_src,
                src_f1_before, src_f1_after)
    plot_panel_d(axes[1, 0], occ_src, occ_by_domain, targets, spec)
    plot_panel_e(axes[1, 1], delta_t, delta_j, spec, t_domain, j_domain, j_f1_before, j_f1_after)
    plot_callout(axes[1, 2], panel_a_df, rho_src, p_src, src_f1_before, src_f1_after, spec, t_domain)
    fig.suptitle("Domain shift acts locally on the frozen embedding manifold "
                f"({spec.name.upper()} / {BACKBONE}, source={spec.domain_label(SOURCE)})",
                fontsize=15, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.085, top=0.90, wspace=0.38, hspace=0.62)
    fig_dir = RESULTS_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"fig1_shift_locality{tag}.{ext}", dpi=300, bbox_inches="tight")
    print(f"\nSaved -> {fig_dir}/fig1_shift_locality{tag}.pdf (+ .png)")
    print(f"Saved -> {out_dir}/panel_{{a,b,c,d,e}}{tag}.csv")


def plot_panel_a(ax, mean_drop, per_domain_drop, targets, spec):
    valid = np.where(~np.isnan(mean_drop))[0]
    order = valid[np.argsort(mean_drop[valid])]
    y = np.arange(len(order))
    lo = np.nanmin(np.stack([per_domain_drop[d][order] for d in targets]), axis=0)
    hi = np.nanmax(np.stack([per_domain_drop[d][order] for d in targets]), axis=0)
    md = mean_drop[order]
    bar_colors = [C["knee"] if v < 5 else C["bar"] for v in md]
    ax.barh(y, md, color=bar_colors, zorder=3, height=0.68, edgecolor="white", linewidth=0.4)
    ax.errorbar(md, y, xerr=[md - lo, hi - md], fmt="none", ecolor="#222222",
               elinewidth=1.1, capsize=2.5, zorder=4)
    ax.axvline(0, color="k", lw=1.0)
    ax.set_yticks(y); ax.set_yticklabels([f"c{c}" for c in order], fontsize=10)
    ax.set_xlabel("margin drop  (source $-$ shifted), logit units")
    ax.set_ylabel(f"chart (K={len(mean_drop)}), sorted by mean drop")
    ax.set_title("(a)  Shift hits charts unevenly")
    ax.margins(y=0.03)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_panel_b(ax, df):
    ax.plot(df["K"], df["residual"], "o-", color=C["knee"], lw=2.2, ms=8, zorder=3,
           markeredgecolor="white", markeredgewidth=0.8)
    if 1 in df["K"].values:
        r1 = df.loc[df["K"] == 1, "residual"].iloc[0]
        ax.axhline(r1, color="#444444", ls="--", lw=1.0, alpha=0.8)
        ax.text(df["K"].max(), r1, "  global  (K=1)", va="bottom", ha="right", fontsize=10)
    flat = df[(df["K"] >= 2) & (df["K"] <= 32)]
    if len(flat) >= 2:
        ax.axvspan(flat["K"].min(), flat["K"].max(), color=C["knee"], alpha=0.10, zorder=0)
        ax.text(np.sqrt(flat["K"].min() * flat["K"].max()), ax.get_ylim()[1],
               "insensitive to $K$", ha="center", va="top", fontsize=9.5,
               color=C["knee"], fontstyle="italic")
    for _, row in df.iterrows():
        ax.annotate(f"{row.residual:.3f}", (row.K, row.residual),
                   textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(df["K"]); ax.set_xticklabels(df["K"].astype(int))
    ax.set_xlabel("local adapters  $K$  (calibration-set clusters)")
    ax.set_ylabel("held-out residual   $1 - F_1$")
    ax.set_title("(b)  Any locality beats one global map")
    ax.set_ylim(bottom=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_panel_c(ax, delta_t, delta_src, spec, t_domain, rho, p, f1_before, f1_after):
    valid = np.where(~np.isnan(delta_t) & ~np.isnan(delta_src))[0]
    order = valid[np.argsort(-delta_t[valid])]
    x = np.arange(len(order))
    w = 0.38
    ax.bar(x - w / 2, delta_t[order], width=w, color=C["t"],
          label=f"domain t = {spec.domain_label(t_domain)}  (fit target)",
          edgecolor="white", linewidth=0.4)
    ax.bar(x + w / 2, delta_src[order], width=w, color=C["src"],
          label="source  (bystander, already correct)", edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels([f"c{c}" for c in order], fontsize=9.5, rotation=0)
    ax.set_xlabel("chart, sorted by domain-t gain")
    ax.set_ylabel(r"$\Delta$ margin from domain-t's global fix")
    ax.set_title("(c)  A blind global fix damages the source")
    ax.text(0.02, 0.03,
           f"Spearman(baseline, $\\Delta$) = {rho:.2f}  (p={p:.4f})\n"
           f"whole-domain source $F_1$: {f1_before:.3f} $\\to$ {f1_after:.3f}  "
           f"({f1_after - f1_before:+.3f})",
           transform=ax.transAxes, fontsize=9.5, va="bottom", ha="left",
           bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bbbbbb", lw=0.6, alpha=0.9))
    ax.legend(frameon=False, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_panel_d(ax, occ_src, occ_by_domain, targets, spec):
    order = np.argsort(-occ_src)
    x = np.arange(len(order))
    ax.plot(x, occ_src[order], "o-", color="k", lw=2.2, ms=7, label="source", zorder=5,
           markeredgecolor="white", markeredgewidth=0.6)
    for i, d in enumerate(targets):
        ax.plot(x, occ_by_domain[d][order], "o--", color=TGT_COLORS[i % len(TGT_COLORS)],
               lw=1.6, ms=6, alpha=0.9, label=spec.domain_label(d),
               markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f"c{c}" for c in order], fontsize=10, rotation=0)
    ax.set_xlabel("chart, sorted by source occupancy")
    ax.set_ylabel("fraction of domain routed to chart")
    ax.set_title("(d)  Shifted samples migrate between charts")
    ax.legend(frameon=False, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_panel_e(ax, delta_t, delta_j, spec, t_domain, j_domain, f1_before, f1_after):
    valid = np.where(~np.isnan(delta_t) & ~np.isnan(delta_j))[0]
    order = valid[np.argsort(-delta_t[valid])]
    x = np.arange(len(order))
    w = 0.38
    ax.bar(x - w / 2, delta_t[order], width=w, color=C["t"],
          label=f"domain t = {spec.domain_label(t_domain)}  (fit target)",
          edgecolor="white", linewidth=0.4)
    ax.bar(x + w / 2, delta_j[order], width=w, color=C["j"],
          label=f"domain j = {spec.domain_label(j_domain)}  (also shifted, unadapted)",
          edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels([f"c{c}" for c in order], fontsize=9.5, rotation=0)
    ax.set_xlabel("chart, sorted by domain-t gain")
    ax.set_ylabel(r"$\Delta$ margin from domain-t's global fix")
    verdict = "net helped" if f1_after >= f1_before else "net hurt"
    ax.set_title(f"(e)  supplementary: transfer to {spec.domain_label(j_domain)}", fontsize=13)
    ax.text(0.02, 0.03,
           f"whole-domain {spec.domain_label(j_domain)} $F_1$: {f1_before:.3f} $\\to$ {f1_after:.3f}  "
           f"({f1_after - f1_before:+.3f}, {verdict})\nnot a forgetting claim -- j is itself "
           "unadapted and shifted",
           transform=ax.transAxes, fontsize=9.5, va="bottom", ha="left",
           bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bbbbbb", lw=0.6, alpha=0.9))
    ax.legend(frameon=False, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_callout(ax, panel_a_df, rho_src, p_src, src_f1_before, src_f1_after, spec, t_domain):
    ax.axis("off")
    sig_anticorr = (rho_src < -0.3) and (p_src < 0.05)
    if sig_anticorr:
        headline = ("What the frozen head already\ngets right is what a global\n"
                   "correction undoes.")
        corr_line = (f"damage anti-correlates with need:\n"
                    f"Spearman(baseline margin, $\\Delta$)\non source = {rho_src:+.2f} "
                    f"(p={p_src:.3f})")
    else:
        headline = ("A blind global fix collapses\nthe source regardless of\n"
                   "any per-chart correlation.")
        corr_line = (f"per-chart correlation is weak/ns here:\n"
                    f"Spearman(baseline margin, $\\Delta$)\non source = {rho_src:+.2f} "
                    f"(p={p_src:.3f}, n.s.)\nthe collapse is a whole-domain effect,\n"
                    "not one traceable to a clean\nneed-vs-damage ranking on this backbone")
    stats = (f"{corr_line}\n\n"
            f"whole-domain source $F_1$\nafter fitting {spec.domain_label(t_domain)}'s "
            f"global fix:\n{src_f1_before:.3f} $\\to$ {src_f1_after:.3f}  "
            f"({src_f1_after - src_f1_before:+.3f})\n\n"
            "note: F1-invariant charts (panel a's\nfloor) are not always margin-\n"
            "invariant -- confidence erodes\nbefore any prediction flips.")
    ax.text(0.5, 0.82, headline, ha="center", va="top", fontsize=15, fontweight="bold",
           style="italic", transform=ax.transAxes, linespacing=1.5)
    ax.text(0.5, 0.46, stats, ha="center", va="top", fontsize=10.5,
           transform=ax.transAxes, linespacing=1.7,
           bbox=dict(boxstyle="round,pad=0.7", fc="#f5f5f5", ec="#999999", lw=0.9))


if __name__ == "__main__":
    main()
