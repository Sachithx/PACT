"""
exp_patch_sweep.py — Patchwise Concept Assignment Sweep

Same as exp_crossfd_sweep but uses INDIVIDUAL PATCH embeddings (pre-mean-pool)
for concept clustering and adapter correction, then re-pools before the head.

Pipeline per sample:
  Input [B, 1, 5120]
    → patch_embed + pos_drop + 2×TSLANetLayer  →  [B, P=1279, 128]  patch embs
    → assign each patch to its nearest concept centroid
    → apply concept adapter to each patch
    → mean pool corrected patches  →  [B, 128]
    → frozen head  →  [B, 3]

Differences from mean-pool sweep:
  - k-means runs on patches (subsampled to KMEANS_CAP patches for tractability)
  - Adapter corrects each 128-dim patch embedding, not the pooled sample embedding
  - Distance gate: mean patch-level distance to concept centroids per sample
  - Concept thresholds: p-th percentile of per-patch distances on source train

Conditions evaluated per source config (same as exp_crossfd_sweep):
  - No correction
  - PACT (no gate)    — patch-level adapters, always apply
  - PACT              — patch-level adapters + distance gate p=90%

Run:
  CUDA_VISIBLE_DEVICES=1 LD_PRELOAD=/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6 \\
  /home/AD/sachith/.conda/envs/entrope/bin/python exp_patch_sweep.py
"""

import sys, copy
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent / "TSLANet" / "Classification"))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "tft", Path(__file__).parent / "transfer_fd_tslanet.py"
)
tft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tft)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────

DATA_ROOT    = Path(__file__).parent / "dataset"
CKPT_DIR     = Path(__file__).parent / "TSLANet/Classification/output/tslanet"
OUT_DIR      = Path(__file__).parent / "results/patch_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_DOMAINS  = ["FD-A", "FD-B", "FD-C", "FD-D"]
EMB_DIM      = 128
SEED         = 42
K            = 16
PENALTY      = 1e-3
ADAPTER_EP   = 200
ADAPTER_LR   = 1e-3
BATCH        = 64          # smaller batch — each sample yields 1279 patch embeddings
GATE_PCT     = 90.0
CALIB_FRAC   = 0.20

# Subsample cap for k-means (10.5M patches → tractable)
KMEANS_CAP   = 100_000

class Args:
    emb_dim=128; depth=2; patch_size=8; dropout_rate=0.15; masking_ratio=0.4
    ICB=True; ASB=True; adaptive_filter=True
    data_root=str(DATA_ROOT)
    ckpt_dir=str(CKPT_DIR)
    batch_size=256; pretrain=True; pretrain_epochs=40; pretrain_lr=1e-4
    cls_epochs=80; cls_lr=1e-4; retrain=False


# ── Adapter ───────────────────────────────────────────────────────────────────

class IdentityAdapter(nn.Module):
    def __init__(self, dim=EMB_DIM):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    def forward(self, x): return self.linear(x)


# ── Extract patches (pre-mean-pool) ──────────────────────────────────────────

@torch.no_grad()
def extract_patch_embs(model, loader, device):
    """
    Returns:
      patches: np [N*P, 128] — all patches from all samples, flattened
      sample_idx: np [N*P] — which sample each patch belongs to
      labels: np [N]
    """
    model.eval()
    all_patches, all_sidx, all_lbls = [], [], []
    s = 0
    for xb, yb in loader:
        xb = xb.to(device)
        # Run backbone up to pre-mean-pool
        p = model.patch_embed(xb)
        p = model.pos_drop(p + model.pos_embed)
        for blk in model.blocks:
            p = blk(p)
        # p: [B, P, 128]
        B, P, D = p.shape
        all_patches.append(p.reshape(B * P, D).cpu().numpy())
        sidx = np.repeat(np.arange(s, s + B), P)
        all_sidx.append(sidx)
        all_lbls.append(yb.numpy())
        s += B
    return (np.concatenate(all_patches).astype(np.float32),
            np.concatenate(all_sidx),
            np.concatenate(all_lbls))


@torch.no_grad()
def pool_and_classify(model, loader, device):
    """Standard mean-pool evaluation (no adapter)."""
    model.eval()
    preds, targets = [], []
    for xb, yb in loader:
        preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
        targets.extend(yb.numpy())
    return f1_score(targets, preds, average="macro", zero_division=0)


# ── k-means on patch embeddings ───────────────────────────────────────────────

def kmeans_patches(patch_embs, k, seed=SEED, cap=KMEANS_CAP, n_iter=100):
    """
    k-means on (subsampled) patch embeddings.
    Returns centroids [k, 128] in patch embedding space.
    """
    rng = np.random.default_rng(seed)
    if len(patch_embs) > cap:
        idx = rng.choice(len(patch_embs), cap, replace=False)
        sample = patch_embs[idx]
    else:
        sample = patch_embs

    # Initialise from random sample points
    cinit = rng.choice(len(sample), k, replace=False)
    centroids = sample[cinit].copy()
    assignments = np.zeros(len(sample), dtype=int)

    for _ in range(n_iter):
        dists = np.linalg.norm(sample[:, None] - centroids[None], axis=2)
        new_asgn = dists.argmin(1)
        if np.all(new_asgn == assignments): break
        assignments = new_asgn
        for c in range(k):
            m = sample[assignments == c]
            if len(m): centroids[c] = m.mean(0)

    return centroids


def assign_patches(patch_embs, centroids):
    """Assign each patch to nearest centroid. Returns [N_patches] int array."""
    return np.linalg.norm(patch_embs[:, None] - centroids[None], axis=2).argmin(1)


# ── Patchwise adapter training ────────────────────────────────────────────────

def train_patch_adapters(model, calib_loader, centroids, frozen_head, device):
    """
    Train per-concept adapters where each adapter corrects individual patch
    embeddings. Loss: CrossEntropy(head(mean(corrected_patches)), label).
    """
    adapters = nn.ModuleList([IdentityAdapter(EMB_DIM) for _ in range(K)]).to(device)
    opt  = torch.optim.AdamW(adapters.parameters(), lr=ADAPTER_LR)
    crit = nn.CrossEntropyLoss()
    eye  = torch.eye(EMB_DIM, device=device)
    ct   = torch.from_numpy(centroids).to(device)    # [K, 128]

    for ep in range(ADAPTER_EP):
        for xb, yb in calib_loader:
            xb, yb = xb.to(device), yb.to(device)
            adapters.train(); opt.zero_grad()

            with torch.no_grad():
                # Extract patch embeddings: [B, P, 128]
                p = model.patch_embed(xb)
                p = model.pos_drop(p + model.pos_embed)
                for blk in model.blocks:
                    p = blk(p)

            B, P, D = p.shape
            p_flat = p.reshape(B * P, D)    # [B*P, 128]

            # Assign each patch to its concept
            with torch.no_grad():
                dists = torch.cdist(p_flat, ct)   # [B*P, K]
                asgn  = dists.argmin(1)            # [B*P]

            # Apply per-concept adapter to each patch
            corrected = p_flat.clone()
            for c in range(K):
                mask = (asgn == c)
                if mask.any():
                    corrected[mask] = adapters[c](p_flat[mask])

            # Mean pool corrected patches back to sample level
            corrected_samples = corrected.reshape(B, P, D).mean(1)  # [B, 128]

            loss = crit(frozen_head(corrected_samples), yb)
            for a in adapters:
                loss = loss + PENALTY * (torch.sum((a.linear.weight - eye)**2)
                                         + torch.sum(a.linear.bias**2))
            loss.backward(); opt.step()

    return adapters


# ── Patchwise evaluation ──────────────────────────────────────────────────────

@torch.no_grad()
def eval_patch_adapters(model, loader, centroids, adapters, frozen_head, device,
                        thresholds=None):
    """
    Evaluate patchwise adapters. If thresholds provided, gate opens per-sample
    when mean patch distance to assigned centroids exceeds sample threshold.

    Gate logic: mean(patch_distances) > threshold[majority_concept_of_sample]
    where majority_concept = most common patch concept assignment in the sample.
    """
    ct = torch.from_numpy(centroids).to(device)
    preds, targets = [], []

    for xb, yb in loader:
        xb = xb.to(device)

        # Extract patches: [B, P, 128]
        p = model.patch_embed(xb)
        p = model.pos_drop(p + model.pos_embed)
        for blk in model.blocks:
            p = blk(p)

        B, P, D = p.shape
        p_flat = p.reshape(B * P, D)

        # Assign each patch to concept
        dists_to_centroid = torch.cdist(p_flat, ct)   # [B*P, K]
        patch_asgn        = dists_to_centroid.argmin(1)  # [B*P]
        patch_dists       = dists_to_centroid[
            torch.arange(B * P, device=device), patch_asgn]  # [B*P]

        # Reshape to [B, P]
        patch_asgn_2d  = patch_asgn.reshape(B, P)
        patch_dists_2d = patch_dists.reshape(B, P)

        # Per-sample: majority concept and mean distance
        sample_concept = torch.mode(patch_asgn_2d, dim=1).values  # [B]
        sample_meandist = patch_dists_2d.mean(1)                   # [B]

        # Gate decision (per sample)
        if thresholds is not None:
            thr_t    = torch.from_numpy(thresholds).to(device)
            gate_open = sample_meandist > thr_t[sample_concept]    # [B]
        else:
            gate_open = torch.ones(B, dtype=torch.bool, device=device)

        # Apply adapters patch-wise, pooled, then classify
        corrected = p_flat.clone()
        for c in range(K):
            # Only apply for samples where gate is open
            sample_gate = gate_open.repeat_interleave(P)  # [B*P]
            mask = (patch_asgn == c) & sample_gate
            if mask.any():
                adapters[c].eval()
                corrected[mask] = adapters[c](p_flat[mask])

        corrected_samples = corrected.reshape(B, P, D).mean(1)  # [B, 128]
        batch_preds = frozen_head(corrected_samples).argmax(1).cpu().numpy()
        preds.extend(batch_preds)
        targets.extend(yb.numpy())

    return f1_score(targets, preds, average="macro", zero_division=0)


def calibrate_patch_thresholds(model, src_train_loader, centroids, device, p):
    """
    Per-concept threshold: p-th percentile of mean-patch-distances on source
    training samples, grouped by their majority concept.
    """
    ct      = torch.from_numpy(centroids).to(device)
    # Collect (sample_concept, sample_meandist) for all source train samples
    sc_list, sd_list = [], []

    with torch.no_grad():
        for xb, _ in src_train_loader:
            xb = xb.to(device)
            px = model.patch_embed(xb)
            px = model.pos_drop(px + model.pos_embed)
            for blk in model.blocks: px = blk(px)
            B, P, D = px.shape
            pf = px.reshape(B * P, D)
            dc = torch.cdist(pf, ct)
            pa = dc.argmin(1)
            pd = dc[torch.arange(B * P, device=device), pa]
            sc_list.append(torch.mode(pa.reshape(B, P), dim=1).values.cpu().numpy())
            sd_list.append(pd.reshape(B, P).mean(1).cpu().numpy())

    sc = np.concatenate(sc_list)
    sd = np.concatenate(sd_list)

    thresholds = np.zeros(K)
    for c in range(K):
        m = sd[sc == c]
        if len(m):
            thresholds[c] = float(np.percentile(m, p))
    return thresholds


# ── One source config ─────────────────────────────────────────────────────────

def run_source_config(src_domain, device):
    target_domains = [d for d in ALL_DOMAINS if d != src_domain]
    calib_domain   = target_domains[0]

    print(f"\n{'='*60}")
    print(f"  Source: {src_domain}  |  Calib: {calib_domain}")
    print(f"{'='*60}")

    # Load frozen model
    model = tft.train_source(src_domain, Args(), device)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    frozen_head = model.head

    # Loaders
    src_train_ds = tft.FDDataset(src_domain, "train", DATA_ROOT)
    src_train_ld = torch.utils.data.DataLoader(
        src_train_ds, batch_size=BATCH, shuffle=False)
    all_test_lds = {d: torch.utils.data.DataLoader(
        tft.FDDataset(d, "test", DATA_ROOT), batch_size=BATCH, shuffle=False)
        for d in ALL_DOMAINS}

    # Calibration split
    calib_full_ds = tft.FDDataset(calib_domain, "train", DATA_ROOT)
    y_all = calib_full_ds.y.numpy()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - CALIB_FRAC, random_state=SEED)
    calib_idx, _ = next(sss.split(np.zeros(len(y_all)), y_all))
    calib_ds = torch.utils.data.Subset(calib_full_ds, calib_idx)
    calib_ld = torch.utils.data.DataLoader(calib_ds, batch_size=BATCH, shuffle=True)

    # k-means on source training patches (subsampled)
    print(f"  Extracting source train patches for k-means (cap={KMEANS_CAP:,}) …")
    src_patches, src_sidx, _ = extract_patch_embs(model, src_train_ld, device)
    print(f"    Total patches: {len(src_patches):,}  "
          f"(from {src_sidx.max()+1} samples × {len(src_patches)//(src_sidx.max()+1)} patches)")
    centroids = kmeans_patches(src_patches, K, cap=KMEANS_CAP)

    # Per-concept distance thresholds from source training patches
    print(f"  Calibrating patch distance thresholds (p={GATE_PCT}%) …")
    thresholds = calibrate_patch_thresholds(
        model, src_train_ld, centroids, device, GATE_PCT)

    # Train patchwise adapters on calibration data
    print("  Training patchwise adapters …")
    adapters = train_patch_adapters(model, calib_ld, centroids, frozen_head, device)

    # Evaluate
    results = {}
    print(f"\n  {'Condition':<22} {'src':>7} {'drift_avg':>10}  "
          + "  ".join(f"{d:>7}" for d in ALL_DOMAINS))
    print(f"  {'-'*72}")

    for cond_name, use_thresholds in [
            ("No correction",   None),          # standard mean-pool, no adapter
            ("PACT (no gate)",  False),          # patchwise adapters, no gate
            ("PACT",            True),           # patchwise adapters + gate
    ]:
        row = {}
        for d, ld in all_test_lds.items():
            if cond_name == "No correction":
                row[d] = pool_and_classify(model, ld, device)
            else:
                thr = thresholds if use_thresholds else None
                row[d] = eval_patch_adapters(
                    model, ld, centroids, adapters, frozen_head, device, thresholds=thr)

        results[cond_name] = row
        src_f1    = row[src_domain]
        avg_drift = np.mean([row[d] for d in target_domains])
        print(f"  {cond_name:<22} {src_f1:>7.4f} {avg_drift:>10.4f}  "
              + "  ".join(f"{row[d]:>7.4f}" for d in ALL_DOMAINS))

    return results, target_domains


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    conditions = ["No correction", "PACT (no gate)", "PACT"]
    all_src_f1   = {c: [] for c in conditions}
    all_drift_f1 = {c: [] for c in conditions}
    all_results  = {}

    for src in ALL_DOMAINS:
        res, target_domains = run_source_config(src, device)
        all_results[src] = (res, target_domains)
        for cond in conditions:
            all_src_f1[cond].append(res[cond][src])
            all_drift_f1[cond].append(np.mean([res[cond][d] for d in target_domains]))

    # ── Population summary ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("POPULATION SUMMARY — PATCHWISE (mean ± std, 4 source configs)")
    print(f"{'='*70}")
    print(f"  {'Condition':<22} {'src_F1':>14} {'drift_F1':>16}")
    print(f"  {'-'*54}")

    lines = ["POPULATION SUMMARY — PATCHWISE (mean ± std, 4 source configs)\n",
             f"  {'Condition':<22} {'src_F1':>14} {'drift_F1':>16}", "  " + "-"*54]

    for cond in conditions:
        sm = np.mean(all_src_f1[cond]);   ss = np.std(all_src_f1[cond])
        dm = np.mean(all_drift_f1[cond]); ds = np.std(all_drift_f1[cond])
        line = f"  {cond:<22} {sm:.4f}±{ss:.4f}  {dm:.4f}±{ds:.4f}"
        print(line); lines.append(line)

    (OUT_DIR / "population_summary.txt").write_text("\n".join(lines))

    # ── Per-source table ──────────────────────────────────────────────────────
    table_lines = ["Per-source results (patchwise)\n"]
    for src in ALL_DOMAINS:
        res, target_domains = all_results[src]
        table_lines.append(f"\nSource: {src}")
        table_lines.append(f"  {'Condition':<22} {'src':>7} {'drift':>7}  "
                           + "  ".join(f"{d:>7}" for d in ALL_DOMAINS))
        for cond in conditions:
            src_f1    = res[cond][src]
            avg_drift = np.mean([res[cond][d] for d in target_domains])
            row = "  ".join(f"{res[cond][d]:>7.4f}" for d in ALL_DOMAINS)
            table_lines.append(f"  {cond:<22} {src_f1:>7.4f} {avg_drift:>7.4f}  {row}")
    (OUT_DIR / "per_source_table.txt").write_text("\n".join(table_lines))

    # ── Figure — compare patchwise vs mean-pool (hardcoded mean-pool numbers) ──
    # Mean-pool numbers from exp_crossfd_sweep population summary
    meanpool = {
        "No correction":  (0.9968, 0.6083),
        "PACT (no gate)": (0.7001, 0.7160),
        "PACT":           (0.9679, 0.8150),
    }
    patchwise = {c: (np.mean(all_src_f1[c]), np.mean(all_drift_f1[c])) for c in conditions}
    patchwise_std = {c: (np.std(all_src_f1[c]), np.std(all_drift_f1[c])) for c in conditions}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")
    x      = np.arange(len(conditions))
    width  = 0.35
    colors = {"mean-pool": "#aec6cf", "patchwise": "#e05c5c"}
    cond_colors = {"No correction": "#888888", "PACT (no gate)": "#2196F3", "PACT": "#1b7a2e"}

    for ax, key, title in zip(axes, [0, 1],
            ["Source domain F1  (no forgetting ↑)",
             "Drifted domains F1  (adapts to drift ↑)"]):
        mp_vals = [meanpool[c][key] for c in conditions]
        pw_vals = [patchwise[c][key] for c in conditions]
        pw_stds = [patchwise_std[c][key] for c in conditions]

        bars1 = ax.bar(x - width/2, mp_vals, width, label="Mean-pool emb",
                       color=[cond_colors[c] for c in conditions],
                       edgecolor="white", linewidth=0.5, alpha=0.55)
        bars2 = ax.bar(x + width/2, pw_vals, width, yerr=pw_stds, capsize=4,
                       label="Patchwise emb",
                       color=[cond_colors[c] for c in conditions],
                       edgecolor="black", linewidth=0.7,
                       error_kw=dict(elinewidth=1.0, ecolor="#555"))

        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=9)
        ax.set_ylim(0.5, 1.08)
        ax.set_ylabel("Macro F1", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
        ax.set_facecolor("#f8f8f8")
        for bar, v in zip(bars2, pw_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

    fig.suptitle("Patchwise vs Mean-Pool Concept Assignment — Population Sweep\n"
                 "(mean ± std over 4 FD source configs, hatched = patchwise)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "patch_vs_meanpool.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    fig.savefig(OUT_DIR / "patch_vs_meanpool.pdf", bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"\nFigure  → {OUT_DIR}/patch_vs_meanpool.{{png,pdf}}")
    print(f"Summary → {OUT_DIR}/population_summary.txt")
    print(f"Table   → {OUT_DIR}/per_source_table.txt")


if __name__ == "__main__":
    main()
