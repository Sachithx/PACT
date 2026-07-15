"""
exp_crossfd_sweep.py — Cross-FD Population Sweep

Evaluates the catastrophic-forgetting vs frozen-host-adapter trade-off across
ALL four source→target configurations (FD-A, FD-B, FD-C, FD-D each as source,
remaining three as drifted targets).

For each source domain:
  1. Train TSLANet on source train set (checkpoint reused if already exists)
  2. Use 20% stratified calibration from target_1 (the "calibration domain")
  3. Evaluate 6 conditions on all 4 domains:
       (a) No correction
       (b) Full model fine-tune on calibration slice
       (c) Head-only fine-tune on calibration slice
       (d) Global adapter (frozen host)
       (e) Per-concept adapters k=16 (frozen host)
       (f) PC adapters + gate p=90%

Reports mean ± std of:
  - Source F1 (in-distribution retention)
  - Avg-drifted F1 (adaptation quality)
across the 4 source configs.

Run:
  LD_PRELOAD=/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6 \\
  /home/AD/sachith/.conda/envs/entrope/bin/python exp_crossfd_sweep.py
"""

import sys, copy
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent / "tent"))

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
CKPT_TMPL    = Path(__file__).parent / "TSLANet/Classification/output/tslanet"
OUT_DIR      = Path(__file__).parent / "results/crossfd_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_DOMAINS  = ["FD-A", "FD-B", "FD-C", "FD-D"]
EMB_DIM      = 128
N_CLASSES    = 3
CALIB_FRAC   = 0.20
SEED         = 42
K            = 16
PENALTY      = 1e-3
ADAPTER_EP   = 200
ADAPTER_LR   = 1e-3
FINETUNE_EP  = 80
FINETUNE_LR  = 1e-4
BATCH        = 256
GATE_PCT     = 90.0

class Args:
    emb_dim=128; depth=2; patch_size=8; dropout_rate=0.15; masking_ratio=0.4
    ICB=True; ASB=True; adaptive_filter=True
    # train_source args
    data_root=str(Path(__file__).parent / "dataset")
    ckpt_dir=str(Path(__file__).parent / "TSLANet/Classification/output/tslanet")
    batch_size=256; pretrain=True; pretrain_epochs=40; pretrain_lr=1e-4
    cls_epochs=80; cls_lr=1e-4; retrain=False


# ── Adapter ───────────────────────────────────────────────────────────────────

class IdentityAdapter(nn.Module):
    def __init__(self, dim=EMB_DIM):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


# ── k-means ───────────────────────────────────────────────────────────────────

def kmeans_numpy(embs, k, seed=SEED, n_iter=100):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(embs), k, replace=False)
    centroids = embs[idx].copy()
    assignments = np.zeros(len(embs), dtype=int)
    for _ in range(n_iter):
        dists = np.linalg.norm(embs[:, None] - centroids[None], axis=2)
        new_asgn = dists.argmin(1)
        if np.all(new_asgn == assignments):
            break
        assignments = new_asgn
        for c in range(k):
            m = embs[assignments == c]
            if len(m):
                centroids[c] = m.mean(0)
    return centroids, assignments


def assign_concepts(embs, centroids):
    dists = np.linalg.norm(embs[:, None] - centroids[None], axis=2)
    return dists.argmin(1)


# ── Eval helpers ──────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_f1(model, loader, device):
    model.eval()
    preds, targets = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        out = model(xb).argmax(1).cpu().numpy()
        preds.extend(out); targets.extend(yb.numpy())
    return f1_score(targets, preds, average="macro", zero_division=0)


@torch.no_grad()
def extract_embs(model, loader, device):
    model.eval()
    embs, lbls = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        embs.append(model.extract(xb).cpu().numpy())
        lbls.append(yb.numpy())
    return np.concatenate(embs), np.concatenate(lbls)


@torch.no_grad()
def eval_adapter_f1(embs_np, labels_np, centroids, adapters, frozen_head, device,
                    thresholds=None):
    k = len(centroids)
    asgn = assign_concepts(embs_np, centroids)
    embs = torch.from_numpy(embs_np).to(device)
    corr = embs.clone()

    if thresholds is not None:
        dists    = np.linalg.norm(embs_np - centroids[asgn], axis=1)
        gate_open = dists > thresholds[asgn]
    else:
        gate_open = np.ones(len(embs_np), dtype=bool)

    for c in range(k):
        mask = (asgn == c) & gate_open
        if mask.any():
            adapters[c].eval()
            tidx = torch.from_numpy(mask).to(device)
            corr[tidx] = adapters[c](embs[tidx])

    preds = frozen_head(corr).argmax(1).cpu().numpy()
    return f1_score(labels_np, preds, average="macro", zero_division=0)


def calibrate_per_concept_thresholds(src_embs, centroids, asgn, p):
    thresholds = np.zeros(len(centroids))
    for c in range(len(centroids)):
        m = src_embs[asgn == c]
        if len(m):
            thresholds[c] = float(np.percentile(
                np.linalg.norm(m - centroids[c], axis=1), p))
    return thresholds


# ── Load or train source model ────────────────────────────────────────────────

def load_or_train_source(src_domain, device):
    """Load existing FD checkpoint or train from scratch using tft.train_source."""
    args = Args()
    model = tft.train_source(src_domain, args, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ── EWC-DR (ported from baselines/EWC-DR) ─────────────────────────────────────
# Core algorithm: Fisher importance with Logits Reversal (LR) on source domain,
# then fine-tune on calibration with EWC penalty.
# Ref: "Elastic Weight Consolidation Done Right for Continual Learning" (CVPR 2026)

EWC_LAMBDA  = 1000     # regularisation strength (10000 in orig. for CIFAR; scaled for 3-class FD)
EWC_OMEGA_MAX = 1e-4   # Fisher clamp (same as orig.)
EWC_FT_EP   = 80       # fine-tune epochs (same budget as full_ft / head_ft)
EWC_LR      = 1e-4


def _ewcdr_fisher(model, src_train_loader, device):
    """Compute per-parameter Fisher importance using Logits Reversal (LR)."""
    omega = {n: torch.zeros_like(p) for n, p in model.named_parameters()
             if p.requires_grad}
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    for xb, yb in src_train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        logits = logits * -1          # Logits Reversal — key EWC-DR contribution
        loss   = nn.CrossEntropyLoss()(logits, yb)
        opt.zero_grad()
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                omega[n] += p.grad.pow(2).clone()
    n_batches = len(src_train_loader)
    for n in omega:
        omega[n] = omega[n] / n_batches
        omega[n] = omega[n].clamp(max=EWC_OMEGA_MAX)
    return omega


def run_ewcdr(src_model, src_train_ds, calib_ds, val_loaders, device):
    """
    EWC-DR baseline:
      1. Compute Fisher omega on source domain training data (LR trick).
      2. Fine-tune ALL parameters on calibration slice with EWC penalty.
    Saves per-source results to baselines/output/.
    """
    # Step 1: compute Fisher on source domain
    src_loader = torch.utils.data.DataLoader(
        src_train_ds, batch_size=BATCH, shuffle=False)
    model = copy.deepcopy(src_model)
    for p in model.parameters():
        p.requires_grad_(True)
    print("    Computing EWC-DR Fisher (LR) on source domain …")
    omega = _ewcdr_fisher(model, src_loader, device)
    mean  = {n: p.clone().detach() for n, p in model.named_parameters()
             if p.requires_grad}

    # Step 2: fine-tune on calibration with EWC penalty
    calib_loader = torch.utils.data.DataLoader(calib_ds, batch_size=BATCH, shuffle=True)
    opt  = torch.optim.AdamW(model.parameters(), lr=EWC_LR, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    for ep in range(EWC_FT_EP):
        model.train()
        for xb, yb in calib_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss_clf = crit(logits, yb)
            loss_ewc = sum(
                torch.sum(omega[n] * (p - mean[n]).pow(2)) / 2
                for n, p in model.named_parameters()
                if n in omega
            )
            loss = loss_clf + EWC_LAMBDA * loss_ewc
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    return {name: eval_f1(model, ld, device) for name, ld in val_loaders.items()}


# ── Fine-tune baselines ───────────────────────────────────────────────────────

def run_fullft(src_model, calib_ds, val_loaders, device):
    """Full model fine-tune on calibration slice."""
    model = copy.deepcopy(src_model)
    for p in model.parameters():
        p.requires_grad_(True)
    loader = torch.utils.data.DataLoader(calib_ds, batch_size=BATCH, shuffle=True)
    opt  = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    for ep in range(FINETUNE_EP):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    return {name: eval_f1(model, ld, device) for name, ld in val_loaders.items()}


def run_headft(src_model, calib_ds, val_loaders, device):
    """Head-only fine-tune."""
    model = copy.deepcopy(src_model)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.head.parameters():
        p.requires_grad_(True)
    loader = torch.utils.data.DataLoader(calib_ds, batch_size=BATCH, shuffle=True)
    opt  = torch.optim.AdamW(model.head.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    for ep in range(FINETUNE_EP):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    return {name: eval_f1(model, ld, device) for name, ld in val_loaders.items()}


# ── TENT: test-time entropy minimization (adapted for LayerNorm) ──────────────
# Original TENT targets BatchNorm2d; TSLANet uses LayerNorm — same idea,
# same entropy loss, only the parameter collection differs.

def _tent_setup(src_model, lr=1e-3):
    """Return a deep-copied model with only LayerNorm affine params trainable."""
    model = copy.deepcopy(src_model)
    model.eval()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                m.weight.requires_grad_(True)
            if m.bias is not None:
                m.bias.requires_grad_(True)
    params = [p for m in model.modules()
              if isinstance(m, nn.LayerNorm)
              for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    return model, opt


def run_tent(src_model, calib_ds, val_loaders, device,
             lr=1e-3, n_passes=1):
    """
    Adapt LayerNorm affine params via entropy minimisation on unlabeled
    calibration data (labels are ignored — TENT is label-free).
    n_passes: number of full passes over calib_ds.
    """
    model, opt = _tent_setup(src_model, lr=lr)
    loader = torch.utils.data.DataLoader(calib_ds, batch_size=BATCH, shuffle=True)

    for _ in range(n_passes):
        for xb, _ in loader:          # labels deliberately ignored
            xb = xb.to(device)
            with torch.enable_grad():
                out  = model(xb)
                loss = -(out.softmax(1) * out.log_softmax(1)).sum(1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

    model.eval()
    return {name: eval_f1(model, ld, device) for name, ld in val_loaders.items()}


# ── Adapter training ──────────────────────────────────────────────────────────

def train_adapters(calib_embs, calib_lbls, src_embs, frozen_head,
                   centroids, device, global_only=False):
    """
    Trains global adapter or per-concept k=16 adapters on calibration embeddings.
    Returns adapters (ModuleList of 1 or K) + centroids.
    """
    k = 1 if global_only else K
    adapters = nn.ModuleList([IdentityAdapter(EMB_DIM) for _ in range(k)]).to(device)
    opt  = torch.optim.AdamW(adapters.parameters(), lr=ADAPTER_LR)
    crit = nn.CrossEntropyLoss()

    embs_t = torch.from_numpy(calib_embs).to(device)
    lbls_t = torch.from_numpy(calib_lbls).to(device)

    if global_only:
        asgn = np.zeros(len(calib_embs), dtype=int)
    else:
        asgn = assign_concepts(calib_embs, centroids)

    ds   = torch.utils.data.TensorDataset(embs_t, lbls_t,
               torch.from_numpy(asgn.astype(np.int64)))
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True)

    for ep in range(ADAPTER_EP):
        for eb, yb, cb in loader:
            adapters.train()
            opt.zero_grad()
            corr = eb.clone()
            for c in range(k):
                m = (cb == c)
                if m.any():
                    corr[m] = adapters[c](eb[m])
            loss = crit(frozen_head(corr), yb)
            for a in adapters:
                w = a.linear.weight
                b = a.linear.bias
                loss = loss + PENALTY * (torch.sum((w - torch.eye(EMB_DIM, device=device))**2)
                                         + torch.sum(b**2))
            loss.backward()
            opt.step()
    return adapters


# ── One source-domain experiment ──────────────────────────────────────────────

def run_source_config(src_domain, device):
    print(f"\n{'='*60}")
    print(f"  Source: {src_domain}")
    print(f"{'='*60}")

    target_domains = [d for d in ALL_DOMAINS if d != src_domain]
    calib_domain   = target_domains[0]   # first target = calibration domain

    # Load/train source model
    src_model = load_or_train_source(src_domain, device)
    frozen_head = src_model.head

    # Build data loaders for all domains
    all_test_ds = {d: tft.FDDataset(d, "test", DATA_ROOT) for d in ALL_DOMAINS}
    all_test_ld = {d: torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False)
                   for d, ds in all_test_ds.items()}
    src_train_ds = tft.FDDataset(src_domain, "train", DATA_ROOT)
    src_train_ld = torch.utils.data.DataLoader(src_train_ds, batch_size=BATCH, shuffle=False)

    # Calibration split from calib_domain train set
    calib_full_ds = tft.FDDataset(calib_domain, "train", DATA_ROOT)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-CALIB_FRAC, random_state=SEED)
    y_all = calib_full_ds.y.numpy()
    calib_idx, _ = next(sss.split(np.zeros(len(y_all)), y_all))
    calib_ds = torch.utils.data.Subset(calib_full_ds, calib_idx)
    calib_ld = torch.utils.data.DataLoader(calib_ds, batch_size=BATCH, shuffle=False)

    # Extract embeddings
    print("  Extracting embeddings …")
    src_train_embs, src_train_lbls = extract_embs(src_model, src_train_ld, device)
    calib_embs, calib_lbls         = extract_embs(src_model, calib_ld,     device)
    all_test_embs = {d: extract_embs(src_model, all_test_ld[d], device)
                     for d in ALL_DOMAINS}

    # k-means on source train embeddings
    centroids, src_asgn = kmeans_numpy(src_train_embs, K)
    thresholds = calibrate_per_concept_thresholds(src_train_embs, centroids, src_asgn, GATE_PCT)

    # Val loaders for fine-tune eval
    val_loaders = all_test_ld

    # ── Condition results ─────────────────────────────────────────────────────
    results = {}

    # (a) No correction
    print("  (a) No correction …")
    results["no_corr"] = {}
    for d, (embs, lbls) in all_test_embs.items():
        results["no_corr"][d] = f1_score(lbls,
            frozen_head(torch.from_numpy(embs).to(device)).argmax(1).cpu().numpy(),
            average="macro", zero_division=0)

    # (b) Full fine-tune
    print("  (b) Full fine-tune …")
    results["full_ft"] = run_fullft(src_model, calib_ds, val_loaders, device)

    # (b2) EWC-DR
    print("  (b2) EWC-DR …")
    results["ewcdr"] = run_ewcdr(src_model, src_train_ds, calib_ds, val_loaders, device)

    # (c) Head-only fine-tune
    print("  (c) Head-only fine-tune …")
    results["head_ft"] = run_headft(src_model, calib_ds, val_loaders, device)

    # (d) TENT (label-free, entropy minimization on calib data)
    print("  (d) TENT (label-free) …")
    results["tent"] = run_tent(src_model, calib_ds, val_loaders, device)

    # (e) Global adapter
    print("  (e) Global adapter …")
    global_centroids = np.array([src_train_embs.mean(0)])
    gadapters = train_adapters(calib_embs, calib_lbls, src_train_embs,
                               frozen_head, global_centroids, device, global_only=True)
    results["global_adap"] = {}
    for d, (embs, lbls) in all_test_embs.items():
        results["global_adap"][d] = eval_adapter_f1(
            embs, lbls, global_centroids, gadapters, frozen_head, device)

    # (f) Per-concept adapters k=16
    print(f"  (f) Per-concept adapters k={K} …")
    pc_adapters = train_adapters(calib_embs, calib_lbls, src_train_embs,
                                 frozen_head, centroids, device)
    results["pc_adap"] = {}
    for d, (embs, lbls) in all_test_embs.items():
        results["pc_adap"][d] = eval_adapter_f1(
            embs, lbls, centroids, pc_adapters, frozen_head, device)

    # (g) PC + gate p=90%
    print(f"  (g) PC + gate p={GATE_PCT}% …")
    results["pc_gate"] = {}
    for d, (embs, lbls) in all_test_embs.items():
        results["pc_gate"][d] = eval_adapter_f1(
            embs, lbls, centroids, pc_adapters, frozen_head, device,
            thresholds=thresholds)

    # ── Save per-source results to baselines/output/ ─────────────────────────
    baseline_out = Path(__file__).parent / "baselines" / "output"
    baseline_out.mkdir(parents=True, exist_ok=True)
    cond_order = ["no_corr", "full_ft", "ewcdr", "head_ft", "tent",
                  "global_adap", "pc_adap", "pc_gate"]
    cond_lbl   = {"no_corr": "No correction", "full_ft": "Full fine-tune",
                  "ewcdr": "EWC-DR", "head_ft": "Head-only FT",
                  "tent": "TENT (label-free)", "global_adap": "Global adapter",
                  "pc_adap": "PACT (no gate)", "pc_gate": "PACT"}
    lines = [f"Source: {src_domain}  |  calib: {calib_domain} (20% train)\n",
             f"  {'Condition':<22} {'src':>6} {'drift_avg':>10}  "
             + "  ".join(f"{d:>7}" for d in ALL_DOMAINS)]
    for cond in cond_order:
        if cond not in results:
            continue
        sf  = results[cond][src_domain]
        dfm = np.mean([results[cond][d] for d in target_domains])
        row = "  ".join(f"{results[cond][d]:>7.4f}" for d in ALL_DOMAINS)
        lines.append(f"  {cond_lbl[cond]:<22} {sf:>6.4f} {dfm:>10.4f}  {row}")
    (baseline_out / f"results_source_{src_domain}.txt").write_text("\n".join(lines))

    return results, src_domain, target_domains


# ── Aggregate and report ──────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    conditions = ["no_corr", "full_ft", "ewcdr", "head_ft", "tent",
                  "global_adap", "pc_adap", "pc_gate"]
    cond_labels = {
        "no_corr":     "No correction",
        "full_ft":     "Full fine-tune",
        "ewcdr":       "EWC-DR",
        "head_ft":     "Head-only FT",
        "tent":        "TENT (label-free)",
        "global_adap": "Global adapter",
        "pc_adap":     "PACT (no gate)",
        "pc_gate":     "PACT",
    }

    all_src_f1   = {c: [] for c in conditions}
    all_drift_f1 = {c: [] for c in conditions}
    all_results  = {}

    for src in ALL_DOMAINS:
        res, src_domain, target_domains = run_source_config(src, device)
        all_results[src] = res

        print(f"\n  Results for source={src}:")
        print(f"  {'Condition':<22} {'src':>6} {'drift_avg':>10} "
              + "  ".join(f"{d:>7}" for d in ALL_DOMAINS))
        print(f"  {'-'*80}")

        for cond in conditions:
            src_f1   = res[cond][src_domain]
            drift_f1 = np.mean([res[cond][d] for d in target_domains])
            all_src_f1[cond].append(src_f1)
            all_drift_f1[cond].append(drift_f1)
            row = "  ".join(f"{res[cond][d]:>7.4f}" for d in ALL_DOMAINS)
            print(f"  {cond_labels[cond]:<22} {src_f1:>6.4f} {drift_f1:>10.4f}  {row}")

    # ── Population summary ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("POPULATION SUMMARY (mean ± std over 4 source configurations)")
    print(f"{'='*70}")
    print(f"  {'Condition':<22} {'src_F1 (mean±std)':>22} {'drift_F1 (mean±std)':>22}")
    print(f"  {'-'*70}")

    summary_lines = ["POPULATION SUMMARY (mean ± std over 4 source configurations)\n"]
    summary_lines.append(f"  {'Condition':<22} {'src_F1 mean':>12} {'src_F1 std':>10} "
                         f"{'drift_F1 mean':>14} {'drift_F1 std':>12}")
    summary_lines.append("  " + "-"*70)

    for cond in conditions:
        src_mean  = np.mean(all_src_f1[cond])
        src_std   = np.std(all_src_f1[cond])
        dft_mean  = np.mean(all_drift_f1[cond])
        dft_std   = np.std(all_drift_f1[cond])
        print(f"  {cond_labels[cond]:<22} {src_mean:.4f} ± {src_std:.4f}     "
              f"{dft_mean:.4f} ± {dft_std:.4f}")
        summary_lines.append(
            f"  {cond_labels[cond]:<22} {src_mean:.4f} ± {src_std:.4f}     "
            f"{dft_mean:.4f} ± {dft_std:.4f}")

    (OUT_DIR / "population_summary.txt").write_text("\n".join(summary_lines))
    # mirror to baselines/output
    baseline_out = Path(__file__).parent / "baselines" / "output"
    baseline_out.mkdir(parents=True, exist_ok=True)
    (baseline_out / "population_summary.txt").write_text("\n".join(summary_lines))

    # ── Per-source table ──────────────────────────────────────────────────────
    table_lines = [f"Per-source results ({len(conditions)} conditions × 4 domains)\n"]
    for src in ALL_DOMAINS:
        res = all_results[src]
        target_domains = [d for d in ALL_DOMAINS if d != src]
        table_lines.append(f"\nSource: {src}")
        table_lines.append(f"  {'Condition':<22} " + "  ".join(f"{d:>7}" for d in ALL_DOMAINS))
        for cond in conditions:
            row = "  ".join(f"{res[cond][d]:>7.4f}" for d in ALL_DOMAINS)
            table_lines.append(f"  {cond_labels[cond]:<22} {row}")
    (OUT_DIR / "per_source_table.txt").write_text("\n".join(table_lines))

    # ── Figure: grouped bar chart ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x     = np.arange(len(conditions))
    means = {"src": [np.mean(all_src_f1[c]) for c in conditions],
             "dft": [np.mean(all_drift_f1[c]) for c in conditions]}
    stds  = {"src": [np.std(all_src_f1[c]) for c in conditions],
             "dft": [np.std(all_drift_f1[c]) for c in conditions]}

    colors = ["#aec6cf", "#ff6b6b", "#e8965a", "#ffa07a", "#f4d35e",
              "#c8a2c8", "#2196F3", "#1b7a2e"]
    labels_short = ["No corr", "Full FT", "EWC-DR", "Head FT", "TENT",
                    "Global", "PACT\n(no gate)", "PACT"]

    for ax, key, title in zip(axes, ["src", "dft"],
                               ["Source domain F1 (retention)", "Drifted domains F1 (avg)"]):
        bars = ax.bar(x, means[key], yerr=stds[key], capsize=5,
                      color=colors, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_short, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Macro F1")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axhline(y=0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        for bar, m in zip(bars, means[key]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Cross-FD Population: Source Retention vs Drift Recovery\n"
                 "(mean ± std over 4 source configs, error bars = std across configs)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "population_figure.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT_DIR / "population_figure.pdf", bbox_inches="tight")
    plt.close()

    print(f"\nFigure saved → {OUT_DIR}/population_figure.{{png,pdf}}")
    print(f"Summary saved → {OUT_DIR}/population_summary.txt")
    print(f"Table saved   → {OUT_DIR}/per_source_table.txt")


if __name__ == "__main__":
    main()
