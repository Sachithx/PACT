"""
exp_label_budget.py — PACT Label Budget Sweep

For each source domain (FD-A, B, C, D):
  - Load frozen TSLANet checkpoint
  - For each label budget [1%, 5%, 10%, 20%, 50%, 100%] of the
    calibration domain's training set:
      * Train PACT (no gate) and PACT adapters on that slice
      * Evaluate on all 4 FD test domains
  - No correction included as budget-independent baseline

Reports population mean ± std (over 4 source configs) at each budget.
Produces a line plot: x = label budget, y = src F1 / drift F1.

Run:
  LD_PRELOAD=/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6 \\
  /home/AD/sachith/.conda/envs/entrope/bin/python exp_label_budget.py
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

DATA_ROOT   = Path(__file__).parent / "dataset"
OUT_DIR     = Path(__file__).parent / "results/label_budget"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_DOMAINS = ["FD-A", "FD-B", "FD-C", "FD-D"]
EMB_DIM     = 128
SEED        = 42
K           = 16
PENALTY     = 1e-3
ADAPTER_EP  = 200
ADAPTER_LR  = 1e-3
BATCH       = 256
GATE_PCT    = 90.0

# Label budgets to sweep
BUDGETS = [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]

class Args:
    emb_dim=128; depth=2; patch_size=8; dropout_rate=0.15; masking_ratio=0.4
    ICB=True; ASB=True; adaptive_filter=True
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
def eval_adapter_f1(embs_np, labels_np, centroids, adapters, frozen_head,
                    device, thresholds=None):
    k = len(centroids)
    asgn  = assign_concepts(embs_np, centroids)
    embs  = torch.from_numpy(embs_np).to(device)
    corr  = embs.clone()
    if thresholds is not None:
        dists     = np.linalg.norm(embs_np - centroids[asgn], axis=1)
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


def calibrate_thresholds(src_embs, centroids, asgn, p):
    thresholds = np.zeros(len(centroids))
    for c in range(len(centroids)):
        m = src_embs[asgn == c]
        if len(m):
            thresholds[c] = float(np.percentile(
                np.linalg.norm(m - centroids[c], axis=1), p))
    return thresholds


# ── Adapter training ──────────────────────────────────────────────────────────

def train_adapters(calib_embs, calib_lbls, centroids, frozen_head, device):
    adapters = nn.ModuleList([IdentityAdapter(EMB_DIM) for _ in range(K)]).to(device)
    opt  = torch.optim.AdamW(adapters.parameters(), lr=ADAPTER_LR)
    crit = nn.CrossEntropyLoss()
    asgn = assign_concepts(calib_embs, centroids)

    embs_t = torch.from_numpy(calib_embs).to(device)
    lbls_t = torch.from_numpy(calib_lbls).to(device)
    asgn_t = torch.from_numpy(asgn.astype(np.int64))
    ds     = torch.utils.data.TensorDataset(embs_t, lbls_t, asgn_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True)

    for _ in range(ADAPTER_EP):
        for eb, yb, cb in loader:
            adapters.train()
            opt.zero_grad()
            corr = eb.clone()
            for c in range(K):
                m = (cb == c)
                if m.any():
                    corr[m] = adapters[c](eb[m])
            loss = crit(frozen_head(corr), yb)
            for a in adapters:
                w, b = a.linear.weight, a.linear.bias
                loss = loss + PENALTY * (
                    torch.sum((w - torch.eye(EMB_DIM, device=device))**2)
                    + torch.sum(b**2))
            loss.backward()
            opt.step()
    return adapters


# ── One source-domain, one budget ────────────────────────────────────────────

def run_one(src_domain, budget, frozen_model, src_train_embs, src_train_asgn,
            all_test_embs, all_test_lds, target_domains, device):
    """
    Returns dict: {domain: {no_corr, pact_ngate, pact}} for one (source, budget).
    """
    calib_domain = target_domains[0]
    frozen_head  = frozen_model.head

    # ── Calibration split ────────────────────────────────────────────────────
    calib_full_ds = tft.FDDataset(calib_domain, "train", DATA_ROOT)
    y_all = calib_full_ds.y.numpy()

    if budget >= 1.0:
        calib_idx = np.arange(len(y_all))
    else:
        # stratified split: keep `budget` fraction
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=1.0 - budget, random_state=SEED)
        calib_idx, _ = next(sss.split(np.zeros(len(y_all)), y_all))

    calib_ds = torch.utils.data.Subset(calib_full_ds, calib_idx)
    calib_ld = torch.utils.data.DataLoader(calib_ds, batch_size=BATCH, shuffle=False)
    calib_embs, calib_lbls = extract_embs(frozen_model, calib_ld, device)

    n_calib = len(calib_lbls)

    # ── Centroids and thresholds (from source domain, fixed) ─────────────────
    centroids  = src_train_embs[
        np.array([np.where(src_train_asgn == c)[0][0] for c in range(K)
                  if (src_train_asgn == c).any()])
    ]
    # Recompute properly
    centroids = np.array([
        src_train_embs[src_train_asgn == c].mean(0)
        if (src_train_asgn == c).any()
        else np.zeros(EMB_DIM)
        for c in range(K)
    ])
    thresholds = calibrate_thresholds(src_train_embs, centroids, src_train_asgn, GATE_PCT)

    # ── Train adapters ────────────────────────────────────────────────────────
    adapters = train_adapters(calib_embs, calib_lbls, centroids, frozen_head, device)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = {}
    for d, (embs, lbls) in all_test_embs.items():
        no_corr = f1_score(lbls,
            frozen_head(torch.from_numpy(embs).to(device)).argmax(1).cpu().numpy(),
            average="macro", zero_division=0)
        pact_ng = eval_adapter_f1(embs, lbls, centroids, adapters, frozen_head, device)
        pact    = eval_adapter_f1(embs, lbls, centroids, adapters, frozen_head, device,
                                   thresholds=thresholds)
        results[d] = {"no_corr": no_corr, "pact_ngate": pact_ng, "pact": pact}

    return results, n_calib


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # results[src][budget][domain][method] = F1
    all_results = {}

    for src_domain in ALL_DOMAINS:
        print(f"{'='*60}")
        print(f"  Source: {src_domain}")
        print(f"{'='*60}")
        all_results[src_domain] = {}

        target_domains = [d for d in ALL_DOMAINS if d != src_domain]

        # Load frozen model
        args = Args()
        frozen_model = tft.train_source(src_domain, args, device)
        frozen_model.eval()
        for p in frozen_model.parameters():
            p.requires_grad_(False)

        # Build test loaders and extract embeddings once
        all_test_lds = {d: torch.utils.data.DataLoader(
            tft.FDDataset(d, "test", DATA_ROOT), batch_size=BATCH, shuffle=False)
            for d in ALL_DOMAINS}
        print("  Extracting test embeddings …")
        all_test_embs = {d: extract_embs(frozen_model, ld, device)
                         for d, ld in all_test_lds.items()}

        # k-means on source training embeddings (fixed for all budgets)
        src_train_ds = tft.FDDataset(src_domain, "train", DATA_ROOT)
        src_train_ld = torch.utils.data.DataLoader(src_train_ds, batch_size=BATCH, shuffle=False)
        src_train_embs, _ = extract_embs(frozen_model, src_train_ld, device)
        _, src_train_asgn  = kmeans_numpy(src_train_embs, K)

        for budget in BUDGETS:
            pct = int(budget * 100)
            print(f"  Budget {pct:3d}% ", end="", flush=True)
            res, n_calib = run_one(
                src_domain, budget, frozen_model,
                src_train_embs, src_train_asgn,
                all_test_embs, all_test_lds, target_domains, device)
            all_results[src_domain][budget] = res
            # Quick per-domain summary
            src_f1 = res[src_domain]["pact"]
            drift  = np.mean([res[d]["pact"] for d in target_domains])
            print(f"→ n_calib={n_calib:5d}  "
                  f"PACT: src={src_f1:.4f}  drift={drift:.4f}")

        print()

    # ── Aggregate population stats per budget ────────────────────────────────
    methods = ["no_corr", "pact_ngate", "pact"]
    method_labels = {"no_corr": "No correction",
                     "pact_ngate": "PACT (no gate)",
                     "pact": "PACT"}

    # pop[budget][method] = {"src": [4 values], "drift": [4 values]}
    pop = {b: {m: {"src": [], "drift": []} for m in methods} for b in BUDGETS}

    for src_domain in ALL_DOMAINS:
        target_domains = [d for d in ALL_DOMAINS if d != src_domain]
        for budget in BUDGETS:
            res = all_results[src_domain][budget]
            for m in methods:
                pop[budget][m]["src"].append(res[src_domain][m])
                pop[budget][m]["drift"].append(
                    np.mean([res[d][m] for d in target_domains]))

    # ── Print population table ────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("POPULATION SUMMARY (mean ± std over 4 source configs)")
    print(f"{'='*72}")
    header = f"  {'Budget':>7}  {'Method':<18}  {'src_F1':>10}  {'drift_F1':>10}"
    print(header)
    print("  " + "-"*68)

    table_lines = ["PACT Label Budget Sweep — Population Summary\n",
                   header, "  " + "-"*68]

    for budget in BUDGETS:
        pct = int(budget * 100)
        for m in methods:
            sm = np.mean(pop[budget][m]["src"])
            ss = np.std(pop[budget][m]["src"])
            dm = np.mean(pop[budget][m]["drift"])
            ds = np.std(pop[budget][m]["drift"])
            line = (f"  {pct:6d}%  {method_labels[m]:<18}  "
                    f"{sm:.4f}±{ss:.4f}  {dm:.4f}±{ds:.4f}")
            print(line)
            table_lines.append(line)
        print()
        table_lines.append("")

    (OUT_DIR / "population_summary.txt").write_text("\n".join(table_lines))

    # ── Per-source table ──────────────────────────────────────────────────────
    src_lines = ["Per-source results\n"]
    for src_domain in ALL_DOMAINS:
        target_domains = [d for d in ALL_DOMAINS if d != src_domain]
        src_lines.append(f"Source: {src_domain}")
        src_lines.append(f"  {'Budget':>7}  {'Method':<18}  "
                         + "  ".join(f"{d:>7}" for d in ALL_DOMAINS)
                         + "  drift_avg")
        for budget in BUDGETS:
            pct = int(budget * 100)
            res = all_results[src_domain][budget]
            for m in methods:
                row = "  ".join(f"{res[d][m]:>7.4f}" for d in ALL_DOMAINS)
                drift = np.mean([res[d][m] for d in target_domains])
                src_lines.append(f"  {pct:6d}%  {method_labels[m]:<18}  {row}  {drift:.4f}")
        src_lines.append("")
    (OUT_DIR / "per_source_table.txt").write_text("\n".join(src_lines))

    # ── Figure ────────────────────────────────────────────────────────────────
    pct_labels = [f"{int(b*100)}%" for b in BUDGETS]
    x = np.arange(len(BUDGETS))

    COLORS = {"no_corr": "#888888", "pact_ngate": "#2196F3", "pact": "#1b7a2e"}
    LSTYLE = {"no_corr": "--",      "pact_ngate": "-",       "pact": "-"}
    MARKER = {"no_corr": "o",       "pact_ngate": "D",       "pact": "*"}
    MSIZE  = {"no_corr": 6,         "pact_ngate": 7,         "pact": 10}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")

    for ax, key, title in zip(
            axes,
            ["src", "drift"],
            ["Source domain F1  (no forgetting ↑)",
             "Drifted domains F1  (adapts to drift ↑)"]):
        for m in methods:
            means = [np.mean(pop[b][m][key]) for b in BUDGETS]
            stds  = [np.std(pop[b][m][key])  for b in BUDGETS]
            ax.plot(x, means,
                    color=COLORS[m], linestyle=LSTYLE[m],
                    marker=MARKER[m], markersize=MSIZE[m],
                    linewidth=2.0, label=method_labels[m], zorder=3)
            ax.fill_between(x,
                             np.array(means) - np.array(stds),
                             np.array(means) + np.array(stds),
                             alpha=0.12, color=COLORS[m], zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels(pct_labels, fontsize=10)
        ax.set_xlabel("Label budget (% of calibration domain train set)", fontsize=10)
        ax.set_ylabel("Macro F1", fontsize=10)
        ax.set_ylim(0.35, 1.05)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        ax.set_facecolor("#f8f8f8")

    fig.suptitle("PACT Label Efficiency\n"
                 "Population mean ± std over 4 FD source/target configs",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "label_budget_figure.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    fig.savefig(OUT_DIR / "label_budget_figure.pdf", bbox_inches="tight",
                facecolor="white")
    plt.close()

    print(f"Figure  → {OUT_DIR}/label_budget_figure.{{png,pdf}}")
    print(f"Summary → {OUT_DIR}/population_summary.txt")
    print(f"Table   → {OUT_DIR}/per_source_table.txt")


if __name__ == "__main__":
    main()
