"""
exp_crosshhar_sweep.py — Cross-User HHAR Population Sweep (PACT vs baselines)

Runs the same catastrophic-forgetting / frozen-host-adapter comparison as
exp_crossfd_sweep.py, but on the HHAR dataset with 9 user domains.

Uses the 10 cross-user scenarios from AdaTime's HHAR benchmark:
  (0,6),(1,6),(2,7),(3,8),(4,5),(5,0),(6,1),(7,4),(8,3),(0,2)
  (domain ID = user index: a=0, b=1, …, i=8)

For each scenario (src→tgt):
  1. Train / load TSLANet on src user domain
  2. Take 20% stratified calibration slice from tgt train set
  3. Evaluate 6 conditions:
       (a) No correction          — frozen source model, zero adaptation
       (b) Full fine-tune         — all params, calibration labels
       (c) Head-only fine-tune    — head only, calibration labels
       (d) TENT                   — LayerNorm entropy min, label-free
       (e) PACT (no gate)         — frozen host + k=16 per-concept adapters
       (f) PACT                   — frozen host + adapters + distance gate p=90%

Outputs → results/crosshhar_sweep/
  population_summary.txt
  per_scenario_table.txt
  population_figure.{png,pdf}

Run:
  CUDA_VISIBLE_DEVICES=1 \\
  LD_PRELOAD=/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6 \\
  /home/AD/sachith/.conda/envs/entrope/bin/python exp_crosshhar_sweep.py
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
    "tht", Path(__file__).parent / "transfer_hhar_tslanet.py")
tht = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tht)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_ROOT  = Path(__file__).parent / "dataset"
CKPT_DIR   = Path(__file__).parent / "TSLANet" / "Classification" / "output" / "tslanet"
OUT_DIR    = Path(__file__).parent / "results" / "crosshhar_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 10 AdaTime HHAR cross-user scenarios (src_id, tgt_id)
SCENARIOS  = [(0,6),(1,6),(2,7),(3,8),(4,5),(5,0),(6,1),(7,4),(8,3),(0,2)]

EMB_DIM    = 128
N_CLASSES  = 6
CALIB_FRAC = 0.20
SEED       = 42
K          = 16
PENALTY    = 1e-3
ADAPTER_EP = 200
ADAPTER_LR = 1e-3
FINETUNE_EP = 80
FINETUNE_LR = 1e-4
BATCH      = 256
GATE_PCT   = 90.0

USER_NAMES = {i: f"User-{chr(ord('a')+i)}" for i in range(9)}


class Args:
    emb_dim=128; depth=2; patch_size=8; dropout_rate=0.15; masking_ratio=0.4
    ICB=True; ASB=True; adaptive_filter=True
    data_root=str(DATA_ROOT)
    ckpt_dir=str(CKPT_DIR)
    batch_size=256; pretrain=True; pretrain_epochs=40; pretrain_lr=1e-4
    cls_epochs=80; cls_lr=1e-4; retrain=False


# ── k-means ────────────────────────────────────────────────────────────────────

class IdentityAdapter(nn.Module):
    def __init__(self, dim=EMB_DIM):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    def forward(self, x): return self.linear(x)


def kmeans_numpy(embs, k, seed=SEED, n_iter=100):
    rng = np.random.default_rng(seed)
    centroids = embs[rng.choice(len(embs), k, replace=False)].copy()
    asgn = np.zeros(len(embs), dtype=int)
    for _ in range(n_iter):
        d = np.linalg.norm(embs[:, None] - centroids[None], axis=2)
        new = d.argmin(1)
        if np.all(new == asgn): break
        asgn = new
        for c in range(k):
            m = embs[asgn == c]
            if len(m): centroids[c] = m.mean(0)
    return centroids, asgn


def assign_concepts(embs, centroids):
    return np.linalg.norm(embs[:, None] - centroids[None], axis=2).argmin(1)


# ── Eval helpers ───────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_f1(model, loader, device):
    model.eval()
    preds, targets = [], []
    for xb, yb in loader:
        preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
        targets.extend(yb.numpy())
    return f1_score(targets, preds, average="macro", zero_division=0)


@torch.no_grad()
def extract_embs(model, loader, device):
    model.eval()
    embs, lbls = [], []
    for xb, yb in loader:
        embs.append(model.extract(xb.to(device)).cpu().numpy())
        lbls.append(yb.numpy())
    return np.concatenate(embs), np.concatenate(lbls)


@torch.no_grad()
def eval_adapter_f1(embs_np, labels_np, centroids, adapters, frozen_head, device,
                    thresholds=None):
    asgn = assign_concepts(embs_np, centroids)
    embs = torch.from_numpy(embs_np).to(device)
    corr = embs.clone()
    gate = (np.linalg.norm(embs_np - centroids[asgn], axis=1) > thresholds[asgn]
            if thresholds is not None
            else np.ones(len(embs_np), dtype=bool))
    for c in range(len(centroids)):
        mask = (asgn == c) & gate
        if mask.any():
            adapters[c].eval()
            idx = torch.from_numpy(mask).to(device)
            corr[idx] = adapters[c](embs[idx])
    preds = frozen_head(corr).argmax(1).cpu().numpy()
    return f1_score(labels_np, preds, average="macro", zero_division=0)


def calibrate_thresholds(src_embs, centroids, asgn, p):
    thr = np.zeros(K)
    for c in range(K):
        m = src_embs[asgn == c]
        if len(m): thr[c] = np.percentile(np.linalg.norm(m - centroids[c], axis=1), p)
    return thr


# ── Baselines ──────────────────────────────────────────────────────────────────

def run_fullft(src_model, calib_ds, src_ld, tgt_ld, device):
    model = copy.deepcopy(src_model)
    for p in model.parameters(): p.requires_grad_(True)
    opt   = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
    crit  = nn.CrossEntropyLoss()
    ld    = torch.utils.data.DataLoader(calib_ds, batch_size=BATCH, shuffle=True)
    for _ in range(FINETUNE_EP):
        model.train()
        for xb, yb in ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
    model.eval()
    return eval_f1(model, src_ld, device), eval_f1(model, tgt_ld, device)


def run_headft(src_model, calib_ds, src_ld, tgt_ld, device):
    model = copy.deepcopy(src_model)
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.head.parameters(): p.requires_grad_(True)
    opt   = torch.optim.AdamW(model.head.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
    crit  = nn.CrossEntropyLoss()
    ld    = torch.utils.data.DataLoader(calib_ds, batch_size=BATCH, shuffle=True)
    for _ in range(FINETUNE_EP):
        model.train()
        for xb, yb in ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
    model.eval()
    return eval_f1(model, src_ld, device), eval_f1(model, tgt_ld, device)


def run_tent(src_model, calib_ds, src_ld, tgt_ld, device, lr=1e-3, n_passes=1):
    model = copy.deepcopy(src_model)
    model.eval(); model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            if m.weight is not None: m.weight.requires_grad_(True)
            if m.bias   is not None: m.bias.requires_grad_(True)
    params = [p for m in model.modules()
              if isinstance(m, nn.LayerNorm) for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    ld  = torch.utils.data.DataLoader(calib_ds, batch_size=BATCH, shuffle=True)
    for _ in range(n_passes):
        for xb, _ in ld:
            xb = xb.to(device)
            with torch.enable_grad():
                out  = model(xb)
                loss = -(out.softmax(1) * out.log_softmax(1)).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return eval_f1(model, src_ld, device), eval_f1(model, tgt_ld, device)


def train_adapters(calib_embs, calib_lbls, centroids, frozen_head, device):
    adapters = nn.ModuleList([IdentityAdapter() for _ in range(K)]).to(device)
    opt  = torch.optim.AdamW(adapters.parameters(), lr=ADAPTER_LR)
    crit = nn.CrossEntropyLoss()
    asgn = assign_concepts(calib_embs, centroids)
    et   = torch.from_numpy(calib_embs).to(device)
    lt   = torch.from_numpy(calib_lbls).to(device)
    at   = torch.from_numpy(asgn.astype(np.int64))
    ds   = torch.utils.data.TensorDataset(et, lt, at)
    ld   = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True)
    eye  = torch.eye(EMB_DIM, device=device)
    for _ in range(ADAPTER_EP):
        for eb, yb, cb in ld:
            adapters.train(); opt.zero_grad()
            corr = eb.clone()
            for c in range(K):
                m = (cb == c)
                if m.any(): corr[m] = adapters[c](eb[m])
            loss = crit(frozen_head(corr), yb)
            for a in adapters:
                loss = loss + PENALTY * (torch.sum((a.linear.weight - eye)**2)
                                          + torch.sum(a.linear.bias**2))
            loss.backward(); opt.step()
    return adapters


# ── One scenario ───────────────────────────────────────────────────────────────

def run_scenario(src_id, tgt_id, device):
    src_name  = USER_NAMES[src_id]
    tgt_name  = USER_NAMES[tgt_id]
    scen_key  = f"{src_name}→{tgt_name}"
    print(f"\n{'='*60}\n  Scenario: {scen_key}\n{'='*60}")

    # Load / train source model
    src_model = tht.train_source(src_id, Args(), device)
    src_model.eval()
    for p in src_model.parameters(): p.requires_grad_(False)
    frozen_head = src_model.head

    # Data loaders
    def make_ld(uid, split, shuffle=False):
        ds = tht.HHARDataset(uid, split, DATA_ROOT)
        return ds, torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=shuffle)

    src_tr_ds, src_tr_ld = make_ld(src_id, "train")
    _,         src_te_ld = make_ld(src_id, "test")
    tgt_tr_ds, _         = make_ld(tgt_id, "train")
    _,         tgt_te_ld = make_ld(tgt_id, "test")

    # 20% stratified calibration from target train
    y_all = tgt_tr_ds.y.numpy()
    sss   = StratifiedShuffleSplit(n_splits=1, test_size=1-CALIB_FRAC, random_state=SEED)
    calib_idx, _ = next(sss.split(np.zeros(len(y_all)), y_all))
    calib_ds  = torch.utils.data.Subset(tgt_tr_ds, calib_idx)
    calib_ld  = torch.utils.data.DataLoader(calib_ds, batch_size=BATCH, shuffle=False)

    # Source embeddings + k-means
    src_embs, _   = extract_embs(src_model, src_tr_ld, device)
    centroids, asgn = kmeans_numpy(src_embs, K)
    thresholds    = calibrate_thresholds(src_embs, centroids, asgn, GATE_PCT)

    # Calibration embeddings
    calib_embs, calib_lbls = extract_embs(src_model, calib_ld, device)

    # Source F1 before any adaptation
    f1_src_before = eval_f1(src_model, src_te_ld, device)
    f1_tgt_nocorr = eval_f1(src_model, tgt_te_ld, device)
    print(f"  Source F1 (before): {f1_src_before:.4f}  "
          f"Target F1 (no adapt): {f1_tgt_nocorr:.4f}")

    results = {"no_corr": (f1_src_before, f1_tgt_nocorr, 0.0)}

    def _record(key, src_f1, tgt_f1):
        forget = f1_src_before - src_f1
        results[key] = (src_f1, tgt_f1, forget)
        print(f"  [{key:<14}] src={src_f1:.4f}  tgt={tgt_f1:.4f}  forget={forget:+.4f}")

    print("  [full_ft       ] ", end="", flush=True)
    _record("full_ft",  *run_fullft(src_model, calib_ds, src_te_ld, tgt_te_ld, device))

    print("  [head_ft       ] ", end="", flush=True)
    _record("head_ft",  *run_headft(src_model, calib_ds, src_te_ld, tgt_te_ld, device))

    print("  [tent          ] ", end="", flush=True)
    _record("tent",     *run_tent(src_model, calib_ds, src_te_ld, tgt_te_ld, device))

    # PACT adapters (shared for no-gate and gate variants)
    print("  [pact_ng/pact  ]  training adapters …", flush=True)
    adapters = train_adapters(calib_embs, calib_lbls, centroids, frozen_head, device)

    def _pact_f1(thr=None):
        src_embs_te, src_lbls_te = extract_embs(src_model, src_te_ld, device)
        tgt_embs_te, tgt_lbls_te = extract_embs(src_model, tgt_te_ld, device)
        return (eval_adapter_f1(src_embs_te, src_lbls_te, centroids, adapters, frozen_head, device, thr),
                eval_adapter_f1(tgt_embs_te, tgt_lbls_te, centroids, adapters, frozen_head, device, thr))

    _record("pact_ng", *_pact_f1(None))
    _record("pact",    *_pact_f1(thresholds))

    return scen_key, results


# ── Main ───────────────────────────────────────────────────────────────────────

CONDITIONS = ["no_corr", "full_ft", "head_ft", "tent", "pact_ng", "pact"]
COND_LABELS = {
    "no_corr":  "No correction",
    "full_ft":  "Full fine-tune",
    "head_ft":  "Head-only FT",
    "tent":     "TENT (label-free)",
    "pact_ng":  "PACT (no gate)",
    "pact":     "PACT",
}


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"HHAR cross-user sweep  |  {len(SCENARIOS)} scenarios  |  {len(CONDITIONS)} conditions\n")

    all_results = {}
    pop_src    = {c: [] for c in CONDITIONS}
    pop_tgt    = {c: [] for c in CONDITIONS}
    pop_forget = {c: [] for c in CONDITIONS}

    for src_id, tgt_id in SCENARIOS:
        scen_key, res = run_scenario(src_id, tgt_id, device)
        all_results[scen_key] = res
        for c in CONDITIONS:
            if c in res:
                s, t, f = res[c]
                pop_src[c].append(s); pop_tgt[c].append(t); pop_forget[c].append(f)

    # ── Population summary ─────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("POPULATION SUMMARY (mean ± std across 10 HHAR scenarios)")
    print(f"{'='*72}")
    hdr = f"  {'Method':<20} {'Src F1':>12}  {'Tgt F1':>12}  {'Forgetting':>12}"
    sep = "  " + "-"*68
    print(hdr); print(sep)

    lines = [f"POPULATION SUMMARY — HHAR cross-user sweep ({len(SCENARIOS)} scenarios)\n",
             hdr, sep]
    for c in CONDITIONS:
        sm = np.mean(pop_src[c]);    ss = np.std(pop_src[c])
        tm = np.mean(pop_tgt[c]);    ts = np.std(pop_tgt[c])
        fm = np.mean(pop_forget[c]); fs = np.std(pop_forget[c])
        row = (f"  {COND_LABELS[c]:<20} {sm:.4f}±{ss:.4f}  "
               f"{tm:.4f}±{ts:.4f}  {fm:+.4f}±{fs:.4f}")
        print(row); lines.append(row)
    (OUT_DIR / "population_summary.txt").write_text("\n".join(lines))

    # ── Per-scenario table ─────────────────────────────────────────────────────
    scen_lines = [f"Per-scenario results — HHAR\n"]
    for sk in all_results:
        scen_lines.append(f"\nScenario: {sk}")
        scen_lines.append(f"  {'Method':<20} {'Src F1':>8}  {'Tgt F1':>8}  {'Forget':>8}")
        for c in CONDITIONS:
            if c in all_results[sk]:
                s, t, f = all_results[sk][c]
                scen_lines.append(f"  {COND_LABELS[c]:<20} {s:>8.4f}  {t:>8.4f}  {f:>+8.4f}")
    (OUT_DIR / "per_scenario_table.txt").write_text("\n".join(scen_lines))

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x      = np.arange(len(CONDITIONS))
    colors = ["#aec6cf", "#ff6b6b", "#ffa07a", "#f4d35e", "#2196F3", "#1b7a2e"]

    for ax, key, title in zip(axes,
                               ["src", "tgt"],
                               ["Source F1 (retention)", "Target F1 (drift recovery)"]):
        means = [np.mean(pop_src[c] if key == "src" else pop_tgt[c]) for c in CONDITIONS]
        stds  = [np.std(pop_src[c]  if key == "src" else pop_tgt[c]) for c in CONDITIONS]
        bars  = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                       edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([COND_LABELS[c] for c in CONDITIONS],
                           rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Macro F1")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axhline(0.9, color="gray", ls="--", lw=0.8, alpha=0.7)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                    f"{m:.3f}", ha="center", fontsize=7)
        ax.grid(axis="y", ls="--", alpha=0.4)

    fig.suptitle("Cross-User HHAR Population Sweep: PACT vs Baselines\n"
                 f"(10 AdaTime scenarios, TSLANet backbone, mean ± std)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "population_figure.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT_DIR / "population_figure.pdf", bbox_inches="tight")
    plt.close()

    print(f"\nOutputs → {OUT_DIR}")
    print(f"  population_summary.txt  per_scenario_table.txt  population_figure.{{png,pdf}}")


if __name__ == "__main__":
    main()
