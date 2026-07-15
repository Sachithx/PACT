"""
exp_crossfd_lg.py — FD Cross-Domain Sweep: Learned Gate @ 5% Labels

10 AdaTime FD scenarios (same as run_unified_eval.py):
  (A→B),(A→D),(B→A),(B→C),(B→D),(C→B),(C→D),(D→A),(D→B),(D→C)
  domain IDs: 0=FD-A, 1=FD-B, 2=FD-C, 3=FD-D

Label budget : 5% stratified target train  (≈409 samples/domain)
Gate         : learned binary MLP per concept
               source embs = negative, 5% calib embs = positive

Conditions:
  no_corr  — frozen source, zero adaptation
  full_ft  — full model fine-tune on 5% calib
  pact_ng  — frozen host + k=16 adapters, no gate (always apply)
  pact_lg  — frozen host + k=16 adapters + learned gate MLP

Reuses existing FD-A/B/C/D checkpoints.
Outputs → results/crossfd_lg/

Run:
  CUDA_VISIBLE_DEVICES=1 \\
  LD_PRELOAD=/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6 \\
  /home/AD/sachith/.conda/envs/entrope/bin/python -u exp_crossfd_lg.py
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
    "tft", Path(__file__).parent / "transfer_fd_tslanet.py")
tft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tft)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_ROOT  = Path(__file__).parent / "dataset"
CKPT_DIR   = Path(__file__).parent / "TSLANet" / "Classification" / "output" / "tslanet"
OUT_DIR    = Path(__file__).parent / "results" / "crossfd_lg"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FD_MAP     = {0: "FD-A", 1: "FD-B", 2: "FD-C", 3: "FD-D"}
SCENARIOS  = [(0,1),(0,3),(1,0),(1,2),(1,3),(2,1),(2,3),(3,0),(3,1),(3,2)]

EMB_DIM    = 128
N_CLASSES  = 3
CALIB_FRAC = 0.05
SEED       = 42
K          = 16
PENALTY    = 1e-3
ADAPTER_EP = 200
ADAPTER_LR = 1e-3
GATE_EP    = 200
GATE_LR    = 1e-3
FINETUNE_EP = 80
FINETUNE_LR = 1e-4
BATCH      = 256


class Args:
    emb_dim=128; depth=2; patch_size=8; dropout_rate=0.15; masking_ratio=0.4
    ICB=True; ASB=True; adaptive_filter=True
    data_root=str(DATA_ROOT)
    ckpt_dir=str(CKPT_DIR)
    batch_size=256; pretrain=True; pretrain_epochs=40; pretrain_lr=1e-4
    cls_epochs=80; cls_lr=1e-4; retrain=False


# ── Modules ────────────────────────────────────────────────────────────────────

class IdentityAdapter(nn.Module):
    def __init__(self, dim=EMB_DIM):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    def forward(self, x): return self.linear(x)


class GateMLP(nn.Module):
    """Binary classifier: 0=source-like (no adapt), 1=drifted (apply adapter)."""
    def __init__(self, in_dim=EMB_DIM, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 2),
        )
    def forward(self, x): return self.net(x)


# ── k-means ────────────────────────────────────────────────────────────────────

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


# ── Helpers ────────────────────────────────────────────────────────────────────

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
    return np.concatenate(embs).astype(np.float32), np.concatenate(lbls)


# ── Adapter training ───────────────────────────────────────────────────────────

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


# ── Learned gate training ──────────────────────────────────────────────────────

def train_learned_gates(src_embs, src_asgn, calib_embs, calib_asgn, device):
    gates = []
    n_empty = 0
    for c in range(K):
        neg = src_embs[src_asgn == c]
        pos = calib_embs[calib_asgn == c]
        if len(pos) == 0:
            gates.append(None); n_empty += 1; continue

        rng = np.random.default_rng(SEED + c)
        if len(neg) > len(pos):
            pos = pos[rng.choice(len(pos), len(neg), replace=True)]
        elif len(pos) > len(neg):
            neg = neg[rng.choice(len(neg), len(pos), replace=True)]

        X = np.concatenate([neg, pos])
        y = np.concatenate([np.zeros(len(neg), dtype=np.int64),
                            np.ones(len(pos),  dtype=np.int64)])
        perm = np.random.default_rng(SEED + c + 100).permutation(len(X))
        X, y = X[perm], y[perm]

        Xt = torch.from_numpy(X).to(device)
        yt = torch.from_numpy(y).to(device)
        ld = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xt, yt),
            batch_size=min(BATCH, len(X)), shuffle=True)

        gate = GateMLP().to(device)
        opt  = torch.optim.AdamW(gate.parameters(), lr=GATE_LR, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        for _ in range(GATE_EP):
            gate.train()
            for xb, yb_g in ld:
                opt.zero_grad(); crit(gate(xb), yb_g).backward(); opt.step()
        gates.append(gate)

    print(f"    Learned gates: {K - n_empty}/{K} concepts have positive calib examples")
    return gates


# ── Gate functions ─────────────────────────────────────────────────────────────

def make_learned_gate(gates, device):
    @torch.no_grad()
    def fn(embs_np, asgn):
        result = np.zeros(len(embs_np), dtype=bool)
        for c in range(K):
            mask = (asgn == c)
            if not mask.any(): continue
            gate = gates[c]
            if gate is None:
                result[mask] = False
            else:
                gate.eval()
                p = gate(torch.from_numpy(embs_np[mask]).to(device)).argmax(1).cpu().numpy()
                result[mask] = (p == 1)
        return result
    return fn


@torch.no_grad()
def eval_with_gate(embs_np, labels_np, centroids, adapters, frozen_head, device, gate_fn):
    asgn  = assign_concepts(embs_np, centroids)
    embs  = torch.from_numpy(embs_np).to(device)
    corr  = embs.clone()
    apply = gate_fn(embs_np, asgn)
    for c in range(K):
        mask = (asgn == c) & apply
        if mask.any():
            adapters[c].eval()
            idx = torch.from_numpy(mask).to(device)
            corr[idx] = adapters[c](embs[idx])
    preds = frozen_head(corr).argmax(1).cpu().numpy()
    return f1_score(labels_np, preds, average="macro", zero_division=0)


# ── Baselines ──────────────────────────────────────────────────────────────────

def run_fullft(src_model, calib_ds, src_ld, tgt_ld, device):
    model = copy.deepcopy(src_model)
    for p in model.parameters(): p.requires_grad_(True)
    opt  = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    ld   = torch.utils.data.DataLoader(calib_ds, batch_size=BATCH, shuffle=True)
    for _ in range(FINETUNE_EP):
        model.train()
        for xb, yb in ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
    model.eval()
    return eval_f1(model, src_ld, device), eval_f1(model, tgt_ld, device)


# ── One scenario ───────────────────────────────────────────────────────────────

def run_scenario(src_id, tgt_id, device):
    src_name = FD_MAP[src_id]
    tgt_name = FD_MAP[tgt_id]
    scen_key = f"{src_name}→{tgt_name}"
    print(f"\n{'='*60}\n  Scenario: {scen_key}  [5% labels]\n{'='*60}")

    # Load cached source model
    src_model = tft.train_source(src_name, Args(), device)
    src_model.eval()
    for p in src_model.parameters(): p.requires_grad_(False)
    frozen_head = src_model.head

    def make_ld(name, split, shuffle=False):
        ds = tft.FDDataset(name, split, DATA_ROOT)
        return ds, torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=shuffle)

    src_tr_ds, src_tr_ld = make_ld(src_name, "train")
    _,         src_te_ld = make_ld(src_name, "test")
    tgt_tr_ds, _         = make_ld(tgt_name, "train")
    _,         tgt_te_ld = make_ld(tgt_name, "test")

    # 5% stratified calibration from target train
    y_all = tgt_tr_ds.y.numpy()
    sss   = StratifiedShuffleSplit(n_splits=1, test_size=1-CALIB_FRAC, random_state=SEED)
    calib_idx, _ = next(sss.split(np.zeros(len(y_all)), y_all))
    calib_ds  = torch.utils.data.Subset(tgt_tr_ds, calib_idx)
    calib_ld  = torch.utils.data.DataLoader(calib_ds, batch_size=BATCH, shuffle=False)
    print(f"  Calib: {len(calib_idx)} samples (5% of {len(y_all)})")

    # Source embeddings + k-means
    src_embs, _         = extract_embs(src_model, src_tr_ld, device)
    centroids, src_asgn = kmeans_numpy(src_embs, K)

    # Calibration embeddings
    calib_embs, calib_lbls = extract_embs(src_model, calib_ld, device)
    calib_asgn = assign_concepts(calib_embs, centroids)
    counts = np.bincount(calib_asgn, minlength=K)
    print(f"  Calib per concept (min/max): {counts.min()}/{counts.max()}")

    # Train adapters
    print("  Training adapters …", flush=True)
    adapters = train_adapters(calib_embs, calib_lbls, centroids, frozen_head, device)

    # Train learned gates
    print("  Training learned gates …", flush=True)
    gates = train_learned_gates(src_embs, src_asgn, calib_embs, calib_asgn, device)
    learned_gate_fn = make_learned_gate(gates, device)

    def always_apply(embs_np, asgn):
        return np.ones(len(embs_np), dtype=bool)

    # Source F1 baseline
    f1_src_before = eval_f1(src_model, src_te_ld, device)
    f1_tgt_nocorr = eval_f1(src_model, tgt_te_ld, device)
    print(f"  Source F1 (before): {f1_src_before:.4f}  "
          f"Target F1 (no adapt): {f1_tgt_nocorr:.4f}")

    results = {"no_corr": (f1_src_before, f1_tgt_nocorr, 0.0)}

    def _record(key, src_f1, tgt_f1):
        forget = f1_src_before - src_f1
        results[key] = (src_f1, tgt_f1, forget)
        print(f"  [{key:<9}] src={src_f1:.4f}  tgt={tgt_f1:.4f}  forget={forget:+.4f}",
              flush=True)

    print("  [full_ft  ] ", end="", flush=True)
    _record("full_ft", *run_fullft(src_model, calib_ds, src_te_ld, tgt_te_ld, device))

    # Extract test embeddings once for both PACT variants
    src_te_embs, src_te_lbls = extract_embs(src_model, src_te_ld, device)
    tgt_te_embs, tgt_te_lbls = extract_embs(src_model, tgt_te_ld, device)

    _record("pact_ng",
            eval_with_gate(src_te_embs, src_te_lbls, centroids, adapters, frozen_head,
                           device, always_apply),
            eval_with_gate(tgt_te_embs, tgt_te_lbls, centroids, adapters, frozen_head,
                           device, always_apply))

    _record("pact_lg",
            eval_with_gate(src_te_embs, src_te_lbls, centroids, adapters, frozen_head,
                           device, learned_gate_fn),
            eval_with_gate(tgt_te_embs, tgt_te_lbls, centroids, adapters, frozen_head,
                           device, learned_gate_fn))

    return scen_key, results


# ── Main ───────────────────────────────────────────────────────────────────────

CONDITIONS  = ["no_corr", "full_ft", "pact_ng", "pact_lg"]
COND_LABELS = {
    "no_corr":  "No correction",
    "full_ft":  "Full FT (5%)",
    "pact_ng":  "PACT (no gate)",
    "pact_lg":  "PACT",
}
PLOT_COLORS = ["#aec6cf", "#ff6b6b", "#2196F3", "#1b7a2e"]


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Label budget: 5% stratified target train  |  "
          f"{len(SCENARIOS)} scenarios  |  {len(CONDITIONS)} conditions\n")

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
    print(f"POPULATION SUMMARY  (5% labels, learned gate  |  {len(SCENARIOS)} scenarios)")
    print(f"{'='*72}")
    hdr = f"  {'Method':<22} {'Src F1':>14}  {'Tgt F1':>14}  {'Forgetting':>14}"
    sep = "  " + "-"*68
    print(hdr); print(sep)

    lines = [f"POPULATION SUMMARY — FD cross-domain sweep, 5% labels, learned gate\n",
             hdr, sep]
    for c in CONDITIONS:
        sm = np.mean(pop_src[c]);    ss = np.std(pop_src[c])
        tm = np.mean(pop_tgt[c]);    ts = np.std(pop_tgt[c])
        fm = np.mean(pop_forget[c]); fs = np.std(pop_forget[c])
        row = (f"  {COND_LABELS[c]:<22} {sm:.4f}±{ss:.4f}  "
               f"{tm:.4f}±{ts:.4f}  {fm:+.4f}±{fs:.4f}")
        print(row); lines.append(row)
    (OUT_DIR / "population_summary.txt").write_text("\n".join(lines))

    # ── Per-scenario table ─────────────────────────────────────────────────────
    scen_lines = ["Per-scenario results — FD 5% labels + learned gate\n"]
    for sk in all_results:
        scen_lines.append(f"\nScenario: {sk}")
        scen_lines.append(f"  {'Method':<22} {'Src F1':>8}  {'Tgt F1':>8}  {'Forget':>8}")
        for c in CONDITIONS:
            if c in all_results[sk]:
                s, t, f = all_results[sk][c]
                scen_lines.append(
                    f"  {COND_LABELS[c]:<22} {s:>8.4f}  {t:>8.4f}  {f:>+8.4f}")
    (OUT_DIR / "per_scenario_table.txt").write_text("\n".join(scen_lines))

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    x = np.arange(len(CONDITIONS))

    for ax, key, title in zip(
            axes[:2],
            ["src", "tgt"],
            ["Source F1 (retention)", "Target F1 (drift recovery)"]):
        data  = pop_src if key == "src" else pop_tgt
        means = [np.mean(data[c]) for c in CONDITIONS]
        stds  = [np.std(data[c])  for c in CONDITIONS]
        bars  = ax.bar(x, means, yerr=stds, capsize=5, color=PLOT_COLORS,
                       edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([COND_LABELS[c] for c in CONDITIONS],
                           rotation=20, ha="right", fontsize=10)
        ax.set_ylim(0, 1.05); ax.set_ylabel("Macro F1")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axhline(0.9, color="gray", ls="--", lw=0.8, alpha=0.6)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                    f"{m:.3f}", ha="center", fontsize=9)
        ax.grid(axis="y", ls="--", alpha=0.4)

    # Forgetting panel
    ax = axes[2]
    f_means = [np.mean(pop_forget[c]) for c in CONDITIONS]
    f_stds  = [np.std(pop_forget[c])  for c in CONDITIONS]
    ax.bar(x, f_means, yerr=f_stds, capsize=5, color=PLOT_COLORS,
           edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[c] for c in CONDITIONS],
                       rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Forgetting (src F1 drop)")
    ax.set_title("Forgetting metric", fontsize=10, fontweight="bold")
    ax.grid(axis="y", ls="--", alpha=0.4)

    fig.suptitle("FD Cross-Domain Sweep: PACT Learned Gate @ 5% Labels\n"
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
