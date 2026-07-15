"""
exp_crosshhar_ksweep.py — Sweep K (number of adapters) for PACT on HHAR.

Runs all 10 AdaTime HHAR scenarios for each K in K_VALUES.
Reports mean±std target F1 for PACT (no gate) and PACT (learned gate).

Outputs → results/crosshhar_ksweep/
  sweep_summary.txt   — table: K × {pact_ng, pact_lg} mean±std tgt F1
  per_k_table.txt     — per-scenario breakdown per K

Run:
  CUDA_VISIBLE_DEVICES=0 \\
  LD_PRELOAD=/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6 \\
  /home/AD/sachith/.conda/envs/entrope/bin/python -u exp_crosshhar_ksweep.py
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
OUT_DIR    = Path(__file__).parent / "results" / "crosshhar_ksweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS  = [(0,6),(1,6),(2,7),(3,8),(4,5),(5,0),(6,1),(7,4),(8,3),(0,2)]

EMB_DIM    = 128
N_CLASSES  = 6
CALIB_FRAC = 0.05
SEED       = 42
PENALTY    = 1e-3
ADAPTER_EP = 200
ADAPTER_LR = 1e-3
GATE_EP    = 200
GATE_LR    = 1e-3
BATCH      = 256

K_VALUES   = [4, 8, 12, 16, 24, 32, 48]

USER_NAMES = {i: f"User-{chr(ord('a')+i)}" for i in range(9)}


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


# ── Adapter + gate training (K-parameterised) ──────────────────────────────────

def train_adapters(calib_embs, calib_lbls, centroids, frozen_head, device, k):
    adapters = nn.ModuleList([IdentityAdapter() for _ in range(k)]).to(device)
    opt  = torch.optim.AdamW(adapters.parameters(), lr=ADAPTER_LR)
    crit = nn.CrossEntropyLoss()
    asgn = assign_concepts(calib_embs, centroids)
    et   = torch.from_numpy(calib_embs).to(device)
    lt   = torch.from_numpy(calib_lbls).to(device)
    at   = torch.from_numpy(asgn.astype(np.int64))
    ds   = torch.utils.data.TensorDataset(et, lt, at)
    ld   = torch.utils.data.DataLoader(ds, batch_size=min(BATCH, len(et)), shuffle=True)
    eye  = torch.eye(EMB_DIM, device=device)
    for _ in range(ADAPTER_EP):
        for eb, yb, cb in ld:
            adapters.train(); opt.zero_grad()
            corr = eb.clone()
            for c in range(k):
                m = (cb == c)
                if m.any(): corr[m] = adapters[c](eb[m])
            loss = crit(frozen_head(corr), yb)
            for a in adapters:
                loss = loss + PENALTY * (torch.sum((a.linear.weight - eye)**2)
                                         + torch.sum(a.linear.bias**2))
            loss.backward(); opt.step()
    return adapters


def train_learned_gates(src_embs, src_asgn, calib_embs, calib_asgn, device, k):
    gates = []
    for c in range(k):
        neg = src_embs[src_asgn == c]
        pos = calib_embs[calib_asgn == c]
        if len(pos) == 0:
            gates.append(None); continue
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
    n_empty = sum(g is None for g in gates)
    print(f"    Gates trained: {k - n_empty}/{k} concepts have calib examples", flush=True)
    return gates


def make_learned_gate_fn(gates, device, k):
    @torch.no_grad()
    def fn(embs_np, asgn):
        result = np.zeros(len(embs_np), dtype=bool)
        for c in range(k):
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
def eval_with_gate(embs_np, labels_np, centroids, adapters, frozen_head,
                   device, gate_fn, k):
    asgn  = assign_concepts(embs_np, centroids)
    embs  = torch.from_numpy(embs_np).to(device)
    corr  = embs.clone()
    apply = gate_fn(embs_np, asgn)
    for c in range(k):
        mask = (asgn == c) & apply
        if mask.any():
            adapters[c].eval()
            idx = torch.from_numpy(mask).to(device)
            corr[idx] = adapters[c](embs[idx])
    preds = frozen_head(corr).argmax(1).cpu().numpy()
    return f1_score(labels_np, preds, average="macro", zero_division=0)


# ── Per-scenario runner for one K ─────────────────────────────────────────────

def run_scenario_k(src_model, src_embs, src_tr_ld, src_te_ld,
                   calib_embs, calib_lbls, calib_ld,
                   tgt_te_embs, tgt_te_lbls, tgt_te_ld,
                   frozen_head, device, k, f1_src_before, f1_tgt_nocorr):

    centroids, src_asgn = kmeans_numpy(src_embs, k)
    calib_asgn = assign_concepts(calib_embs, centroids)
    concept_counts = np.bincount(calib_asgn, minlength=k)
    print(f"    K={k}: calib/concept min={concept_counts.min()} max={concept_counts.max()}",
          flush=True)

    adapters = train_adapters(calib_embs, calib_lbls, centroids, frozen_head, device, k)
    gates    = train_learned_gates(src_embs, src_asgn, calib_embs, calib_asgn, device, k)
    gate_fn  = make_learned_gate_fn(gates, device, k)

    def always_apply(embs_np, asgn):
        return np.ones(len(embs_np), dtype=bool)

    src_te_embs, src_te_lbls = extract_embs(src_model, src_te_ld, device)

    t_ng = eval_with_gate(tgt_te_embs, tgt_te_lbls, centroids, adapters, frozen_head,
                          device, always_apply, k)
    s_ng = eval_with_gate(src_te_embs, src_te_lbls, centroids, adapters, frozen_head,
                          device, always_apply, k)

    t_lg = eval_with_gate(tgt_te_embs, tgt_te_lbls, centroids, adapters, frozen_head,
                          device, gate_fn, k)
    s_lg = eval_with_gate(src_te_embs, src_te_lbls, centroids, adapters, frozen_head,
                          device, gate_fn, k)

    print(f"    K={k}  pact_ng: src={s_ng:.4f} tgt={t_ng:.4f}  "
          f"pact_lg: src={s_lg:.4f} tgt={t_lg:.4f}", flush=True)

    return {
        "pact_ng": (s_ng, t_ng, max(0.0, f1_src_before - s_ng)),
        "pact_lg": (s_lg, t_lg, max(0.0, f1_src_before - s_lg)),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  K sweep: {K_VALUES}\n")

    # Results: k → scen_key → {pact_ng: (s,t,f), pact_lg: (s,t,f)}
    sweep_results = {k: {} for k in K_VALUES}

    for src_id, tgt_id in SCENARIOS:
        src_name = USER_NAMES[src_id]
        tgt_name = USER_NAMES[tgt_id]
        scen_key = f"{src_name}→{tgt_name}"
        print(f"\n{'='*60}\n  Scenario: {scen_key}\n{'='*60}")

        # Load cached source model
        src_model = tht.train_source(src_id, Args(), device)
        src_model.eval()
        for p in src_model.parameters(): p.requires_grad_(False)
        frozen_head = src_model.head

        def make_ld(uid, split, shuffle=False):
            ds = tht.HHARDataset(uid, split, DATA_ROOT)
            return ds, torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=shuffle)

        src_tr_ds, src_tr_ld = make_ld(src_id, "train")
        _,         src_te_ld = make_ld(src_id, "test")
        tgt_tr_ds, _         = make_ld(tgt_id, "train")
        _,         tgt_te_ld = make_ld(tgt_id, "test")

        # 5% stratified calibration
        y_all = tgt_tr_ds.y.numpy()
        sss   = StratifiedShuffleSplit(n_splits=1, test_size=1-CALIB_FRAC, random_state=SEED)
        calib_idx, _ = next(sss.split(np.zeros(len(y_all)), y_all))
        calib_ds = torch.utils.data.Subset(tgt_tr_ds, calib_idx)
        calib_ld = torch.utils.data.DataLoader(calib_ds, batch_size=BATCH, shuffle=False)
        print(f"  Calib: {len(calib_idx)} samples  (5% of {len(y_all)})")

        # Shared embeddings (extracted once)
        src_embs, _              = extract_embs(src_model, src_tr_ld, device)
        calib_embs, calib_lbls   = extract_embs(src_model, calib_ld, device)
        tgt_te_embs, tgt_te_lbls = extract_embs(src_model, tgt_te_ld, device)

        f1_src_before = eval_f1(src_model, src_te_ld, device)
        f1_tgt_nocorr = eval_f1(src_model, tgt_te_ld, device)
        print(f"  src_before={f1_src_before:.4f}  tgt_nocorr={f1_tgt_nocorr:.4f}")

        for k in K_VALUES:
            print(f"\n  -- K={k} --", flush=True)
            res = run_scenario_k(
                src_model, src_embs, src_tr_ld, src_te_ld,
                calib_embs, calib_lbls, calib_ld,
                tgt_te_embs, tgt_te_lbls, tgt_te_ld,
                frozen_head, device, k, f1_src_before, f1_tgt_nocorr)
            sweep_results[k][scen_key] = res

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"K-SWEEP SUMMARY  (HHAR, 5% labels, 10 scenarios, mean±std tgt F1)")
    print(f"{'='*72}")
    hdr = f"  {'K':>4}   {'PACT(ng) tgt':>14}  {'PACT(lg) tgt':>14}  {'ng src':>8}  {'lg src':>8}"
    sep = "  " + "-" * 64
    print(hdr); print(sep)

    lines = ["K-SWEEP SUMMARY — HHAR cross-user, 5% labels, TSLANet\n", hdr, sep]

    # Also save per-K, per-scenario breakdown
    per_k_lines = ["Per-scenario K-sweep results — HHAR 5% labels\n"]

    for k in K_VALUES:
        ng_tgts, lg_tgts = [], []
        ng_srcs, lg_srcs = [], []
        per_k_lines.append(f"\n--- K={k} ---")
        per_k_lines.append(f"  {'Scenario':<20} {'ng_tgt':>8}  {'lg_tgt':>8}  {'ng_src':>8}  {'lg_src':>8}")
        for sk, res in sweep_results[k].items():
            s_ng, t_ng, _ = res["pact_ng"]
            s_lg, t_lg, _ = res["pact_lg"]
            ng_tgts.append(t_ng); lg_tgts.append(t_lg)
            ng_srcs.append(s_ng); lg_srcs.append(s_lg)
            per_k_lines.append(
                f"  {sk:<20} {t_ng:>8.4f}  {t_lg:>8.4f}  {s_ng:>8.4f}  {s_lg:>8.4f}")
        row = (f"  {k:>4}   {np.mean(ng_tgts):>6.4f}±{np.std(ng_tgts):.4f}  "
               f"{np.mean(lg_tgts):>6.4f}±{np.std(lg_tgts):.4f}  "
               f"{np.mean(ng_srcs):>6.4f}  {np.mean(lg_srcs):>6.4f}")
        print(row); lines.append(row)

    (OUT_DIR / "sweep_summary.txt").write_text("\n".join(lines))
    (OUT_DIR / "per_k_table.txt").write_text("\n".join(per_k_lines))

    # ── Plot ───────────────────────────────────────────────────────────────────
    ng_means = [np.mean([sweep_results[k][sk]["pact_ng"][1] for sk in sweep_results[k]])
                for k in K_VALUES]
    lg_means = [np.mean([sweep_results[k][sk]["pact_lg"][1] for sk in sweep_results[k]])
                for k in K_VALUES]
    ng_stds  = [np.std ([sweep_results[k][sk]["pact_ng"][1] for sk in sweep_results[k]])
                for k in K_VALUES]
    lg_stds  = [np.std ([sweep_results[k][sk]["pact_lg"][1] for sk in sweep_results[k]])
                for k in K_VALUES]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(K_VALUES, ng_means, yerr=ng_stds, marker="o", label="PACT (no gate)",
                color="#2196F3", capsize=4)
    ax.errorbar(K_VALUES, lg_means, yerr=lg_stds, marker="s", label="PACT (learned gate)",
                color="#1b7a2e", capsize=4)
    ax.axhline(0.9237, color="red",  ls="--", lw=1.2, label="CDAN† (SOTA, 0.9237)")
    ax.axhline(0.9192, color="orange", ls="--", lw=1.0, label="DANN† (0.9192)")
    ax.set_xlabel("K (number of adapters)", fontsize=12)
    ax.set_ylabel("Mean Macro F1 (target)", fontsize=12)
    ax.set_title("PACT vs K — HHAR Cross-User, 5% Labels\n(10 AdaTime scenarios, TSLANet backbone)",
                 fontsize=11, fontweight="bold")
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=10)
    ax.grid(True, ls="--", alpha=0.4)
    ax.set_ylim(0.5, 1.0)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "ksweep_plot.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT_DIR / "ksweep_plot.pdf", bbox_inches="tight")
    plt.close()

    print(f"\nOutputs → {OUT_DIR}")
    print(f"  sweep_summary.txt  per_k_table.txt  ksweep_plot.{{png,pdf}}")


if __name__ == "__main__":
    main()
