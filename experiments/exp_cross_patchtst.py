"""
exp_cross_patchtst.py — PACT backbone ablation: PatchTST vs TSLANet.

Runs the same PACT experiment as exp_crossfd_lg.py / exp_crosshhar_lg.py /
exp_crosssleep_lg.py but with PatchTST as the backbone instead of TSLANet.

Conditions: no_corr, full_ft, pact_ng, pact_lg  (same as the TSLANet runs)
Label budget: 5%  |  Gate: learned binary MLP

Datasets:
  FD       — 10 AdaTime scenarios  (4 domains, 3 classes, seq=5120, 1 ch)
  HHAR     — 10 AdaTime scenarios  (9 users,   6 classes, seq=128,  3 ch)
  Sleep-EDF— 10 AdaTime scenarios  (20 subjects,5 classes, seq=3000,1 ch)

Outputs → results/cross_patchtst/{fd,hhar,sleep}_{summary,figure}.{txt,png,pdf}

Run:
  CUDA_VISIBLE_DEVICES=0 \\
  LD_PRELOAD=/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6 \\
  /home/AD/sachith/.conda/envs/entrope/bin/python -u exp_cross_patchtst.py
"""

import sys, copy
import torch, torch.nn as nn, numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

_REPO = Path(__file__).parent
sys.path.insert(0, str(_REPO / "TSLANet" / "Classification"))

import importlib.util
def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

tft  = _load("tft",  _REPO / "transfer_fd_tslanet.py")
tht  = _load("tht",  _REPO / "transfer_hhar_tslanet.py")
tst  = _load("tst",  _REPO / "transfer_sleep_tslanet.py")
tpst = _load("tpst", _REPO / "transfer_patchtst.py")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_ROOT  = _REPO / "dataset"
OUT_DIR    = _REPO / "results" / "cross_patchtst"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CALIB_FRAC = 0.05
SEED       = 42
K          = 16
PENALTY    = 1e-3
ADAPTER_EP = 200
ADAPTER_LR = 1e-3
GATE_EP    = 200
GATE_LR    = 1e-3
FT_EP      = 80
FT_LR      = 1e-4
EMB_DIM    = 128   # d_model; same for all datasets via mean-pooling

CONDITIONS  = ["no_corr", "full_ft", "pact_ng", "pact_lg"]
COND_LABELS = {
    "no_corr": "No correction",
    "full_ft": "Full FT (5%)",
    "pact_ng": "PACT (no gate)",
    "pact_lg": "PACT",
}
PLOT_COLORS = ["#aec6cf", "#ff6b6b", "#2196F3", "#1b7a2e"]

DATASETS = [
    # (tag, cfg, scenarios, dataset_fn, domain_name_fn)
    ("FD",
     tpst._FD_CFG,
     [(0,1),(0,3),(1,0),(1,2),(1,3),(2,1),(2,3),(3,0),(3,1),(3,2)],
     lambda sid, split: tft.FDDataset(tft.FD_DATASETS[sid], split, DATA_ROOT),
     lambda sid: tft.FD_DATASETS[sid]),
    ("HHAR",
     tpst._HHAR_CFG,
     [(0,6),(1,6),(2,7),(3,8),(4,5),(5,0),(6,1),(7,4),(8,3),(0,2)],
     lambda sid, split: tht.HHARDataset(sid, split, DATA_ROOT),
     lambda sid: f"User-{chr(ord('a')+sid)}"),
    ("SleepEDF",
     tpst._SLEEP_CFG,
     tst.SLEEP_SCENARIOS,
     lambda sid, split: tst.SleepEDFDataset(sid, split, DATA_ROOT),
     lambda sid: f"Sleep-{sid}"),
]


# ── Shared PACT modules ────────────────────────────────────────────────────────

class IdentityAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(EMB_DIM, EMB_DIM)
        nn.init.eye_(self.linear.weight); nn.init.zeros_(self.linear.bias)
    def forward(self, x): return self.linear(x)


class GateMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMB_DIM, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 2))
    def forward(self, x): return self.net(x)


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
    model.eval(); preds, targets = [], []
    for xb, yb in loader:
        preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
        targets.extend(yb.numpy())
    return f1_score(targets, preds, average="macro", zero_division=0)

@torch.no_grad()
def extract_embs(model, loader, device):
    model.eval(); embs, lbls = [], []
    for xb, yb in loader:
        embs.append(model.extract(xb.to(device)).cpu().numpy())
        lbls.append(yb.numpy())
    return np.concatenate(embs).astype(np.float32), np.concatenate(lbls)

def make_loader(ds, batch, shuffle=False):
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle)

def calib_subset(ds):
    y = torch.stack([ds[i][1] for i in range(len(ds))]).numpy()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-CALIB_FRAC, random_state=SEED)
    idx, _ = next(sss.split(np.zeros(len(y)), y))
    return torch.utils.data.Subset(ds, idx)


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
    ld   = torch.utils.data.DataLoader(ds, batch_size=min(256, len(et)), shuffle=True)
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


def train_gates(src_embs, src_asgn, calib_embs, calib_asgn, device):
    gates = []
    for c in range(K):
        neg = src_embs[src_asgn == c]
        pos = calib_embs[calib_asgn == c]
        if len(pos) == 0:
            gates.append(None); continue
        rng = np.random.default_rng(SEED + c)
        if len(neg) > len(pos): pos = pos[rng.choice(len(pos), len(neg), replace=True)]
        elif len(pos) > len(neg): neg = neg[rng.choice(len(neg), len(pos), replace=True)]
        X = np.concatenate([neg, pos])
        y = np.concatenate([np.zeros(len(neg), dtype=np.int64), np.ones(len(pos), dtype=np.int64)])
        perm = np.random.default_rng(SEED + c + 100).permutation(len(X))
        X, y = X[perm], y[perm]
        Xt = torch.from_numpy(X).to(device); yt = torch.from_numpy(y).to(device)
        ld = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt, yt),
                                         batch_size=min(256, len(X)), shuffle=True)
        gate = GateMLP().to(device)
        opt  = torch.optim.AdamW(gate.parameters(), lr=GATE_LR, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        for _ in range(GATE_EP):
            gate.train()
            for xb, yb_g in ld:
                opt.zero_grad(); crit(gate(xb), yb_g).backward(); opt.step()
        gates.append(gate)
    return gates


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


def make_learned_gate_fn(gates, device):
    @torch.no_grad()
    def fn(embs_np, asgn):
        result = np.zeros(len(embs_np), dtype=bool)
        for c in range(K):
            mask = (asgn == c)
            if not mask.any(): continue
            gate = gates[c]
            if gate is None: result[mask] = False; continue
            gate.eval()
            p = gate(torch.from_numpy(embs_np[mask]).to(device)).argmax(1).cpu().numpy()
            result[mask] = (p == 1)
        return result
    return fn


# ── Full FT baseline ───────────────────────────────────────────────────────────

def run_fullft(src_model, calib_ds, src_ld, tgt_ld, device, batch):
    model = copy.deepcopy(src_model)
    for p in model.parameters(): p.requires_grad_(True)
    opt  = torch.optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    ld   = make_loader(calib_ds, batch=min(batch, len(calib_ds)), shuffle=True)
    for _ in range(FT_EP):
        model.train()
        for xb, yb in ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
    return eval_f1(model, src_ld, device), eval_f1(model, tgt_ld, device)


# ── One scenario ───────────────────────────────────────────────────────────────

def run_scenario(tag, cfg, src_id, tgt_id, dataset_fn, domain_name_fn, device):
    batch    = cfg["batch"]
    src_name = domain_name_fn(src_id)
    tgt_name = domain_name_fn(tgt_id)
    scen_key = f"{src_name}→{tgt_name}"
    print(f"\n{'='*60}\n  [{tag}] {scen_key}  [5% labels, PatchTST]\n{'='*60}")

    src_model = tpst.train_source(src_id, cfg, dataset_fn, device)
    src_model.eval()
    for p in src_model.parameters(): p.requires_grad_(False)
    frozen_head = src_model.head

    src_tr_ds = dataset_fn(src_id, "train")
    src_te_ds = dataset_fn(src_id, "test")
    tgt_tr_ds = dataset_fn(tgt_id, "train")
    tgt_te_ds = dataset_fn(tgt_id, "test")

    src_tr_ld = make_loader(src_tr_ds, batch)
    src_te_ld = make_loader(src_te_ds, batch)
    tgt_te_ld = make_loader(tgt_te_ds, batch)

    calib_ds = calib_subset(tgt_tr_ds)
    calib_ld = make_loader(calib_ds, batch)
    print(f"  Calib: {len(calib_ds)} samples (5% of {len(tgt_tr_ds)} target train)")

    src_embs, _         = extract_embs(src_model, src_tr_ld, device)
    centroids, src_asgn = kmeans_numpy(src_embs, K)
    calib_embs, calib_lbls = extract_embs(src_model, calib_ld, device)
    calib_asgn = assign_concepts(calib_embs, centroids)

    print("  Training adapters …", flush=True)
    adapters = train_adapters(calib_embs, calib_lbls, centroids, frozen_head, device)
    print("  Training gates …", flush=True)
    gates    = train_gates(src_embs, src_asgn, calib_embs, calib_asgn, device)
    gate_fn  = make_learned_gate_fn(gates, device)

    def always_apply(embs_np, asgn): return np.ones(len(embs_np), dtype=bool)

    f1_src_before = eval_f1(src_model, src_te_ld, device)
    f1_tgt_nocorr = eval_f1(src_model, tgt_te_ld, device)
    print(f"  src_before={f1_src_before:.4f}  tgt_nocorr={f1_tgt_nocorr:.4f}")
    results = {"no_corr": (f1_src_before, f1_tgt_nocorr, 0.0)}

    def _record(key, sf, tf):
        fgt = f1_src_before - sf
        results[key] = (sf, tf, fgt)
        print(f"  [{key:<9}] src={sf:.4f}  tgt={tf:.4f}  forget={fgt:+.4f}", flush=True)

    print("  [full_ft  ] ", end="", flush=True)
    _record("full_ft", *run_fullft(src_model, calib_ds, src_te_ld, tgt_te_ld, device, batch))

    src_te_embs, src_te_lbls = extract_embs(src_model, src_te_ld, device)
    tgt_te_embs, tgt_te_lbls = extract_embs(src_model, tgt_te_ld, device)

    s_ng = eval_with_gate(src_te_embs, src_te_lbls, centroids, adapters, frozen_head, device, always_apply)
    t_ng = eval_with_gate(tgt_te_embs, tgt_te_lbls, centroids, adapters, frozen_head, device, always_apply)
    _record("pact_ng", s_ng, t_ng)

    s_lg = eval_with_gate(src_te_embs, src_te_lbls, centroids, adapters, frozen_head, device, gate_fn)
    t_lg = eval_with_gate(tgt_te_embs, tgt_te_lbls, centroids, adapters, frozen_head, device, gate_fn)
    _record("pact_lg", s_lg, t_lg)

    return scen_key, results


# ── Dataset sweep + figure ─────────────────────────────────────────────────────

def run_dataset(tag, cfg, scenarios, dataset_fn, domain_name_fn, device):
    print(f"\n{'#'*64}\n## {tag}  ({len(scenarios)} scenarios, PatchTST backbone)\n{'#'*64}")
    pop = {c: {"src":[], "tgt":[], "fgt":[]} for c in CONDITIONS}
    all_res = {}

    for src_id, tgt_id in scenarios:
        scen_key, res = run_scenario(tag, cfg, src_id, tgt_id, dataset_fn, domain_name_fn, device)
        all_res[scen_key] = res
        for c in CONDITIONS:
            s, t, f = res[c]
            pop[c]["src"].append(s); pop[c]["tgt"].append(t); pop[c]["fgt"].append(f)

    # Population summary
    print(f"\n{'='*64}")
    hdr = f"  {'Method':<22} {'Src F1':>14}  {'Tgt F1':>14}  {'Forgetting':>14}"
    sep = "  " + "-"*68
    print(hdr); print(sep)
    lines = [f"POPULATION SUMMARY — {tag} PatchTST backbone, 5% labels\n", hdr, sep]
    for c in CONDITIONS:
        sm = np.mean(pop[c]["src"]); ss = np.std(pop[c]["src"])
        tm = np.mean(pop[c]["tgt"]); ts = np.std(pop[c]["tgt"])
        fm = np.mean(pop[c]["fgt"]); fs = np.std(pop[c]["fgt"])
        row = f"  {COND_LABELS[c]:<22} {sm:.4f}±{ss:.4f}  {tm:.4f}±{ts:.4f}  {fm:+.4f}±{fs:.4f}"
        print(row); lines.append(row)
    (OUT_DIR / f"{tag.lower()}_summary.txt").write_text("\n".join(lines))

    # Per-scenario table
    scen_lines = [f"Per-scenario — {tag} PatchTST backbone, 5% labels\n"]
    for sk, res in all_res.items():
        scen_lines.append(f"\nScenario: {sk}")
        scen_lines.append(f"  {'Method':<22} {'Src F1':>8}  {'Tgt F1':>8}  {'Forget':>8}")
        for c in CONDITIONS:
            s, t, f = res[c]
            scen_lines.append(f"  {COND_LABELS[c]:<22} {s:>8.4f}  {t:>8.4f}  {f:>+8.4f}")
    (OUT_DIR / f"{tag.lower()}_per_scenario.txt").write_text("\n".join(scen_lines))

    # Figure — 3 panels: src retention, tgt F1, forgetting
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    x = np.arange(len(CONDITIONS))
    for ax, key, title in [
        (axes[0], "src", "Source F1 (retention)"),
        (axes[1], "tgt", "Target F1 (adaptation)"),
    ]:
        data  = pop
        means = [np.mean(data[c][key]) for c in CONDITIONS]
        stds  = [np.std(data[c][key])  for c in CONDITIONS]
        bars  = ax.bar(x, means, yerr=stds, capsize=5, color=PLOT_COLORS,
                       edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([COND_LABELS[c] for c in CONDITIONS], rotation=20, ha="right", fontsize=10)
        ax.set_ylim(0, 1.05); ax.set_ylabel("Macro F1")
        ax.set_title(title, fontsize=10, fontweight="bold")
        for bar, m in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                    f"{m:.3f}", ha="center", fontsize=9)
        ax.grid(axis="y", ls="--", alpha=0.4)

    ax = axes[2]
    fgt_means = [np.mean(pop[c]["fgt"]) for c in CONDITIONS]
    fgt_stds  = [np.std(pop[c]["fgt"])  for c in CONDITIONS]
    ax.bar(x, fgt_means, yerr=fgt_stds, capsize=5, color=PLOT_COLORS,
           edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[c] for c in CONDITIONS], rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Forgetting (src F1 drop)")
    ax.set_title("Catastrophic forgetting", fontsize=10, fontweight="bold")
    ax.grid(axis="y", ls="--", alpha=0.4)

    fig.suptitle(f"{tag} — PatchTST backbone  (5% labels, learned gate, {len(scenarios)} scenarios)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"{tag.lower()}_figure.{ext}", dpi=150, bbox_inches="tight",
                    facecolor="white")
    plt.close()
    print(f"  Outputs → {OUT_DIR}/{tag.lower()}_*.{{txt,png,pdf}}")
    return all_res, pop


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Backbone: PatchTST  (d_model=128, 3 layers, patch_len=16, stride=8)")
    print(f"Label budget: 5%  |  Gate: learned MLP  |  K={K} adapters\n")

    for tag, cfg, scenarios, dataset_fn, domain_name_fn in DATASETS:
        run_dataset(tag, cfg, scenarios, dataset_fn, domain_name_fn, device)

    print(f"\nAll done. Results → {OUT_DIR}")
