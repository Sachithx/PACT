"""
exp7_soft_gate.py — Soft-Mixture Gate over Per-Domain Adapters

Extends exp6 (binary learned gate per concept) to a soft multi-class routing
mechanism across all seen domains. For N+1 domains (source + N targets), each
concept c gets a single SoftGateMLP that outputs N+1 logits. At inference the
corrected embedding is a weighted average of all domain-specific adapter outputs.

    ê = Σ_d  α_d(e, c)  ·  A_{d,c}(e)
      α  = softmax( SoftGateMLP_c(e) / temperature )     shape [B, N+1]
      A_{0,c}(e) = e                                       source domain → identity
      A_{j,c}(e) = IdentityAdapter_{j,c}(e)  (j > 0)

Key properties:
  · Each A_{j,c} is frozen after domain j is trained — zero forgetting by construction.
  · α_d sums to 1 over d, so the correction is always a convex combination.
  · Hard routing (argmax) is the special case temperature → 0.
  · Binary gate (Exp 6) is the special case N=1.

Datasets: FD-A (source), FD-B, FD-C, FD-D (targets), K=16, 20% calib.

Conditions:
  1. no_adapt      — frozen source model, no adapters
  2. oracle        — correct adapter per sample (upper bound)
  3. dist_route    — nearest-domain-stats hard route (PACT / Seq-PACT-LoRA approach)
  4. hard_gate     — argmax of SoftGateMLP → single adapter bank
  5. soft_mix_1    — softmax blend, temperature=1.0
  6. soft_mix_05   — softmax blend, temperature=0.5  (sharper)
  7. binary_dist   — Exp6 binary gate decides apply/bypass,
                     distance routing picks which domain when apply=True

Run:
  LD_PRELOAD=/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6 \\
  /home/AD/sachith/.conda/envs/entrope/bin/python exp7_soft_gate.py
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent / "TSLANet" / "Classification"))

import importlib.util

def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

import argparse

tft = _load_mod("tft", Path(__file__).parent / "transfer_fd_tslanet.py")
tht = _load_mod("tht", Path(__file__).parent / "transfer_hhar_tslanet.py")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constants ─────────────────────────────────────────────────────────────────

REPO      = Path(__file__).parent
DATA_ROOT = REPO / "dataset"
CKPT_DIR  = REPO / "TSLANet" / "Classification" / "output" / "tslanet"

EMB_DIM    = 128
K          = 16
CALIB_FRAC = 0.20
SEED       = 42
GATE_EP    = 200
GATE_LR    = 1e-3
ADAPTER_EP = 200
ADAPTER_LR = 1e-3
PENALTY    = 1e-3
BATCH      = 256

FD_MAP = {0: "FD-A", 1: "FD-B", 2: "FD-C", 3: "FD-D"}


class Args:
    emb_dim=128; depth=2; patch_size=8; dropout_rate=0.15; masking_ratio=0.4
    ICB=True; ASB=True; adaptive_filter=True
    data_root=str(DATA_ROOT); ckpt_dir=str(CKPT_DIR)
    batch_size=256; pretrain=True; pretrain_epochs=40; pretrain_lr=1e-4
    cls_epochs=80; cls_lr=1e-4; retrain=False


# ── Dataset / model config ────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Exp 7: Soft-Mixture Gate")
    p.add_argument("--dataset", choices=["fd", "hhar"], default="fd")
    p.add_argument("--source",  type=int, default=None,
                   help="Source domain ID (FD: 0-3, HHAR: 0-8). Default: 0")
    p.add_argument("--targets", default=None,
                   help="Comma-separated target domain IDs. "
                        "Default FD: 1,2,3  HHAR: 1,2,3")
    return p.parse_args()


def build_config(args):
    """Return a dict with dataset-specific model/data accessors.

    Keys:
        load_model(device)   → frozen nn.Module
        get_train(domain_id) → Dataset for train split
        get_test(domain_id)  → Dataset for test  split
        src_id               → int (source domain)
        tgt_ids              → list[int] (target domains)
        all_ids              → [src_id] + tgt_ids
        domain_labels        → display name per domain, same order as all_ids
        batch_size           → int
        out_tag              → string prefix for output files
    """
    if args.dataset == "fd":
        src_id  = args.source  if args.source  is not None else 0
        tgt_ids = ([int(x) for x in args.targets.split(",")]
                   if args.targets else [1, 2, 3])
        ckpt = CKPT_DIR / f"tslanet_{FD_MAP[src_id]}_ps8_d2_dim128.pt"

        def load_model(device):
            m    = tft._make_model(Args()).to(device)
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            m.load_state_dict(state["model_state_dict"])
            m.eval()
            for p in m.parameters(): p.requires_grad_(False)
            return m

        get_train = lambda did: tft.FDDataset(FD_MAP[did], "train", DATA_ROOT)
        get_test  = lambda did: tft.FDDataset(FD_MAP[did], "test",  DATA_ROOT)
        labels    = [FD_MAP[i] for i in [src_id] + tgt_ids]
        batch_sz  = 256

    elif args.dataset == "hhar":
        src_id  = args.source  if args.source  is not None else 0
        tgt_ids = ([int(x) for x in args.targets.split(",")]
                   if args.targets else [1, 2, 3])

        def load_model(device):
            m = tht.train_source(src_id, Args(), device)
            m.eval()
            for p in m.parameters(): p.requires_grad_(False)
            return m

        get_train = lambda did: tht.HHARDataset(did, "train", DATA_ROOT)
        get_test  = lambda did: tht.HHARDataset(did, "test",  DATA_ROOT)
        labels    = [f"User-{i}" for i in [src_id] + tgt_ids]
        batch_sz  = 256

    else:
        raise ValueError(args.dataset)

    all_ids = [src_id] + tgt_ids
    tag     = f"{args.dataset}_src{src_id}_tgt{'_'.join(str(i) for i in tgt_ids)}"
    return dict(
        load_model=load_model,
        get_train=get_train,
        get_test=get_test,
        src_id=src_id,
        tgt_ids=tgt_ids,
        all_ids=all_ids,
        domain_labels=labels,
        batch_size=batch_sz,
        out_tag=tag,
        dataset=args.dataset,
    )


# ── k-means / concept helpers ─────────────────────────────────────────────────

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


# ── Modules ───────────────────────────────────────────────────────────────────

class IdentityAdapter(nn.Module):
    """Full-rank linear adapter, identity-initialized. Identical to Exp 6."""
    def __init__(self, dim=EMB_DIM):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


class BinaryGateMLP(nn.Module):
    """Per-concept binary classifier: 0 = bypass (source-like), 1 = apply adapter."""
    def __init__(self, in_dim=EMB_DIM, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(hidden, 2),
        )

    def forward(self, x): return self.net(x)


class SoftGateMLP(nn.Module):
    """Per-concept multi-class gate: one logit per domain (source + N targets).

    The same module is used for both hard routing (argmax) and soft mixing
    (softmax with temperature). Architecture kept identical to BinaryGateMLP
    so the only difference is the output width.
    """
    def __init__(self, in_dim=EMB_DIM, hidden=64, n_domains=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(hidden, n_domains),
        )

    def forward(self, x): return self.net(x)


# ── Embedding extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_embs(model, loader, device):
    model.eval()
    embs, lbls = [], []
    for xb, yb in loader:
        embs.append(model.extract(xb.to(device)).cpu().numpy())
        lbls.append(yb.numpy())
    return np.concatenate(embs).astype(np.float32), np.concatenate(lbls)


# ── Adapter training ──────────────────────────────────────────────────────────

def train_adapters(calib_embs, calib_lbls, centroids, frozen_head, device):
    """Train K IdentityAdapters for one domain's calibration data."""
    adapters = nn.ModuleList([IdentityAdapter() for _ in range(K)]).to(device)
    opt  = torch.optim.AdamW(adapters.parameters(), lr=ADAPTER_LR)
    crit = nn.CrossEntropyLoss()
    asgn = assign_concepts(calib_embs, centroids)
    eye  = torch.eye(EMB_DIM, device=device)

    et = torch.from_numpy(calib_embs).to(device)
    lt = torch.from_numpy(calib_lbls).to(device)
    at = torch.from_numpy(asgn.astype(np.int64))
    ld = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(et, lt, at),
        batch_size=BATCH, shuffle=True)

    for _ in range(ADAPTER_EP):
        for eb, yb, cb in ld:
            adapters.train(); opt.zero_grad()
            corr = eb.clone()
            for c in range(K):
                m = (cb == c)
                if m.any(): corr[m] = adapters[c](eb[m])
            loss = crit(frozen_head(corr), yb)
            for a in adapters:
                loss = loss + PENALTY * (torch.sum((a.linear.weight - eye) ** 2)
                                          + torch.sum(a.linear.bias ** 2))
            loss.backward(); opt.step()
    return adapters


# ── Gate training ─────────────────────────────────────────────────────────────

def train_binary_gates(src_embs, src_asgn, calib_per_domain, device):
    """Binary gate: source = class 0, all targets pooled = class 1.

    calib_per_domain: list of (embs, asgn, lbls), one per target domain.
    Returns: list of K BinaryGateMLP (or None when no calibration in concept c).
    """
    gates = []
    for c in range(K):
        neg = src_embs[src_asgn == c]
        pos_parts = [embs[asgn == c]
                     for embs, asgn, _ in calib_per_domain
                     if (asgn == c).any()]
        pos = np.concatenate(pos_parts) if pos_parts else np.empty((0, EMB_DIM), np.float32)

        if len(pos) == 0:
            gates.append(None); continue

        # Balance classes by oversampling the minority
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

        gate = BinaryGateMLP().to(device)
        opt  = torch.optim.AdamW(gate.parameters(), lr=GATE_LR, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        for _ in range(GATE_EP):
            gate.train()
            for xb, yb in ld:
                opt.zero_grad(); crit(gate(xb), yb).backward(); opt.step()
        gates.append(gate)
    return gates


def train_soft_gates(src_embs, src_asgn, calib_per_domain, device):
    """Multi-class gate: logit per domain (0 = source, 1..N = targets).

    calib_per_domain: list of (embs, asgn, lbls), one per target domain.
    Each class is balanced by oversampling to the size of the largest class.
    Returns: list of K SoftGateMLP (or None when no calibration in concept c).
    """
    n_domains = 1 + len(calib_per_domain)
    soft_gates = []

    for c in range(K):
        # Collect per-class data for concept c
        X_by_class = {0: src_embs[src_asgn == c]}
        for j, (embs, asgn, _) in enumerate(calib_per_domain):
            mask = (asgn == c)
            X_by_class[j + 1] = embs[mask]

        has_target = any(len(v) > 0 for k, v in X_by_class.items() if k > 0)
        if not has_target:
            soft_gates.append(None); continue

        # Balance: oversample each non-empty class to max_cls
        max_cls = max(len(v) for v in X_by_class.values() if len(v) > 0)
        rng = np.random.default_rng(SEED + c + 200)
        X_parts, y_parts = [], []
        for cls_idx, X_cls in X_by_class.items():
            if len(X_cls) == 0:
                continue
            if len(X_cls) < max_cls:
                extra = rng.choice(len(X_cls), max_cls - len(X_cls), replace=True)
                X_cls = np.vstack([X_cls, X_cls[extra]])
            X_parts.append(X_cls)
            y_parts.append(np.full(len(X_cls), cls_idx, dtype=np.int64))

        X = np.concatenate(X_parts).astype(np.float32)
        y = np.concatenate(y_parts)
        perm = rng.permutation(len(X)); X, y = X[perm], y[perm]

        Xt = torch.from_numpy(X).to(device)
        yt = torch.from_numpy(y).to(device)
        ld = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xt, yt),
            batch_size=min(BATCH, len(X)), shuffle=True)

        gate = SoftGateMLP(n_domains=n_domains).to(device)
        opt  = torch.optim.AdamW(gate.parameters(), lr=GATE_LR, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        for _ in range(GATE_EP):
            gate.train()
            for xb, yb in ld:
                opt.zero_grad(); crit(gate(xb), yb).backward(); opt.step()
        soft_gates.append(gate)
    return soft_gates


# ── Distance routing stats ────────────────────────────────────────────────────

def build_domain_stats(embs, asgn, centroids, K, pct=95.0):
    """Per-concept mean and percentile radius for distance-based routing."""
    stats = {}
    for c in range(K):
        mask = (asgn == c)
        if mask.any():
            mu = embs[mask].mean(0)
            dists = np.linalg.norm(embs[mask] - mu, axis=1)
            stats[c] = {"mu": mu, "radius": float(np.percentile(dists, pct)),
                        "available": True}
        else:
            stats[c] = {"available": False}
    return stats


# ── Inference forward passes ──────────────────────────────────────────────────

@torch.no_grad()
def eval_oracle(embs_np, labels_np, centroids, adapters_per_domain,
                true_domain_idx, frozen_head, device):
    """Apply the correct domain's adapter bank for every sample.

    true_domain_idx = 0 → source, bypass (identity).
    true_domain_idx = j > 0 → use adapters_per_domain[j-1].
    """
    asgn   = assign_concepts(embs_np, centroids)
    embs_t = torch.from_numpy(embs_np).to(device)
    corr   = embs_t.clone()

    if true_domain_idx > 0:
        adapters = adapters_per_domain[true_domain_idx - 1]
        for c in range(K):
            mask = (asgn == c)
            if mask.any():
                adapters[c].eval()
                corr[mask] = adapters[c](embs_t[mask])

    preds = frozen_head(corr).argmax(1).cpu().numpy()
    return f1_score(labels_np, preds, average="macro", zero_division=0)


@torch.no_grad()
def eval_dist_route(embs_np, labels_np, centroids, adapters_per_domain,
                    domain_stats_list, frozen_head, device,
                    source_bias=0.95, eps=1e-6):
    """Hard nearest-domain routing via per-concept distance statistics.

    domain_stats_list[0] = source stats, [j] = target j's stats.
    Picks argmin normalised distance; domain 0 → bypass.
    """
    n_domains = len(domain_stats_list)
    asgn      = assign_concepts(embs_np, centroids)
    scores    = np.full((n_domains, len(embs_np)), np.inf)

    for j, stats_j in enumerate(domain_stats_list):
        for c in range(K):
            sk = stats_j.get(c, {"available": False})
            if not sk["available"]: continue
            mask = (asgn == c)
            if not mask.any(): continue
            dists = np.linalg.norm(embs_np[mask] - sk["mu"], axis=1)
            scores[j, mask] = dists / (sk["radius"] + eps)
    scores[0] *= source_bias
    routes = scores.argmin(0)

    embs_t = torch.from_numpy(embs_np).to(device)
    corr   = embs_t.clone()
    for j in range(1, n_domains):
        adapters = adapters_per_domain[j - 1]
        for c in range(K):
            mask = (routes == j) & (asgn == c)
            if mask.any():
                adapters[c].eval()
                corr[mask] = adapters[c](embs_t[mask])

    preds = frozen_head(corr).argmax(1).cpu().numpy()
    return f1_score(labels_np, preds, average="macro", zero_division=0)


@torch.no_grad()
def eval_hard_gate(embs_np, labels_np, centroids, soft_gates,
                   adapters_per_domain, frozen_head, device):
    """Hard routing: argmax of SoftGateMLP selects exactly one adapter bank.

    Gate output 0 → bypass (source); j > 0 → adapters_per_domain[j-1].
    """
    asgn   = assign_concepts(embs_np, centroids)
    embs_t = torch.from_numpy(embs_np).to(device)
    corr   = embs_t.clone()

    for c in range(K):
        mask = (asgn == c)
        if not mask.any(): continue
        gate = soft_gates[c]
        if gate is None: continue   # no calibration → bypass

        gate.eval()
        batch  = embs_t[mask]
        routes = gate(batch).argmax(1).cpu().numpy()   # [B] ∈ {0..n_domains-1}
        global_idx = np.where(mask)[0]

        for j in range(1, len(adapters_per_domain) + 1):
            sel = (routes == j)
            if not sel.any(): continue
            adapters_per_domain[j - 1][c].eval()
            corr[global_idx[sel]] = adapters_per_domain[j - 1][c](batch[sel])

    preds = frozen_head(corr).argmax(1).cpu().numpy()
    return f1_score(labels_np, preds, average="macro", zero_division=0)


@torch.no_grad()
def eval_soft_mixture(embs_np, labels_np, centroids, soft_gates,
                      adapters_per_domain, frozen_head, device, temperature=1.0):
    """Soft-mixture forward pass.

        ê_i = Σ_d  α_d(e_i, c_i)  ·  A_{d, c_i}(e_i)

    where α = softmax(gate(e) / temperature), A_{0,c} = I (source identity).

    temperature > 1 → softer (more uniform mixing).
    temperature < 1 → sharper (approaches hard argmax).
    """
    asgn   = assign_concepts(embs_np, centroids)
    embs_t = torch.from_numpy(embs_np).to(device)
    corr   = torch.zeros_like(embs_t)

    for c in range(K):
        mask = (asgn == c)
        if not mask.any(): continue
        gate  = soft_gates[c]
        batch = embs_t[mask]   # [B, D]

        if gate is None:
            corr[mask] = batch  # bypass
            continue

        gate.eval()
        logits  = gate(batch)                                         # [B, n_domains]
        weights = torch.softmax(logits / temperature, dim=1)         # [B, n_domains]

        # Source contribution: α_0 · identity(e) = α_0 · e
        mixed = weights[:, 0:1] * batch                              # [B, D]

        # Target contributions: α_j · A_{j,c}(e)
        for j, domain_adapters in enumerate(adapters_per_domain):
            domain_adapters[c].eval()
            adapted = domain_adapters[c](batch)                       # [B, D]
            mixed   = mixed + weights[:, j + 1 : j + 2] * adapted

        corr[mask] = mixed

    preds = frozen_head(corr).argmax(1).cpu().numpy()
    return f1_score(labels_np, preds, average="macro", zero_division=0)


@torch.no_grad()
def eval_binary_dist(embs_np, labels_np, centroids, binary_gates,
                     adapters_per_domain, domain_stats_list,
                     frozen_head, device, source_bias=0.95, eps=1e-6):
    """Binary gate (Exp 6) decides apply/bypass; distance routing picks domain.

    Two-stage:
      1. BinaryGateMLP_c(e) ∈ {bypass=0, apply=1}
      2. For apply=1 samples: argmin distance to target-domain stats selects adapter.

    This extends Exp 6 to multi-domain without changing the binary gate.
    """
    asgn   = assign_concepts(embs_np, centroids)
    embs_t = torch.from_numpy(embs_np).to(device)
    corr   = embs_t.clone()

    # Stage 1: binary gate decision
    apply = np.zeros(len(embs_np), dtype=bool)
    for c in range(K):
        mask = (asgn == c)
        if not mask.any(): continue
        gate = binary_gates[c]
        if gate is None: continue
        gate.eval()
        p = gate(embs_t[mask]).argmax(1).cpu().numpy()
        apply[mask] = (p == 1)

    if not apply.any():
        preds = frozen_head(corr).argmax(1).cpu().numpy()
        return f1_score(labels_np, preds, average="macro", zero_division=0)

    # Stage 2: route apply-flagged samples to nearest target domain
    n_targets    = len(adapters_per_domain)
    apply_embs   = embs_np[apply]
    apply_asgn   = asgn[apply]
    apply_embs_t = embs_t[apply]

    scores = np.full((n_targets, len(apply_embs)), np.inf)
    for j_tgt in range(n_targets):
        stats_j = domain_stats_list[j_tgt + 1]   # skip source at index 0
        for c in range(K):
            sk = stats_j.get(c, {"available": False})
            if not sk["available"]: continue
            mask_c = (apply_asgn == c)
            if not mask_c.any(): continue
            dists = np.linalg.norm(apply_embs[mask_c] - sk["mu"], axis=1)
            scores[j_tgt, mask_c] = dists / (sk["radius"] + eps)

    target_routes = scores.argmin(0)   # [n_apply] index into adapters_per_domain
    corr_apply    = apply_embs_t.clone()

    for j_tgt in range(n_targets):
        sel = (target_routes == j_tgt)
        if not sel.any(): continue
        for c in range(K):
            mask_c = sel & (apply_asgn == c)
            if not mask_c.any(): continue
            adapters_per_domain[j_tgt][c].eval()
            local_idx = np.where(mask_c)[0]
            corr_apply[local_idx] = adapters_per_domain[j_tgt][c](
                apply_embs_t[local_idx])

    corr[apply] = corr_apply
    preds = frozen_head(corr).argmax(1).cpu().numpy()
    return f1_score(labels_np, preds, average="macro", zero_division=0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = build_config(args)

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_id        = cfg["src_id"]
    tgt_ids       = cfg["tgt_ids"]
    all_ids       = cfg["all_ids"]
    domain_labels = cfg["domain_labels"]   # display names, same order as all_ids
    batch_sz      = cfg["batch_size"]
    out_tag       = cfg["out_tag"]

    out_dir = REPO / "results" / "exp7_soft_gate"
    out_dir.mkdir(parents=True, exist_ok=True)

    src_label     = domain_labels[0]
    tgt_labels    = domain_labels[1:]
    n_domains     = len(all_ids)

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}  source={src_label}  "
          f"targets={tgt_labels}")
    print(f"K={K}  calib={CALIB_FRAC*100:.0f}%  "
          f"adapter_ep={ADAPTER_EP}  gate_ep={GATE_EP}")

    # ── Load frozen source model ─────────────────────────────────────────────
    print(f"Loading frozen model for {src_label} ...")
    model       = cfg["load_model"](device)
    frozen_head = model.head

    # ── Source embeddings + concept space ────────────────────────────────────
    print(f"Extracting {src_label} train embeddings ...")
    src_ld   = torch.utils.data.DataLoader(
        cfg["get_train"](src_id), batch_size=batch_sz)
    src_embs, _ = extract_embs(model, src_ld, device)

    print(f"K-means (K={K}) ...")
    centroids, src_asgn = kmeans_numpy(src_embs, K)
    src_stats = build_domain_stats(src_embs, src_asgn, centroids, K)

    # ── Test embeddings (extracted once, cached by label) ────────────────────
    print("Extracting test embeddings ...")
    test_embs, test_lbls = {}, {}
    for did, lbl in zip(all_ids, domain_labels):
        ld = torch.utils.data.DataLoader(
            cfg["get_test"](did), batch_size=batch_sz)
        test_embs[lbl], test_lbls[lbl] = extract_embs(model, ld, device)

    # ── Calibration splits + routing stats for each target domain ────────────
    print(f"Building {CALIB_FRAC*100:.0f}% calibration splits ...")
    calib_per_domain  = []           # (embs, asgn, lbls) per target domain
    domain_stats_list = [src_stats]  # index 0 = source

    for did, lbl in zip(tgt_ids, tgt_labels):
        ds    = cfg["get_train"](did)
        y_all = ds.y.numpy()
        sss   = StratifiedShuffleSplit(n_splits=1, test_size=1 - CALIB_FRAC,
                                       random_state=SEED)
        idx, _ = next(sss.split(np.zeros(len(y_all)), y_all))
        calib_ld = torch.utils.data.DataLoader(
            torch.utils.data.Subset(ds, idx), batch_size=batch_sz)
        embs, _  = extract_embs(model, calib_ld, device)
        lbls_c   = y_all[idx]
        asgn     = assign_concepts(embs, centroids)
        calib_per_domain.append((embs, asgn, lbls_c))
        domain_stats_list.append(build_domain_stats(embs, asgn, centroids, K))
        print(f"  {lbl}: {len(embs)} calibration samples")

    # ── Per-domain adapter banks ─────────────────────────────────────────────
    print("Training per-domain adapters ...")
    adapters_per_domain = []
    for i, lbl in enumerate(tgt_labels):
        embs, asgn, lbls_c = calib_per_domain[i]
        print(f"  {lbl} ...", flush=True)
        adapters_per_domain.append(
            train_adapters(embs, lbls_c, centroids, frozen_head, device))

    # ── Gate training ────────────────────────────────────────────────────────
    print("Training binary gate (source vs all targets pooled) ...")
    binary_gates = train_binary_gates(src_embs, src_asgn, calib_per_domain, device)
    print(f"  {K - sum(g is None for g in binary_gates)}/{K} concepts have a gate")

    print(f"Training soft {n_domains}-way gate ({'/'.join(domain_labels)}) ...")
    soft_gates = train_soft_gates(src_embs, src_asgn, calib_per_domain, device)
    print(f"  {K - sum(g is None for g in soft_gates)}/{K} concepts have a gate")

    # ── Evaluation ───────────────────────────────────────────────────────────
    print("\nEvaluation:")
    results = {}

    for d_idx, (did, lbl) in enumerate(zip(all_ids, domain_labels)):
        embs = test_embs[lbl]
        lbls = test_lbls[lbl]

        with torch.no_grad():
            no_adapt = f1_score(
                lbls,
                frozen_head(torch.from_numpy(embs).to(device)).argmax(1).cpu().numpy(),
                average="macro", zero_division=0)

        oracle   = eval_oracle(embs, lbls, centroids, adapters_per_domain,
                               d_idx, frozen_head, device)
        dist     = eval_dist_route(embs, lbls, centroids, adapters_per_domain,
                                   domain_stats_list, frozen_head, device)
        hard     = eval_hard_gate(embs, lbls, centroids, soft_gates,
                                  adapters_per_domain, frozen_head, device)
        soft_1   = eval_soft_mixture(embs, lbls, centroids, soft_gates,
                                     adapters_per_domain, frozen_head, device,
                                     temperature=1.0)
        soft_05  = eval_soft_mixture(embs, lbls, centroids, soft_gates,
                                     adapters_per_domain, frozen_head, device,
                                     temperature=0.5)
        bin_dist = eval_binary_dist(embs, lbls, centroids, binary_gates,
                                    adapters_per_domain, domain_stats_list,
                                    frozen_head, device)

        results[lbl] = dict(
            no_adapt=no_adapt, oracle=oracle, dist=dist,
            hard=hard, soft_1=soft_1, soft_05=soft_05, bin_dist=bin_dist)

        print(f"  {lbl:<10}  no_adapt={no_adapt:.4f}  oracle={oracle:.4f}  "
              f"dist={dist:.4f}  hard={hard:.4f}  "
              f"soft(t=1)={soft_1:.4f}  soft(t=0.5)={soft_05:.4f}  "
              f"bin_dist={bin_dist:.4f}")

    # ── Summary table ────────────────────────────────────────────────────────
    CONDITIONS = [
        ("no_adapt", "No correction"),
        ("oracle",   "Oracle routing"),
        ("dist",     "Dist routing"),
        ("hard",     "Hard gate (argmax)"),
        ("soft_1",   "Soft mixture t=1.0"),
        ("soft_05",  "Soft mixture t=0.5"),
        ("bin_dist", "Binary+dist (Exp6)"),
    ]

    print(f"\n{'='*72}")
    print(f"SUMMARY — {args.dataset.upper()}  src={src_label}  "
          f"tgt={tgt_labels}  K={K}  calib={CALIB_FRAC*100:.0f}%")
    print(f"{'='*72}")
    hdr = (f"  {'Condition':<22} {src_label:>9}"
           + "".join(f"  {d:>9}" for d in tgt_labels)
           + f"  {'avg_tgt':>8}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))

    lines = [f"Exp 7 — Soft-Mixture Gate  {args.dataset.upper()} "
             f"src={src_label}  tgt={tgt_labels}\n", hdr,
             "  " + "-" * (len(hdr) - 2)]
    for key, name in CONDITIONS:
        src_f1  = results[src_label][key]
        tgt_f1s = [results[d][key] for d in tgt_labels]
        avg_tgt = np.mean(tgt_f1s)
        row = (f"  {name:<22} {src_f1:>9.4f}"
               + "".join(f"  {v:>9.4f}" for v in tgt_f1s)
               + f"  {avg_tgt:>8.4f}")
        print(row); lines.append(row)

    (out_dir / f"{out_tag}_results.txt").write_text("\n".join(lines))

    # ── Bar chart ────────────────────────────────────────────────────────────
    short_names = ["No corr", "Oracle", "Dist", "Hard", "Soft t=1", "Soft t=0.5", "Bin+Dist"]
    keys        = [k for k, _ in CONDITIONS]
    colors      = ["#aec6cf", "#2ecc71", "#f39c12", "#e74c3c",
                   "#8e44ad", "#6c3483", "#3498db"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(CONDITIONS))

    for ax, metric_key, title in [
            (axes[0], src_label, f"Source F1 ({src_label} retention)"),
            (axes[1], "avg",     f"Target F1 (avg {'/'.join(tgt_labels)})"),
    ]:
        vals = ([results[src_label][k] for k in keys] if metric_key == src_label
                else [np.mean([results[d][k] for d in tgt_labels]) for k in keys])
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Macro F1"); ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", ls="--", alpha=0.4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        f"Exp 7: Soft-Mixture Gate — {args.dataset.upper()}\n"
        f"src={src_label}  tgt={tgt_labels}  K={K}  calib={CALIB_FRAC*100:.0f}%",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"{out_tag}_comparison.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close()

    # ── Temperature sensitivity ───────────────────────────────────────────────
    temps = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    temp_f1 = {}
    print("\nTemperature sweep (avg target F1) ...")
    for t in temps:
        avg = float(np.mean([
            eval_soft_mixture(test_embs[d], test_lbls[d], centroids, soft_gates,
                              adapters_per_domain, frozen_head, device, temperature=t)
            for d in tgt_labels]))
        temp_f1[t] = avg
        print(f"  t={t:.2f}  avg_tgt={avg:.4f}")

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.semilogx(temps, [temp_f1[t] for t in temps], "o-", color="#8e44ad", lw=2,
                 label="Soft mixture")
    ax2.axhline(np.mean([results[d]["hard"]     for d in tgt_labels]),
                ls="--", color="#e74c3c", label="Hard gate (argmax)")
    ax2.axhline(np.mean([results[d]["dist"]     for d in tgt_labels]),
                ls="--", color="#f39c12", label="Dist routing")
    ax2.axhline(np.mean([results[d]["no_adapt"] for d in tgt_labels]),
                ls=":",  color="gray",    label="No correction")
    ax2.set_xlabel("Softmax temperature"); ax2.set_ylabel("Avg target Macro F1")
    ax2.set_title(f"Temperature sensitivity — {args.dataset.upper()}\n"
                  f"(avg {'/'.join(tgt_labels)})", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(ls="--", alpha=0.4)
    fig2.tight_layout()
    for ext in ["png", "pdf"]:
        fig2.savefig(out_dir / f"{out_tag}_temperature.{ext}",
                     dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nResults → {out_dir}/{out_tag}_results.txt")
    print(f"Figures → {out_dir}/{out_tag}_comparison.{{png,pdf}}")
    print(f"         {out_dir}/{out_tag}_temperature.{{png,pdf}}")


if __name__ == "__main__":
    main()
