"""PACT: per-concept adapters + soft-mixture gate over a frozen backbone.

This is the core of the method, consolidated from the original
exp7_soft_gate.py. Everything operates on the frozen backbone's pre-head
embeddings, so the host classifier is never modified (zero host-level
forgetting by construction).

Pipeline:
    1. K-means concepts on the source embeddings (see models.concepts).
    2. Per-domain, per-concept adapter banks trained on a small labelled
       calibration slice (train_adapters).
    3. A per-concept soft gate that mixes all domains' adapter outputs
       (train_soft_gates + eval_soft_mixture).

Config is passed via a PactConfig dataclass so nothing relies on module
globals (the original used mutable module-level constants).
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from ..models.adapters import make_adapter_bank
from ..models.concepts import assign_concepts
from ..models.gate import BinaryGateMLP, SoftGateMLP


@dataclass
class PactConfig:
    k: int = 16
    emb_dim: int = 128
    calib_frac: float = 0.05
    calib_n: int | None = None      # fixed label budget; overrides calib_frac when set
    adapter_ep: int = 200
    adapter_lr: float = 1e-3
    gate_ep: int = 200
    gate_lr: float = 1e-3
    penalty: float = 1e-3
    batch: int = 256
    seed: int = 42
    adapter_kind: str = "lora"       # "lora" (default rank-8) | "identity" (full affine)
    lora_rank: int = 8
    kmeans_n_init: int = 10
    min_chart_domain_n: int = 5    # omit chart-domain gate output below this support
    gate_kind: str = "learned"       # "learned" (MLP, rehearsal) | "prototype" |
                                      # "prototype_diag" | "prototype_lowrank" |
                                      # "prototype_masked" | "prototype_mixture" |
                                      # "analytic" | "confidence" (all rehearsal-free)
    analytic_n_features: int = 1536  # random-feature width for gate_kind="analytic"
    analytic_lam_rel: float = 1e-2   # relative ridge penalty for gate_kind="analytic"


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

def train_adapters(calib_embs, calib_lbls, centroids, frozen_head, device, cfg: PactConfig):
    """Train K adapters for one domain's calibration data."""
    adapters = make_adapter_bank(cfg.k, cfg.emb_dim, cfg.adapter_kind, cfg.lora_rank).to(device)
    opt = torch.optim.AdamW(adapters.parameters(), lr=cfg.adapter_lr)
    crit = nn.CrossEntropyLoss()
    asgn = assign_concepts(calib_embs, centroids)

    et = torch.from_numpy(calib_embs).to(device)
    lt = torch.from_numpy(calib_lbls).to(device)
    at = torch.from_numpy(asgn.astype(np.int64))
    ld = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(et, lt, at),
        batch_size=cfg.batch, shuffle=True)

    for _ in range(cfg.adapter_ep):
        for eb, yb, cb in ld:
            adapters.train(); opt.zero_grad()
            corr = eb.clone()
            for c in range(cfg.k):
                m = (cb == c)
                if m.any():
                    corr[m] = adapters[c](eb[m])
            loss = crit(frozen_head(corr), yb)
            for a in adapters:
                loss = loss + cfg.penalty * a.identity_penalty(device)
            loss.backward(); opt.step()
    return adapters


# ── Gate training ─────────────────────────────────────────────────────────────

def _calib_ea(entry):
    """Normalize a calib buffer entry to (embs, asgn). Labels are never required."""
    return entry[0], entry[1]


def train_binary_gates(src_embs, src_asgn, calib_per_domain, device, cfg: PactConfig):
    """Binary gate: source = class 0, all targets pooled = class 1."""
    gates = []
    for c in range(cfg.k):
        neg = src_embs[src_asgn == c]
        pos_parts = [embs[asgn == c] for embs, asgn in (_calib_ea(e) for e in calib_per_domain)
                     if (asgn == c).any()]
        pos = np.concatenate(pos_parts) if pos_parts else np.empty((0, cfg.emb_dim), np.float32)
        if len(pos) == 0:
            gates.append(None); continue

        rng = np.random.default_rng(cfg.seed + c)
        if len(neg) > len(pos):
            pos = pos[rng.choice(len(pos), len(neg), replace=True)]
        elif len(pos) > len(neg):
            neg = neg[rng.choice(len(neg), len(pos), replace=True)]

        X = np.concatenate([neg, pos])
        y = np.concatenate([np.zeros(len(neg), np.int64), np.ones(len(pos), np.int64)])
        perm = np.random.default_rng(cfg.seed + c + 100).permutation(len(X))
        X, y = X[perm], y[perm]

        Xt = torch.from_numpy(X).to(device)
        yt = torch.from_numpy(y).to(device)
        ld = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xt, yt),
            batch_size=min(cfg.batch, len(X)), shuffle=True)

        gate = BinaryGateMLP(cfg.emb_dim).to(device)
        opt = torch.optim.AdamW(gate.parameters(), lr=cfg.gate_lr, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        for _ in range(cfg.gate_ep):
            gate.train()
            for xb, yb in ld:
                opt.zero_grad(); crit(gate(xb), yb).backward(); opt.step()
        gates.append(gate)
    return gates


def train_soft_gates(src_embs, src_asgn, calib_per_domain, device, cfg: PactConfig):
    """Multi-class gate: logit per domain (0 = source, 1..N = targets).

    calib_per_domain: list of (embs, asgn) — embedding keys only, no class labels.
    Each class is balanced by oversampling to the largest class.
    Chart-domain pairs with fewer than cfg.min_chart_domain_n calibration samples
    are omitted from gate training; at inference their logits are masked to
    -inf so routing falls back to available branches.
    Returns a list of K SoftGateMLP (None where a chart has no target support).
    """
    n_domains = 1 + len(calib_per_domain)
    soft_gates = []
    for c in range(cfg.k):
        X_by_class = {0: src_embs[src_asgn == c]}
        active_targets = []
        for j, entry in enumerate(calib_per_domain):
            embs, asgn = _calib_ea(entry)
            X_cls = embs[asgn == c]
            if len(X_cls) >= cfg.min_chart_domain_n:
                active_targets.append(j + 1)
                X_by_class[j + 1] = X_cls

        if not active_targets:
            soft_gates.append(None)
            continue

        max_cls = max(len(v) for v in X_by_class.values() if len(v) > 0)
        rng = np.random.default_rng(cfg.seed + c + 200)
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
        perm = rng.permutation(len(X))
        X, y = X[perm], y[perm]

        Xt = torch.from_numpy(X).to(device)
        yt = torch.from_numpy(y).to(device)
        ld = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xt, yt),
            batch_size=min(cfg.batch, len(X)), shuffle=True)

        gate = SoftGateMLP(cfg.emb_dim, n_domains=n_domains).to(device)
        if len(active_targets) < n_domains - 1:
            gate.register_buffer(
                "_domain_mask",
                torch.tensor(
                    [True] + [d in active_targets for d in range(1, n_domains)],
                    dtype=torch.bool,
                ),
            )
        opt = torch.optim.AdamW(gate.parameters(), lr=cfg.gate_lr, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        for _ in range(cfg.gate_ep):
            gate.train()
            for xb, yb in ld:
                opt.zero_grad()
                crit(gate(xb), yb).backward()
                opt.step()
        soft_gates.append(gate)
    return soft_gates


# ── Inference forward passes ──────────────────────────────────────────────────

@torch.no_grad()
def eval_oracle(embs_np, labels_np, centroids, adapters_per_domain,
                true_domain_idx, frozen_head, device, k):
    """Apply the correct domain's adapter bank for every sample (upper bound)."""
    asgn = assign_concepts(embs_np, centroids)
    embs_t = torch.from_numpy(embs_np).to(device)
    corr = embs_t.clone()
    if true_domain_idx > 0:
        adapters = adapters_per_domain[true_domain_idx - 1]
        for c in range(k):
            mask = (asgn == c)
            if mask.any():
                adapters[c].eval()
                corr[mask] = adapters[c](embs_t[mask])
    preds = frozen_head(corr).argmax(1).cpu().numpy()
    return f1_score(labels_np, preds, average="macro", zero_division=0)


@torch.no_grad()
def eval_hard_gate(embs_np, labels_np, centroids, soft_gates,
                   adapters_per_domain, frozen_head, device, k):
    """Hard routing: argmax of the soft gate selects exactly one adapter bank."""
    asgn = assign_concepts(embs_np, centroids)
    embs_t = torch.from_numpy(embs_np).to(device)
    corr = embs_t.clone()
    for c in range(k):
        mask = (asgn == c)
        if not mask.any():
            continue
        gate = soft_gates[c]
        if gate is None:
            continue
        gate.eval()
        batch = embs_t[mask]
        routes = gate(batch).argmax(1).cpu().numpy()
        global_idx = np.where(mask)[0]
        for j in range(1, len(adapters_per_domain) + 1):
            sel = (routes == j)
            if not sel.any():
                continue
            adapters_per_domain[j - 1][c].eval()
            corr[global_idx[sel]] = adapters_per_domain[j - 1][c](batch[sel])
    preds = frozen_head(corr).argmax(1).cpu().numpy()
    return f1_score(labels_np, preds, average="macro", zero_division=0)


@torch.no_grad()
@torch.no_grad()
def transport_embeddings(embs_np, centroids, soft_gates, adapters_per_domain,
                         device, k, temperature=0.5):
    """Same soft-mixture forward pass as eval_soft_mixture, but returns the
    transported embeddings themselves (numpy) instead of scoring them --
    for representation-space visualization (t-SNE/UMAP/PCA figures)."""
    asgn = assign_concepts(embs_np, centroids)
    embs_t = torch.from_numpy(embs_np).to(device)
    corr = torch.zeros_like(embs_t)
    for c in range(k):
        mask = (asgn == c)
        if not mask.any():
            continue
        gate = soft_gates[c]
        batch = embs_t[mask]
        if gate is None:
            corr[mask] = batch
            continue
        gate.eval()
        weights = torch.softmax(gate(batch) / temperature, dim=1)
        mixed = weights[:, 0:1] * batch
        for j, domain_adapters in enumerate(adapters_per_domain):
            domain_adapters[c].eval()
            mixed = mixed + weights[:, j + 1: j + 2] * domain_adapters[c](batch)
        corr[mask] = mixed
    return corr.cpu().numpy()


def eval_soft_mixture(embs_np, labels_np, centroids, soft_gates,
                      adapters_per_domain, frozen_head, device, k, temperature=0.5):
    """Soft-mixture forward pass (the main PACT inference path).

        e_hat_i = sum_d  alpha_d(e_i, c_i) * A_{d, c_i}(e_i)
        alpha   = softmax(gate(e) / temperature),  A_{0,c} = identity.
    """
    asgn = assign_concepts(embs_np, centroids)
    embs_t = torch.from_numpy(embs_np).to(device)
    corr = torch.zeros_like(embs_t)
    for c in range(k):
        mask = (asgn == c)
        if not mask.any():
            continue
        gate = soft_gates[c]
        batch = embs_t[mask]
        if gate is None:
            corr[mask] = batch
            continue
        gate.eval()
        weights = torch.softmax(gate(batch) / temperature, dim=1)
        mixed = weights[:, 0:1] * batch
        for j, domain_adapters in enumerate(adapters_per_domain):
            domain_adapters[c].eval()
            mixed = mixed + weights[:, j + 1: j + 2] * domain_adapters[c](batch)
        corr[mask] = mixed
    preds = frozen_head(corr).argmax(1).cpu().numpy()
    return f1_score(labels_np, preds, average="macro", zero_division=0)


@torch.no_grad()
def no_adapt_f1(embs_np, labels_np, frozen_head, device):
    logits = frozen_head(torch.from_numpy(embs_np).to(device))
    return f1_score(labels_np, logits.argmax(1).cpu().numpy(),
                    average="macro", zero_division=0)


@torch.no_grad()
def eval_uniform_mixture(embs_np, labels_np, centroids, adapters_per_domain,
                         frozen_head, device, k):
    """No-gate control: replace the learned gate with UNIFORM weights over
    {identity, all domain adapters}. Identical adapters to PACT, so any gap to
    the soft mixture isolates exactly what the gate contributes (interference
    avoidance / routing). With no adapters this reduces to the frozen head.
    """
    asgn = assign_concepts(embs_np, centroids)
    embs_t = torch.from_numpy(embs_np).to(device)
    n_src = 1  # the identity ("source") branch
    w = 1.0 / (n_src + len(adapters_per_domain))
    corr = torch.zeros_like(embs_t)
    for c in range(k):
        mask = (asgn == c)
        if not mask.any():
            continue
        batch = embs_t[mask]
        mixed = w * batch
        for domain_adapters in adapters_per_domain:
            domain_adapters[c].eval()
            mixed = mixed + w * domain_adapters[c](batch)
        corr[mask] = mixed
    preds = frozen_head(corr).argmax(1).cpu().numpy()
    return f1_score(labels_np, preds, average="macro", zero_division=0)
