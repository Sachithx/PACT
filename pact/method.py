"""PACT: chart-local transports + soft gate over a frozen backbone's embeddings.

Everything operates on frozen, standardized embeddings z. Neither the backbone
nor the source head is ever updated, so the source classifier is preserved by
construction.

    1. Charts: k-means (K centroids) on the SOURCE training embeddings. Every
       embedding is routed to its nearest centroid c(z).
    2. Transports: when target domain d arrives, a bank of K LoRA maps
       T_{d,c}(z) = z + B A z (rank r, initialised to the identity) is trained
       on a small labelled calibration slice of d, through the frozen head.
       The bank is then frozen.
    3. Gate: for each chart c, an MLP predicts which domain (0 = source,
       1..t = targets) an embedding came from. It is refit at every step on
       the source embeddings plus the accumulated calibration embeddings
       (embeddings only, no class labels).
    4. Inference (soft mixture):
           z_hat = w_0(z) z + sum_d w_d(z) T_{d,c(z)}(z),
           w = softmax(gate_c(z) / temperature),
       and the prediction is head(z_hat).
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from .data import calib_indices

NEG_INF = -1e9  # gate logit for (chart, domain) cells without calibration support


@dataclass
class PactConfig:
    k: int = 8                  # number of charts (FD main table: K=8)
    emb_dim: int = 768
    calib_frac: float = 0.05    # labelled fraction of each target domain's train split
    lora_rank: int = 8
    adapter_ep: int = 200
    adapter_lr: float = 1e-3
    gate_ep: int = 200
    gate_lr: float = 1e-3
    penalty: float = 1e-3       # weight of ||BA||_F^2 (keeps transports near identity)
    batch: int = 64
    seed: int = 42
    kmeans_n_init: int = 10
    min_chart_domain_n: int = 5  # a domain needs >= this many samples in a chart to get a gate output
    temperature: float = 0.5


# ── Charts (k-means on source embeddings) ────────────────────────────────────

def assign_charts(embs, centroids):
    return np.linalg.norm(embs[:, None] - centroids[None], axis=2).argmin(1)


def _lloyd(embs, centroids, n_iter=100):
    asgn = np.zeros(len(embs), dtype=int)
    for _ in range(n_iter):
        new = np.linalg.norm(embs[:, None] - centroids[None], axis=2).argmin(1)
        if np.all(new == asgn):
            break
        asgn = new
        for c in range(len(centroids)):
            m = embs[asgn == c]
            if len(m):
                centroids[c] = m.mean(0)
    inertia = float(((embs - centroids[asgn]) ** 2).sum())
    return centroids, asgn, inertia


def kmeans(embs, k, seed=42, n_init=10):
    """Lloyd's k-means with n_init restarts; returns the lowest-inertia run."""
    best = (None, None, np.inf)
    for init in range(max(1, n_init)):
        rng = np.random.default_rng(seed + init * 9973)
        centroids = embs[rng.choice(len(embs), k, replace=False)].copy()
        centroids, asgn, inertia = _lloyd(embs, centroids)
        if inertia < best[2]:
            best = (centroids, asgn, inertia)
    return best[0], best[1]


# ── Modules ──────────────────────────────────────────────────────────────────

class LoRATransport(nn.Module):
    """T(z) = z + B A z, with B = 0 at init so T starts as the identity."""

    def __init__(self, dim, rank=8):
        super().__init__()
        self.A = nn.Linear(dim, rank, bias=False)
        self.B = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.A.weight, std=1.0 / max(rank, 1))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return x + self.B(self.A(x))

    def identity_penalty(self):
        return torch.sum((self.B.weight @ self.A.weight) ** 2)


class ChartGate(nn.Module):
    """Per-chart domain classifier: one logit per domain (0 = source)."""

    def __init__(self, in_dim, n_domains, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(hidden, n_domains))

    def forward(self, x):
        logits = self.net(x)
        mask = getattr(self, "_domain_mask", None)
        if mask is not None:
            logits = logits.masked_fill(~mask.to(logits.device), NEG_INF)
        return logits


# ── Training ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embs(model, loader, device):
    model.eval()
    embs, lbls = [], []
    for xb, yb in loader:
        embs.append(model.extract(xb.to(device)).cpu().numpy())
        lbls.append(yb.numpy())
    return np.concatenate(embs).astype(np.float32), np.concatenate(lbls)


def train_transports(calib_embs, calib_lbls, centroids, head, device, cfg):
    """Train one domain's bank of K chart-local LoRA transports."""
    bank = nn.ModuleList([LoRATransport(cfg.emb_dim, cfg.lora_rank)
                          for _ in range(cfg.k)]).to(device)
    opt = torch.optim.AdamW(bank.parameters(), lr=cfg.adapter_lr)
    crit = nn.CrossEntropyLoss()
    asgn = assign_charts(calib_embs, centroids)
    ld = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(calib_embs).to(device),
                                       torch.from_numpy(calib_lbls).to(device),
                                       torch.from_numpy(asgn.astype(np.int64))),
        batch_size=cfg.batch, shuffle=True)
    for _ in range(cfg.adapter_ep):
        for eb, yb, cb in ld:
            bank.train(); opt.zero_grad()
            z = eb.clone()
            for c in range(cfg.k):
                m = (cb == c)
                if m.any():
                    z[m] = bank[c](eb[m])
            loss = crit(head(z), yb)
            for t in bank:
                loss = loss + cfg.penalty * t.identity_penalty()
            loss.backward(); opt.step()
    return bank


def train_gates(src_embs, src_asgn, calib_buffer, device, cfg):
    """Fit one gate per chart on (source embeddings, accumulated target
    calibration embeddings). Classes are balanced by oversampling."""
    n_domains = 1 + len(calib_buffer)
    gates = []
    for c in range(cfg.k):
        X_by_dom = {0: src_embs[src_asgn == c]}
        active = []
        for j, (embs, asgn) in enumerate(calib_buffer):
            Xc = embs[asgn == c]
            if len(Xc) >= cfg.min_chart_domain_n:
                active.append(j + 1)
                X_by_dom[j + 1] = Xc
        if not active:
            gates.append(None)            # no target support: chart stays identity
            continue

        max_n = max(len(v) for v in X_by_dom.values() if len(v) > 0)
        rng = np.random.default_rng(cfg.seed + c + 200)
        X_parts, y_parts = [], []
        for d, Xd in X_by_dom.items():
            if len(Xd) == 0:
                continue
            if len(Xd) < max_n:
                Xd = np.vstack([Xd, Xd[rng.choice(len(Xd), max_n - len(Xd), replace=True)]])
            X_parts.append(Xd)
            y_parts.append(np.full(len(Xd), d, dtype=np.int64))
        X = np.concatenate(X_parts).astype(np.float32)
        y = np.concatenate(y_parts)
        perm = rng.permutation(len(X))
        X, y = X[perm], y[perm]

        ld = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X).to(device),
                                           torch.from_numpy(y).to(device)),
            batch_size=min(cfg.batch, len(X)), shuffle=True)
        gate = ChartGate(cfg.emb_dim, n_domains).to(device)
        present = [True] + [d in active for d in range(1, n_domains)]
        if not all(present):
            gate.register_buffer("_domain_mask", torch.tensor(present, dtype=torch.bool))
        opt = torch.optim.AdamW(gate.parameters(), lr=cfg.gate_lr, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        for _ in range(cfg.gate_ep):
            gate.train()
            for xb, yb in ld:
                opt.zero_grad(); crit(gate(xb), yb).backward(); opt.step()
        gates.append(gate)
    return gates


# ── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(embs_np, centroids, gates, banks, head, device, cfg):
    """Soft-mixture PACT prediction (class indices)."""
    asgn = assign_charts(embs_np, centroids)
    z = torch.from_numpy(embs_np).to(device)
    z_hat = torch.zeros_like(z)
    for c in range(cfg.k):
        m = (asgn == c)
        if not m.any():
            continue
        zc = z[m]
        if gates[c] is None:
            z_hat[m] = zc
            continue
        gates[c].eval()
        w = torch.softmax(gates[c](zc) / cfg.temperature, dim=1)
        mixed = w[:, 0:1] * zc                        # source branch = identity
        for j, bank in enumerate(banks):
            bank[c].eval()
            mixed = mixed + w[:, j + 1:j + 2] * bank[c](zc)
        z_hat[m] = mixed
    return head(z_hat).argmax(1).cpu().numpy()


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


# ── Continual-learning metrics ───────────────────────────────────────────────

def cl_metrics(A, t):
    """AA, BWT, F, RF, PRF after step t from the macro-F1 matrix A
    (A[i, j] = score on domain j after adapting through domain i)."""
    seen = t + 1
    AA = float(np.nanmean(A[t, :seen]))
    bwt, f, rf, prf = [], [], [], []
    for j in range(seen):
        if j == t:
            continue
        bwt.append(A[t, j] - A[j, j])
        peak = max(A[i, j] for i in range(j, t + 1))
        drop = max(0.0, peak - A[t, j])
        f.append(drop)
        rf.append(drop / (abs(peak) + 1e-9))
        prf.append((1.0 - A[t, j]) * rf[-1])
    mean = lambda v: float(np.mean(v)) if v else 0.0
    return AA, mean(bwt), mean(f), mean(rf), mean(prf)


# ── Domain-incremental stream ────────────────────────────────────────────────

def run_stream(model, make_dataset, domain_order, names, device, cfg, verbose=True):
    """Run PACT over one domain order. model.head must already be fitted on
    domain_order[0] and frozen. Returns the macro-F1 matrix and metric rows."""
    head = model.head
    T = len(domain_order)

    src_ld = torch.utils.data.DataLoader(make_dataset(domain_order[0], "train"),
                                         batch_size=cfg.batch)
    src_embs, _ = extract_embs(model, src_ld, device)     # source labels are not kept
    centroids, src_asgn = kmeans(src_embs, cfg.k, seed=cfg.seed, n_init=cfg.kmeans_n_init)

    test = {}
    for d in domain_order:
        ld = torch.utils.data.DataLoader(make_dataset(d, "test"), batch_size=cfg.batch)
        test[d] = extract_embs(model, ld, device)

    A = np.full((T, T), np.nan)
    with torch.no_grad():
        e0, y0 = test[domain_order[0]]
        A[0, 0] = macro_f1(y0, head(torch.from_numpy(e0).to(device)).argmax(1).cpu().numpy())
    rows = [dict(step=0, domain=names[domain_order[0]], AA=A[0, 0],
                 BWT=0.0, F=0.0, RF=0.0, PRF=0.0)]
    if verbose:
        print(f"  step 0  {names[domain_order[0]]:<5} (source)  F1={A[0, 0]:.4f}")

    calib_buffer, banks = [], []
    for t in range(1, T):
        d = domain_order[t]
        ds = make_dataset(d, "train")
        y_all = ds.y.numpy()
        idx = calib_indices(y_all, cfg.calib_frac, cfg.seed)
        calib_ld = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx),
                                               batch_size=cfg.batch)
        calib_embs, _ = extract_embs(model, calib_ld, device)
        calib_buffer.append((calib_embs, assign_charts(calib_embs, centroids)))

        banks.append(train_transports(calib_embs, y_all[idx], centroids, head, device, cfg))
        gates = train_gates(src_embs, src_asgn, calib_buffer, device, cfg)

        for j in range(t + 1):
            e, y = test[domain_order[j]]
            A[t, j] = macro_f1(y, predict(e, centroids, gates, banks, head, device, cfg))

        AA, BWT, F, RF, PRF = cl_metrics(A, t)
        rows.append(dict(step=t, domain=names[d], AA=AA, BWT=BWT, F=F, RF=RF, PRF=PRF))
        if verbose:
            print(f"  step {t}  {names[d]:<5} n_calib={len(idx):<4} AA={AA:.4f} "
                  f"BWT={BWT:+.4f} F={F:.4f} PRF={PRF:.4f}")
    return A, rows
