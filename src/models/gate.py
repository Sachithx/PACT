"""Per-concept routing gates (the soft-mixture gate — PACT's core contribution).

For each concept c a small MLP decides how to combine the source (identity)
and every target domain's adapter output. The soft gate outputs one logit per
domain seen so far; a softmax with temperature yields mixing weights so the
corrected embedding is a convex combination of all domains' adapter outputs:

    e_hat = sum_d  alpha_d(e, c) * A_{d,c}(e),   alpha = softmax(g_c(e) / T)

with A_{0,c} = identity (source). Hard routing (argmax) and the binary
apply/bypass gate are special cases.

PrototypeGate below is a parameter-free alternative: instead of an MLP trained
by re-fitting on all accumulated calibration data every step (rehearsal), each
domain gets one Gaussian prototype (mean + scale) per concept, computed once
from that domain's own calibration slice. Appending a domain touches nothing
from earlier domains.
"""

import numpy as np
import torch
import torch.nn as nn

NEG_INF = -1e9  # mask value for absent (chart, domain) cells


class BinaryGateMLP(nn.Module):
    """Per-concept binary classifier: 0 = bypass (source-like), 1 = apply adapter."""

    def __init__(self, in_dim: int = 128, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


class PrototypeGate:
    """Parameter-free per-concept gate: one Gaussian prototype (mean, scale)
    per domain, in log-density form so it slots into the same softmax/
    temperature machinery as SoftGateMLP with no other code changes:

        logit_d(z) = -||z - m_d||^2 / (2 * sigma_d^2)

    No backprop, no rehearsal buffer beyond the prototypes themselves
    (K * n_domains * (dim + 1) floats total). `.eval()` is a no-op kept only
    for interface parity with the nn.Module-based gates.
    """

    def __init__(self, means: torch.Tensor, scales: torch.Tensor):
        self.means = means      # (n_domains, dim)
        self.scales = scales    # (n_domains,)

    def eval(self):
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        d2 = ((x.unsqueeze(1) - self.means.unsqueeze(0)) ** 2).sum(-1)   # (N, n_domains)
        return -d2 / (2 * self.scales.unsqueeze(0) ** 2)


def _chart_dispersion(embs: np.ndarray, floor: float) -> float:
    """RMS distance to the chart's own mean; None if too few points to trust."""
    if len(embs) < 5:
        return None
    mu = embs.mean(0)
    d2 = ((embs - mu) ** 2).sum(1)
    return max(float(np.sqrt(d2.mean())), floor)


def build_prototype_gates(src_embs, src_asgn, calib_per_domain, k: int, emb_dim: int,
                          device, floor: float = 1e-2):
    """Parameter-free counterpart to train_soft_gates: one PrototypeGate per
    concept, fit directly from embeddings (no optimizer, no accumulated-
    calibration re-fit). calib_per_domain: list of (embs, asgn, lbls), one per
    past/current target domain, same as train_soft_gates. Returns a list of K
    PrototypeGate (or None where no domain has any sample in that concept)."""
    domain_embs_asgn = [(src_embs, src_asgn)] + [(e, a) for e, a, _ in calib_per_domain]
    gates = []
    for c in range(k):
        pooled = np.concatenate([e[a == c] for e, a in domain_embs_asgn if (a == c).any()]) \
            if any((a == c).any() for _, a in domain_embs_asgn) else np.empty((0, emb_dim))
        fallback = _chart_dispersion(pooled, floor) or floor

        means, scales, any_domain = [], [], False
        for embs, asgn in domain_embs_asgn:
            sub = embs[asgn == c]
            if len(sub) > 0:
                means.append(sub.mean(0))
                scales.append(_chart_dispersion(sub, floor) or fallback)
                any_domain = True
            else:
                means.append(np.zeros(emb_dim, dtype=np.float32))
                scales.append(fallback)
        if not any_domain:
            gates.append(None); continue

        means_t = torch.from_numpy(np.stack(means).astype(np.float32)).to(device)
        scales_t = torch.tensor(scales, dtype=torch.float32, device=device)
        gates.append(PrototypeGate(means_t, scales_t))
    return gates


def _per_dim_variance(embs: np.ndarray, floor: float) -> np.ndarray:
    """Per-dimension variance with a floor to prevent division by zero."""
    if len(embs) < 3:
        return None
    var = embs.var(axis=0)
    return np.maximum(var, floor)


def build_prototype_gates_diag(src_embs, src_asgn, calib_per_domain, k: int,
                               emb_dim: int, device, floor: float = 1e-4):
    """Diagonal Mahalanobis prototype gate: per-dimension variance."""
    domain_embs_asgn = [(src_embs, src_asgn)] + [(e, a) for e, a, _ in calib_per_domain]
    gates = []
    for c in range(k):
        pooled = np.concatenate([e[a == c] for e, a in domain_embs_asgn if (a == c).any()]) \
            if any((a == c).any() for _, a in domain_embs_asgn) else np.empty((0, emb_dim))
        fallback_var = _per_dim_variance(pooled, floor)
        if fallback_var is None:
            fallback_var = np.full(emb_dim, floor, dtype=np.float32)

        means, variances, any_domain = [], [], False
        for embs, asgn in domain_embs_asgn:
            sub = embs[asgn == c]
            if len(sub) > 0:
                means.append(sub.mean(0))
                v = _per_dim_variance(sub, floor)
                variances.append(v if v is not None else fallback_var)
                any_domain = True
            else:
                means.append(np.zeros(emb_dim, dtype=np.float32))
                variances.append(fallback_var)
        if not any_domain:
            gates.append(None); continue

        means_t = torch.from_numpy(np.stack(means).astype(np.float32)).to(device)
        var_t = torch.from_numpy(np.stack(variances).astype(np.float32)).to(device)
        gates.append(PrototypeGateDiag(means_t, var_t))
    return gates


def build_prototype_gates_lowrank(src_embs, src_asgn, calib_per_domain, k: int,
                                  emb_dim: int, device, rank: int = 16,
                                  floor: float = 1e-2):
    """Low-rank projected prototype gate: discriminative subspace from
    between-domain scatter of the concept means."""
    domain_embs_asgn = [(src_embs, src_asgn)] + [(e, a) for e, a, _ in calib_per_domain]
    gates = []
    for c in range(k):
        means, scales, any_domain = [], [], False
        pooled = np.concatenate([e[a == c] for e, a in domain_embs_asgn if (a == c).any()]) \
            if any((a == c).any() for _, a in domain_embs_asgn) else np.empty((0, emb_dim))
        fallback = _chart_dispersion(pooled, floor) or floor

        for embs, asgn in domain_embs_asgn:
            sub = embs[asgn == c]
            if len(sub) > 0:
                means.append(sub.mean(0))
                scales.append(_chart_dispersion(sub, floor) or fallback)
                any_domain = True
            else:
                means.append(np.zeros(emb_dim, dtype=np.float32))
                scales.append(fallback)
        if not any_domain:
            gates.append(None); continue

        means_np = np.stack(means).astype(np.float32)  # (n_domains, dim)
        # Between-domain scatter: PCA on centered means
        grand_mean = means_np.mean(axis=0, keepdims=True)
        centered = means_np - grand_mean
        # SVD to get top-rank directions
        actual_rank = min(rank, len(means_np) - 1, emb_dim)
        if actual_rank < 1:
            # Fall back to isotropic
            means_t = torch.from_numpy(means_np).to(device)
            scales_t = torch.tensor(scales, dtype=torch.float32, device=device)
            gates.append(PrototypeGate(means_t, scales_t))
            continue

        # Use the full pooled embeddings for a richer covariance if available
        if len(pooled) > actual_rank:
            cov_data = pooled - pooled.mean(axis=0, keepdims=True)
            # Add between-domain scatter emphasis
            cov_data = np.vstack([cov_data, centered * np.sqrt(len(pooled))])
        else:
            cov_data = centered

        U, S, Vt = np.linalg.svd(cov_data, full_matrices=False)
        P = Vt[:actual_rank].T.astype(np.float32)  # (dim, rank)

        proj_means = means_np @ P  # (n_domains, rank)
        proj_t = torch.from_numpy(P).to(device)
        proj_means_t = torch.from_numpy(proj_means).to(device)
        scales_t = torch.tensor(scales, dtype=torch.float32, device=device)
        gates.append(PrototypeGateLowRank(proj_t, proj_means_t, scales_t))
    return gates


def build_confidence_gates(adapters_per_domain, frozen_head, k: int):
    """Confidence-based routing: no stored embeddings, no parameters.
    Returns a list of K ConfidenceGate objects."""
    return [ConfidenceGate(adapters_per_domain, frozen_head, c) for c in range(k)]


class PrototypeGateDiag:
    """Diagonal Mahalanobis prototype gate: per-dimension variance instead of
    scalar scale.  logit_d(z) = -Σ_i (z_i - μ_{d,i})² / (2 * var_{d,i}).

    Captures dimension-wise structure so overlapping domains can be separated
    along their discriminating dimensions. Same O(1) per new domain as the
    scalar PrototypeGate, no rehearsal.
    """

    def __init__(self, means: torch.Tensor, variances: torch.Tensor):
        self.means = means          # (n_domains, dim)
        self.variances = variances  # (n_domains, dim)

    def eval(self):
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - self.means.unsqueeze(0)    # (N, n_domains, dim)
        return -(diff ** 2 / (2 * self.variances.unsqueeze(0))).sum(-1)  # (N, n_domains)


class PrototypeGateLowRank:
    """Low-rank projected prototype gate: project embeddings into a discriminative
    subspace before computing isotropic Gaussian logits.  The projection P is
    derived from between-domain scatter (PCA on stacked domain means per concept).

    logit_d(z) = -||P^T z - P^T μ_d||² / (2 * σ_d²)
    """

    def __init__(self, proj: torch.Tensor, proj_means: torch.Tensor,
                 scales: torch.Tensor):
        self.proj = proj              # (dim, rank)
        self.proj_means = proj_means  # (n_domains, rank)
        self.scales = scales          # (n_domains,)

    def eval(self):
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        z_proj = x @ self.proj                          # (N, rank)
        diff = z_proj.unsqueeze(1) - self.proj_means.unsqueeze(0)  # (N, n_domains, rank)
        d2 = (diff ** 2).sum(-1)                        # (N, n_domains)
        return -d2 / (2 * self.scales.unsqueeze(0) ** 2)


class ConfidenceGate:
    """Test-time confidence routing: forward each adapter and pick the one
    whose corrected embedding yields the highest classifier confidence.

    No stored embeddings, no parameters, no rehearsal. At test time:
        logit_d = max(softmax(head(A_{d,c}(z))))
    Naturally produces logits for softmax mixing in the same interface.
    """

    def __init__(self, adapters_per_domain, frozen_head, concept_idx: int):
        self.adapters = adapters_per_domain
        self.head = frozen_head
        self.c = concept_idx

    def eval(self):
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # source branch (identity)
        logits_src = torch.softmax(self.head(x), dim=1).max(dim=1).values  # (N,)
        all_conf = [logits_src]
        for domain_adapters in self.adapters:
            adapter = domain_adapters[self.c]
            adapter.eval()
            adapted = adapter(x)
            conf = torch.softmax(self.head(adapted), dim=1).max(dim=1).values
            all_conf.append(conf)
        return torch.stack(all_conf, dim=1)  # (N, n_domains)


class SoftGateMLP(nn.Module):
    """Per-concept multi-class gate: one logit per domain (source + N targets).

    Used for both hard routing (argmax) and soft mixing (softmax/temperature);
    architecture matches BinaryGateMLP so the only difference is output width.
    """

    def __init__(self, in_dim: int = 128, hidden: int = 64, n_domains: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(hidden, n_domains),
        )

    def forward(self, x):
        logits = self.net(x)
        mask = getattr(self, "_domain_mask", None)
        if mask is not None:
            logits = logits.masked_fill(~mask.to(logits.device), NEG_INF)
        return logits


# ── Mask fix for the isotropic/diag/lowrank prototype gates ──────────────────
#
# build_prototype_gates (and _diag/_lowrank) place a phantom mean-zero
# prototype for any (chart, domain) cell with no calibration samples.
# Standardized embeddings have their global mean near the origin, so that
# phantom competes for softmax mass in central charts instead of being
# excluded. MaskedGate forces absent cells to NEG_INF; build_prototype_
# gates_masked is the corrected isotropic builder (kept separate from
# build_prototype_gates so the mask's effect is its own measurable ablation).

class MaskedGate:
    """Wrap any gate; force absent domains to -inf instead of a phantom
    zero-mean Gaussian at the (standardized) data origin."""

    def __init__(self, inner, present_mask: torch.Tensor):
        self.inner = inner
        self.mask = present_mask

    def eval(self):
        return self

    def __call__(self, x):
        return self.inner(x).masked_fill(~self.mask.unsqueeze(0), NEG_INF)


def build_prototype_gates_masked(src_embs, src_asgn, calib_per_domain, k: int,
                                 emb_dim: int, device, floor: float = 1e-2):
    """Corrected isotropic builder: identical to build_prototype_gates except
    absent (chart, domain) cells are masked instead of modeled at zero."""
    domain_embs_asgn = [(src_embs, src_asgn)] + [(e, a) for e, a, _ in calib_per_domain]
    gates = []
    for c in range(k):
        pooled_list = [e[a == c] for e, a in domain_embs_asgn if (a == c).any()]
        pooled = np.concatenate(pooled_list) if pooled_list else np.empty((0, emb_dim))
        fallback = _chart_dispersion(pooled, floor) or floor

        means, scales, present = [], [], []
        for embs, asgn in domain_embs_asgn:
            sub = embs[asgn == c]
            present.append(len(sub) > 0)
            if len(sub) > 0:
                means.append(sub.mean(0))
                scales.append(_chart_dispersion(sub, floor) or fallback)
            else:
                means.append(np.zeros(emb_dim, dtype=np.float32))  # masked anyway
                scales.append(fallback)
        if not any(present):
            gates.append(None); continue
        inner = PrototypeGate(
            torch.from_numpy(np.stack(means).astype(np.float32)).to(device),
            torch.tensor(scales, dtype=torch.float32, device=device))
        gates.append(MaskedGate(inner, torch.tensor(present, dtype=torch.bool,
                                                    device=device)))
    return gates


# ── Class-conditional mixture prototype gate ──────────────────────────────────
#
# A single Gaussian per (chart, domain) smears together whatever per-class
# clusters exist within a domain's slice of a chart. MixturePrototypeGate
# uses the calibration labels (already available, currently unused by the
# other prototype builders) to fit one Gaussian per (chart, domain, class)
# and route by logsumexp over classes -- still closed-form, still
# append-only, still statistics rather than stored samples.

class MixturePrototypeGate:
    """logit_d(z) = logsumexp_y [ -||z - m_{d,y}||^2 / (2 sigma_{d,y}^2) ]
    One Gaussian per (domain, class) inside this chart."""

    def __init__(self, means, scales, comp_dom, present_mask):
        self.means = means            # (n_comp, dim) all components stacked
        self.scales = scales          # (n_comp,)
        self.comp_dom = comp_dom      # (n_comp,) long: component -> domain
        self.mask = present_mask      # (n_domains,) bool

    def eval(self):
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        d2 = ((x.unsqueeze(1) - self.means.unsqueeze(0)) ** 2).sum(-1)
        comp_logit = -d2 / (2 * self.scales.unsqueeze(0) ** 2)      # (N, n_comp)
        D = int(self.mask.numel())
        out = torch.full((x.shape[0], D), NEG_INF, device=x.device, dtype=x.dtype)
        for d in range(D):
            sel = self.comp_dom == d
            if sel.any():
                out[:, d] = torch.logsumexp(comp_logit[:, sel], dim=1)
        return out


def build_mixture_prototype_gates(src_embs, src_asgn, calib_per_domain, k, emb_dim,
                                  device, src_lbls=None, floor: float = 1e-2,
                                  min_per_comp: int = 3):
    """Class-conditional prototypes. src_lbls optional: if None, the source
    contributes a single (unlabeled) Gaussian, targets use their labels."""
    doms = [(src_embs, src_asgn, src_lbls)] + list(calib_per_domain)
    gates = []
    for c in range(k):
        means, scales, comp_dom, present = [], [], [], []
        for d, (embs, asgn, lbls) in enumerate(doms):
            sub_idx = asgn == c
            sub = embs[sub_idx]
            present.append(len(sub) > 0)
            if len(sub) == 0:
                continue
            groups = ([sub] if lbls is None else
                      [sub[lbls[sub_idx] == y] for y in np.unique(lbls[sub_idx])])
            groups = [g for g in groups if len(g) >= min_per_comp] or [sub]
            for g in groups:
                mu = g.mean(0)
                s = float(np.sqrt(((g - mu) ** 2).sum(1).mean())) if len(g) > 1 else floor
                means.append(mu); scales.append(max(s, floor)); comp_dom.append(d)
        if not any(present):
            gates.append(None); continue
        gates.append(MixturePrototypeGate(
            torch.from_numpy(np.stack(means).astype(np.float32)).to(device),
            torch.tensor(scales, dtype=torch.float32, device=device),
            torch.tensor(comp_dom, dtype=torch.long, device=device),
            torch.tensor(present, dtype=torch.bool, device=device)))
    return gates


# ── Analytic (RLS) gate ────────────────────────────────────────────────────────
#
# A ridge-regression domain classifier over fixed random ReLU features,
# maintained via sufficient statistics (G = H^T H, C = H^T Y) rather than
# stored samples -- the RanPAC/analytic-continual-learning recipe, pointed at
# domain routing instead of class prediction. Rehearsal-free in the same
# sense RanPAC is: G is (n_features, n_features), CONSTANT in the number of
# domains; C gains one column per domain. Discriminative (unlike the
# isotropic/diag/lowrank Gaussians above), and because normal equations are
# additive in the data, the streaming solution at every step is exactly the
# jointly-trained least-squares solution -- order-invariant by construction.

class RandomFeatures:
    """h(z) = ReLU(W z + b), W fixed Gaussian. Create ONCE per experiment run
    (same instance for every chart and every domain step) so that G/C
    statistics accumulated at different steps live in the same feature
    space. Persist (W, b) alongside the prototypes if you checkpoint."""

    def __init__(self, in_dim: int, n_features: int = 1536, seed: int = 0,
                 device: str = "cpu"):
        g = torch.Generator(device="cpu").manual_seed(seed)
        W = torch.randn(in_dim, n_features, generator=g) / np.sqrt(in_dim)
        b = torch.randn(n_features, generator=g) * 0.1
        self.W = W.to(device)
        self.b = b.to(device)
        self.n_features = n_features

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ self.W + self.b)


class AnalyticGateState:
    """Sufficient statistics for one chart's gate.

    G : (M, M)   accumulated H^T H over ALL domains seen -- constant size in T
    C : (M, D)   accumulated per-domain H^T y column      -- one col per domain
    n : list     per-domain sample counts in this chart (masking/balancing)

    add_domain() appends a column and folds in the new domain's rows. Old
    columns of C and all of G are only ever added to, never revisited, so no
    past embedding is needed: rehearsal-free by construction.
    """

    def __init__(self, n_features: int, device: str = "cpu"):
        self.M = n_features
        self.device = device
        self.G = torch.zeros(n_features, n_features, device=device, dtype=torch.float64)
        self.C = torch.zeros(n_features, 0, device=device, dtype=torch.float64)
        self.n = []

    def add_domain(self, H):
        """H: (n_d, M) random features of the new domain's chart-c calibration
        embeddings, or None / empty if this domain has no samples here."""
        D_new = len(self.n) + 1
        C_new = torch.zeros(self.M, D_new, device=self.device, dtype=torch.float64)
        C_new[:, :D_new - 1] = self.C
        if H is not None and len(H) > 0:
            H64 = H.to(self.device, torch.float64)
            self.G += H64.T @ H64
            C_new[:, D_new - 1] = H64.sum(dim=0)  # H^T y, y = one-hot(new)
            self.n.append(len(H))
        else:
            self.n.append(0)
        self.C = C_new

    def solve(self, lam_rel: float = 1e-2, balance: bool = True):
        """Return AnalyticGate. lam = lam_rel * tr(G)/M keeps regularization
        scale-free; with tiny per-cell n this does real work.

        balance=True reweights columns of C toward equal effective per-domain
        mass, countering the source's much larger sample count. Column
        scaling (not full row-reweighted least squares) is the
        statistics-only surrogate; it does not touch G."""
        tr = torch.diagonal(self.G).sum()
        lam = float(lam_rel) * (tr / self.M if tr > 0 else 1.0)
        A = self.G + lam * torch.eye(self.M, device=self.device, dtype=torch.float64)
        C = self.C.clone()
        if balance and any(self.n):
            n = torch.tensor([max(x, 1) for x in self.n], device=self.device,
                             dtype=torch.float64)
            C = C * (n.mean() / n).sqrt().unsqueeze(0)
        Wg = torch.linalg.solve(A, C)  # (M, D)
        mask = torch.tensor([x > 0 for x in self.n], device=self.device)
        return AnalyticGate(Wg.float(), mask)


class AnalyticGate:
    """Callable gate: logits = phi(z) @ W, absent domains masked to -inf."""

    def __init__(self, W: torch.Tensor, present_mask: torch.Tensor,
                 features=None):
        self.W = W                    # (M, D)
        self.mask = present_mask      # (D,) bool
        self.features = features      # set by builder

    def eval(self):
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x) if self.features is not None else x
        logits = h @ self.W.to(h.dtype)
        return logits.masked_fill(~self.mask.unsqueeze(0), NEG_INF)


def build_analytic_gates(src_embs, src_asgn, calib_per_domain, k: int,
                         emb_dim: int, device, n_features: int = 1536,
                         lam_rel: float = 1e-2, seed: int = 0,
                         features: "RandomFeatures | None" = None,
                         states=None):
    """Drop-in counterpart to train_soft_gates / build_prototype_gates.
    Returns (gates, features, states).

    Stateless usage (rebuild from scratch each step, matching how this
    pipeline currently calls the other prototype builders): pass all domains
    seen so far in calib_per_domain and leave states=None. Streaming usage
    (true O(1) per step): keep `features` and `states` across calls and pass
    ONLY the newest domain -- results are identical (joint == streaming for
    least squares), which is the point.
    """
    feats = features or RandomFeatures(emb_dim, n_features, seed, str(device))
    domain_embs_asgn = [(src_embs, src_asgn)] + [(e, a) for e, a, _ in calib_per_domain]

    if states is None:
        states = [AnalyticGateState(feats.n_features, str(device)) for _ in range(k)]
        to_add = domain_embs_asgn          # rebuild: add every domain
    else:
        to_add = domain_embs_asgn[-1:]     # streaming: newest only

    for embs, asgn in to_add:
        for c in range(k):
            sub = embs[asgn == c]
            H = feats(torch.from_numpy(sub.astype(np.float32)).to(device)) \
                if len(sub) > 0 else None
            states[c].add_domain(H)

    gates = []
    for c in range(k):
        if not any(n > 0 for n in states[c].n):
            gates.append(None)
            continue
        g = states[c].solve(lam_rel=lam_rel)
        g.features = feats
        gates.append(g)
    return gates, feats, states
