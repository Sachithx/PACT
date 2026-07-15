"""dp_utils.py — Differentially-private prototype utilities for DP-PACT.

Privacy guarantee: embeddings stored in the prototype bank are
(epsilon_total, delta)-DP under RDP composition of T_mech independent
Gaussian mechanisms (one per stored domain).

What is NOT covered:
  (a) The current domain's raw calibration embeddings, used transiently for
      adapter training and as its own gate class, then discarded.
  (b) The frozen TSLANet encoder weights, treated as public knowledge.

Threat model: honest-but-curious adversary with read access to the stored
prototype bank after all domains have arrived.

DP accounting: Rényi Differential Privacy (Mironov, CSF 2017) via opacus.
Composition: T_mech Gaussian mechanisms, each with L2 sensitivity = clip_norm.

References:
  Mironov (2017) "Rényi Differential Privacy." CSF.
  Abadi et al. (2016) "Deep Learning with Differential Privacy." CCS.
"""

import warnings
import numpy as np

# ── DP accounting backend ─────────────────────────────────────────────────────

try:
    from opacus.accountants import RDPAccountant as _RDPBackend
    _BACKEND = "opacus"
except ImportError:
    try:
        from dp_accounting.rdp import RdpAccountant as _RDPBackend
        _BACKEND = "dp_accounting"
    except ImportError:
        raise ImportError(
            "Neither opacus nor dp_accounting is installed. "
            "Install one: pip install opacus  or  pip install dp_accounting"
        )


def _make_accountant(noise_multiplier: float, T_mech: int):
    """Build RDP accountant after T_mech Gaussian steps (sample_rate=1 = no subsampling)."""
    if _BACKEND == "opacus":
        acc = _RDPBackend()
        for _ in range(T_mech):
            acc.step(noise_multiplier=noise_multiplier, sample_rate=1.0)
        return acc
    else:
        raise NotImplementedError("dp_accounting backend not yet wired; use opacus.")


# ── σ calibration ─────────────────────────────────────────────────────────────

def compute_sigma_from_epsilon(
    eps_total: float,
    delta: float,
    T_mech: int,
    clip_norm: float,
) -> float:
    """Binary-search σ s.t. T_mech composed Gaussian steps achieve (eps_total, δ)-DP.

    Uses RDP accountant (opacus) for tight RDP composition — much tighter than
    basic composition (which would require σ ∝ T_mech).

    noise_multiplier = σ / clip_norm  (opacus convention).

    Args:
        eps_total: target total privacy budget across all T_mech domains.
        delta:     failure probability (e.g. 1e-5).
        T_mech:    number of stored domains = number of Gaussian mechanisms applied.
        clip_norm: L2 sensitivity bound (each prototype clipped to this norm).

    Returns:
        sigma: noise std dev to add to each clipped prototype.
    """
    if T_mech == 0:
        return 0.0

    lo, hi = 1e-4, 5000.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(80):
            mid = (lo + hi) / 2.0
            acc = _make_accountant(mid / clip_norm, T_mech)
            if acc.get_epsilon(delta) <= eps_total:
                hi = mid
            else:
                lo = mid

    sigma = hi

    # Round-trip verification — catches misconfiguration (e.g. wrong units).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acc_check = _make_accountant(sigma / clip_norm, T_mech)
        eps_actual = float(acc_check.get_epsilon(delta))

    tol = max(0.02 * eps_total, 0.01)
    if abs(eps_actual - eps_total) > tol:
        raise ValueError(
            f"σ calibration failed: target ε={eps_total:.4f}, "
            f"achieved ε={eps_actual:.4f} (|diff|={abs(eps_actual-eps_total):.4f} > tol={tol:.4f}). "
            f"Check accountant configuration."
        )
    return sigma


def verify_privacy_spent(sigma: float, clip_norm: float, T_mech: int, delta: float) -> float:
    """Return the effective (ε, δ)-DP budget after T_mech Gaussian steps.

    Args:
        sigma:     noise std dev used per step.
        clip_norm: L2 sensitivity bound.
        T_mech:    number of mechanism applications.
        delta:     failure probability.

    Returns:
        effective_epsilon (float; inf if sigma == 0 or T_mech == 0).
    """
    if sigma <= 0.0 or T_mech == 0:
        return float("inf")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acc = _make_accountant(sigma / clip_norm, T_mech)
        return float(acc.get_epsilon(delta))


# ── Embedding ops ─────────────────────────────────────────────────────────────

def clip_embeddings(protos: np.ndarray, clip_norm: float) -> np.ndarray:
    """L2-clip each row of protos to norm ≤ clip_norm (clips down, never up)."""
    norms = np.linalg.norm(protos, axis=1, keepdims=True)
    scale = np.minimum(1.0, clip_norm / (norms + 1e-12))
    return (protos * scale).astype(np.float32)


def add_gaussian_noise(protos: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add iid N(0, σ²) noise to every element of protos."""
    if sigma <= 0.0:
        return protos.copy()
    noise = rng.standard_normal(protos.shape).astype(np.float32) * sigma
    return protos + noise


# ── Prototype selection ───────────────────────────────────────────────────────

def select_prototypes(
    embs_c: np.ndarray,
    k: int,
    method: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select up to k representative prototype indices from in-concept embeddings.

    Args:
        embs_c: (M_c, D) embeddings belonging to concept c.
        k:      max prototypes to select.
        method: 'kmeans' | 'random' | 'farthest'.
        rng:    numpy Generator (consumed in-place; deterministic given seed).

    Returns:
        indices: 1-D integer array into embs_c, length min(k, M_c).
    """
    M = len(embs_c)
    if M == 0:
        return np.array([], dtype=np.intp)
    if M <= k:
        return np.arange(M, dtype=np.intp)

    if method == "random":
        return rng.choice(M, k, replace=False).astype(np.intp)

    elif method == "kmeans":
        # Mini k-means; return sample nearest each centroid.
        seed_int = int(rng.integers(2 ** 31))
        km_rng = np.random.default_rng(seed_int)
        centroids = embs_c[km_rng.choice(M, k, replace=False)].copy().astype(np.float64)
        asgn = np.zeros(M, dtype=np.intp)
        embs_f64 = embs_c.astype(np.float64)
        for _ in range(30):
            d = np.linalg.norm(embs_f64[:, None] - centroids[None], axis=2)
            new_asgn = d.argmin(1).astype(np.intp)
            if np.all(new_asgn == asgn):
                break
            asgn = new_asgn
            for ci in range(k):
                pts = embs_f64[asgn == ci]
                if len(pts):
                    centroids[ci] = pts.mean(0)
        # Select sample nearest each centroid (deduplicate rare ties).
        chosen = []
        for ci in range(k):
            d = np.linalg.norm(embs_f64 - centroids[ci], axis=1)
            chosen.append(int(d.argmin()))
        return np.array(sorted(set(chosen)), dtype=np.intp)

    elif method == "farthest":
        # Greedy max-min farthest-point sampling.
        # Seed with sample closest to in-concept centroid.
        centroid = embs_c.astype(np.float64).mean(0)
        first = int(np.linalg.norm(embs_c.astype(np.float64) - centroid, axis=1).argmin())
        selected = [first]
        min_dists = np.linalg.norm(
            embs_c.astype(np.float64) - embs_c[first].astype(np.float64), axis=1
        )
        min_dists[first] = -np.inf  # exclude from future selection
        for _ in range(k - 1):
            nxt = int(min_dists.argmax())
            selected.append(nxt)
            new_d = np.linalg.norm(
                embs_c.astype(np.float64) - embs_c[nxt].astype(np.float64), axis=1
            )
            min_dists = np.minimum(min_dists, new_d)
            min_dists[nxt] = -np.inf
        return np.array(selected, dtype=np.intp)

    else:
        raise ValueError(f"Unknown prototype_method: {method!r}. "
                         f"Choose 'kmeans', 'random', or 'farthest'.")
