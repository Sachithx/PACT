"""Concept discovery and routing in the frozen embedding space.

Concepts are K-means clusters of the *source* domain's embeddings. Every
incoming sample is routed to its nearest concept centroid; PACT keeps one
adapter and one gate class per concept. Per-concept distance statistics are
also used for the distance-routing baseline / trigger.
"""

import numpy as np


def kmeans_numpy(embs, k, seed=42, n_iter=100):
    """Lloyd's k-means (numpy). Returns (centroids [k, D], assignments [N])."""
    rng = np.random.default_rng(seed)
    centroids = embs[rng.choice(len(embs), k, replace=False)].copy()
    asgn = np.zeros(len(embs), dtype=int)
    for _ in range(n_iter):
        d = np.linalg.norm(embs[:, None] - centroids[None], axis=2)
        new = d.argmin(1)
        if np.all(new == asgn):
            break
        asgn = new
        for c in range(k):
            m = embs[asgn == c]
            if len(m):
                centroids[c] = m.mean(0)
    return centroids, asgn


def assign_concepts(embs, centroids):
    """Nearest-centroid concept assignment for a batch of embeddings."""
    return np.linalg.norm(embs[:, None] - centroids[None], axis=2).argmin(1)


def build_domain_stats(embs, asgn, centroids, k, pct=95.0):
    """Per-concept mean and percentile radius, used for distance-based routing."""
    stats = {}
    for c in range(k):
        mask = (asgn == c)
        if mask.any():
            mu = embs[mask].mean(0)
            dists = np.linalg.norm(embs[mask] - mu, axis=1)
            stats[c] = {"mu": mu, "radius": float(np.percentile(dists, pct)),
                        "available": True}
        else:
            stats[c] = {"available": False}
    return stats
