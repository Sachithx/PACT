"""Lightweight numpy k-means (no sklearn dependency)."""

import numpy as np


def kmeans(embeddings: np.ndarray, k: int, seed: int = 42,
           n_iter: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """
    K-means on L2-metric embeddings.

    Args:
        embeddings: [N, D] float32 array
        k:          number of clusters
        seed:       RNG seed
        n_iter:     maximum iterations

    Returns:
        centroids: [K, D]
        assignments: [N] int32
    """
    rng = np.random.default_rng(seed)
    centroids = embeddings[rng.choice(len(embeddings), k, replace=False)].copy()
    assignments = np.zeros(len(embeddings), dtype=np.int32)

    for _ in range(n_iter):
        dists = np.linalg.norm(embeddings[:, None] - centroids[None], axis=2)
        new_asgn = dists.argmin(axis=1).astype(np.int32)
        if np.all(new_asgn == assignments):
            break
        assignments = new_asgn
        for c in range(k):
            pts = embeddings[assignments == c]
            if len(pts):
                centroids[c] = pts.mean(0)

    return centroids, assignments


def assign(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each embedding to its nearest centroid. Returns [N] int32."""
    dists = np.linalg.norm(embeddings[:, None] - centroids[None], axis=2)
    return dists.argmin(axis=1).astype(np.int32)
