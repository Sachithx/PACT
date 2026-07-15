"""Continual-learning metrics computed from retention matrices."""

import numpy as np


def compute_cl_metrics(A, t, include_source=True):
    """Return AA, BWT, F, RF, and PRF through step ``t``.

    ``A`` is a NaN-padded retention matrix. BWT compares the final score with
    the first-seen score. F, RF, and PRF use the running maximum before the
    final evaluation as their reference.
    """
    seen = t + 1
    aa = float(np.nanmean(A[t, :seen]))
    start_j = 0 if include_source else 1

    bwt_values = []
    for j in range(start_j, seen):
        if j == t or np.isnan(A[j, j]) or np.isnan(A[t, j]):
            continue
        bwt_values.append(A[t, j] - A[j, j])
    bwt = float(np.mean(bwt_values)) if bwt_values else 0.0

    drops, relative_drops, performance_weighted_drops = [], [], []
    for j in range(start_j, seen):
        if j == t or np.isnan(A[t, j]):
            continue
        history = [A[i, j] for i in range(j, t) if not np.isnan(A[i, j])]
        if not history:
            continue
        peak = max(history)
        drop = max(0.0, peak - A[t, j])
        relative_drop = drop / (peak + 1e-9)
        drops.append(drop)
        relative_drops.append(relative_drop)
        performance_weighted_drops.append((1.0 - A[t, j]) * relative_drop)

    forgetting = float(np.mean(drops)) if drops else 0.0
    relative_forgetting = float(np.mean(relative_drops)) if relative_drops else 0.0
    performance_weighted = (
        float(np.mean(performance_weighted_drops))
        if performance_weighted_drops
        else 0.0
    )
    return aa, bwt, forgetting, relative_forgetting, performance_weighted
