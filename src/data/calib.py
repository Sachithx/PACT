"""Stratified calibration splits with a per-class floor.

Some AdaTime domains (e.g. WISDM subject 23) have small train sets where
5% rounds down below the number of classes; sklearn then rejects the split.
"""

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def calib_indices(labels, calib_frac: float, seed: int, min_size: int | None = None,
                  calib_n: int | None = None):
    """Return indices of a stratified calibration subset from `labels`.

    If `calib_n` is set, use min(n, max(n_classes, calib_n)) labels.
    Otherwise size is max(min_size or n_classes, ceil(n * calib_frac)), capped at n.
    """
    y = np.asarray(labels)
    n = len(y)
    if n == 0:
        return np.arange(n)
    n_classes = len(np.unique(y))
    floor = n_classes if min_size is None else max(min_size, n_classes)
    if calib_n is not None:
        n_calib = min(n, max(floor, calib_n))
    elif calib_frac >= 1.0:
        return np.arange(n)
    else:
        n_calib = min(n, max(floor, int(np.ceil(n * calib_frac))))
    if n_calib >= n:
        return np.arange(n)
    if n - n_calib < n_classes:
        # Remainder too small to stratify (sklearn needs >= 1 sample per
        # class in the complement too) -- nothing meaningful is left to
        # hold out, so just use everything.
        return np.arange(n)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_calib, random_state=seed)
    idx, _ = next(sss.split(np.zeros(n), y))
    return idx
