"""Benchmark loaders and the labelled calibration split.

Each dataset is a folder of per-domain files ``{train,test}_{id}.pt``. Every
file is a dict with ``samples`` and ``labels``. ``download_data.py`` writes
them under ``dataset/<name>/``, which is the layout this loader expects.

    FD         4 operating conditions (0..3 = FD-A..FD-D), 3 classes, length 5120, 1 channel
    WISDM      36 subjects (0..35), 6 classes, length 128, 3 channels
    HHAR       9 users (0..8), 6 classes, length 128, 3 channels
    UCIHAR     30 subjects (1..30), 6 classes, length 128, 9 channels
    Sleep-EDF  20 subjects (0..19), 5 classes, length 3000, 1 channel
"""

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit


def calib_indices(labels, calib_frac: float, seed: int, calib_n: int | None = None):
    """Stratified labelled calibration subset.

    With ``calib_n``, the budget is ``min(N, max(n_classes, calib_n))``
    (WISDM uses 450, UCI-HAR uses 100). Otherwise it is
    ``max(n_classes, ceil(calib_frac * N))``, which is the 5% protocol used
    on FD, HHAR, and Sleep-EDF. At least one sample per class is kept.
    """
    y = np.asarray(labels)
    n = len(y)
    if n == 0:
        return np.arange(n)
    n_classes = len(np.unique(y))
    if calib_n is not None:
        n_calib = min(n, max(n_classes, int(calib_n)))
    elif calib_frac >= 1.0:
        return np.arange(n)
    else:
        n_calib = min(n, max(n_classes, int(np.ceil(n * calib_frac))))
    if n_calib >= n or n - n_calib < n_classes:
        return np.arange(n)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_calib, random_state=seed)
    idx, _ = next(sss.split(np.zeros(n), y))
    return idx


def _as_nct(samples):
    """Return samples as a float tensor of shape (N, C, T)."""
    x = torch.as_tensor(samples).float()
    if x.dim() == 2:
        return x.unsqueeze(1)                       # (N, T) -> (N, 1, T)
    if x.dim() != 3:
        raise ValueError(f"expected samples of rank 2 or 3, got {tuple(x.shape)}")
    if x.shape[-1] == 1 and x.shape[1] > 1:
        return x.squeeze(-1).unsqueeze(1)           # (N, T, 1) -> (N, 1, T)
    if x.shape[1] > x.shape[2] and x.shape[2] <= 16:
        return x.transpose(1, 2).contiguous()       # (N, T, C) -> (N, C, T)
    return x


class DomainDataset(torch.utils.data.Dataset):
    def __init__(self, domain_id: int, split: str, data_dir):
        path = Path(data_dir) / f"{split}_{domain_id}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}. Run `python download_data.py` or pass --data_dir."
            )
        d = torch.load(path, map_location="cpu", weights_only=False)
        self.x = _as_nct(d["samples"])
        self.y = torch.as_tensor(d["labels"]).long().view(-1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]
