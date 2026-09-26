"""FD (Paderborn bearing) data in the AdaTime layout, plus the calibration split.

Expected files: <data_dir>/{train,test}_{0,1,2,3}.pt, each a dict with
"samples" (N, 5120) or (N, 1, 5120) and "labels" (N,). Domains 0..3 are the
four operating conditions FD-A..FD-D; 3 classes.
"""

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

FD_NAMES = {0: "FD-A", 1: "FD-B", 2: "FD-C", 3: "FD-D"}
FD_NUM_CLASSES = 3


class FDDataset(torch.utils.data.Dataset):
    def __init__(self, domain_id: int, split: str, data_dir):
        d = torch.load(Path(data_dir) / f"{split}_{domain_id}.pt",
                       map_location="cpu", weights_only=False)
        x = torch.as_tensor(d["samples"])
        if x.dim() == 2:
            x = x.unsqueeze(1)          # [N, T] -> [N, 1, T]
        self.x = x.float()
        self.y = torch.as_tensor(d["labels"]).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def calib_indices(labels, calib_frac: float, seed: int):
    """Stratified labelled calibration subset: ceil(calib_frac * N) samples,
    at least one per class."""
    y = np.asarray(labels)
    n = len(y)
    n_classes = len(np.unique(y))
    n_calib = min(n, max(n_classes, int(np.ceil(n * calib_frac))))
    if n_calib >= n or n - n_calib < n_classes:
        return np.arange(n)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_calib, random_state=seed)
    idx, _ = next(sss.split(np.zeros(n), y))
    return idx
