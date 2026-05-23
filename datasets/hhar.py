"""HHAR (Heterogeneity Activity Recognition) dataset loader.

9 user domains, 6 activity classes, 3-axis accelerometer.
Input shape: [N, 3, 128]

Data directory layout (AdaTime format):
    <data_root>/HHAR_processed/train_0.pt
    <data_root>/HHAR_processed/test_0.pt
    ...
    <data_root>/HHAR_processed/train_8.pt

Each .pt file: {"samples": Tensor[N, 3, 128], "labels": Tensor[N]}
"""

from pathlib import Path
import torch

HHAR_USERS   = list(range(9))          # 0 = User-a … 8 = User-i
HHAR_CLASSES = ["bike", "sit", "stand", "walk", "stairsup", "stairsdown"]


class HHARDataset(torch.utils.data.Dataset):
    """
    One split of one HHAR user domain.

    Args:
        user_id:   Integer 0–8 (maps to User-a … User-i).
        split:     "train" or "test".
        data_root: Root directory containing HHAR_processed/.

    Returns (x [3, 128], y).
    """

    def __init__(self, user_id: int, split: str, data_root: str | Path):
        assert 0 <= user_id <= 8, f"user_id must be 0–8, got {user_id}"
        path = Path(data_root) / "HHAR_processed" / f"{split}_{user_id}.pt"
        d    = torch.load(path, map_location="cpu", weights_only=False)
        self.x = d["samples"].float()   # [N, 3, 128]
        self.y = d["labels"].long()     # [N]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
