"""FD (Fault Diagnosis) dataset loader.

Each domain is a bearing fault dataset split into train / val / test.
Input shape: [N, 1, 5120]  Labels: 0=normal, 1=ball, 2=inner

Data directory layout (AdaTime format):
    <data_root>/FD-A/train_a.pt
    <data_root>/FD-A/val_a.pt
    <data_root>/FD-A/test_a.pt
    <data_root>/FD-B/train_b.pt   ...etc.

Each .pt file: {"samples": Tensor[N, T], "labels": Tensor[N]}
"""

from pathlib import Path
import torch

FD_DOMAINS = ["FD-A", "FD-B", "FD-C", "FD-D"]
_SUFFIX    = {"FD-A": "a", "FD-B": "b", "FD-C": "c", "FD-D": "d"}


class FDDataset(torch.utils.data.Dataset):
    """
    One split of one FD domain.

    Args:
        domain:    One of "FD-A", "FD-B", "FD-C", "FD-D".
        split:     "train", "val", or "test".
        data_root: Root directory containing the FD subdirectories.

    Returns (x [1, 5120], y).
    """

    def __init__(self, domain: str, split: str, data_root: str | Path):
        assert domain in FD_DOMAINS, f"Unknown domain {domain!r}. Choose from {FD_DOMAINS}"
        path = Path(data_root) / domain / f"{split}_{_SUFFIX[domain]}.pt"
        d    = torch.load(path, map_location="cpu", weights_only=False)
        x    = d["samples"]
        if x.dim() == 2:
            x = x.unsqueeze(1)          # [N, T] → [N, 1, T]
        self.x = x.float()
        self.y = d["labels"].long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
