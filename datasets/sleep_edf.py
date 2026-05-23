"""Sleep-EDF (EEG sleep staging) dataset loader.

20 subject domains, 5 sleep stages, single EEG channel.
Input shape: [N, 1, 3000]

Data directory layout (AdaTime format):
    <data_root>/Sleep-EDF/EEG/train_0.pt
    <data_root>/Sleep-EDF/EEG/test_0.pt
    ...
    <data_root>/Sleep-EDF/EEG/train_19.pt

Each .pt file: {"samples": Tensor[N, 3000], "labels": Tensor[N]}
"""

from pathlib import Path
import torch

SLEEP_CLASSES   = ["W", "N1", "N2", "N3", "REM"]
SLEEP_SCENARIOS = [           # 10 AdaTime cross-subject pairs
    (0, 11), (2, 5), (4, 12), (6, 1),  (8, 14),
    (10, 7), (12, 3), (14, 9), (16, 0), (18, 6),
]


class SleepEDFDataset(torch.utils.data.Dataset):
    """
    One split of one Sleep-EDF subject domain.

    Args:
        subject_id: Integer 0–19.
        split:      "train" or "test".
        data_root:  Root directory containing Sleep-EDF/EEG/.

    Returns (x [1, 3000], y).
    """

    def __init__(self, subject_id: int, split: str, data_root: str | Path):
        assert 0 <= subject_id <= 19, f"subject_id must be 0–19, got {subject_id}"
        path = Path(data_root) / "Sleep-EDF" / "EEG" / f"{split}_{subject_id}.pt"
        d    = torch.load(path, map_location="cpu", weights_only=False)
        x    = torch.as_tensor(d["samples"]).float()
        if x.dim() == 2:
            x = x.unsqueeze(1)          # [N, T] → [N, 1, T]
        self.x = x
        self.y = torch.as_tensor(d["labels"]).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
