"""Dataset classes + registry for FD / HHAR / Sleep-EDF / UCIHAR / WISDM.

All store dicts with keys {"samples", "labels"} in per-domain .pt files.
Domains are addressed by integer id (subject / user / condition id); the
registry maps ids to human-readable labels and to the correct file path.

    FD          4 domains (0=FD-A .. 3=FD-D),  3 classes, seq 5120, 1 channel
    HHAR        9 domains (users 0..8),        6 classes, seq 128,  3 channels
    Sleep-EDF   20 domains (subjects 0..19),   5 classes, seq 3000, 1 channel
    UCIHAR      30 subjects (1..30),           6 classes, seq 128,  9 channels
    WISDM       36 subjects (0..35),          6 classes, seq 128,  3 channels

Input convention returned to models: x is [num_channels, seq_len].
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List

import torch


# ── Dataset classes ────────────────────────────────────────────────────────────

_FD_SUFFIX = {0: "a", 1: "b", 2: "c", 3: "d"}
_FD_NAME = {0: "FD-A", 1: "FD-B", 2: "FD-C", 3: "FD-D"}


def _first_existing(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(
        "None of the candidate dataset paths exist:\n  "
        + "\n  ".join(str(p) for p in paths)
    )


def _fd_path(data_root: Path, domain_id: int, split: str) -> Path:
    return _first_existing(
        data_root / _FD_NAME[domain_id] / f"{split}_{_FD_SUFFIX[domain_id]}.pt",
        data_root / "FD" / "FD" / f"{split}_{domain_id}.pt",
    )


def _hhar_path(data_root: Path, domain_id: int, split: str) -> Path:
    return _first_existing(
        data_root / "HHAR_processed" / f"{split}_{domain_id}.pt",
        data_root / "HHAR" / "HHAR_SA" / f"{split}_{domain_id}.pt",
    )


def _sleep_path(data_root: Path, domain_id: int, split: str) -> Path:
    return _first_existing(
        data_root / "Sleep-EDF" / "EEG" / f"{split}_{domain_id}.pt",
        data_root / "Sleep" / "EEG" / f"{split}_{domain_id}.pt",
    )


class FDDataset(torch.utils.data.Dataset):
    """Paderborn bearing fault diagnosis, one operating condition per domain."""

    def __init__(self, domain_id: int, split: str, data_root: Path):
        path = _fd_path(Path(data_root), domain_id, split)
        d = torch.load(path, map_location="cpu", weights_only=False)
        x = d["samples"]
        if x.dim() == 2:
            x = x.unsqueeze(1)          # [N, T] -> [N, 1, T]
        self.x = x.float()
        self.y = d["labels"].long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class HHARDataset(torch.utils.data.Dataset):
    """Heterogeneity Human Activity Recognition, one user per domain."""

    def __init__(self, domain_id: int, split: str, data_root: Path):
        path = _hhar_path(Path(data_root), domain_id, split)
        d = torch.load(path, map_location="cpu", weights_only=False)
        x = torch.as_tensor(d["samples"]).float()
        if x.ndim == 3 and x.shape[1] != 3 and x.shape[2] == 3:
            x = x.transpose(1, 2)   # [N, 128, 3] -> [N, 3, 128]
        self.x = x
        self.y = torch.as_tensor(d["labels"]).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


_ADATIME_ROOT = "ADATIME_downloaded_datasets"


class UCIHARDataset(torch.utils.data.Dataset):
    """UCI HAR (AdaTime HAR protocol), one subject per domain."""

    def __init__(self, domain_id: int, split: str, data_root: Path):
        path = (Path(data_root) / _ADATIME_ROOT / "UCIHAR" / "HAR"
                / f"{split}_{domain_id}.pt")
        d = torch.load(path, map_location="cpu", weights_only=False)
        x = torch.as_tensor(d["samples"]).float()   # [N, 9, 128]
        self.x = x
        self.y = torch.as_tensor(d["labels"]).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class WISDMDataset(torch.utils.data.Dataset):
    """WISDM activity recognition (AdaTime protocol), one subject per domain."""

    def __init__(self, domain_id: int, split: str, data_root: Path):
        path = (Path(data_root) / _ADATIME_ROOT / "WISDM" / "WISDM"
                / f"{split}_{domain_id}.pt")
        d = torch.load(path, map_location="cpu", weights_only=False)
        x = torch.as_tensor(d["samples"]).float()
        if x.ndim == 3 and x.shape[1] != 3 and x.shape[2] == 3:
            x = x.transpose(1, 2)   # [N, T, C] -> [N, C, T]
        self.x = x
        self.y = torch.as_tensor(d["labels"]).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class SleepEDFDataset(torch.utils.data.Dataset):
    """Sleep-EDF sleep-stage classification, one subject per domain."""

    def __init__(self, domain_id: int, split: str, data_root: Path):
        path = _sleep_path(Path(data_root), domain_id, split)
        d = torch.load(path, map_location="cpu", weights_only=False)
        x = torch.as_tensor(d["samples"]).float()
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1).unsqueeze(1)   # (N, 3000, 1) -> (N, 1, 3000)
        self.x = x
        self.y = torch.as_tensor(d["labels"]).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


# ── Registry ─────────────────────────────────────────────────────────────────

@dataclass
class DatasetSpec:
    name: str
    seq_len: int
    num_channels: int
    num_classes: int
    n_domains: int
    class_names: List[str]
    dataset_cls: type
    splits: List[str]
    default_batch: int
    _label_fmt: Callable[[int], str] = field(default=lambda i: str(i))
    has_val: bool = True

    def domain_label(self, domain_id: int) -> str:
        return self._label_fmt(domain_id)

    def get_dataset(self, domain_id: int, split: str, data_root):
        return self.dataset_cls(domain_id, split, data_root)


REGISTRY = {
    "fd": DatasetSpec(
        name="fd", seq_len=5120, num_channels=1, num_classes=3, n_domains=4,
        class_names=["healthy", "fault-1", "fault-2"],
        dataset_cls=FDDataset, splits=["train", "val", "test"], default_batch=256,
        _label_fmt=lambda i: _FD_NAME[i], has_val=True,
    ),
    "hhar": DatasetSpec(
        name="hhar", seq_len=128, num_channels=3, num_classes=6, n_domains=9,
        class_names=["bike", "sit", "stand", "walk", "stairsup", "stairsdown"],
        dataset_cls=HHARDataset, splits=["train", "val", "test"], default_batch=256,
        _label_fmt=lambda i: f"User-{i}", has_val=True,
    ),
    "sleep": DatasetSpec(
        name="sleep", seq_len=3000, num_channels=1, num_classes=5, n_domains=20,
        class_names=["W", "N1", "N2", "N3", "REM"],
        dataset_cls=SleepEDFDataset, splits=["train", "test"], default_batch=64,
        _label_fmt=lambda i: f"Subj-{i}", has_val=False,
    ),
    "ucihar": DatasetSpec(
        name="ucihar", seq_len=128, num_channels=9, num_classes=6, n_domains=30,
        class_names=["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
                     "SITTING", "STANDING", "LAYING"],
        dataset_cls=UCIHARDataset, splits=["train", "test"], default_batch=256,
        _label_fmt=lambda i: f"Subj-{i}", has_val=False,
    ),
    "wisdm": DatasetSpec(
        name="wisdm", seq_len=128, num_channels=3, num_classes=6, n_domains=36,
        class_names=["walking", "jogging", "stairs", "sitting", "standing", "typing"],
        dataset_cls=WISDMDataset, splits=["train", "test"], default_batch=256,
        _label_fmt=lambda i: f"Subj-{i}", has_val=False,
    ),
}

DATASET_NAMES = list(REGISTRY)


def get_spec(dataset: str) -> DatasetSpec:
    if dataset not in REGISTRY:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose from {list(REGISTRY)}.")
    return REGISTRY[dataset]


# Cross-subject/user/condition scenarios (AdaTime protocol) for the benchmark.
SCENARIOS = {
    "fd": [(0, 1), (0, 3), (1, 0), (1, 2), (1, 3),
           (2, 1), (2, 3), (3, 0), (3, 1), (3, 2)],
    "hhar": [(0, 6), (1, 6), (2, 7), (3, 8), (4, 5),
             (5, 0), (6, 1), (7, 4), (8, 3), (0, 2)],
    "sleep": [(0, 11), (7, 18), (9, 14), (12, 5), (16, 1),
              (3, 19), (18, 12), (13, 17), (5, 15), (6, 2)],
    "ucihar": [(2, 11), (6, 23), (7, 13), (9, 18), (12, 16),
               (18, 27), (20, 5), (24, 8), (28, 27), (30, 20)],
    "wisdm": [(7, 18), (20, 30), (35, 31), (17, 23), (6, 19),
              (2, 11), (33, 12), (5, 26), (28, 4), (23, 32)],
}


def adatime_chain(dataset):
    """Build a single domain-incremental chain covering every domain that the
    AdaTime pair set for `dataset` touches.

    The order is chosen to *maximise* the number of AdaTime pairs (s->t) that
    run forward (s before t) in the chain, using the Eades-Lin-Smyth greedy
    minimum-feedback-arc-set heuristic. Because a full T x T retention matrix
    measures every forward pair (not just adjacent ones), this embeds as many of
    the benchmark's chosen shifts as possible into one continual sequence.
    Cyclic pairs (e.g. FD has both (0,1) and (1,0)) force a few backward edges.

    Returns: (order, stats) where stats = dict(n_domains, n_pairs, forward, backward).
    """
    import collections

    pairs = [(s, t) for (s, t) in SCENARIOS[dataset] if s != t]
    nodes = []
    for s, t in pairs:
        for x in (s, t):
            if x not in nodes:
                nodes.append(x)

    out = collections.defaultdict(set)
    inn = collections.defaultdict(set)
    for s, t in set(pairs):
        out[s].add(t)
        inn[t].add(s)

    rem = set(nodes)
    s1, s2 = [], []
    while rem:
        moved = True
        while moved:
            moved = False
            for n in [x for x in nodes if x in rem and not (out[x] & rem)]:  # sinks
                s2.insert(0, n); rem.discard(n); moved = True
            for n in [x for x in nodes if x in rem and not (inn[x] & rem)]:  # sources
                s1.append(n); rem.discard(n); moved = True
        if rem:
            best = max(rem, key=lambda n: len(out[n] & rem) - len(inn[n] & rem))
            s1.append(best); rem.discard(best)
    order = s1 + s2

    pos = {d: i for i, d in enumerate(order)}
    fwd = sum(1 for s, t in set(pairs) if pos[s] < pos[t])
    bwd = len(set(pairs)) - fwd
    stats = dict(n_domains=len(order), n_pairs=len(set(pairs)),
                 forward=fwd, backward=bwd)
    return order, stats
