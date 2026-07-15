"""Backbone factory: build any supported backbone and resolve checkpoint paths.

Checkpoint layout is kept identical to the original repo so pre-trained
checkpoints (symlinked under checkpoints/) are reused without retraining:

    tslanet  fd    -> checkpoints/tslanet/tslanet_{FD-X}_ps8_d2_dim128.pt
    tslanet  hhar  -> checkpoints/tslanet/tslanet_hhar/tslanet_hhar_u{id}_ps8_d2_dim128.pt
    tslanet  sleep -> checkpoints/tslanet/tslanet_sleepedf/tslanet_sleepedf_s{id}_ps8_d2_dim128.pt
    patchtst *     -> checkpoints/patchtst/patchtst_{dataset}_{id}.pt
    timesnet *     -> checkpoints/timesnet/timesnet_{dataset}_{id}.pt
"""

from pathlib import Path

import torch

from ..data.datasets import DatasetSpec, _FD_NAME
from .patchtst import make_patchtst
from .timesnet import make_timesnet
from .tslanet import make_tslanet

BACKBONES = ["tslanet", "patchtst", "timesnet"]

# TSLANet architecture defaults (match paper / existing checkpoints).
TSLANET_DEFAULTS = dict(emb_dim=128, depth=2, patch_size=8,
                        dropout_rate=0.15, masking_ratio=0.4,
                        use_icb=True, use_asb=True, adaptive_filter=True)


def build_backbone(backbone: str, spec: DatasetSpec) -> torch.nn.Module:
    """Construct an (untrained) backbone sized for the given dataset."""
    if backbone == "tslanet":
        return make_tslanet(spec.seq_len, spec.num_channels, spec.num_classes,
                            **TSLANET_DEFAULTS)
    if backbone == "patchtst":
        return make_patchtst(spec.seq_len, spec.num_channels, spec.num_classes)
    if backbone == "timesnet":
        return make_timesnet(spec.seq_len, spec.num_channels, spec.num_classes)
    raise ValueError(f"Unknown backbone {backbone!r}. Choose from {BACKBONES}.")


def checkpoint_path(ckpt_root, backbone: str, dataset: str, domain_id: int) -> Path:
    ckpt_root = Path(ckpt_root)
    if backbone == "tslanet":
        base = ckpt_root / "tslanet"
        if dataset == "fd":
            return base / f"tslanet_{_FD_NAME[domain_id]}_ps8_d2_dim128.pt"
        if dataset == "hhar":
            return base / "tslanet_hhar" / f"tslanet_hhar_u{domain_id}_ps8_d2_dim128.pt"
        if dataset == "sleep":
            return base / "tslanet_sleepedf" / f"tslanet_sleepedf_s{domain_id}_ps8_d2_dim128.pt"
    if backbone in ("patchtst", "timesnet"):
        return ckpt_root / backbone / f"{backbone}_{dataset}_{domain_id}.pt"
    raise ValueError(f"Unknown (backbone={backbone}, dataset={dataset})")


def load_backbone(backbone: str, spec: DatasetSpec, ckpt_root, dataset: str,
                  domain_id: int, device, freeze: bool = True) -> torch.nn.Module:
    """Build a backbone and load its cached source checkpoint (must exist)."""
    ckpt = checkpoint_path(ckpt_root, backbone, dataset, domain_id)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No {backbone} checkpoint for {dataset} domain {domain_id} at {ckpt}. "
            f"Train it first: scripts/train_backbone.py --backbone {backbone} "
            f"--dataset {dataset} --domain {domain_id}")
    model = build_backbone(backbone, spec).to(device)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)
    return model
