"""
TSLANet backbone wrapper for PACT.

TSLANet is the backbone used in the PACT paper. This module provides:
  - TSLANetBackbone: full TSLANet model that satisfies the PACT interface
      .extract(x)  → [B, emb_dim]     embedding before the linear head
      .forward(x)  → [B, n_classes]   full classification output
      .head        → nn.Linear         the final classification head
  - load_tslanet(ckpt_path, cfg): load a pre-trained checkpoint

PACT interface contract:
    Any backbone passed to PACT must implement:
      backbone.extract(x: Tensor) → Tensor[B, emb_dim]
      backbone(x: Tensor)         → Tensor[B, n_classes]
      backbone.head               → nn.Linear (used for adapter-supervised loss)

Pre-training:
    Source-domain models are trained with the scripts in ../.. (EvoConcept repo).
    Checkpoints are saved in:
      checkpoints/tslanet_{dataset}_{domain_id}.pt

Alternatively, any backbone can be used by implementing the above three attributes.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

# ── TSLANet architecture (self-contained, no external TSLANet repo needed) ─────
# Re-implemented here so PACT can run without the full TSLANet codebase.
# Architecture matches the paper: ICB + ASB blocks + linear head.


def _trunc_normal(t, std=.02):
    return nn.init.trunc_normal_(t, std=std)


class _DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.p = drop_prob

    def forward(self, x):
        if self.p == 0. or not self.training:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x / keep * torch.floor(torch.rand(shape, dtype=x.dtype, device=x.device) + keep)


class _ICB(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, hidden_dim, 1)
        self.conv2 = nn.Conv1d(dim, hidden_dim, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_dim, dim, 1)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(drop)

    def forward(self, x):
        x = x.transpose(1, 2)
        o = self.conv3(self.conv1(x) * self.drop(self.act(self.conv2(x)))
                       + self.conv2(x) * self.drop(self.act(self.conv1(x))))
        return o.transpose(1, 2)


class _PatchEmbed(nn.Module):
    def __init__(self, seq_len, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Linear(patch_size * in_chans, embed_dim, bias=False)
        n = (seq_len // patch_size)
        self.pos = nn.Parameter(torch.zeros(1, n, embed_dim))
        _trunc_normal(self.pos)

    def forward(self, x):           # x: [B, C, T]
        B, C, T = x.shape
        p = T // self.proj.in_features * C
        if C > 1:
            x = x.permute(0, 2, 1)                  # [B, T, C]
            x = x.reshape(B, -1, self.proj.in_features // C * C)
        else:
            x = x.squeeze(1).reshape(B, -1, self.proj.in_features)
        return self.proj(x) + self.pos


class _ASB(nn.Module):
    def __init__(self, dim, adaptive_filter=True):
        super().__init__()
        self.norm1  = nn.LayerNorm(dim)
        self.norm2  = nn.LayerNorm(dim)
        self.filter = nn.Parameter(torch.ones(dim)) if adaptive_filter else None
        self.mlp    = nn.Sequential(nn.Linear(dim, dim * 3), nn.GELU(),
                                    nn.Linear(dim * 3, dim))

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        if self.filter is not None:
            xf = torch.fft.rfft(x, dim=1)
            xf = xf * self.filter[:xf.shape[-1] if xf.shape[-1] <= len(self.filter) else len(self.filter)]
            x  = torch.fft.irfft(xf, n=x.shape[1], dim=1)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class _Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3., drop=0., drop_path=0.,
                 use_icb=True, use_asb=True, adaptive_filter=True):
        super().__init__()
        self.norm1   = nn.LayerNorm(dim)
        self.icb     = _ICB(dim, int(dim * mlp_ratio), drop) if use_icb else nn.Identity()
        self.drop    = _DropPath(drop_path)
        self.asb     = _ASB(dim, adaptive_filter) if use_asb else nn.Identity()

    def forward(self, x):
        x = x + self.drop(self.icb(self.norm1(x)))
        x = self.asb(x)
        return x


class TSLANetBackbone(nn.Module):
    """
    TSLANet classifier satisfying the PACT backbone interface.

    interface:
        .extract(x) → [B, emb_dim]
        .forward(x) → [B, n_classes]
        .head        → nn.Linear
    """

    def __init__(self, seq_len: int, n_vars: int, n_classes: int,
                 emb_dim: int = 128, depth: int = 2, patch_size: int = 8,
                 mlp_ratio: float = 3., dropout: float = 0.15,
                 use_icb: bool = True, use_asb: bool = True,
                 adaptive_filter: bool = True):
        super().__init__()
        self.patch_embed = _PatchEmbed(seq_len, patch_size, n_vars, emb_dim)
        self.blocks = nn.Sequential(*[
            _Block(emb_dim, mlp_ratio, dropout, use_icb=use_icb,
                   use_asb=use_asb, adaptive_filter=adaptive_filter)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, n_classes)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """[B, n_vars, seq_len] → [B, emb_dim]"""
        x = self.patch_embed(x)          # [B, n_patches, D]
        x = self.blocks(x)
        x = self.norm(x)
        return x.mean(dim=1)             # global average pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract(x))


# ── Dataset configurations ─────────────────────────────────────────────────────

@dataclass
class DatasetCfg:
    seq_len:   int
    n_vars:    int
    n_classes: int
    batch:     int
    name:      str


DATASET_CFGS = {
    "fd":    DatasetCfg(seq_len=5120, n_vars=1, n_classes=3, batch=64,  name="fd"),
    "hhar":  DatasetCfg(seq_len=128,  n_vars=3, n_classes=6, batch=256, name="hhar"),
    "sleep": DatasetCfg(seq_len=3000, n_vars=1, n_classes=5, batch=64,  name="sleep"),
}


def make_tslanet(dataset: str, emb_dim: int = 128, depth: int = 2,
                 patch_size: int = 8) -> TSLANetBackbone:
    """Create an untrained TSLANetBackbone for a given dataset name."""
    cfg = DATASET_CFGS[dataset]
    return TSLANetBackbone(
        seq_len=cfg.seq_len, n_vars=cfg.n_vars, n_classes=cfg.n_classes,
        emb_dim=emb_dim, depth=depth, patch_size=patch_size)


def load_tslanet(ckpt_path: str | Path, dataset: str,
                 emb_dim: int = 128, depth: int = 2,
                 patch_size: int = 8,
                 device: str | torch.device = "cpu") -> TSLANetBackbone:
    """
    Load a pre-trained TSLANet checkpoint.

    The checkpoint must contain a "model_state_dict" key (format written by
    the source-training scripts in the EvoConcept repo).

    Args:
        ckpt_path:  Path to .pt checkpoint file.
        dataset:    "fd", "hhar", or "sleep".
        emb_dim:    Embedding dimension (must match checkpoint).
        depth:      Number of TSLANet blocks (must match checkpoint).
        patch_size: Patch size (must match checkpoint).
        device:     Device to load the model onto.

    Returns:
        Frozen TSLANetBackbone ready for PACT.
    """
    model = make_tslanet(dataset, emb_dim, depth, patch_size)
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd    = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)
