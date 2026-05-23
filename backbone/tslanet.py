"""
TSLANet backbone wrapper for PACT.

Architecture matches the checkpoints produced by transfer_hhar_tslanet.py /
transfer_fd_tslanet.py in the EvoConcept repo.  Key layout:
  pos_embed, patch_embed.proj, blocks.N.{norm1,asb,norm2,icb}, head

PACT interface contract:
    backbone.extract(x: Tensor) → Tensor[B, emb_dim]
    backbone(x: Tensor)         → Tensor[B, n_classes]
    backbone.head               → nn.Linear
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


# ── Building blocks ────────────────────────────────────────────────────────────

class _DropPath(nn.Module):
    def __init__(self, p=0.):
        super().__init__()
        self.p = p

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
        self.drop  = nn.Dropout(drop)
        self.act   = nn.GELU()

    def forward(self, x):
        x  = x.transpose(1, 2)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return self.conv3(x1 * self.drop(self.act(x2))
                          + x2 * self.drop(self.act(x1))).transpose(1, 2)


class _PatchEmbed(nn.Module):
    def __init__(self, seq_len, patch_size, in_chans, embed_dim):
        super().__init__()
        stride = patch_size // 2
        self.num_patches = int((seq_len - patch_size) / stride + 1)
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)   # [B, n_patches, D]


class _ASB(nn.Module):
    """Adaptive Spectral Block — matches checkpoint keys complex_weight / complex_weight_h / threshold_param."""
    def __init__(self, dim, adaptive_filter=True):
        super().__init__()
        self.adaptive_filter  = adaptive_filter
        self.complex_weight   = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight_h = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        nn.init.trunc_normal_(self.complex_weight,   std=.02)
        nn.init.trunc_normal_(self.complex_weight_h, std=.02)
        self.threshold_param  = nn.Parameter(torch.rand(1))

    def _hi_freq_mask(self, x_fft):
        B      = x_fft.shape[0]
        energy = torch.abs(x_fft).pow(2).sum(-1)
        med    = energy.view(B, -1).median(1, keepdim=True)[0].view(B, 1)
        norm_e = energy / (med + 1e-6)
        mask   = ((norm_e > self.threshold_param).float() - self.threshold_param).detach() \
                 + self.threshold_param
        return mask.unsqueeze(-1)

    def forward(self, x_in):
        B, N, C = x_in.shape
        x       = x_in.to(torch.float32)
        x_fft   = torch.fft.rfft(x, dim=1, norm='ortho')
        weight  = torch.view_as_complex(self.complex_weight)
        x_w     = x_fft * weight
        if self.adaptive_filter:
            w_h  = torch.view_as_complex(self.complex_weight_h)
            x_w += x_fft * self._hi_freq_mask(x_fft).to(x.device) * w_h
        return torch.fft.irfft(x_w, n=N, dim=1, norm='ortho').to(x_in.dtype).view(B, N, C)


class _Block(nn.Module):
    """TSLANet layer: norm1 → ASB → norm2 → ICB with residual (+ optional drop_path)."""
    def __init__(self, dim, mlp_ratio=3., drop=0., drop_path=0.,
                 use_icb=True, use_asb=True, adaptive_filter=True):
        super().__init__()
        self.use_icb   = use_icb
        self.use_asb   = use_asb
        self.norm1     = nn.LayerNorm(dim)
        self.asb       = _ASB(dim, adaptive_filter)
        self.drop_path = _DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2     = nn.LayerNorm(dim)
        self.icb       = _ICB(dim, int(dim * mlp_ratio), drop)

    def forward(self, x):
        if self.use_icb and self.use_asb:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        elif self.use_icb:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        elif self.use_asb:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        return x


# ── Backbone ───────────────────────────────────────────────────────────────────

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
        num_patches      = self.patch_embed.num_patches
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_patches, emb_dim))
        self.pos_drop    = nn.Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, dropout, depth)]
        self.blocks = nn.ModuleList([
            _Block(emb_dim, mlp_ratio, dropout, drop_path=dpr[i],
                   use_icb=use_icb, use_asb=use_asb, adaptive_filter=adaptive_filter)
            for i in range(depth)])
        self.head = nn.Linear(emb_dim, n_classes)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """[B, n_vars, seq_len] → [B, emb_dim]"""
        x = self.pos_drop(self.patch_embed(x) + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        return x.mean(dim=1)

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
    cfg = DATASET_CFGS[dataset]
    return TSLANetBackbone(
        seq_len=cfg.seq_len, n_vars=cfg.n_vars, n_classes=cfg.n_classes,
        emb_dim=emb_dim, depth=depth, patch_size=patch_size)


def load_tslanet(ckpt_path: str | Path, dataset: str,
                 emb_dim: int = 128, depth: int = 2,
                 patch_size: int = 8,
                 device: str | torch.device = "cpu") -> TSLANetBackbone:
    """Load a pre-trained TSLANet checkpoint (format: {'model_state_dict': ...})."""
    model = make_tslanet(dataset, emb_dim, depth, patch_size)
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd    = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)
