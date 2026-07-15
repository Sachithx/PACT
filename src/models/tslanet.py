"""TSLANet backbone (consolidated).

Single self-contained TSLANet implementation shared across all datasets
(FD / HHAR / Sleep-EDF). Previously this exact architecture was copy-pasted
into transfer_fd_tslanet.py, transfer_hhar_tslanet.py and
transfer_sleep_tslanet.py; the only per-dataset differences were seq_len,
num_channels and num_classes, which are now passed as constructor arguments.

Reference: Eldele et al., "TSLANet: Rethinking Transformers for Time Series
Representation Learning" (ICML 2024).

Backbone contract used by the PACT plugin:
    extract(x) -> [B, emb_dim]   (frozen pre-head embedding)
    forward(x) -> [B, num_classes]
    head                          nn.Linear(emb_dim, num_classes)
Input convention: x is [B, num_channels, seq_len].
"""

import torch
import torch.nn as nn

from ..utils.misc import random_masking_3D


class DropPath(nn.Module):
    """Stochastic depth (drop entire sample paths during training)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = torch.rand(shape, dtype=x.dtype, device=x.device)
        return x / keep_prob * torch.floor(rand + keep_prob)


def trunc_normal_(tensor, std=0.02, mean=0.0, a=-2.0, b=2.0):
    return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


class ICBModule(nn.Module):
    """Inverted Convolutional Block with cross-gating (matches original TSLANet)."""

    def __init__(self, in_features, hidden_features, drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = self.conv3(x1 * self.drop(self.act(x2)) + x2 * self.drop(self.act(x1)))
        return out.transpose(1, 2)


class PatchEmbed(nn.Module):
    def __init__(self, seq_len, patch_size=8, in_chans=1, embed_dim=128):
        super().__init__()
        stride = patch_size // 2
        self.num_patches = int((seq_len - patch_size) / stride + 1)
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class AdaptiveSpectralBlock(nn.Module):
    """FFT-domain adaptive filtering with a learned high-frequency mask."""

    def __init__(self, dim, adaptive_filter=True):
        super().__init__()
        self.adaptive_filter = adaptive_filter
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight_h = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight, std=0.02)
        trunc_normal_(self.complex_weight_h, std=0.02)
        self.threshold_param = nn.Parameter(torch.rand(1))

    def _hi_freq_mask(self, x_fft):
        B = x_fft.shape[0]
        energy = torch.abs(x_fft).pow(2).sum(-1)
        med = energy.view(B, -1).median(1, keepdim=True)[0].view(B, 1)
        norm_e = energy / (med + 1e-6)
        mask = ((norm_e > self.threshold_param).float() - self.threshold_param).detach() \
            + self.threshold_param
        return mask.unsqueeze(-1)

    def forward(self, x_in):
        B, N, C = x_in.shape
        x = x_in.to(torch.float32)
        x_fft = torch.fft.rfft(x, dim=1, norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x_w = x_fft * weight
        if self.adaptive_filter:
            w_h = torch.view_as_complex(self.complex_weight_h)
            x_w = x_w + x_fft * self._hi_freq_mask(x_fft).to(x.device) * w_h
        x = torch.fft.irfft(x_w, n=N, dim=1, norm="ortho")
        return x.to(x_in.dtype).view(B, N, C)


class TSLANetLayer(nn.Module):
    def __init__(self, dim, mlp_ratio=3.0, drop=0.0, drop_path=0.0,
                 norm_layer=nn.LayerNorm, use_icb=True, use_asb=True,
                 adaptive_filter=True):
        super().__init__()
        self.use_icb = use_icb
        self.use_asb = use_asb
        self.norm1 = norm_layer(dim)
        self.asb = AdaptiveSpectralBlock(dim, adaptive_filter)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.icb = ICBModule(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        if self.use_icb and self.use_asb:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        elif self.use_icb:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        elif self.use_asb:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        return x


class TSLANet(nn.Module):
    """Self-contained TSLANet taking explicit config (no global args)."""

    def __init__(self, seq_len, num_channels, num_classes,
                 emb_dim=128, depth=2, patch_size=8,
                 dropout_rate=0.15, masking_ratio=0.4,
                 use_icb=True, use_asb=True, adaptive_filter=True):
        super().__init__()
        self.emb_dim = emb_dim
        self.masking_ratio = masking_ratio

        self.patch_embed = PatchEmbed(seq_len, patch_size, num_channels, emb_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, emb_dim))
        self.pos_drop = nn.Dropout(p=dropout_rate)

        dpr = [x.item() for x in torch.linspace(0, dropout_rate, depth)]
        self.blocks = nn.ModuleList([
            TSLANetLayer(dim=emb_dim, drop=dropout_rate, drop_path=dpr[i],
                         use_icb=use_icb, use_asb=use_asb,
                         adaptive_filter=adaptive_filter)
            for i in range(depth)
        ])
        self.head = nn.Linear(emb_dim, num_classes)

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def extract(self, x):
        """Backbone: patch embed -> positional encoding -> blocks -> mean pool."""
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        return x.mean(1)

    def forward(self, x):
        return self.head(self.extract(x))

    def pretrain_fwd(self, x):
        """Masked reconstruction forward; returns (pred, target, bool_mask)."""
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        x_masked, _, mask, _ = random_masking_3D(x, self.masking_ratio)
        mask = mask.bool()
        for blk in self.blocks:
            x_masked = blk(x_masked)
        return x_masked, x, mask


def make_tslanet(seq_len, num_channels, num_classes, emb_dim=128, depth=2,
                 patch_size=8, dropout_rate=0.15, masking_ratio=0.4,
                 use_icb=True, use_asb=True, adaptive_filter=True) -> TSLANet:
    return TSLANet(
        seq_len=seq_len, num_channels=num_channels, num_classes=num_classes,
        emb_dim=emb_dim, depth=depth, patch_size=patch_size,
        dropout_rate=dropout_rate, masking_ratio=masking_ratio,
        use_icb=use_icb, use_asb=use_asb, adaptive_filter=adaptive_filter,
    )
