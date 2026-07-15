"""PatchTST backbone (self-contained) for the PACT backbone ablation.

Drop-in replacement for TSLANet: same backbone contract
    extract(x) -> [B, d_model]   (mean-pool over vars x patches)
    forward(x) -> [B, n_classes]
    head                          nn.Linear(d_model, n_classes)
Input convention: x is [B, n_vars, seq_len].

Reference: Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting
with Transformers" (PatchTST, ICLR 2023).
"""

import math

import torch
import torch.nn as nn


class _MHA(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        H, dk = self.h, self.dk
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, L, H, dk).transpose(1, 2)
        k = k.view(B, L, H, dk).transpose(1, 2)
        v = v.view(B, L, H, dk).transpose(1, 2)
        attn = self.drop((q @ k.transpose(-2, -1)) / math.sqrt(dk)).softmax(-1)
        return self.out((attn @ v).transpose(1, 2).reshape(B, L, D))


class _EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.attn = _MHA(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class _PatchEmbed(nn.Module):
    """Patch + positional embedding for multivariate time series."""

    def __init__(self, seq_len, d_model, patch_len, stride, dropout=0.0):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.pad = nn.ReplicationPad1d((0, stride))
        self.proj = nn.Linear(patch_len, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        n_patches = (seq_len + stride - patch_len) // stride + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, C, L = x.shape
        x = self.pad(x)
        x = x.unfold(-1, self.patch_len, self.stride)      # [B, C, N, patch_len]
        N = x.shape[2]
        x = x.reshape(B * C, N, self.patch_len)
        x = self.proj(x)                                   # [B*C, N, d_model]
        return self.drop(x + self.pos_embed[:, :N]), C


class PatchTSTClassifier(nn.Module):
    """Self-contained PatchTST for time-series classification."""

    def __init__(self, seq_len: int, n_vars: int, n_classes: int,
                 d_model: int = 128, n_heads: int = 8, e_layers: int = 3,
                 d_ff: int = 256, dropout: float = 0.1,
                 patch_len: int = 16, stride: int = 8):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.emb_dim = d_model  # PACT backbone contract
        self.patch_embed = _PatchEmbed(seq_len, d_model, patch_len, stride, dropout)
        self.blocks = nn.Sequential(
            *[_EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        tok, n_vars = self.patch_embed(x)          # [B*C, N, d_model]
        tok = self.norm(self.blocks(tok))          # [B*C, N, d_model]
        tok = tok.mean(dim=1)                       # [B*C, d_model]
        tok = tok.reshape(B, n_vars, self.d_model)  # [B, C, d_model]
        return tok.mean(dim=1)                      # [B, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.extract(x)))


def make_patchtst(seq_len, n_vars, n_classes, d_model=128, n_heads=8,
                  e_layers=3, d_ff=256, dropout=0.1, patch_len=16,
                  stride=8) -> PatchTSTClassifier:
    return PatchTSTClassifier(
        seq_len=seq_len, n_vars=n_vars, n_classes=n_classes, d_model=d_model,
        n_heads=n_heads, e_layers=e_layers, d_ff=d_ff, dropout=dropout,
        patch_len=patch_len, stride=stride)
