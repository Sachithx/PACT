"""
TimesNet backbone wrapper for PACT.

Uses TimesBlock and DataEmbedding directly from the cloned Time-Series-Library.
PACT interface:
    backbone.extract(x: Tensor[B, C, T]) → Tensor[B, d_model]
    backbone(x: Tensor[B, C, T])         → Tensor[B, n_classes]
    backbone.head                         → nn.Linear
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Import from Time-Series-Library ───────────────────────────────────────────
_TSL = Path(__file__).parent.parent / "Time-Series-Library"
if str(_TSL) not in sys.path:
    sys.path.insert(0, str(_TSL))

from layers.Embed import DataEmbedding          # noqa: E402
from layers.Conv_Blocks import Inception_Block_V1  # noqa: E402


def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, seq_len, d_model, d_ff, top_k, num_kernels):
        super().__init__()
        self.seq_len = seq_len
        self.k       = top_k
        self.conv    = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        res = []
        for period in period_list:
            if T % period != 0:
                length = (T // period + 1) * period
                padding = torch.zeros(B, length - T, N, device=x.device, dtype=x.dtype)
                out = torch.cat([x, padding], dim=1)
            else:
                length, out = T, x
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :T, :])
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).expand(B, T, N, self.k)
        res = (res * period_weight).sum(-1) + x
        return res


# ── Backbone ───────────────────────────────────────────────────────────────────

class TimesNetBackbone(nn.Module):
    """
    TimesNet classifier satisfying the PACT backbone interface.

    Input convention: [B, n_vars, seq_len]  (PACT / sensor format, C-first).
    Internally transposed to [B, seq_len, n_vars] for TimesNet.

    extract() → mean-pools over time → [B, d_model]   (emb_dim = d_model)
    forward() → head(extract(x))     → [B, n_classes]
    head      → nn.Linear(d_model, n_classes)
    """

    def __init__(self, seq_len: int, n_vars: int, n_classes: int,
                 d_model: int = 128, e_layers: int = 2,
                 d_ff: int = 256, top_k: int = 3, num_kernels: int = 6,
                 dropout: float = 0.1):
        super().__init__()
        self.enc_embedding = DataEmbedding(n_vars, d_model, embed_type='fixed',
                                           freq='h', dropout=dropout)
        self.blocks     = nn.ModuleList([
            TimesBlock(seq_len, d_model, d_ff, top_k, num_kernels)
            for _ in range(e_layers)])
        self.layer_norm = nn.LayerNorm(d_model)
        self.act        = nn.GELU()
        self.dropout    = nn.Dropout(dropout)
        self.head       = nn.Linear(d_model, n_classes)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """[B, n_vars, seq_len] → [B, d_model]"""
        x = x.permute(0, 2, 1)                      # [B, T, C]
        x = self.enc_embedding(x, None)              # [B, T, d_model]
        for blk in self.blocks:
            x = self.layer_norm(blk(x))
        x = self.act(x)
        x = self.dropout(x)
        return x.mean(dim=1)                         # [B, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract(x))


# ── Dataset configurations ─────────────────────────────────────────────────────

@dataclass
class DatasetCfg:
    seq_len:   int
    n_vars:    int
    n_classes: int
    name:      str


DATASET_CFGS = {
    "fd":    DatasetCfg(seq_len=5120, n_vars=1, n_classes=3, name="fd"),
    "hhar":  DatasetCfg(seq_len=128,  n_vars=3, n_classes=6, name="hhar"),
    "sleep": DatasetCfg(seq_len=3000, n_vars=1, n_classes=5, name="sleep"),
}


def make_timesnet(dataset: str, d_model: int = 128, e_layers: int = 2,
                  d_ff: int = 256, top_k: int = 3,
                  num_kernels: int = 6) -> TimesNetBackbone:
    cfg = DATASET_CFGS[dataset]
    return TimesNetBackbone(
        seq_len=cfg.seq_len, n_vars=cfg.n_vars, n_classes=cfg.n_classes,
        d_model=d_model, e_layers=e_layers, d_ff=d_ff,
        top_k=top_k, num_kernels=num_kernels)


def load_timesnet(ckpt_path: str | Path, dataset: str,
                  d_model: int = 128, e_layers: int = 2,
                  d_ff: int = 256, top_k: int = 3, num_kernels: int = 6,
                  device: str | torch.device = "cpu") -> TimesNetBackbone:
    """Load a pre-trained TimesNet checkpoint (format: {'model_state_dict': ...})."""
    model = make_timesnet(dataset, d_model, e_layers, d_ff, top_k, num_kernels)
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd    = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)
