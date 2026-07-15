"""TimesNet backbone (self-contained scaffold) for the PACT backbone ablation.

Reference: Wu et al., "TimesNet: Temporal 2D-Variation Modeling for General
Time Series Analysis" (ICLR 2023).

STATUS / SCAFFOLD NOTE
----------------------
This is a self-contained re-implementation of TimesNet adapted to the PACT
backbone contract:
    extract(x) -> [B, emb_dim]   (mean-pool over time of the encoder output)
    forward(x) -> [B, n_classes]
    head                          nn.Linear(emb_dim, n_classes)
Input convention: x is [B, n_vars, seq_len]  (same as TSLANet / PatchTST).

It differs from the original TimesNet classification head (which flattens
d_model * seq_len) so that it produces the fixed `emb_dim`-dimensional
pre-head embedding that the PACT adapters/gate require. It is fully trainable,
but **no TimesNet checkpoints ship with this repo yet** — train one with
`scripts/train_backbone.py --backbone timesnet ...` before running PACT on it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _fft_for_period(x, k=2):
    """x: [B, T, C]. Returns (period_list, amplitude_weights)."""
    xf = torch.fft.rfft(x, dim=1)
    freq = abs(xf).mean(0).mean(-1)
    freq[0] = 0
    _, top_list = torch.topk(freq, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class _InceptionBlock(nn.Module):
    """Multi-kernel 2D convolution block (Inception_Block_V1)."""

    def __init__(self, in_ch, out_ch, num_kernels=6):
        super().__init__()
        self.kernels = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=2 * i + 1, padding=i)
            for i in range(num_kernels)
        ])

    def forward(self, x):
        return torch.stack([k(x) for k in self.kernels], dim=-1).mean(-1)


class _TimesBlock(nn.Module):
    def __init__(self, seq_len, d_model, d_ff, top_k=3, num_kernels=6):
        super().__init__()
        self.seq_len = seq_len
        self.k = top_k
        self.conv = nn.Sequential(
            _InceptionBlock(d_model, d_ff, num_kernels),
            nn.GELU(),
            _InceptionBlock(d_ff, d_model, num_kernels),
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = _fft_for_period(x, self.k)
        res = []
        for i in range(self.k):
            period = max(int(period_list[i]), 1)
            if T % period != 0:
                length = (T // period + 1) * period
                pad = torch.zeros([B, length - T, N], device=x.device)
                out = torch.cat([x, pad], dim=1)
            else:
                length = T
                out = x
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :T, :])
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        return res + x


class _TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.conv = nn.Conv1d(c_in, d_model, kernel_size=3, padding=1,
                              padding_mode="circular", bias=False)

    def forward(self, x):  # x: [B, T, C]
        return self.conv(x.permute(0, 2, 1)).transpose(1, 2)


class TimesNetClassifier(nn.Module):
    """TimesNet for classification, PACT-compatible."""

    def __init__(self, seq_len: int, n_vars: int, n_classes: int,
                 d_model: int = 128, d_ff: int = 256, e_layers: int = 2,
                 top_k: int = 3, num_kernels: int = 6, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.emb_dim = d_model
        self.enc_embedding = _TokenEmbedding(n_vars, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            _TimesBlock(seq_len, d_model, d_ff, top_k, num_kernels)
            for _ in range(e_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """[B, n_vars, seq_len] -> [B, d_model] (mean over time)."""
        x = x.transpose(1, 2)                     # [B, T, C]
        enc = self.drop(self.enc_embedding(x) + self.pos_embed[:, : x.shape[1]])
        for blk in self.blocks:
            enc = self.layer_norm(blk(enc))
        return enc.mean(dim=1)                    # [B, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract(x))


def make_timesnet(seq_len, n_vars, n_classes, d_model=128, d_ff=256,
                  e_layers=2, top_k=3, num_kernels=6, dropout=0.1) -> TimesNetClassifier:
    return TimesNetClassifier(
        seq_len=seq_len, n_vars=n_vars, n_classes=n_classes, d_model=d_model,
        d_ff=d_ff, e_layers=e_layers, top_k=top_k, num_kernels=num_kernels,
        dropout=dropout)
