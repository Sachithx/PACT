"""
transfer_patchtst.py — Self-contained PatchTST backbone for PACT ablation.

Fully self-contained (no TSLib imports) to avoid utils.py naming conflicts.
Drop-in replacement for TSLANet across all 3 datasets.

PACT compatibility:
  extract(x) → [B, 128]  (mean-pool over vars × patches; always 128-dim)
  forward(x) → [B, n_classes]
  Input: x [B, n_vars, seq_len]  (same convention as TSLANet)

Hyperparams: d_model=128, n_heads=8, e_layers=3, d_ff=256, patch_len=16, stride=8

Checkpoints: TSLANet/Classification/output/patchtst/patchtst_{dataset}_{id}.pt
"""

import sys, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, random_split

_REPO = Path(__file__).parent


# ── Minimal Transformer (self-contained, no external deps) ────────────────────

class _MHA(nn.Module):
    """Multi-head self-attention."""
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h  = n_heads
        self.dk = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape; H = self.h; dk = self.dk
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, L, H, dk).transpose(1, 2)
        k = k.view(B, L, H, dk).transpose(1, 2)
        v = v.view(B, L, H, dk).transpose(1, 2)
        attn = self.drop((q @ k.transpose(-2, -1)) / math.sqrt(dk)).softmax(-1)
        return self.out((attn @ v).transpose(1, 2).reshape(B, L, D))


class _EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.attn  = _MHA(d_model, n_heads, dropout)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class _PatchEmbed(nn.Module):
    """Patch + positional embedding for multivariate time series."""
    def __init__(self, seq_len, d_model, patch_len, stride, dropout=0.0):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        self.pad       = nn.ReplicationPad1d((0, stride))
        self.proj      = nn.Linear(patch_len, d_model, bias=False)
        self.drop      = nn.Dropout(dropout)
        # Pre-compute n_patches and register positional embedding properly
        n_patches = (seq_len + stride - patch_len) // stride + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, C, L = x.shape
        x = self.pad(x)
        x = x.unfold(-1, self.patch_len, self.stride)           # [B, C, N, patch_len]
        N = x.shape[2]
        x = x.reshape(B * C, N, self.patch_len)
        x = self.proj(x)                                        # [B*C, N, d_model]
        return self.drop(x + self.pos_embed[:, :N]), C


# ── PatchTST classifier ────────────────────────────────────────────────────────

class PatchTSTClassifier(nn.Module):
    """
    Self-contained PatchTST for time-series classification.

    Input:  x [B, n_vars, seq_len]
    embed:  mean-pool over vars × patches → [B, d_model=128]
    output: [B, n_classes]
    """

    def __init__(self, seq_len: int, n_vars: int, n_classes: int,
                 d_model: int = 128, n_heads: int = 8, e_layers: int = 3,
                 d_ff: int = 256, dropout: float = 0.1,
                 patch_len: int = 16, stride: int = 8):
        super().__init__()
        self.n_vars  = n_vars
        self.d_model = d_model
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
                if m.bias is not None: nn.init.zeros_(m.bias)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """[B, n_vars, seq_len] → [B, d_model]  (mean over vars and patches)."""
        B, C, L = x.shape
        tok, n_vars = self.patch_embed(x)           # [B*C, N, d_model]
        tok = self.norm(self.blocks(tok))            # [B*C, N, d_model]
        tok = tok.mean(dim=1)                        # [B*C, d_model]  mean over patches
        tok = tok.reshape(B, n_vars, self.d_model)   # [B, C, d_model]
        return tok.mean(dim=1)                       # [B, d_model]    mean over vars

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.extract(x)))


# ── Dataset configs ────────────────────────────────────────────────────────────

_FD_CFG    = dict(seq_len=5120, n_vars=1, n_classes=3, batch=64,  name="fd")
_HHAR_CFG  = dict(seq_len=128,  n_vars=3, n_classes=6, batch=256, name="hhar")
_SLEEP_CFG = dict(seq_len=3000, n_vars=1, n_classes=5, batch=64,  name="sleep")


def _make_model(cfg: dict, d_model: int = 128) -> PatchTSTClassifier:
    return PatchTSTClassifier(
        seq_len=cfg["seq_len"], n_vars=cfg["n_vars"], n_classes=cfg["n_classes"],
        d_model=d_model, n_heads=8, e_layers=3, d_ff=256, dropout=0.1,
        patch_len=16, stride=8)


def _ckpt_path(cfg: dict, domain_id) -> Path:
    out = _REPO / "TSLANet" / "Classification" / "output" / "patchtst"
    out.mkdir(parents=True, exist_ok=True)
    return out / f"patchtst_{cfg['name']}_{domain_id}.pt"


@torch.no_grad()
def _eval_f1(model, loader, device) -> float:
    model.eval(); preds, lbls = [], []
    for x, y in loader:
        preds.extend(model(x.to(device)).argmax(1).cpu().tolist())
        lbls.extend(y.tolist())
    return f1_score(lbls, preds, average="macro", zero_division=0)


def _best_state(model):
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


# ── Training ───────────────────────────────────────────────────────────────────

def train_source(domain_id, cfg: dict, dataset_fn, device,
                 train_epochs: int = 100, lr: float = 1e-4,
                 retrain: bool = False) -> PatchTSTClassifier:
    """Train PatchTST on one source domain. Caches checkpoint."""
    ckpt  = _ckpt_path(cfg, domain_id)
    model = _make_model(cfg).to(device)

    if ckpt.exists() and not retrain:
        print(f"  Loading PatchTST checkpoint: {ckpt.name}")
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model_state_dict"])
        return model

    print(f"\n{'─'*60}")
    print(f"  Training PatchTST — {cfg['name'].upper()} domain {domain_id}")
    print(f"  params={sum(p.numel() for p in model.parameters()):,}")
    print(f"{'─'*60}")

    full_ds = dataset_fn(domain_id, "train")
    val_n   = max(1, int(0.15 * len(full_ds)))
    tr_ds, va_ds = random_split(full_ds, [len(full_ds)-val_n, val_n],
                                generator=torch.Generator().manual_seed(42))
    tr_ld  = DataLoader(tr_ds, batch_size=cfg["batch"], shuffle=True,  drop_last=True)
    va_ld  = DataLoader(va_ds, batch_size=cfg["batch"], shuffle=False)
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_epochs)
    crit   = nn.CrossEntropyLoss()
    best_val, best = float("inf"), None

    for ep in range(1, train_epochs + 1):
        model.train()
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            va_loss = sum(crit(model(x.to(device)), y.to(device)).item() * len(x)
                          for x, y in va_ld) / len(va_ds)
        if va_loss < best_val:
            best_val = va_loss; best = _best_state(model)
        if ep % 20 == 0:
            f1 = _eval_f1(model, va_ld, device)
            print(f"    ep {ep:3d}/{train_epochs}  val_loss={va_loss:.4f}  val_f1={f1:.4f}")

    model.load_state_dict({k: v.to(device) for k, v in best.items()})
    f1 = _eval_f1(model, va_ld, device)
    print(f"  Done — val F1={f1:.4f}")
    torch.save({"domain_id": domain_id, "model_state_dict": model.state_dict()}, ckpt)
    print(f"  Saved → {ckpt}")
    return model
