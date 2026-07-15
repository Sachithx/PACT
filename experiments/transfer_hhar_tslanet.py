"""
transfer_hhar_tslanet.py — TSLANet training / loading for HHAR cross-user domains.

Analogous to transfer_fd_tslanet.py but for HHAR:
  - 9 user domains (0–8, maps a→0 … i→8)
  - 6 activity classes: bike, sit, stand, walk, stairsup, stairsdown
  - 3-channel accelerometer input (x, y, z)
  - Sequence length 128 (window-based preprocessing)
  - patch_size=8, stride=4  → 31 patches

Checkpoints cached at:
  TSLANet/Classification/output/tslanet_hhar/tslanet_hhar_u{N}_ps8_d2_dim128.pt

Usage (as a module):
  import importlib.util, sys
  sys.path.insert(0, "TSLANet/Classification")
  spec = importlib.util.spec_from_file_location("tht", "transfer_hhar_tslanet.py")
  tht  = importlib.util.module_from_spec(spec); spec.loader.exec_module(tht)
  model = tht.train_source(user_id, args, device)    # user_id: int 0–8
  ds    = tht.HHARDataset(user_id, "train", data_root)
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

_here = Path(__file__).parent
if str(_here / "TSLANet" / "Classification") not in sys.path:
    sys.path.insert(0, str(_here / "TSLANet" / "Classification"))
from utils import random_masking_3D, str2bool   # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────

HHAR_N_USERS   = 9
HHAR_N_CLASSES = 6
HHAR_SEQ_LEN   = 128
HHAR_CHANNELS  = 3
HHAR_CLASS_NAMES = ["bike", "sit", "stand", "walk", "stairsup", "stairsdown"]


# ── TSLANet (identical implementation, different constants) ────────────────────

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        r = torch.rand(shape, dtype=x.dtype, device=x.device)
        return x / keep * torch.floor(r + keep)


def trunc_normal_(t, std=.02, mean=0., a=-2., b=2.):
    return torch.nn.init.trunc_normal_(t, mean=mean, std=std, a=a, b=b)


class ICBModule(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop  = nn.Dropout(drop)
        self.act   = nn.GELU()

    def forward(self, x):
        x  = x.transpose(1, 2)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = self.conv3(x1 * self.drop(self.act(x2)) + x2 * self.drop(self.act(x1)))
        return out.transpose(1, 2)


class PatchEmbed(nn.Module):
    def __init__(self, seq_len, patch_size=8, in_chans=3, embed_dim=128):
        super().__init__()
        stride = patch_size // 2
        self.num_patches = int((seq_len - patch_size) / stride + 1)
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class AdaptiveSpectralBlock(nn.Module):
    def __init__(self, dim, adaptive_filter=True):
        super().__init__()
        self.adaptive_filter  = adaptive_filter
        self.complex_weight   = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight_h = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight,   std=.02)
        trunc_normal_(self.complex_weight_h, std=.02)
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


class TSLANetLayer(nn.Module):
    def __init__(self, dim, mlp_ratio=3., drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, use_icb=True, use_asb=True,
                 adaptive_filter=True):
        super().__init__()
        self.use_icb   = use_icb;  self.use_asb = use_asb
        self.norm1     = norm_layer(dim)
        self.asb       = AdaptiveSpectralBlock(dim, adaptive_filter)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2     = norm_layer(dim)
        self.icb       = ICBModule(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        if self.use_icb and self.use_asb:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        elif self.use_icb:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        elif self.use_asb:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        return x


class TSLANet(nn.Module):
    def __init__(self, seq_len, num_channels, num_classes,
                 emb_dim=128, depth=2, patch_size=8,
                 dropout_rate=0.15, masking_ratio=0.4,
                 use_icb=True, use_asb=True, adaptive_filter=True):
        super().__init__()
        self.emb_dim       = emb_dim
        self.masking_ratio = masking_ratio
        self.patch_embed   = PatchEmbed(seq_len, patch_size, num_channels, emb_dim)
        num_patches        = self.patch_embed.num_patches
        self.pos_embed     = nn.Parameter(torch.zeros(1, num_patches, emb_dim))
        self.pos_drop      = nn.Dropout(p=dropout_rate)
        dpr = [x.item() for x in torch.linspace(0, dropout_rate, depth)]
        self.blocks = nn.ModuleList([
            TSLANetLayer(dim=emb_dim, drop=dropout_rate, drop_path=dpr[i],
                         use_icb=use_icb, use_asb=use_asb, adaptive_filter=adaptive_filter)
            for i in range(depth)])
        self.head = nn.Linear(emb_dim, num_classes)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)

    def extract(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        return x.mean(1)

    def forward(self, x):
        return self.head(self.extract(x))

    def pretrain_fwd(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        x_masked, _, mask, _ = random_masking_3D(x, self.masking_ratio)
        mask = mask.bool()
        for blk in self.blocks:
            x_masked = blk(x_masked)
        return x_masked, x, mask


# ── Dataset ────────────────────────────────────────────────────────────────────

class HHARDataset(torch.utils.data.Dataset):
    """Loads one split for one HHAR user domain.

    File: dataset/HHAR_processed/{split}_{user_id}.pt
    Returns (x [3, 128], y).
    """
    def __init__(self, user_id: int, split: str, data_root: Path):
        path = Path(data_root) / "HHAR_processed" / f"{split}_{user_id}.pt"
        d    = torch.load(path, map_location="cpu", weights_only=False)
        self.x = d["samples"].float()   # [N, 3, 128]
        self.y = d["labels"].long()     # [N]

    def __len__(self):   return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_model(args) -> TSLANet:
    return TSLANet(
        seq_len=HHAR_SEQ_LEN, num_channels=HHAR_CHANNELS, num_classes=HHAR_N_CLASSES,
        emb_dim=args.emb_dim, depth=args.depth, patch_size=args.patch_size,
        dropout_rate=args.dropout_rate, masking_ratio=args.masking_ratio,
        use_icb=args.ICB, use_asb=args.ASB, adaptive_filter=args.adaptive_filter,
    )


def _ckpt_path(user_id: int, args) -> Path:
    return (Path(args.ckpt_dir) / "tslanet_hhar" /
            f"tslanet_hhar_u{user_id}_ps{args.patch_size}_d{args.depth}_dim{args.emb_dim}.pt")


def _best_state(model):
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, float]:
    model.eval()
    preds, lbls = [], []
    for x, y in loader:
        preds.extend(model(x.to(device)).argmax(1).cpu().tolist())
        lbls.extend(y.tolist())
    acc = sum(p == l for p, l in zip(preds, lbls)) / max(len(lbls), 1)
    f1  = f1_score(lbls, preds, average="macro", zero_division=0)
    return acc, f1


# ── Source model training ──────────────────────────────────────────────────────

def train_source(user_id: int, args, device: torch.device) -> TSLANet:
    """Train TSLANet on one HHAR user domain. Returns trained (frozen) model."""
    ckpt_path = _ckpt_path(user_id, args)

    if ckpt_path.exists() and not args.retrain:
        print(f"  Loading checkpoint: {ckpt_path.name}")
        model = _make_model(args).to(device)
        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        return model

    print(f"\n{'─'*60}")
    print(f"  Training TSLANet on HHAR user {user_id}")
    print(f"{'─'*60}")

    tr_ds     = HHARDataset(user_id, "train", Path(args.data_root))
    va_ds     = HHARDataset(user_id, "val",   Path(args.data_root))
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    model     = _make_model(args).to(device)

    # Phase 1: masked pretraining
    if args.pretrain:
        print(f"  Phase 1/2: Pretraining ({args.pretrain_epochs} epochs) …")
        opt      = torch.optim.AdamW(model.parameters(), lr=args.pretrain_lr, weight_decay=1e-4)
        best_val = float("inf"); best = None
        for ep in range(1, args.pretrain_epochs + 1):
            model.train()
            for x, _ in tr_loader:
                x = x.to(device)
                pred, tgt, mask = model.pretrain_fwd(x)
                loss = ((pred - tgt) ** 2).mean(-1)
                loss = (loss * mask).sum() / mask.sum()
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            va_loss = 0.; n = 0
            for x, _ in va_loader:
                x = x.to(device)
                with torch.no_grad():
                    pred, tgt, mask = model.pretrain_fwd(x)
                    l = ((pred - tgt) ** 2).mean(-1)
                    va_loss += ((l * mask).sum() / mask.sum()).item() * len(x); n += len(x)
            va_loss /= n
            if va_loss < best_val: best_val = va_loss; best = _best_state(model)
            if ep % 10 == 0: print(f"    ep {ep:3d}/{args.pretrain_epochs}  val_loss={va_loss:.4f}")
        model.load_state_dict({k: v.to(device) for k, v in best.items()})
        print(f"  Pretraining done (best val {best_val:.4f})")

    # Phase 2: classifier fine-tuning
    print(f"  Phase 2/2: Fine-tuning ({args.cls_epochs} epochs) …")
    opt      = torch.optim.AdamW(model.parameters(), lr=args.cls_lr, weight_decay=1e-4)
    best_val = float("inf"); best = None
    for ep in range(1, args.cls_epochs + 1):
        model.train()
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); criterion(model(x), y).backward(); opt.step()
        model.eval()
        va_loss = sum(criterion(model(x.to(device)), y.to(device)).item() * len(x)
                      for x, y in va_loader) / len(va_ds)
        if va_loss < best_val: best_val = va_loss; best = _best_state(model)
        if ep % 10 == 0:
            _, va_f1 = evaluate(model, va_loader, device)
            print(f"    ep {ep:3d}/{args.cls_epochs}  val_loss={va_loss:.4f}  val_f1={va_f1:.4f}")

    model.load_state_dict({k: v.to(device) for k, v in best.items()})
    _, va_f1 = evaluate(model, va_loader, device)
    print(f"  Fine-tuning done  val F1={va_f1:.4f}")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"user_id": user_id, "model_state_dict": model.state_dict()}, ckpt_path)
    print(f"  Saved → {ckpt_path}")
    return model
