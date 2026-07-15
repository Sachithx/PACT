"""
transfer_fd_tslanet.py — TSLANet cross-domain transfer on FD-A / FD-B / FD-C / FD-D.

For every ordered (src, tgt) pair (12 total):

  domain_shift  — source-trained TSLANet classifier applied directly to
                  target test data WITHOUT retraining.

  transfer      — source TSLANet backbone (frozen) + fresh linear head
                  trained on target train/val, evaluated on target test.

Source models are trained (optional pretrain + finetune) once per dataset
and cached as checkpoints under:
  <ckpt_dir>/tslanet_<dataset>_ps<patch_size>_d<depth>_dim<emb_dim>.pt

Usage:
  python transfer_fd_tslanet.py                          # all 12 pairs
  python transfer_fd_tslanet.py --src FD-A --tgt FD-B   # single pair
  python transfer_fd_tslanet.py --label_fraction 0.05   # 5%% target labels
  python transfer_fd_tslanet.py --no_pretrain            # skip MAE pretraining
"""

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

# DropPath and trunc_normal_ are from timm in the original TSLANet, but timm
# can't be imported in this environment due to a libstdc++ version mismatch.
# Both utilities are trivially re-implemented below using only PyTorch.

class DropPath(nn.Module):
    """Stochastic depth (drop entire sample paths during training)."""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand  = torch.rand(shape, dtype=x.dtype, device=x.device)
        rand  = torch.floor(rand + keep_prob)
        return x / keep_prob * rand


def trunc_normal_(tensor: torch.Tensor, std: float = .02,
                  mean: float = 0., a: float = -2., b: float = 2.) -> torch.Tensor:
    return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


# utils.py (random_masking_3D, str2bool) lives in the same directory
_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
from utils import random_masking_3D, str2bool


# ── Constants ─────────────────────────────────────────────────────────────────

FD_DATASETS  = ["FD-A", "FD-B", "FD-C", "FD-D"]
FD_SUFFIX    = {"FD-A": "a", "FD-B": "b", "FD-C": "c", "FD-D": "d"}
FD_N_CLASSES = 3
FD_SEQ_LEN   = 5120
FD_CHANNELS  = 1


# ── Model (no global args dependency) ────────────────────────────────────────

class ICBModule(nn.Module):
    """Inverted Convolutional Block — matches original TSLANet exactly."""
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
        # cross-gating: raw output of each branch gated by dropped activation of the other
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
        x = torch.fft.irfft(x_w, n=N, dim=1, norm='ortho')
        return x.to(x_in.dtype).view(B, N, C)


class TSLANetLayer(nn.Module):
    def __init__(self, dim, mlp_ratio=3., drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, use_icb=True, use_asb=True,
                 adaptive_filter=True):
        super().__init__()
        self.use_icb   = use_icb
        self.use_asb   = use_asb
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
    """Self-contained TSLANet that takes explicit config (no global args)."""

    def __init__(self, seq_len, num_channels, num_classes,
                 emb_dim=128, depth=2, patch_size=8,
                 dropout_rate=0.15, masking_ratio=0.4,
                 use_icb=True, use_asb=True, adaptive_filter=True):
        super().__init__()
        self.emb_dim       = emb_dim
        self.masking_ratio = masking_ratio

        self.patch_embed = PatchEmbed(seq_len, patch_size, num_channels, emb_dim)
        num_patches      = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, emb_dim))
        self.pos_drop  = nn.Dropout(p=dropout_rate)

        dpr = [x.item() for x in torch.linspace(0, dropout_rate, depth)]
        self.blocks = nn.ModuleList([
            TSLANetLayer(
                dim=emb_dim, drop=dropout_rate, drop_path=dpr[i],
                use_icb=use_icb, use_asb=use_asb, adaptive_filter=adaptive_filter,
            )
            for i in range(depth)
        ])
        self.head = nn.Linear(emb_dim, num_classes)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def extract(self, x):
        """Backbone: patch embed → positional encoding → blocks → mean pool."""
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


# ── FD Dataset ────────────────────────────────────────────────────────────────

class FDDataset(torch.utils.data.Dataset):
    """Loads a single split from an FD-{A,B,C,D} .pt file.

    File naming convention: <split>_<letter>.pt  (e.g. train_a.pt)
    Stored as {"samples": [N, T], "labels": [N]}.
    Returns (x [1, T], y) — channel-first for TSLANet.
    """

    def __init__(self, dataset: str, split: str, data_root: Path):
        suffix = FD_SUFFIX[dataset]
        path   = data_root / dataset / f"{split}_{suffix}.pt"
        d      = torch.load(path, map_location="cpu", weights_only=False)
        x      = d["samples"]
        y      = d["labels"]
        if x.dim() == 2:
            x = x.unsqueeze(1)   # [N, T] → [N, 1, T]
        self.x = x.float()
        self.y = y.long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


# ── Stratified subset (same logic as PRECEPT train_pipeline.py) ───────────────

def stratified_subset(
    x: torch.Tensor,
    labels: torch.Tensor,
    fraction: float,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a stratified subset keeping `fraction` of samples per class (min 1)."""
    if fraction >= 1.0:
        return x, labels
    rng  = torch.Generator().manual_seed(seed)
    keep = []
    for c in labels.unique():
        idx    = (labels == c).nonzero(as_tuple=True)[0]
        n_keep = max(1, round(len(idx) * fraction))
        perm   = torch.randperm(len(idx), generator=rng)
        keep.append(idx[perm[:n_keep]])
    keep = torch.cat(keep)
    return x[keep], labels[keep]


# ── Training utilities ────────────────────────────────────────────────────────

def pretrain_epoch(model: TSLANet, loader: DataLoader,
                   optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total, n = 0., 0
    for x, _ in loader:
        x    = x.to(device)
        pred, target, mask = model.pretrain_fwd(x)
        loss = ((pred - target) ** 2).mean(-1)
        loss = (loss * mask).sum() / mask.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(x)
        n     += len(x)
    return total / n


@torch.no_grad()
def pretrain_eval(model: TSLANet, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, n = 0., 0
    for x, _ in loader:
        x    = x.to(device)
        pred, target, mask = model.pretrain_fwd(x)
        loss = ((pred - target) ** 2).mean(-1)
        loss = (loss * mask).sum() / mask.sum()
        total += loss.item() * len(x)
        n     += len(x)
    return total / n


def finetune_epoch(model: TSLANet, loader: DataLoader,
                   optimizer: torch.optim.Optimizer,
                   criterion: nn.Module, device: torch.device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@torch.no_grad()
def eval_loss(model: TSLANet, loader: DataLoader,
              criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total, n = 0., 0
    for x, y in loader:
        x, y  = x.to(device), y.to(device)
        total += criterion(model(x), y).item() * len(x)
        n     += len(x)
    return total / n


@torch.no_grad()
def evaluate(model: TSLANet, loader: DataLoader,
             device: torch.device) -> Tuple[float, float]:
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        all_preds.extend(model(x.to(device)).argmax(1).cpu().tolist())
        all_labels.extend(y.tolist())
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / max(len(all_labels), 1)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1


# ── Model construction / checkpoint helpers ───────────────────────────────────

def _make_model(args) -> TSLANet:
    return TSLANet(
        seq_len=FD_SEQ_LEN, num_channels=FD_CHANNELS, num_classes=FD_N_CLASSES,
        emb_dim=args.emb_dim, depth=args.depth, patch_size=args.patch_size,
        dropout_rate=args.dropout_rate, masking_ratio=args.masking_ratio,
        use_icb=args.ICB, use_asb=args.ASB, adaptive_filter=args.adaptive_filter,
    )


def _ckpt_path(dataset: str, args) -> Path:
    return (Path(args.ckpt_dir) /
            f"tslanet_{dataset}_ps{args.patch_size}_d{args.depth}_dim{args.emb_dim}.pt")


def _best_state(model: nn.Module) -> Dict:
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


# ── Source model: train or load ───────────────────────────────────────────────

def train_source(dataset: str, args, device: torch.device) -> TSLANet:
    """Train TSLANet on one FD domain (pretrain + finetune). Caches checkpoint."""
    ckpt_path = _ckpt_path(dataset, args)

    if ckpt_path.exists() and not args.retrain:
        print(f"  Loading checkpoint: {ckpt_path.name}")
        model = _make_model(args).to(device)
        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        return model

    print(f"\n{'─'*60}")
    print(f"  Training TSLANet on {dataset}")
    print(f"{'─'*60}")

    tr_ds     = FDDataset(dataset, "train", Path(args.data_root))
    va_ds     = FDDataset(dataset, "val",   Path(args.data_root))
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)

    model = _make_model(args).to(device)

    # ── Phase 1: Masked pretraining ───────────────────────────────────────────
    if args.pretrain:
        print(f"  Phase 1/2: Pretraining ({args.pretrain_epochs} epochs)…")
        pre_opt    = torch.optim.AdamW(model.parameters(), lr=args.pretrain_lr, weight_decay=1e-4)
        best_val   = float("inf")
        best       = None
        for ep in range(1, args.pretrain_epochs + 1):
            tr_loss = pretrain_epoch(model, tr_loader, pre_opt, device)
            va_loss = pretrain_eval(model, va_loader, device)
            if va_loss < best_val:
                best_val = va_loss
                best     = _best_state(model)
            if ep % 10 == 0:
                print(f"    ep {ep:3d}/{args.pretrain_epochs}"
                      f"  tr={tr_loss:.4f}  va={va_loss:.4f}")
        model.load_state_dict({k: v.to(device) for k, v in best.items()})
        print(f"  Pretraining done (best val loss {best_val:.4f})")

    # ── Phase 2: Classifier fine-tuning ──────────────────────────────────────
    print(f"  Phase 2/2: Fine-tuning ({args.cls_epochs} epochs)…")
    criterion = nn.CrossEntropyLoss()
    cls_opt   = torch.optim.AdamW(model.parameters(), lr=args.cls_lr, weight_decay=1e-4)
    best_val  = float("inf")
    best      = None
    for ep in range(1, args.cls_epochs + 1):
        finetune_epoch(model, tr_loader, cls_opt, criterion, device)
        va_loss = eval_loss(model, va_loader, criterion, device)
        if va_loss < best_val:
            best_val = va_loss
            best     = _best_state(model)
        if ep % 10 == 0:
            va_acc, va_f1 = evaluate(model, va_loader, device)
            print(f"    ep {ep:3d}/{args.cls_epochs}"
                  f"  va_loss={va_loss:.4f}  va_acc={va_acc:.4f}  va_f1={va_f1:.4f}")

    model.load_state_dict({k: v.to(device) for k, v in best.items()})
    va_acc, va_f1 = evaluate(model, va_loader, device)
    print(f"  Fine-tuning done  val acc={va_acc:.4f}  val F1={va_f1:.4f}")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"dataset": dataset, "model_state_dict": model.state_dict()}, ckpt_path)
    print(f"  Saved → {ckpt_path}")
    return model


# ── Per-pair transfer runner ──────────────────────────────────────────────────

def run_pair(src: str, tgt: str, src_model: TSLANet,
             args, device: torch.device) -> Dict:
    print(f"\n{'='*64}")
    print(f"  {src}  →  {tgt}")
    print(f"{'='*64}")

    te_ds     = FDDataset(tgt, "test",  Path(args.data_root))
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False)

    # ── Exp 1: Domain shift (no retraining) ───────────────────────────────────
    print(f"\n  [Exp 1] Domain shift — {src} classifier on {tgt} test (no retrain)")
    ds_acc, ds_f1 = evaluate(src_model, te_loader, device)
    print(f"    acc={ds_acc:.4f}  macro-F1={ds_f1:.4f}")

    # ── Exp 2: Transfer — frozen backbone + fresh head on target ──────────────
    print(f"\n  [Exp 2] Transfer — frozen {src} backbone + new head on {tgt}")
    tr_ds = FDDataset(tgt, "train", Path(args.data_root))
    va_ds = FDDataset(tgt, "val",   Path(args.data_root))

    # Stratified subsample of training data
    x_sub, y_sub = stratified_subset(tr_ds.x, tr_ds.y, args.label_fraction, args.seed)
    if args.label_fraction < 1.0:
        print(f"    [label_fraction={args.label_fraction}]"
              f" {len(y_sub)}/{len(tr_ds)} training samples")

    sub_loader = DataLoader(TensorDataset(x_sub, y_sub),
                            batch_size=args.batch_size, shuffle=True, drop_last=False)
    va_loader  = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)

    # Build transfer model from source weights; freeze backbone; replace head
    tfer = _make_model(args).to(device)
    tfer.load_state_dict(src_model.state_dict())
    tfer.requires_grad_(False)
    tfer.head = nn.Linear(tfer.emb_dim, FD_N_CLASSES).to(device)  # fresh, unfrozen

    criterion = nn.CrossEntropyLoss()
    head_opt  = torch.optim.AdamW(tfer.head.parameters(), lr=args.cls_lr, weight_decay=1e-4)

    best_val  = float("inf")
    best_head = _best_state(tfer.head)

    tfer.eval()  # backbone in eval (frozen); head is Linear so mode doesn't matter
    for ep in range(1, args.cls_epochs + 1):
        for x, y in sub_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(tfer(x), y)
            head_opt.zero_grad()
            loss.backward()
            head_opt.step()
        va_loss = eval_loss(tfer, va_loader, criterion, device)
        if va_loss < best_val:
            best_val  = va_loss
            best_head = _best_state(tfer.head)

    tfer.head.load_state_dict({k: v.to(device) for k, v in best_head.items()})
    xfer_acc, xfer_f1 = evaluate(tfer, te_loader, device)
    print(f"    acc={xfer_acc:.4f}  macro-F1={xfer_f1:.4f}")

    return {
        "src": src, "tgt": tgt,
        "domain_shift_acc": ds_acc,   "domain_shift_f1": ds_f1,
        "transfer_acc":     xfer_acc, "transfer_f1":     xfer_f1,
    }


# ── Results display ───────────────────────────────────────────────────────────

def print_table(results: List[Dict]):
    w = 13
    hdr = f"{'src→tgt':<10}{'domain_shift':>{w}}{'transfer':>{w}}"
    print(f"\n{'='*40}")
    print("  TSLANet Transfer Results  (test accuracy)")
    print(f"{'='*40}")
    print(hdr)
    print("-" * 40)
    for r in results:
        pair = f"{r['src']}→{r['tgt']}"
        print(f"{pair:<10}  {r['domain_shift_acc']:>{w-2}.4f}  {r['transfer_acc']:>{w-2}.4f}")
    print(f"{'='*40}")


def save_csv(results: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["src", "tgt",
              "domain_shift_acc", "domain_shift_f1",
              "transfer_acc",     "transfer_f1"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"\n  Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",  default=None, choices=FD_DATASETS,
                        help="Run only this source dataset (default: all)")
    parser.add_argument("--tgt",  default=None, choices=FD_DATASETS,
                        help="Run only this target dataset (default: all)")
    parser.add_argument("--data_root",  default="../../PRECEPT/dataset",
                        help="Root dir containing FD-A/B/C/D sub-folders")
    parser.add_argument("--ckpt_dir",   default="output/tslanet",
                        help="Where to save/load source model checkpoints")
    parser.add_argument("--results_dir", default="results/transfer",
                        help="Directory for CSV output")

    # Source training
    parser.add_argument("--pretrain",        action="store_true", default=True)
    parser.add_argument("--no_pretrain",     dest="pretrain", action="store_false")
    parser.add_argument("--pretrain_epochs", type=int,   default=50)
    parser.add_argument("--pretrain_lr",     type=float, default=1e-3)
    parser.add_argument("--cls_epochs",      type=int,   default=100)
    parser.add_argument("--cls_lr",          type=float, default=1e-3)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--retrain",         action="store_true",
                        help="Ignore existing checkpoints and retrain from scratch")

    # Model architecture (must match the paper defaults)
    parser.add_argument("--emb_dim",         type=int,   default=128)
    parser.add_argument("--depth",           type=int,   default=2)
    parser.add_argument("--patch_size",      type=int,   default=8)
    parser.add_argument("--dropout_rate",    type=float, default=0.15)
    parser.add_argument("--masking_ratio",   type=float, default=0.4)
    parser.add_argument("--ICB",             type=str2bool, default=True)
    parser.add_argument("--ASB",             type=str2bool, default=True)
    parser.add_argument("--adaptive_filter", type=str2bool, default=True)

    # Transfer
    parser.add_argument("--label_fraction",  type=float, default=1.0,
                        help="Fraction of target train labels to use (e.g. 0.05 for 5%%)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # Build pair list and deduplicate source domains
    if args.src and args.tgt:
        if args.src == args.tgt:
            raise ValueError("--src and --tgt must differ")
        pairs = [(args.src, args.tgt)]
        srcs  = [args.src]
    elif args.src:
        pairs = [(args.src, t) for t in FD_DATASETS if t != args.src]
        srcs  = [args.src]
    elif args.tgt:
        pairs = [(s, args.tgt) for s in FD_DATASETS if s != args.tgt]
        srcs  = [s for s in FD_DATASETS if s != args.tgt]
    else:
        pairs = [(s, t) for s in FD_DATASETS for t in FD_DATASETS if s != t]
        srcs  = FD_DATASETS

    # Train / load one source model per source domain
    src_models: Dict[str, TSLANet] = {}
    for src in srcs:
        src_models[src] = train_source(src, args, device)

    # Run transfer experiments
    results = []
    for src, tgt in pairs:
        try:
            results.append(run_pair(src, tgt, src_models[src], args, device))
        except Exception as exc:
            print(f"  ERROR {src}→{tgt}: {exc}")
            import traceback; traceback.print_exc()

    if results:
        print_table(results)
        out_name = "transfer_fd_tslanet"
        if args.label_fraction < 1.0:
            pct      = int(args.label_fraction * 100)
            out_name += f"_{pct}pct"
        save_csv(results, Path(args.results_dir) / f"{out_name}.csv")


if __name__ == "__main__":
    main()
