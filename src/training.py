"""Source-backbone training / loading.

Trains one backbone on one source domain and caches the checkpoint (reusing an
existing one if present). TSLANet uses masked-autoencoding pretraining followed
by supervised fine-tuning; PatchTST / TimesNet use plain supervised training
with a cosine schedule.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, random_split

from .data.datasets import get_spec
from .models.factory import build_backbone, checkpoint_path


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, lbls = [], []
    for x, y in loader:
        preds.extend(model(x.to(device)).argmax(1).cpu().tolist())
        lbls.extend(y.tolist())
    acc = sum(p == l for p, l in zip(preds, lbls)) / max(len(lbls), 1)
    f1 = f1_score(lbls, preds, average="macro", zero_division=0)
    return acc, f1


def _best_state(model):
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def _train_val_loaders(spec, dataset, domain_id, data_root, batch, seed=42):
    """Return (train_loader, val_loader). Sleep-EDF carves val out of train."""
    if spec.has_val:
        try:
            tr = spec.get_dataset(domain_id, "train", data_root)
            va = spec.get_dataset(domain_id, "val", data_root)
        except FileNotFoundError:
            full = spec.get_dataset(domain_id, "train", data_root)
            val_n = max(1, int(0.15 * len(full)))
            tr, va = random_split(full, [len(full) - val_n, val_n],
                                  generator=torch.Generator().manual_seed(seed))
    else:
        full = spec.get_dataset(domain_id, "train", data_root)
        val_n = max(1, int(0.15 * len(full)))
        tr, va = random_split(full, [len(full) - val_n, val_n],
                              generator=torch.Generator().manual_seed(seed))
    tr_ld = DataLoader(tr, batch_size=batch, shuffle=True, drop_last=True)
    va_ld = DataLoader(va, batch_size=batch, shuffle=False)
    return tr_ld, va_ld


def _mae_pretrain(model, tr_ld, va_ld, device, epochs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val, best = float("inf"), None
    for ep in range(1, epochs + 1):
        model.train()
        for x, _ in tr_ld:
            x = x.to(device)
            pred, tgt, mask = model.pretrain_fwd(x)
            loss = ((pred - tgt) ** 2).mean(-1)
            loss = (loss * mask).sum() / mask.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        va, n = 0.0, 0
        with torch.no_grad():
            for x, _ in va_ld:
                x = x.to(device)
                pred, tgt, mask = model.pretrain_fwd(x)
                l = ((pred - tgt) ** 2).mean(-1)
                va += ((l * mask).sum() / mask.sum()).item() * len(x); n += len(x)
        va /= max(n, 1)
        if va < best_val:
            best_val, best = va, _best_state(model)
        if ep % 10 == 0:
            print(f"    [pretrain] ep {ep:3d}/{epochs}  val_loss={va:.4f}")
    if best is not None:
        model.load_state_dict({k: v.to(device) for k, v in best.items()})
    print(f"  Pretraining done (best val {best_val:.4f})")


def _supervised(model, tr_ld, va_ld, device, epochs, lr, cosine=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
             if cosine else None)
    crit = nn.CrossEntropyLoss()
    best_val, best = float("inf"), None
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        if sched:
            sched.step()
        model.eval()
        with torch.no_grad():
            tot, n = 0.0, 0
            for x, y in va_ld:
                x, y = x.to(device), y.to(device)
                tot += crit(model(x), y).item() * len(x); n += len(x)
            va = tot / max(n, 1)
        if va < best_val:
            best_val, best = va, _best_state(model)
        if ep % 20 == 0:
            _, f1 = evaluate(model, va_ld, device)
            print(f"    [finetune] ep {ep:3d}/{epochs}  val_loss={va:.4f}  val_f1={f1:.4f}")
    if best is not None:
        model.load_state_dict({k: v.to(device) for k, v in best.items()})


def train_or_load_source(backbone, dataset, domain_id, ckpt_root, data_root,
                         device, retrain=False, pretrain=True,
                         pretrain_epochs=50, pretrain_lr=1e-3,
                         cls_epochs=100, cls_lr=1e-3, batch=None, seed=42):
    """Train (or load cached) a source backbone for one domain. Returns the model."""
    spec = get_spec(dataset)
    ckpt = checkpoint_path(ckpt_root, backbone, dataset, domain_id)

    model = build_backbone(backbone, spec).to(device)
    if ckpt.exists() and not retrain:
        print(f"  Loading checkpoint: {ckpt}")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        return model

    batch = batch or spec.default_batch
    print(f"\n  Training {backbone} on {dataset} domain {domain_id} "
          f"({spec.domain_label(domain_id)})")
    tr_ld, va_ld = _train_val_loaders(spec, dataset, domain_id, data_root, batch, seed)

    if backbone == "tslanet":
        if pretrain:
            _mae_pretrain(model, tr_ld, va_ld, device, pretrain_epochs, pretrain_lr)
        _supervised(model, tr_ld, va_ld, device, cls_epochs, cls_lr, cosine=False)
    else:
        # PatchTST / TimesNet: supervised only, cosine schedule, lower LR default.
        _supervised(model, tr_ld, va_ld, device, cls_epochs,
                    lr=min(cls_lr, 1e-4), cosine=True)

    acc, f1 = evaluate(model, va_ld, device)
    print(f"  Done — val acc={acc:.4f}  val F1={f1:.4f}")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"dataset": dataset, "domain_id": domain_id,
                "model_state_dict": model.state_dict()}, ckpt)
    print(f"  Saved -> {ckpt}")
    return model
