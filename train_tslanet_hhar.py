"""
train_tslanet_hhar.py — Train TSLANet source-domain backbones on HHAR.

Saves checkpoints to:
  checkpoints/tslanet_hhar/tslanet_hhar_u{N}_ps8_d2_dim128.pt

Usage:
  python train_tslanet_hhar.py                   # all 9 users
  python train_tslanet_hhar.py --users 0 1 6     # specific users
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from backbone.tslanet import make_tslanet
from datasets import HHARDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--users",      type=int, nargs="+", default=list(range(9)))
    p.add_argument("--data_root",  default=".")
    p.add_argument("--d_model",    type=int, default=128)
    p.add_argument("--depth",      type=int, default=2)
    p.add_argument("--patch_size", type=int, default=8)
    p.add_argument("--epochs",     type=int, default=100)
    p.add_argument("--batch",      type=int, default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir",    default="checkpoints/tslanet_hhar")
    return p.parse_args()


def train_one(user_id, args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    train_ds = HHARDataset(user_id, "train", args.data_root)
    val_ds   = HHARDataset(user_id, "val",   args.data_root)
    test_ds  = HHARDataset(user_id, "test",  args.data_root)

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  drop_last=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False)
    test_ld  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False)

    model = make_tslanet("hhar", args.d_model, args.depth, args.patch_size).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit  = nn.CrossEntropyLoss()

    best_val_f1, best_sd = 0., None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        sched.step()

        if epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for x, y in val_ld:
                    preds.append(model(x.to(device)).argmax(1).cpu())
                    trues.append(y)
            val_f1 = f1_score(torch.cat(trues), torch.cat(preds), average="macro")
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  u{user_id} epoch {epoch:3d}  val_f1={val_f1:.4f}  best={best_val_f1:.4f}")

    model.load_state_dict(best_sd)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in test_ld:
            preds.append(model(x.to(device)).argmax(1).cpu())
            trues.append(y)
    test_f1 = f1_score(torch.cat(trues), torch.cat(preds), average="macro")
    print(f"  u{user_id}  TEST f1={test_f1:.4f}")
    return best_sd, test_f1


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = f"tslanet_hhar_u{{u}}_ps{args.patch_size}_d{args.depth}_dim{args.d_model}.pt"

    for u in args.users:
        print(f"\n{'='*50}\nTraining user {u}\n{'='*50}")
        sd, f1 = train_one(u, args)
        ckpt_path = out_dir / ckpt_name.format(u=u)
        torch.save({"user_id": u, "model_state_dict": sd, "test_f1": f1}, ckpt_path)
        print(f"  Saved → {ckpt_path}")


if __name__ == "__main__":
    main()
