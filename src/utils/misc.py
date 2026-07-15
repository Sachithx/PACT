"""Shared low-level utilities: seeding, device, masking, arg parsing."""

import argparse
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Seed python / numpy / torch (and CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def random_masking_3D(xb, mask_ratio):
    """Random patch masking for masked-autoencoding pretraining.

    xb: [B, num_patch, dim]. Returns (x_masked, x_kept, mask, ids_restore),
    where mask is [B, num_patch] with 1 = removed, 0 = kept.
    (Adapted from PatchTST / TSLANet.)
    """
    bs, L, D = xb.shape
    x = xb.clone()
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(bs, L, device=xb.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    x_removed = torch.zeros(bs, L - len_keep, D, device=xb.device)
    x_ = torch.cat([x_kept, x_removed], dim=1)
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))

    mask = torch.ones([bs, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, x_kept, mask, ids_restore
