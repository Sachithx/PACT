"""ORTCL (orthogonal rotation) adapted for TSLANet on the domain-incremental chain.

Applies an SVD-based orthogonal rotation to the last ICB conv3 projection,
following the TSLANet port in experiments/run_ortcl_tslanet.py. Feature
collection uses unlabeled train data from seen domains (source) and the
incoming domain (target).
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

SVD_CAP = 50_000
SVD_SEED = 42


@torch.no_grad()
def _collect_icb_features(model, loader, device, cap=SVD_CAP):
    """Return K_2d, V_2d each [emb_dim, N_tokens] from last ICB conv3."""
    icb = model.blocks[-1].icb
    k_list, v_list = [], []

    def k_hook(_module, inp, _out):
        x = inp[0].detach().cpu()
        b, h, n = x.shape
        emb = h // 3
        k_list.append(x.view(b, 3, emb, n).mean(1))

    def v_hook(_module, _inp, out):
        v_list.append(out.detach().cpu())

    h1 = icb.conv3.register_forward_hook(k_hook)
    h2 = icb.conv3.register_forward_hook(v_hook)
    model.eval()
    for xb, _ in loader:
        model(xb.to(device))
    h1.remove()
    h2.remove()

    k_raw = torch.cat(k_list, dim=0)
    v_raw = torch.cat(v_list, dim=0)
    d = k_raw.shape[1]
    k_2d = k_raw.permute(1, 0, 2).reshape(d, -1)
    v_2d = v_raw.permute(1, 0, 2).reshape(d, -1)

    total = k_2d.shape[1]
    if total > cap:
        rng = np.random.default_rng(SVD_SEED)
        idx = torch.from_numpy(rng.choice(total, cap, replace=False))
        k_2d = k_2d[:, idx]
        v_2d = v_2d[:, idx]
    return k_2d.to(device), v_2d.to(device)


def apply_ortcl_rotation(model, src_loader, tgt_loader, device, batch_size=128):
    """Rotate last ICB conv3 in-place; returns orthogonality Frobenius error."""
    icb = model.blocks[-1].icb
    w = icb.conv3.weight.data.squeeze(-1)
    d = w.shape[0]
    eps = torch.eye(d, device=device) * 1e-6

    src_ld = DataLoader(src_loader.dataset, batch_size=batch_size, shuffle=False)
    tgt_ld = DataLoader(tgt_loader.dataset, batch_size=batch_size, shuffle=False)

    k_a, v_a = _collect_icb_features(model, src_ld, device)
    k_b, v_b = _collect_icb_features(model, tgt_ld, device)

    u0, _s0, v0_t = torch.linalg.svd(w, full_matrices=False)
    k_combined = torch.cat([k_a, k_b], dim=1)
    k_proj = u0 @ (u0.T @ k_combined)

    _u_k, _s_k, x_t = torch.linalg.svd(k_proj.T, full_matrices=False)
    s_mat = x_t.T

    a = k_a
    b_mat = k_b
    m = v_a @ a.T + v_b @ b_mat.T + eps
    u_m, _s_m, v_m_t = torch.linalg.svd(m)
    r = u_m @ v_m_t

    icb.conv3.weight.data = (r @ w).unsqueeze(-1)
    return torch.norm(r.T @ r - torch.eye(d, device=device), p="fro").item()


def finetune_head(model, calib_ds, device, epochs=20, lr=1e-4, batch_size=128):
    """Head-only fine-tune on labelled calibration data."""
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.head.parameters():
        p.requires_grad_(True)
    ld = DataLoader(calib_ds, batch_size=min(batch_size, max(1, len(calib_ds))),
                    shuffle=True)
    opt = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            ce(model(xb), yb).backward()
            opt.step()
    model.eval()


def seen_train_loader(spec, order, t, data_root, batch_size):
    """Concatenate train splits for domains order[0..t-1] (unlabeled use)."""
    ds = ConcatDataset([
        spec.get_dataset(order[j], "train", data_root) for j in range(t)
    ])
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def new_domain_loader(spec, did, data_root, batch_size):
    ds = spec.get_dataset(did, "train", data_root)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)
