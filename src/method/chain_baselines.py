"""Domain-incremental baselines on frozen TSFM embeddings (B3/B10/B12).

Lightweight PACT-compatible implementations inspired by:
  sprompts   — S-Prompts (Wang et al., NeurIPS'22): per-domain heads + KNN domain id
  dualprompt — DualPrompt / L2P (Wang et al., ECCV'22): invariant + domain prompts
  eata       — EATA (Niu et al., ICML'22): entropy minimization on calibration
  cotta      — CoTTA (Wang et al., CVPR'22): EMA teacher + noisy consistency
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearHead(nn.Module):
    def __init__(self, emb_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, z, domain_idx=None):
        return self.fc(z)


class SPromptsPool(nn.Module):
    """One linear head per seen domain; inference picks head via KNN to domain centers."""

    def __init__(self, emb_dim, num_classes):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.heads = nn.ModuleList()
        self.register_buffer("_centers", torch.zeros(0, emb_dim), persistent=True)

    def n_domains(self):
        return len(self.heads)

    def add_domain(self):
        self.heads.append(nn.Linear(self.emb_dim, self.num_classes))

    def set_center(self, idx, center):
        c = center.detach().float().view(1, -1)
        if self._centers.numel() == 0:
            self._centers = c
        else:
            self._centers = torch.cat([self._centers, c], dim=0)

    def nearest_domain(self, z):
        if self._centers.shape[0] == 0:
            return torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        if self._centers.shape[0] == 1:
            return torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        d = torch.cdist(z, self._centers)
        return d.argmin(dim=1)

    def forward(self, z, domain_idx=None):
        if domain_idx is not None:
            if isinstance(domain_idx, int):
                return self.heads[domain_idx](z)
            out = z.new_zeros(z.shape[0], self.num_classes)
            for i, d in enumerate(domain_idx):
                out[i] = self.heads[int(d)](z[i:i + 1]).squeeze(0)
            return out
        idx = self.nearest_domain(z)
        out = z.new_zeros(z.shape[0], self.num_classes)
        for d in idx.unique().tolist():
            m = idx == d
            out[m] = self.heads[d](z[m])
        return out


class DualPromptHead(nn.Module):
    """Shared invariant prompt + per-domain specific prompt (DualPrompt-style on embeddings)."""

    def __init__(self, emb_dim, num_classes, max_domains=64):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.fc = nn.Linear(emb_dim, num_classes)
        self.prompt_inv = nn.Parameter(torch.zeros(emb_dim))
        self.prompt_spec = nn.Parameter(torch.zeros(max_domains, emb_dim))
        self.register_buffer("_centers", torch.zeros(0, emb_dim), persistent=True)
        self.max_domains = max_domains

    def prompt(self, domain_idx):
        return self.prompt_inv + self.prompt_spec[domain_idx]

    def set_center(self, idx, center):
        c = center.detach().float().view(1, -1)
        if self._centers.numel() == 0:
            self._centers = c
        else:
            self._centers = torch.cat([self._centers, c], dim=0)

    def nearest_domain(self, z):
        if self._centers.shape[0] <= 1:
            return torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        return torch.cdist(z, self._centers).argmin(dim=1)

    def forward(self, z, domain_idx=None):
        if domain_idx is not None:
            if isinstance(domain_idx, int):
                return self.fc(z + self.prompt(domain_idx))
            logits = []
            for i, d in enumerate(domain_idx):
                logits.append(self.fc(z[i:i + 1] + self.prompt(int(d))))
            return torch.cat(logits, dim=0)
        idx = self.nearest_domain(z)
        out = z.new_zeros(z.shape[0], self.num_classes)
        for d in idx.unique().tolist():
            m = idx == d
            out[m] = self.fc(z[m] + self.prompt(d))
        return out


def entropy_loss(logits):
    p = F.softmax(logits, dim=1)
    return -(p * (p + 1e-8).log()).sum(dim=1).mean()


def train_linear_head(module, z, y, device, epochs=80, lr=1e-3, domain_idx=None, method=None):
    opt = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(z, y), batch_size=256, shuffle=True)
    module.train()
    for _ in range(epochs):
        for zb, yb in dl:
            opt.zero_grad()
            if method == "dualprompt":
                loss = ce(module(zb, domain_idx=domain_idx), yb)
            else:
                loss = ce(module(zb), yb)
            loss.backward()
            opt.step()
    module.eval()


def adapt_eata(head, z, y, device, steps=20, lr=1e-4, fisher_anchor=None, fisher_alpha=200.0):
    """Entropy minimization on calibration embeddings (EATA-style, head-only)."""
    params = [p for p in head.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    head.train()
    for _ in range(steps):
        logits = head(z)
        loss = entropy_loss(logits)
        if fisher_anchor is not None:
            for n, p in head.named_parameters():
                if n in fisher_anchor and p.requires_grad:
                    loss = loss + fisher_alpha * 0.5 * ((p - fisher_anchor[n]) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval()


def adapt_cotta(head, z, y, device, steps=20, lr=1e-4, noise_std=0.01):
    """CoTTA-style: CE + consistency to EMA teacher under embedding noise."""
    teacher = copy.deepcopy(head).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=lr)
    ce = nn.CrossEntropyLoss()
    head.train()
    for step in range(steps):
        noisy = z + noise_std * torch.randn_like(z)
        student_logits = head(noisy)
        with torch.no_grad():
            teacher_logits = teacher(z)
        loss = ce(student_logits, y) + F.mse_loss(
            F.softmax(student_logits, 1), F.softmax(teacher_logits, 1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for tp, sp in zip(teacher.parameters(), head.parameters()):
                tp.data.mul_(0.999).add_(sp.data, alpha=0.001)
    head.eval()


def register_domain(head, z, y, device, method, epochs=80, lr=1e-3, domain_idx=0):
    """Fit / extend head for domain `domain_idx` and store domain center."""
    center = z.mean(0)
    if method == "sprompts":
        if domain_idx == len(head.heads):
            head.add_domain()
            head.to(device)
        train_linear_head(head.heads[domain_idx], z, y, device, epochs, lr)
        head.set_center(domain_idx, center)
        return head
    if method == "dualprompt":
        train_linear_head(head, z, y, device, epochs, lr, domain_idx=domain_idx, method=method)
        head.set_center(domain_idx, center)
        return head
    if method == "eata":
        anchor = {n: p.detach().clone() for n, p in head.named_parameters()}
        train_linear_head(head, z, y, device, epochs, lr)
        adapt_eata(head, z, y, device, fisher_anchor=anchor)
        return head
    if method == "cotta":
        train_linear_head(head, z, y, device, epochs, lr)
        adapt_cotta(head, z, y, device)
        return head
    raise ValueError(method)


def make_head(method, emb_dim, num_classes, max_domains=64):
    if method == "sprompts":
        h = SPromptsPool(emb_dim, num_classes)
        h.add_domain()
        return h
    if method == "dualprompt":
        return DualPromptHead(emb_dim, num_classes, max_domains=max_domains)
    return LinearHead(emb_dim, num_classes)


def forward_logits(head, z, method, domain_idx=None):
    if method in ("sprompts", "dualprompt"):
        return head(z, domain_idx=domain_idx)
    return head(z)
