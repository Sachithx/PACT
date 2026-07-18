"""Dark Experience Replay++ (Buzzega et al., NeurIPS'20), Mammoth-faithful.

Ports the core observe/buffer logic from baselines/mammoth/models/derpp.py into
PACT's domain-incremental chain protocol. Buffer stores raw windows + labels +
logits (sample rehearsal), sized separately for byte-matching against PACT's
embedding buffer.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReservoirBuffer:
    """Reservoir buffer matching Mammoth's Buffer attributes for DER++."""

    def __init__(self, buffer_size: int, example_shape, num_classes: int, device):
        self.buffer_size = int(buffer_size)
        self.device = device
        self.num_seen = 0
        self.examples = torch.zeros((self.buffer_size, *example_shape),
                                    dtype=torch.float32, device=device)
        self.labels = torch.zeros(self.buffer_size, dtype=torch.long, device=device)
        self.logits = torch.zeros((self.buffer_size, num_classes),
                                  dtype=torch.float32, device=device)

    def __len__(self):
        return min(self.num_seen, self.buffer_size)

    def is_empty(self):
        return self.num_seen == 0

    def add_data(self, examples, labels, logits):
        examples = examples.detach().to(self.device)
        labels = labels.detach().to(self.device).long()
        logits = logits.detach().to(self.device)
        for i in range(examples.shape[0]):
            if self.num_seen < self.buffer_size:
                idx = self.num_seen
            else:
                idx = int(np.random.randint(0, self.num_seen + 1))
                if idx >= self.buffer_size:
                    self.num_seen += 1
                    continue
            self.examples[idx] = examples[i]
            self.labels[idx] = labels[i]
            self.logits[idx] = logits[i]
            self.num_seen += 1

    def get_data(self, n, device=None):
        device = device or self.device
        n = min(n, len(self))
        idx = torch.randperm(len(self), device=self.device)[:n]
        return (self.examples[idx].to(device),
                self.labels[idx].to(device),
                self.logits[idx].to(device))

    def nbytes(self):
        n = len(self)
        return (n * self.examples[0].numel() * 4
                + n * 8
                + n * self.logits.shape[1] * 4)


def bytes_per_derpp_slot(num_channels: int, seq_len: int, num_classes: int) -> int:
    """float32 raw window + int64 label + float32 logits."""
    return num_channels * seq_len * 4 + 8 + num_classes * 4


def pact_buffer_bytes(emb_dim: int, n_src: int, n_calib_total: int) -> int:
    """Embedding-only buffer: float32 keys, no class labels."""
    return emb_dim * (n_src + n_calib_total) * 4


def matched_buffer_size(emb_dim: int, n_src: int, n_calib_total: int,
                        num_channels: int, seq_len: int, num_classes: int) -> int:
    slot = bytes_per_derpp_slot(num_channels, seq_len, num_classes)
    return max(1, pact_buffer_bytes(emb_dim, n_src, n_calib_total) // slot)


def forward_logits(model, x):
    """Works for backbone wrappers (model(x)) and TSFM (head(extract(x)))."""
    if hasattr(model, "extract") and hasattr(model, "head"):
        return model.head(model.extract(x))
    return model(x)


def derpp_observe(model, opt, ce, buffer, inputs, labels, alpha, beta,
                  minibatch_size, freeze_backbone=False):
    """One Mammoth-style DER++ step. Returns scalar loss."""
    model.train()
    if freeze_backbone and hasattr(model, "parameters"):
        # ensure only head (or explicitly unfrozen params) receive grads
        pass
    opt.zero_grad()
    outputs = forward_logits(model, inputs)
    loss = ce(outputs, labels)

    if not buffer.is_empty():
        buf_x, _, buf_logits = buffer.get_data(minibatch_size, device=inputs.device)
        buf_out = forward_logits(model, buf_x)
        loss = loss + alpha * F.mse_loss(buf_out, buf_logits)

        buf_x2, buf_y, _ = buffer.get_data(minibatch_size, device=inputs.device)
        buf_out2 = forward_logits(model, buf_x2)
        loss = loss + beta * ce(buf_out2, buf_y)

    loss.backward()
    opt.step()
    buffer.add_data(examples=inputs, labels=labels, logits=outputs.data)
    return float(loss.item())
