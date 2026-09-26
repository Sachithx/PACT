"""Frozen Sundial backbone (thuml/sundial-base-128m) used as a feature extractor.

Each channel is embedded independently as a univariate sequence; the
last-token hidden state is the embedding. Series longer than the 512-step
context (FD has length 5120) are split into evenly spaced windows whose
embeddings are mean-pooled. Embeddings are z-scored with statistics fitted on
the source domain. The encoder is never updated.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SundialBackbone(nn.Module):
    def __init__(self, num_classes, model_name="thuml/sundial-base-128m",
                 window=512, enc_bs=64):
        super().__init__()
        from transformers import AutoModelForCausalLM
        full = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.enc = full.model if hasattr(full, "model") else full
        self.enc.eval()
        for p in self.enc.parameters():
            p.requires_grad_(False)
        self.window = window
        self.enc_bs = enc_bs
        with torch.no_grad():
            d = self.enc(input_ids=torch.zeros(1, window)).last_hidden_state.shape[-1]
        self.emb_dim = int(d)
        # Source classifier: a single linear layer, trained once on the source
        # domain and then frozen for the rest of the stream.
        self.head = nn.Linear(self.emb_dim, num_classes)
        self._mean = None
        self._std = None

    def set_standardizer(self, mean, std):
        dev = next(self.head.parameters()).device
        self._mean = torch.as_tensor(mean, dtype=torch.float32, device=dev)
        self._std = torch.as_tensor(std, dtype=torch.float32, device=dev).clamp_min(1e-6)

    def freeze_head(self):
        for p in self.head.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _embed_ctx(self, seqs):
        """seqs: (M, window) -> (M, D), last-token hidden state."""
        outs = []
        for i in range(0, seqs.shape[0], self.enc_bs):
            h = self.enc(input_ids=seqs[i:i + self.enc_bs]).last_hidden_state
            outs.append(h[:, -1, :].float())
        return torch.cat(outs, 0)

    @torch.no_grad()
    def extract(self, x):
        """x: (B, C, L) -> standardized embeddings (B, D)."""
        x = x.float()
        B, C, L = x.shape
        if L > self.window:
            nw = math.ceil(L / self.window)
            starts = torch.linspace(0, L - self.window, nw).long()
            wins = torch.stack([x[..., s:s + self.window] for s in starts], 2)
            seqs = wins.reshape(B * C * nw, self.window)
            e = self._embed_ctx(seqs).reshape(B, C, nw, -1).mean(2).mean(1)
        elif L != self.window:
            xw = F.interpolate(x, size=self.window, mode="linear", align_corners=False)
            e = self._embed_ctx(xw.reshape(B * C, self.window)).reshape(B, C, -1).mean(1)
        else:
            e = self._embed_ctx(x.reshape(B * C, L)).reshape(B, C, -1).mean(1)
        e = e.to(next(self.head.parameters()).device)
        if self._mean is not None:
            e = (e - self._mean) / self._std
        return e
