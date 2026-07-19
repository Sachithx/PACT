"""Frozen Time-Series Foundation Model backbones for PACT.

Wraps a pretrained TSFM as a frozen feature extractor exposing the same
interface the PACT method expects from any backbone:
    .extract(x) -> pre-head embeddings   (frozen, no grad)
    .head       -> trainable/​frozen classification head (emb -> logits)
    .emb_dim    -> embedding dimension

Backbones: MOMENT (AutonLab/MOMENT-1-large), Chronos-Bolt (amazon/chronos-bolt-base),
Sundial (thuml/sundial-base-128m), Time-MoE (Maple728/TimeMoE-200M), and Moirai
(Salesforce/moirai-1.1-R-large) -- the largest published checkpoint of each.
Every backbone has a fixed/native context window. Two design choices make all
of them competitive for classification of long / high-frequency series:

  * multi-window pooling: series longer than the window are split into
    (overlapping) same-length windows, each embedded, then mean-pooled ->
    keeps full temporal resolution instead of destroying it by downsampling.
  * embedding standardization: raw transformer embeddings are z-scored using
    statistics fit on the SOURCE domain, which sharpens the k-means concept
    geometry and the gate's routing (both live in embedding space).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MomentBackbone(nn.Module):
    def __init__(self, num_classes, moment_len=512, multi_window=True,
                 enc_bs=128, model_name="AutonLab/MOMENT-1-large"):
        super().__init__()
        from momentfm import MOMENTPipeline
        self.enc = MOMENTPipeline.from_pretrained(
            model_name, model_kwargs={"task_name": "embedding"})
        self.enc.init()
        self.enc.eval()
        for p in self.enc.parameters():
            p.requires_grad_(False)
        self.moment_len = moment_len
        self.multi_window = multi_window
        self.enc_bs = enc_bs
        with torch.no_grad():
            d = self.enc(x_enc=torch.zeros(1, 1, moment_len)).embeddings.shape[-1]
        self.emb_dim = int(d)
        self.head = nn.Linear(self.emb_dim, num_classes)
        self._mean = None
        self._std = None

    def set_standardizer(self, mean, std):
        dev = next(self.head.parameters()).device
        self._mean = torch.as_tensor(mean, dtype=torch.float32, device=dev)
        self._std = torch.as_tensor(std, dtype=torch.float32, device=dev).clamp_min(1e-6)

    @torch.no_grad()
    def _encode(self, x):
        """Run the frozen encoder in chunks; x: (M, C, moment_len) -> (M, D)."""
        outs = []
        for i in range(0, x.shape[0], self.enc_bs):
            outs.append(self.enc(x_enc=x[i:i + self.enc_bs]).embeddings.float())
        return torch.cat(outs, 0)

    @torch.no_grad()
    def extract(self, x):
        x = x.float()
        B, C, L = x.shape
        if L == self.moment_len:
            e = self._encode(x)
        elif L < self.moment_len:
            e = self._encode(F.interpolate(x, size=self.moment_len, mode="linear",
                                           align_corners=False))
        elif self.multi_window:
            nw = math.ceil(L / self.moment_len)
            starts = torch.linspace(0, L - self.moment_len, nw).long()
            wins = torch.stack([x[..., s:s + self.moment_len] for s in starts], 1)
            wins = wins.reshape(B * nw, C, self.moment_len)
            e = self._encode(wins).reshape(B, nw, -1).mean(1)   # pool over windows
        else:
            e = self._encode(F.interpolate(x, size=self.moment_len, mode="linear",
                                           align_corners=False))
        if self._mean is not None:
            e = (e - self._mean) / self._std
        return e

    def forward(self, x):
        return self.head(self.extract(x))

    def freeze_head(self):
        for p in self.head.parameters():
            p.requires_grad_(False)


class ChronosBackbone(nn.Module):
    """Amazon Chronos (T5) as a frozen encoder. Univariate: each channel is
    embedded independently and pooled. Long series use multi-window pooling."""

    def __init__(self, num_classes, model_name="amazon/chronos-t5-base",
                 ctx=512, multi_window=True, enc_bs=64, device="cuda"):
        super().__init__()
        import torch as _t
        from chronos import BaseChronosPipeline
        self.pipe = BaseChronosPipeline.from_pretrained(
            model_name, device_map=device, torch_dtype=_t.float32)
        self.ctx = ctx
        self.multi_window = multi_window
        self.enc_bs = enc_bs
        with torch.no_grad():
            e, _ = self.pipe.embed(torch.zeros(1, ctx))
        self.emb_dim = int(e.shape[-1])
        self.head = nn.Linear(self.emb_dim, num_classes)
        self._mean = None
        self._std = None

    def set_standardizer(self, mean, std):
        dev = next(self.head.parameters()).device
        self._mean = torch.as_tensor(mean, dtype=torch.float32, device=dev)
        self._std = torch.as_tensor(std, dtype=torch.float32, device=dev).clamp_min(1e-6)

    @torch.no_grad()
    def _embed_ctx(self, seqs):
        """seqs: (M, Lctx) -> (M, D), mean-pooled over time."""
        outs = []
        for i in range(0, seqs.shape[0], self.enc_bs):
            emb, _ = self.pipe.embed(seqs[i:i + self.enc_bs].cpu())
            outs.append(emb.mean(1).float().to(seqs.device))
        return torch.cat(outs, 0)

    @torch.no_grad()
    def extract(self, x):
        x = x.float()
        B, C, L = x.shape
        if L > self.ctx and self.multi_window:
            nw = math.ceil(L / self.ctx)
            starts = torch.linspace(0, L - self.ctx, nw).long()
            wins = torch.stack([x[..., s:s + self.ctx] for s in starts], 2)  # (B,C,nw,ctx)
            seqs = wins.reshape(B * C * nw, self.ctx)
            e = self._embed_ctx(seqs).reshape(B, C, nw, -1).mean(2).mean(1)
        else:
            seqs = x.reshape(B * C, L)
            e = self._embed_ctx(seqs).reshape(B, C, -1).mean(1)
        e = e.to(next(self.head.parameters()).device)
        if self._mean is not None:
            e = (e - self._mean) / self._std
        return e

    def forward(self, x):
        return self.head(self.extract(x))

    def freeze_head(self):
        for p in self.head.parameters():
            p.requires_grad_(False)


class CausalLMTSFMBackbone(nn.Module):
    """Frozen decoder-only causal-LM-style TSFM backbone (Sundial, Time-MoE):
    HF `AutoModelForCausalLM` + `trust_remote_code`, takes a plain univariate
    (M, window) sequence of raw values as `input_ids` and exposes
    `.last_hidden_state`. Univariate like Chronos: each channel is embedded
    independently (last-token pooled) and pooled across channels; long series
    use multi-window pooling, same scheme as MomentBackbone/ChronosBackbone."""

    def __init__(self, num_classes, model_name, window=512, multi_window=True,
                 enc_bs=64):
        super().__init__()
        from transformers import AutoModelForCausalLM
        full = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.enc = full.model if hasattr(full, "model") else full
        self.enc.eval()
        for p in self.enc.parameters():
            p.requires_grad_(False)
        self.window = window
        self.multi_window = multi_window
        self.enc_bs = enc_bs
        with torch.no_grad():
            d = self.enc(input_ids=torch.zeros(1, window)).last_hidden_state.shape[-1]
        self.emb_dim = int(d)
        self.head = nn.Linear(self.emb_dim, num_classes)
        self._mean = None
        self._std = None

    def set_standardizer(self, mean, std):
        dev = next(self.head.parameters()).device
        self._mean = torch.as_tensor(mean, dtype=torch.float32, device=dev)
        self._std = torch.as_tensor(std, dtype=torch.float32, device=dev).clamp_min(1e-6)

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
        x = x.float()
        B, C, L = x.shape
        if L > self.window and self.multi_window:
            nw = math.ceil(L / self.window)
            starts = torch.linspace(0, L - self.window, nw).long()
            wins = torch.stack([x[..., s:s + self.window] for s in starts], 2)  # (B,C,nw,window)
            seqs = wins.reshape(B * C * nw, self.window)
            e = self._embed_ctx(seqs).reshape(B, C, nw, -1).mean(2).mean(1)
        elif L != self.window:
            xw = F.interpolate(x, size=self.window, mode="linear", align_corners=False)
            seqs = xw.reshape(B * C, self.window)
            e = self._embed_ctx(seqs).reshape(B, C, -1).mean(1)
        else:
            seqs = x.reshape(B * C, L)
            e = self._embed_ctx(seqs).reshape(B, C, -1).mean(1)
        e = e.to(next(self.head.parameters()).device)
        if self._mean is not None:
            e = (e - self._mean) / self._std
        return e

    def forward(self, x):
        return self.head(self.extract(x))

    def freeze_head(self):
        for p in self.head.parameters():
            p.requires_grad_(False)


class MoiraiBackbone(nn.Module):
    """Salesforce Moirai (uni2ts) as a frozen encoder. Moirai is natively
    multivariate via packed variate/time ids, but each channel is packed and
    embedded as its own (unpacked) sample here, matching the univariate
    treatment used for Chronos/Sundial/Time-MoE -- keeps the pooling semantics
    identical across every TSFM backbone. Each `window`-length slice is
    patchified into `window / patch_size` tokens; the encoder (not the
    distribution head) output is mean-pooled over patches for the embedding."""

    def __init__(self, num_classes, model_name="Salesforce/moirai-1.1-R-large",
                 window=512, patch_size=32, multi_window=True, enc_bs=64):
        super().__init__()
        from uni2ts.model.moirai import MoiraiModule
        from uni2ts.common.torch_util import packed_attention_mask
        self._packed_attention_mask = packed_attention_mask
        self.enc = MoiraiModule.from_pretrained(model_name)
        self.enc.eval()
        for p in self.enc.parameters():
            p.requires_grad_(False)
        if window % patch_size != 0:
            raise ValueError(f"window ({window}) must be a multiple of patch_size ({patch_size})")
        self.window = window
        self.patch_size = patch_size
        self.num_patches = window // patch_size
        self.multi_window = multi_window
        self.enc_bs = enc_bs
        self.emb_dim = int(self.enc.d_model)
        self.head = nn.Linear(self.emb_dim, num_classes)
        self._mean = None
        self._std = None

    def set_standardizer(self, mean, std):
        dev = next(self.head.parameters()).device
        self._mean = torch.as_tensor(mean, dtype=torch.float32, device=dev)
        self._std = torch.as_tensor(std, dtype=torch.float32, device=dev).clamp_min(1e-6)

    @torch.no_grad()
    def _embed_ctx(self, seqs):
        """seqs: (M, window) raw values -> (M, D), mean-pooled over patches."""
        outs = []
        max_patch = max(self.enc.patch_sizes)
        for i in range(0, seqs.shape[0], self.enc_bs):
            chunk = seqs[i:i + self.enc_bs]
            M, dev = chunk.shape[0], chunk.device
            target = chunk.reshape(M, self.num_patches, self.patch_size)
            target = F.pad(target, (0, max_patch - self.patch_size))
            observed_mask = torch.zeros(M, self.num_patches, max_patch, dtype=torch.bool, device=dev)
            observed_mask[..., :self.patch_size] = True
            sample_id = torch.ones(M, self.num_patches, dtype=torch.long, device=dev)
            time_id = torch.arange(self.num_patches, device=dev).unsqueeze(0).expand(M, -1)
            variate_id = torch.zeros(M, self.num_patches, dtype=torch.long, device=dev)
            prediction_mask = torch.zeros(M, self.num_patches, dtype=torch.bool, device=dev)
            patch_size_t = torch.full((M, self.num_patches), self.patch_size, dtype=torch.long, device=dev)

            loc, scale = self.enc.scaler(
                target, observed_mask * ~prediction_mask.unsqueeze(-1), sample_id, variate_id)
            scaled = (target - loc) / scale
            reprs = self.enc.in_proj(scaled, patch_size_t)
            reprs = self.enc.encoder(reprs, self._packed_attention_mask(sample_id),
                                     time_id=time_id, var_id=variate_id)
            outs.append(reprs.mean(1).float())  # pool over patches
        return torch.cat(outs, 0)

    @torch.no_grad()
    def extract(self, x):
        x = x.float()
        B, C, L = x.shape
        if L > self.window and self.multi_window:
            nw = math.ceil(L / self.window)
            starts = torch.linspace(0, L - self.window, nw).long()
            wins = torch.stack([x[..., s:s + self.window] for s in starts], 2)  # (B,C,nw,window)
            seqs = wins.reshape(B * C * nw, self.window)
            e = self._embed_ctx(seqs).reshape(B, C, nw, -1).mean(2).mean(1)
        elif L != self.window:
            xw = F.interpolate(x, size=self.window, mode="linear", align_corners=False)
            seqs = xw.reshape(B * C, self.window)
            e = self._embed_ctx(seqs).reshape(B, C, -1).mean(1)
        else:
            seqs = x.reshape(B * C, L)
            e = self._embed_ctx(seqs).reshape(B, C, -1).mean(1)
        e = e.to(next(self.head.parameters()).device)
        if self._mean is not None:
            e = (e - self._mean) / self._std
        return e

    def forward(self, x):
        return self.head(self.extract(x))

    def freeze_head(self):
        for p in self.head.parameters():
            p.requires_grad_(False)


class MantisBackbone(nn.Module):
    """Mantis (Feofanov et al.) -- a lightweight, classification-targeted TS
    foundation model, as a frozen encoder. Natively univariate
    (n_channels=1); each channel is embedded independently and pooled,
    matching the Chronos/Sundial/Time-MoE/Moirai treatment. seq_len must be a
    multiple of num_patches (default 32), enforced via the same
    multi-window / interpolate scheme as the other backbones."""

    def __init__(self, num_classes, model_name="paris-noah/MantisV2",
                 window=512, multi_window=True, enc_bs=128):
        super().__init__()
        from mantis.architecture import MantisV2
        net = MantisV2(device="cpu")
        self.enc = net.from_pretrained(model_name)
        self.enc.eval()
        for p in self.enc.parameters():
            p.requires_grad_(False)
        num_patches = self.enc.num_patches  # from_pretrained loads this from the checkpoint config
        if window % num_patches != 0:
            raise ValueError(f"window ({window}) must be a multiple of num_patches ({num_patches})")
        self.window = window
        self.multi_window = multi_window
        self.enc_bs = enc_bs
        with torch.no_grad():
            d = self.enc(torch.zeros(1, 1, window)).shape[-1]
        self.emb_dim = int(d)
        self.head = nn.Linear(self.emb_dim, num_classes)
        self._mean = None
        self._std = None

    def to(self, device, *args, **kwargs):
        out = super().to(device, *args, **kwargs)
        out.enc.to(device)
        return out

    def set_standardizer(self, mean, std):
        dev = next(self.head.parameters()).device
        self._mean = torch.as_tensor(mean, dtype=torch.float32, device=dev)
        self._std = torch.as_tensor(std, dtype=torch.float32, device=dev).clamp_min(1e-6)

    @torch.no_grad()
    def _embed_ctx(self, seqs):
        """seqs: (M, window) -> (M, D)."""
        outs = []
        for i in range(0, seqs.shape[0], self.enc_bs):
            outs.append(self.enc(seqs[i:i + self.enc_bs].unsqueeze(1)).float())
        return torch.cat(outs, 0)

    @torch.no_grad()
    def extract(self, x):
        x = x.float()
        B, C, L = x.shape
        if L > self.window and self.multi_window:
            nw = math.ceil(L / self.window)
            starts = torch.linspace(0, L - self.window, nw).long()
            wins = torch.stack([x[..., s:s + self.window] for s in starts], 2)  # (B,C,nw,window)
            seqs = wins.reshape(B * C * nw, self.window)
            e = self._embed_ctx(seqs).reshape(B, C, nw, -1).mean(2).mean(1)
        elif L != self.window:
            xw = F.interpolate(x, size=self.window, mode="linear", align_corners=False)
            seqs = xw.reshape(B * C, self.window)
            e = self._embed_ctx(seqs).reshape(B, C, -1).mean(1)
        else:
            seqs = x.reshape(B * C, L)
            e = self._embed_ctx(seqs).reshape(B, C, -1).mean(1)
        e = e.to(next(self.head.parameters()).device)
        if self._mean is not None:
            e = (e - self._mean) / self._std
        return e

    def forward(self, x):
        return self.head(self.extract(x))

    def freeze_head(self):
        for p in self.head.parameters():
            p.requires_grad_(False)


def make_moment(num_classes, seq_len=None, model_name="AutonLab/MOMENT-1-large",
                multi_window=True):
    return MomentBackbone(num_classes, model_name=model_name,
                          multi_window=multi_window)


TSFM_MODELS = {
    "moment": dict(cls=MomentBackbone, model_name="AutonLab/MOMENT-1-large"),
    "chronos-bolt": dict(cls=ChronosBackbone, model_name="amazon/chronos-bolt-base",
                         ctx=2048, enc_bs=128),
    "sundial": dict(cls=CausalLMTSFMBackbone, model_name="thuml/sundial-base-128m"),
    "time-moe": dict(cls=CausalLMTSFMBackbone, model_name="Maple728/TimeMoE-200M"),
    "moirai": dict(cls=MoiraiBackbone, model_name="Salesforce/moirai-1.1-R-large"),
    "mantis": dict(cls=MantisBackbone, model_name="paris-noah/MantisV2"),
}


def make_tsfm(name, num_classes, device="cuda"):
    if name not in TSFM_MODELS:
        raise ValueError(f"Unknown TSFM {name!r}. Choose from {list(TSFM_MODELS)}.")
    cfg = dict(TSFM_MODELS[name])
    cls = cfg.pop("cls")
    if cls is ChronosBackbone:
        return cls(num_classes, device=device, **cfg)
    return cls(num_classes, **cfg)
