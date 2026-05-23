"""
PACT — Prototype-Aware Concept Transformation

Backbone-agnostic plugin for post-hoc distribution shift correction.
Any PyTorch model with .extract(x) → [B, D] and .forward(x) → [B, C] works.

Typical usage:
    pact = PACT(backbone=frozen_model, k=16, emb_dim=128, n_classes=6, device=device)
    pact.fit(src_loader, calib_loader)        # calibrate on 5% labeled target
    f1 = pact.evaluate(tgt_test_loader)       # drift recovery
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from .kmeans import kmeans, assign


# ── Building-block modules ─────────────────────────────────────────────────────

class IdentityAdapter(nn.Module):
    """
    Linear adapter initialised to the identity transform.
    Learns a residual correction in embedding space.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class GateMLP(nn.Module):
    """
    Per-concept binary gate: 0 = source-like (skip adapter), 1 = drifted (apply adapter).
    Trained with source embeddings as negatives and calib embeddings as positives.
    """
    def __init__(self, emb_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Main PACT class ────────────────────────────────────────────────────────────

class PACT:
    """
    Prototype-Aware Concept Transformation.

    After fitting, the backbone is wrapped with K concept-specific adapters
    and per-concept learned gates. Inference is non-destructive — the backbone
    weights are never modified.

    Args:
        backbone:   Frozen PyTorch model. Must implement:
                      backbone.extract(x) → [B, emb_dim]
                      backbone(x)         → [B, n_classes]
        k:          Number of concept prototypes (clusters). Recommended: 4–16.
        emb_dim:    Embedding dimension (must match backbone.extract output).
        n_classes:  Number of output classes (only used for adapter training).
        penalty:    Identity regularisation weight (keeps adapters near I).
        device:     torch.device for training.
        seed:       Random seed for k-means reproducibility.
    """

    def __init__(self, backbone: nn.Module, k: int = 16, emb_dim: int = 128,
                 n_classes: int = None, penalty: float = 1e-3,
                 device: torch.device | str = "cuda", seed: int = 42):
        self.backbone  = backbone.eval()
        self.k         = k
        self.emb_dim   = emb_dim
        self.n_classes = n_classes
        self.penalty   = penalty
        self.device    = torch.device(device)
        self.seed      = seed

        self.centroids: np.ndarray | None = None
        self.adapters:  nn.ModuleList | None = None
        self.gates:     list[GateMLP | None] | None = None
        self._fitted = False

    # ── Fitting ────────────────────────────────────────────────────────────────

    def fit(self, src_loader, calib_loader, *,
            adapter_epochs: int = 200, adapter_lr: float = 1e-3,
            gate_epochs: int = 200, gate_lr: float = 1e-3,
            batch_size: int = 256) -> "PACT":
        """
        Calibrate PACT.

        Args:
            src_loader:     DataLoader over the source-domain training set
                            (unlabeled; used for k-means and gate negatives).
            calib_loader:   DataLoader over the labeled calibration set
                            (5% stratified slice of target-domain train).
            adapter_epochs: Training epochs for the adapter MLP bank.
            adapter_lr:     AdamW learning rate for adapters.
            gate_epochs:    Training epochs per per-concept gate MLP.
            gate_lr:        AdamW learning rate for gates.
            batch_size:     Mini-batch size for adapter training.
        """
        print(f"[PACT] Extracting source embeddings …")
        src_embs, _              = self._extract(src_loader)
        print(f"[PACT] Extracting calibration embeddings …")
        calib_embs, calib_labels = self._extract(calib_loader)

        print(f"[PACT] K-means (K={self.k}) on {len(src_embs)} source embeddings …")
        self.centroids, src_asgn = kmeans(src_embs, self.k, seed=self.seed)
        calib_asgn = assign(calib_embs, self.centroids)

        n_per_concept = np.bincount(calib_asgn, minlength=self.k)
        print(f"[PACT] Calib per concept: min={n_per_concept.min()}  "
              f"max={n_per_concept.max()}  empty={int((n_per_concept == 0).sum())}")

        print(f"[PACT] Training {self.k} adapters …")
        self.adapters = self._train_adapters(
            calib_embs, calib_labels, calib_asgn,
            adapter_epochs, adapter_lr, batch_size)

        print(f"[PACT] Training {self.k} concept gates …")
        self.gates = self._train_gates(
            src_embs, src_asgn, calib_embs, calib_asgn,
            gate_epochs, gate_lr, batch_size)

        n_trained = sum(g is not None for g in self.gates)
        print(f"[PACT] Fit done — {n_trained}/{self.k} gates trained.")
        self._fitted = True
        return self

    # ── Inference ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with PACT correction applied.

        Args:
            x: input batch [B, ...] on any device
        Returns:
            logits [B, n_classes]
        """
        assert self._fitted, "Call .fit() before .predict()"
        x = x.to(self.device)
        embs = self.backbone.extract(x).cpu().numpy().astype(np.float32)
        asgn = assign(embs, self.centroids)
        corrected = self._apply_correction(embs, asgn)
        return self.backbone.head(corrected)

    @torch.no_grad()
    def evaluate(self, loader) -> float:
        """Macro F1 on a DataLoader using PACT-corrected predictions."""
        assert self._fitted
        all_preds, all_labels = [], []
        for x, y in loader:
            logits = self.predict(x)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(y.tolist())
        return f1_score(all_labels, all_preds, average="macro", zero_division=0)

    @torch.no_grad()
    def evaluate_backbone(self, loader) -> float:
        """Macro F1 using the bare backbone (no PACT), for baseline / forgetting."""
        self.backbone.eval()
        all_preds, all_labels = [], []
        for x, y in loader:
            preds = self.backbone(x.to(self.device)).argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(y.tolist())
        return f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # ── Internal helpers ───────────────────────────────────────────────────────

    @torch.no_grad()
    def _extract(self, loader) -> tuple[np.ndarray, np.ndarray]:
        self.backbone.eval()
        embs, labels = [], []
        for x, y in loader:
            embs.append(self.backbone.extract(x.to(self.device)).cpu().numpy())
            labels.append(y.numpy())
        return (np.concatenate(embs).astype(np.float32),
                np.concatenate(labels))

    def _train_adapters(self, calib_embs, calib_labels, calib_asgn,
                        epochs, lr, batch_size) -> nn.ModuleList:
        adapters = nn.ModuleList(
            [IdentityAdapter(self.emb_dim) for _ in range(self.k)]
        ).to(self.device)
        opt  = torch.optim.AdamW(adapters.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()
        eye  = torch.eye(self.emb_dim, device=self.device)

        et = torch.from_numpy(calib_embs).to(self.device)
        lt = torch.from_numpy(calib_labels.astype(np.int64)).to(self.device)
        at = torch.from_numpy(calib_asgn)
        ds = torch.utils.data.TensorDataset(et, lt, at)
        ld = torch.utils.data.DataLoader(
            ds, batch_size=min(batch_size, len(et)), shuffle=True)

        for _ in range(epochs):
            for eb, yb, cb in ld:
                adapters.train(); opt.zero_grad()
                corr = eb.clone()
                for c in range(self.k):
                    m = (cb == c)
                    if m.any():
                        corr[m] = adapters[c](eb[m])
                loss = crit(self.backbone.head(corr), yb)
                for a in adapters:
                    loss = loss + self.penalty * (
                        torch.sum((a.linear.weight - eye) ** 2)
                        + torch.sum(a.linear.bias ** 2))
                loss.backward(); opt.step()
        return adapters

    def _train_gates(self, src_embs, src_asgn, calib_embs, calib_asgn,
                     epochs, lr, batch_size) -> list:
        gates = []
        rng   = np.random.default_rng(self.seed)

        for c in range(self.k):
            neg = src_embs[src_asgn == c]
            pos = calib_embs[calib_asgn == c]
            if len(pos) == 0:
                gates.append(None)
                continue

            # Balance classes by oversampling minority
            if len(neg) > len(pos):
                pos = pos[rng.choice(len(pos), len(neg), replace=True)]
            elif len(pos) > len(neg):
                neg = neg[rng.choice(len(neg), len(pos), replace=True)]

            X = np.concatenate([neg, pos])
            y = np.concatenate([np.zeros(len(neg), dtype=np.int64),
                                np.ones(len(pos),  dtype=np.int64)])
            perm = np.random.default_rng(self.seed + c + 100).permutation(len(X))
            X, y = X[perm], y[perm]

            Xt = torch.from_numpy(X).to(self.device)
            yt = torch.from_numpy(y).to(self.device)
            ld = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(Xt, yt),
                batch_size=min(batch_size, len(X)), shuffle=True)

            gate = GateMLP(self.emb_dim).to(self.device)
            opt  = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=1e-4)
            crit = nn.CrossEntropyLoss()
            for _ in range(epochs):
                gate.train()
                for xb, yb in ld:
                    opt.zero_grad()
                    crit(gate(xb), yb).backward()
                    opt.step()
            gates.append(gate)

        return gates

    @torch.no_grad()
    def _apply_correction(self, embs_np: np.ndarray,
                          asgn: np.ndarray) -> torch.Tensor:
        embs     = torch.from_numpy(embs_np).to(self.device)
        corrected = embs.clone()
        apply    = self._gate_decision(embs_np, asgn)
        for c in range(self.k):
            mask = (asgn == c) & apply
            if mask.any():
                self.adapters[c].eval()
                idx = torch.from_numpy(mask).to(self.device)
                corrected[idx] = self.adapters[c](embs[idx])
        return corrected

    @torch.no_grad()
    def _gate_decision(self, embs_np: np.ndarray,
                       asgn: np.ndarray) -> np.ndarray:
        result = np.zeros(len(embs_np), dtype=bool)
        for c in range(self.k):
            mask = (asgn == c)
            if not mask.any():
                continue
            gate = self.gates[c]
            if gate is None:
                result[mask] = False          # no calib data → keep source behaviour
            else:
                gate.eval()
                p = gate(torch.from_numpy(embs_np[mask]).to(self.device)
                         ).argmax(1).cpu().numpy()
                result[mask] = (p == 1)
        return result

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save fitted PACT state (centroids, adapters, gates) to a .pt file."""
        assert self._fitted, "PACT is not fitted yet."
        torch.save({
            "k":          self.k,
            "emb_dim":    self.emb_dim,
            "penalty":    self.penalty,
            "seed":       self.seed,
            "centroids":  self.centroids,
            "adapters":   self.adapters.state_dict(),
            "gates":      [g.state_dict() if g is not None else None
                           for g in self.gates],
        }, path)
        print(f"[PACT] Saved → {path}")

    def load(self, path: str) -> "PACT":
        """Restore PACT state from a .pt file."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.k          = ckpt["k"]
        self.emb_dim    = ckpt["emb_dim"]
        self.centroids  = ckpt["centroids"]
        self.adapters   = nn.ModuleList(
            [IdentityAdapter(self.emb_dim) for _ in range(self.k)])
        self.adapters.load_state_dict(ckpt["adapters"])
        self.adapters.to(self.device)
        self.gates = []
        for sd in ckpt["gates"]:
            if sd is None:
                self.gates.append(None)
            else:
                g = GateMLP(self.emb_dim).to(self.device)
                g.load_state_dict(sd)
                self.gates.append(g)
        self._fitted = True
        print(f"[PACT] Loaded ← {path}")
        return self
