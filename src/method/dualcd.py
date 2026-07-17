"""DualCD (Liu et al., WWW'26) as a frozen-encoder, head-level CL baseline.

Reference: baselines/DualCD (paper: "We Need a More Robust Classifier: Dual
Causal Learning Empowers Domain-Incremental Time Series Classification").

Same head-only setting as the other scripts/run_cl_baselines_tsfm.py methods
(head-ft, lwf, srcdistill, ewc, mas): the TSFM encoder is frozen and only a
lightweight head is retrained on each new domain's calibration embeddings.
DualCDHead additionally disentangles the frozen embedding Z into a causal
component Z_R and a spurious component Z_I via two orthogonal soft masks
(Eq. 4), and trains through two causal-intervention variants built from
within-class and across-class swaps (Eq. 6-9) instead of plain CE, so the
classifier is pushed to rely on Z_R alone -- which is also what it falls back
to at inference/eval time (y=None), matching the paper's Sec 5.6 analysis.
"""

import torch
from torch import nn


class DualCDHead(nn.Module):
    def __init__(self, emb_dim, num_classes):
        super().__init__()
        self.decompose = nn.Linear(emb_dim, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def disentangle(self, z):
        """Eq. 4: returns (Z_R, Z_I)."""
        m = self.decompose(z)
        mask_r = torch.sigmoid(m)
        mask_i = torch.sigmoid(-m)
        return mask_r * z, mask_i * z

    @staticmethod
    def _same_class_index(y):
        """For every sample i, an index i' != i with y[i'] == y[i] (Eq. 6).

        Small calibration batches can contain only one member of some class;
        when no distinct partner exists, fall back to i itself (no
        perturbation for that sample this step) rather than failing.
        """
        b = y.shape[0]
        out = torch.arange(b, device=y.device)
        for i in range(b):
            idx = (y == y[i]).nonzero(as_tuple=True)[0]
            idx = idx[idx != i]
            if idx.numel() > 0:
                out[i] = idx[torch.randint(0, idx.numel(), (1,), device=y.device)]
        return out

    @staticmethod
    def _other_class_index(y):
        """For every sample i, an index from a class different from y[i] (Eq. 8)."""
        b = y.shape[0]
        out = torch.arange(b, device=y.device)
        for i in range(b):
            idx = (y != y[i]).nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                out[i] = idx[torch.randint(0, idx.numel(), (1,), device=y.device)]
        return out

    def forward(self, z, y=None):
        """Eval / plain inference (y=None): logits from the causal feature
        Z_R alone -- `model(x) = head(extract(x))` keeps working unmodified.

        Training (y given): returns (logits_intra, logits_inter) from the two
        perturbed variants (Eq. 6-9) for the caller to combine into
        L = lam * CE(logits_intra, y) + (1 - lam) * CE(logits_inter, y).
        """
        z_r, z_i = self.disentangle(z)
        if self.training and y is not None:
            intra_variant = z_r + z_i[self._same_class_index(y)]
            inter_variant = z_r + z_r[self._other_class_index(y)]
            return self.classifier(intra_variant), self.classifier(inter_variant)
        return self.classifier(z_r)
