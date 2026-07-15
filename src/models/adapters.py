"""Per-concept embedding adapters (the "A" in the PACT plugin).

An adapter is a small map on the frozen backbone's pre-head embedding space
that corrects a target domain's embeddings so the frozen head classifies them
correctly. Two flavours:

* IdentityAdapter — full-rank linear layer, identity-initialised (default).
* LoRAAdapter     — identity + low-rank residual B @ A (parameter-efficient
                    ablation; see run_sequential_pact_lora).

Each domain gets a bank of K adapters (one per concept). Once a domain's bank
is trained it is frozen forever, which is what gives PACT zero host-level
forgetting by construction.
"""

import torch
import torch.nn as nn


class IdentityAdapter(nn.Module):
    """Full-rank linear adapter, identity-initialised."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

    def identity_penalty(self, device=None):
        """Sum of squared deviation from identity (weight) and zero (bias)."""
        eye = torch.eye(self.linear.weight.shape[0], device=self.linear.weight.device)
        return torch.sum((self.linear.weight - eye) ** 2) + torch.sum(self.linear.bias ** 2)


class LoRAAdapter(nn.Module):
    """Identity + low-rank residual: x + (x @ A^T) @ B^T.

    B is zero-initialised so the adapter starts as the identity. Only 2*dim*r
    parameters per adapter versus dim*dim for the full-rank variant.
    """

    def __init__(self, dim: int = 128, rank: int = 8):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(dim, rank, bias=False)
        self.B = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.A.weight, std=1.0 / max(rank, 1))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return x + self.B(self.A(x))

    def identity_penalty(self, device=None):
        return torch.sum(self.B.weight ** 2)


def make_adapter_bank(k: int, dim: int = 128, kind: str = "identity",
                      rank: int = 8) -> nn.ModuleList:
    """Build a bank of K adapters for one domain."""
    if kind == "identity":
        return nn.ModuleList([IdentityAdapter(dim) for _ in range(k)])
    if kind == "lora":
        return nn.ModuleList([LoRAAdapter(dim, rank) for _ in range(k)])
    raise ValueError(f"Unknown adapter kind: {kind!r}")
