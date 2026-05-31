"""
Time embedding modules for Flow Matching models.
"""

import math

import torch
import torch.nn as nn


def inv_softplus(y: float) -> float:
    """Inverse softplus: log(exp(y) - 1)"""
    return math.log(math.expm1(y))


class SinusoidalTimeEmbedding(nn.Module):
    """
    Maps t in R^[B] to sinusoidal embedding in R^[B, emb_dim].
    """
    def __init__(self, emb_dim: int = 32, max_period: float = 10000.0):
        super().__init__()
        if emb_dim % 2 != 0:
            raise ValueError("emb_dim must be even.")
        self.emb_dim = emb_dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B]
        returns: [B, emb_dim]
        """
        if t.ndim != 1:
            raise ValueError(f"Expected t shape [B], got {t.shape}")

        half = self.emb_dim // 2
        device = t.device
        dtype = t.dtype

        freqs = torch.exp(
            -torch.log(torch.tensor(self.max_period, device=device, dtype=dtype))
            * torch.arange(half, device=device, dtype=dtype) / half
        )  # [half]

        args = t[:, None] * freqs[None, :]  # [B, half]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return emb
