"""
Velocity field models for Flow Matching.
"""

from typing import Literal

import torch
import torch.nn as nn
from .embeddings import SinusoidalTimeEmbedding


class MLP(nn.Module):
    """
    Simple configurable MLP with SiLU activation and optional LayerNorm.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        use_layernorm: bool = False,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers = []

        if num_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))

            for _ in range(num_layers - 2):
                layers.append(nn.SiLU())
                if use_layernorm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.Linear(hidden_dim, hidden_dim))

            layers.append(nn.SiLU())
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VelocityMLP(nn.Module):
    """
    Velocity field model for toy 2D data:
        v_theta(x_t, t)

    Input:
        x: [B, x_dim]
        t: [B]

    Output:
        v: [B, x_dim]
    """
    def __init__(
        self,
        x_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 4,
        time_conditioning: Literal["scalar", "sinusoidal"] = "scalar",
        time_emb_dim: int = 32,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.time_conditioning = time_conditioning

        if self.time_conditioning == "scalar":
            self.time_embed = None
            in_dim = x_dim + 1
        elif self.time_conditioning == "sinusoidal":
            self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
            in_dim = x_dim + time_emb_dim
        else:
            raise ValueError(f"Unknown time_conditioning: {time_conditioning}")

        self.net = MLP(
            in_dim=in_dim,
            out_dim=x_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_layernorm=use_layernorm,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: [B, x_dim]
        t: [B]
        """
        if x.ndim != 2:
            raise ValueError(f"Expected x shape [B, x_dim], got {x.shape}")
        if t.ndim != 1:
            raise ValueError(f"Expected t shape [B], got {t.shape}")
        if x.shape[0] != t.shape[0]:
            raise ValueError(f"Batch mismatch: x.shape={x.shape}, t.shape={t.shape}")

        if self.time_conditioning == "scalar":
            t_feat = t[:, None]
        else:
            t_feat = self.time_embed(t)

        x_in = torch.cat([x, t_feat], dim=-1)
        return self.net(x_in)
