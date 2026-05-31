"""
Sigma models for Explicit Flow Matching.

Includes:
- SpatialSigmaModel: sigma(x0, x1) - spatially varying noise scale
- TimeSigmaMultiplier: f(t) - temporal modulation of sigma
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import SinusoidalTimeEmbedding, inv_softplus


class SpatialSigmaModel(nn.Module):
    """
    Controlled family of sigma_space(x0, x1) models.

    Supported modes:
        - "constant" : fixed scalar sigma
        - "scalar"   : learnable global scalar sigma
        - "pair_mlp" : learnable sigma(x0, x1) via small MLP

    Input:
        x0, x1: [..., D]
    Output:
        sigma_space: [...]
    """
    def __init__(
        self,
        mode: Literal["constant", "scalar", "pair_mlp"] = "scalar",
        init_sigma: float = 0.1,
        min_sigma: float = 1e-4,
    ):
        super().__init__()

        if init_sigma <= 0:
            raise ValueError("init_sigma must be positive.")
        if min_sigma <= 0:
            raise ValueError("min_sigma must be positive.")

        self.mode = mode
        self.min_sigma = min_sigma

        if mode == "constant":
            self.register_buffer(
                "sigma_const",
                torch.tensor(float(init_sigma), dtype=torch.float32),
                persistent=True,
            )
        elif mode == "scalar":
            # log + exp instead of softplus?
            raw_init = torch.log(torch.expm1(torch.tensor(init_sigma - min_sigma, dtype=torch.float32)))
            self.sigma_param = nn.Parameter(raw_init)
        else:
            raise ValueError(f"Unknown sigma mode: {mode}")

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        # add check for broadcastibility
        if x0.ndim < 2:
            raise ValueError(f"Expected x0, x1 with shape [..., D], got {x0.shape}")

        target_shape = x0.shape[:-1]

        if self.mode == "constant":
            sigma = self.sigma_const.expand(target_shape)
        elif self.mode == "scalar":
            sigma_scalar = F.softplus(self.sigma_param) + self.min_sigma
            sigma = sigma_scalar.expand(target_shape)
        else:
            raise RuntimeError(f"Invalid sigma mode: {self.mode}")

        if sigma.shape != target_shape:
            raise RuntimeError(
                f"SigmaModel returned wrong shape: expected {target_shape}, got {sigma.shape}"
            )

        return sigma.clamp_min(self.min_sigma)


# =========================
# Time multiplier model f(t)
# =========================

class TimeSigmaMultiplier(nn.Module):
    """
    Temporal modulation of sigma: f(t) * t * (1-t)

    Input:
        t: [B]
    Output:
        f_t: [B]
    """
    def __init__(
        self,
        time_emb_dim: int = 32,
        hidden_dim: int = 32,
        num_layers: int = 2,
        init_value: float = 1.0,
        min_value: float = 1e-4,
        max_value: float | None = None,
        use_sinusoidal: bool = True,
        mode: str = "multiplier"
    ):
        super().__init__()
        self.use_sinusoidal = use_sinusoidal
        self.min_value = min_value
        self.max_value = max_value
        self.mode = mode

        if use_sinusoidal:
            self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
            input_dim = time_emb_dim
        else:
            self.time_emb = None
            input_dim = 1

        layers = []
        dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.LayerNorm(hidden_dim))
            dim = hidden_dim

        self.final = nn.Linear(dim, 1)
        layers.append(self.final)
        self.net = nn.Sequential(*layers)
        self._init_final()

    def _init_final(self, init_value: float = 1.0, init_addition: float = 1e-3) -> None:
        """
        mode == "multiplier":
            f(t) = min_value + softplus(raw)

        mode == "additive":
            f(t) = 1 + softplus(raw)
        """
        with torch.no_grad():
            nn.init.zeros_(self.final.weight)

            if self.mode == "multiplier":
                raw_init = inv_softplus(init_value - self.min_value)
                self.final.bias.fill_(raw_init)

            elif self.mode == "additive":
                raw_init = inv_softplus(init_addition)
                self.final.bias.fill_(raw_init)

            else:
                raise ValueError(f"Unknown mode={self.mode}")

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError(f"Expected t shape [B], got {t.shape}")

        if self.use_sinusoidal:
            h = self.time_emb(t)
        else:
            h = t[:, None]

        raw = self.net(h).squeeze(-1)

        if self.mode == "multiplier":
            # f(t) starts near init_value, e.g. 1.0
            f_t = self.min_value + F.softplus(raw)

        elif self.mode == "additive":
            # f(t) starts near 1 + init_addition, e.g. 1.001
            f_t = 1.0 + F.softplus(raw)

        if self.max_value is not None:
            f_t = torch.clamp(f_t, max=self.max_value)

        return f_t
