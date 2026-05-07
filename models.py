from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Time embedding
# =========================

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


# =========================
# Generic MLP block
# =========================

class MLP(nn.Module):
    """
    Simple configurable MLP, but intentionally restricted:
    - activation = SiLU
    - optional LayerNorm
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


# =========================
# Velocity model v_theta(x_t, t)
# =========================

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


# =========================
# Pairwise sigma model sigma_spatial(x0, x1)
# =========================

class PairFeatureBuilder(nn.Module):
    """
    Builds simple handcrafted pair features:
        [x0, x1, x1 - x0, |x1 - x0|, x0 * x1]
    """
    def __init__(self):
        super().__init__()

    def output_dim(self, x_dim: int) -> int:
        return 5 * x_dim

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        x0, x1: [..., D]
        returns: [..., 5D]
        """
        return torch.cat([
            x0,
            x1,
            x1 - x0,
            torch.abs(x1 - x0),
            x0 * x1,
        ], dim=-1)


class PairSigmaMLP(nn.Module):
    """
    sigma_spatial(x0, x1) > 0

    Input:
        x0: [..., D]
        x1: [..., D]

    Output:
        sigma: [...]
    """
    def __init__(
        self,
        x_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        min_sigma: float = 1e-4,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.min_sigma = min_sigma
        self.features = PairFeatureBuilder()

        self.net = MLP(
            in_dim=self.features.output_dim(x_dim),
            out_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_layernorm=use_layernorm,
        )

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        feats = self.features(x0, x1)        # [..., F]
        raw = self.net(feats).squeeze(-1)    # [...]
        sigma = F.softplus(raw) + self.min_sigma
        return sigma


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
        x_dim: int = 2,
        init_sigma: float = 0.1,
        hidden_dim: int = 64,
        num_layers: int = 3,
        min_sigma: float = 1e-4,
        use_layernorm: bool = False,
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
        elif mode == "pair_mlp":
            self.sigma_net = PairSigmaMLP(
                x_dim=x_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                min_sigma=min_sigma,
                use_layernorm=use_layernorm,
            )
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
        elif self.mode == "pair_mlp":
            sigma = self.sigma_net(x0, x1)
        else:
            raise RuntimeError(f"Invalid sigma mode: {self.mode}")

        if sigma.shape != target_shape:
            raise RuntimeError(
                f"SigmaModel returned wrong shape: expected {target_shape}, got {sigma.shape}"
            )

        return sigma.clamp_min(self.min_sigma)
