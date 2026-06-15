from typing import Literal, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import SinusoidalTimeEmbedding, inv_softplus


def pad_t_like_x(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if isinstance(t, (float, int)):
        return t
    return t.view(-1, *([1] * (x.dim() - 1)))


def bridge_scale(t: torch.Tensor, eta: float = 1e-5) -> torch.Tensor:
    return torch.sqrt((t + eta) * (1.0 - t + eta))


def squared_event_norm(
    diff: torch.Tensor,
    event_ndim: int = 1,
) -> torch.Tensor:
    """
    Sum squared norm over event dimensions.
    Examples:
        2D:
            diff: [B, n0, n1, D], event_ndim=1 -> [B, n0, n1]
        images:
            diff: [B, n, C, H, W], event_ndim=3 -> [B, n]
    """
    if event_ndim < 1:
        raise ValueError(f"event_ndim must be >= 1, got {event_ndim}")
    if diff.ndim < event_ndim:
        raise ValueError(
            f"diff.ndim must be >= event_ndim, got {diff.ndim=} and {event_ndim=}"
        )

    reduce_dims = tuple(range(diff.ndim - event_ndim, diff.ndim))
    return (diff ** 2).sum(dim=reduce_dims)


def pad_batch_like_candidates(
    batch_tensor: torch.Tensor,
    x: torch.Tensor,
    event_ndim: int,
) -> torch.Tensor:
    """
    Reshape batch tensor of shape [B] (or [B, ...]) so that it is broadcastable
    to candidate dimensions of x, where
        x: [B, *candidate_dims, *event_dims].

    Returns:
        batch_tensor reshaped to [B, ..., 1, ..., 1]
        with as many trailing singleton dims as there are candidate dims in x.
    """
    if x.ndim < 1 + event_ndim:
        raise ValueError(
            f"Expected x.ndim > event_ndim, got {x.ndim=} and {event_ndim=}"
        )
    if batch_tensor.shape[0] != x.shape[0]:
        raise ValueError(
            f"Batch mismatch: {batch_tensor.shape[0]=} vs {x.shape[0]=}"
        )

    n_candidate = x.ndim - 1 - event_ndim
    return batch_tensor.view(*batch_tensor.shape, *([1] * n_candidate))


def extract_xt_batch(
    xt: torch.Tensor,
    event_ndim: int,
) -> torch.Tensor:
    """
    Extract one event tensor per batch item from
        xt: [B, *broadcast_candidate_dims, *event_dims]
    assuming candidate dims are broadcast copies.
    Returns:
        [B, *event_dims]
    """
    if event_ndim < 1:
        raise ValueError(f"event_ndim must be >= 1, got {event_ndim}")
    if xt.ndim < 1 + event_ndim:
        raise ValueError(
            f"Expected xt.ndim >= 1 + event_ndim, got {xt.ndim=} and {event_ndim=}"
        )

    n_candidate = xt.ndim - 1 - event_ndim
    out = xt
    for _ in range(n_candidate):
        out = out.select(dim=1, index=0)
    return out


class BaseSigmaModel(nn.Module):
    def __init__(
        self,
        mode: Literal["constant", "scalar"] = "scalar",
        init_sigma: float = 0.4,
        min_sigma: float = 1e-4,
    ):
        super().__init__()

        if init_sigma <= min_sigma:
            raise ValueError(
                f"init_sigma must be > min_sigma, got "
                f"{init_sigma=} and {min_sigma=}"
            )

        self.mode = mode
        self.min_sigma = min_sigma

        if mode == "constant":
            self.register_buffer(
                "sigma_const",
                torch.tensor(float(init_sigma), dtype=torch.float32),
                persistent=True,
            )
        elif mode == "scalar":
            raw_init = inv_softplus(init_sigma - min_sigma)
            self.sigma_param = nn.Parameter(
                torch.tensor(float(raw_init), dtype=torch.float32)
            )
        else:
            raise ValueError(f"Unknown sigma mode: {mode}")

    def forward(self) -> torch.Tensor:
        if self.mode == "constant":
            return self.sigma_const
        if self.mode == "scalar":
            sigma = self.min_sigma + F.softplus(self.sigma_param)
            return sigma
        raise RuntimeError(f"Invalid sigma mode: {self.mode}")


class ChebyshevTimeScale(nn.Module):
    """
    Low-parametric time scale:
        s(t) = min_value +  sum_k a_k T_k(2t - 1)
    """
    def __init__(
        self,
        degree: int = 8,
        init_value: float = 1.0,
        min_value: float = 1e-4,
        max_value: Optional[float] = 10.0,
    ):
        super().__init__()

        if degree < 0:
            raise ValueError(f"degree must be non-negative, got {degree}")
        if init_value <= min_value:
            raise ValueError(
                f"init_value must be > min_value, got {init_value=} and {min_value=}"
            )

        self.degree = degree
        self.min_value = min_value
        self.max_value = max_value

        self.coeffs = nn.Parameter(torch.zeros(degree + 1))

        radius = 1.0 / degree if degree > 0 else 0.0
        with torch.no_grad():
            nn.init.constant_(self.coeffs[0], init_value)
            if self.degree >= 1 and radius > 0:
                nn.init.uniform_(self.coeffs[1:], -radius, radius)

    def _basis(self, t: torch.Tensor) -> torch.Tensor:
        z = 2.0 * t - 1.0

        basis = [torch.ones_like(z)]
        if self.degree >= 1:
            basis.append(z)
        for _ in range(2, self.degree + 1):
            basis.append(2.0 * z * basis[-1] - basis[-2])

        return torch.stack(basis, dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError(f"Expected t shape [B], got {t.shape}")

        basis = self._basis(t)
        out = basis @ self.coeffs

        if self.max_value is None:
            out = torch.clamp(out, min=self.min_value)
        else:
            out = torch.clamp(out, min=self.min_value, max=self.max_value)

        return out


class TimeMLPScale(nn.Module):
    def __init__(
        self,
        mode: Literal["positive", "residual"] = "positive",
        time_emb_dim: int = 32,
        hidden_dim: int = 32,
        num_layers: int = 2,
        min_value: float = 1e-4,
        max_value: Optional[float] = 10.0,
        use_sinusoidal: bool = True,
        use_layernorm: bool = True,
    ):
        super().__init__()

        if mode not in {"positive", "residual"}:
            raise ValueError(f"Unknown mode={mode}")

        self.mode = mode
        self.use_sinusoidal = use_sinusoidal
        self.min_value = min_value
        self.max_value = max_value

        if use_sinusoidal:
            self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
            input_dim = time_emb_dim
        else:
            self.time_emb = None
            input_dim = 1

        layers: list[nn.Module] = []
        dim = input_dim

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.SiLU())
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            dim = hidden_dim

        self.final = nn.Linear(dim, 1)
        layers.append(self.final)
        self.net = nn.Sequential(*layers)

        self._init_final()  # без - 03-gaussian_t_const-mult-e_wo_final

    def _init_final(self) -> None:
        with torch.no_grad():
            # # 03-gaussian_t_const-mult-e_hid_xavier
            # for m in self.net.modules():
            #     if isinstance(m, nn.Linear):
            #         nn.init.xavier_uniform_(m.weight)
            #         if m.bias is not None:
            #             nn.init.zeros_(m.bias)

            nn.init.zeros_(self.final.weight)
            # nn.init.normal_(self.final.weight, mean=0.0, std=1e-3) # 03-gaussian_t_const-mult-c_final_w_normal

            if self.mode == "positive":
                raw_init = inv_softplus(1.0 - self.min_value)
                self.final.bias.fill_(float(raw_init))
            elif self.mode == "residual":
                self.final.bias.zero_()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError(f"Expected t shape [B], got {t.shape}")

        h = self.time_emb(t) if self.use_sinusoidal else t[:, None]
        raw = self.net(h).squeeze(-1)

        if self.mode == "positive":
            out = self.min_value + F.softplus(raw)
        elif self.mode == "residual":
            out = 1.0 + torch.tanh(raw)
        else:
            raise RuntimeError(f"Invalid mode={self.mode}")

        if self.max_value is not None:
            out = torch.clamp(out, min=self.min_value, max=self.max_value)

        return out


class TimeXMLPScale(nn.Module):
    def __init__(
        self,
        x_dim: int,
        time_emb_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 2,
        init_value: float = 1.0,
        min_value: float = 1e-4,
        max_value: Optional[float] = 10.0,
        use_layernorm: bool = True,
    ):
        super().__init__()

        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.min_value = min_value
        self.max_value = max_value

        input_dim = time_emb_dim + x_dim

        layers: list[nn.Module] = []
        dim = input_dim

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.SiLU())
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            dim = hidden_dim

        self.final = nn.Linear(dim, 1)
        layers.append(self.final)
        self.net = nn.Sequential(*layers)

        self._init_final(init_value)

    def _init_final(self, init_value: float) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.final.weight)
            raw_init = inv_softplus(init_value - self.min_value)
            self.final.bias.fill_(float(raw_init))

    def forward(self, t: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        xt_flat = xt.reshape(xt.shape[0], -1)
        h = torch.cat([t_emb, xt_flat], dim=-1)

        raw = self.net(h).squeeze(-1)
        out = self.min_value + F.softplus(raw)

        if self.max_value is not None:
            out = torch.clamp(out, max=self.max_value)

        return out


class BaseRBFKernel(nn.Module):
    """
    Base RBF kernel:
        logw = -||x_t - mu_t||^2 / (2 sigma^2)
    Child classes only implement sigma().
    """
    def __init__(
        self,
        event_ndim: int = 1,
        min_sigma: float = 1e-6,
        use_prefactor: bool = False,
    ):
        super().__init__()

        if event_ndim < 1:
            raise ValueError(f"event_ndim must be >= 1, got {event_ndim}")

        self.event_ndim = event_ndim
        self.min_sigma = min_sigma
        self.use_prefactor = use_prefactor

    def sigma(
        self,
        xt: torch.Tensor,
        mu_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def event_dim(self, xt: torch.Tensor) -> int:
        return math.prod(xt.shape[-self.event_ndim:])

    def forward(
        self,
        xt: torch.Tensor,
        mu_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            xt: [B, *candidate_xt_dims, *event_dims]
            mu_t: [B, *candidate_dims, *event_dims]
            t: [B]

        Returns:
            log_w: [B, *candidate_dims]
        """
        if t.ndim != 1:
            raise ValueError(f"Expected t shape [B], got {t.shape}")
        if xt.shape[0] != t.shape[0] or mu_t.shape[0] != t.shape[0]:
            raise ValueError(
                f"Batch mismatch: {xt.shape[0]=}, {mu_t.shape[0]=}, {t.shape[0]=}"
            )

        diff = xt - mu_t
        dist2 = squared_event_norm(diff, event_ndim=self.event_ndim)

        sigma = self.sigma(
            xt=xt,
            mu_t=mu_t,
            t=t,
        ).clamp_min(self.min_sigma)

        logw = -dist2 / (2.0 * sigma ** 2)

        if self.use_prefactor:
            D = self.event_dim(xt)
            logw = logw - D * torch.log(sigma)

        return logw


class TimeRBFKernel(BaseRBFKernel):
    """
    RBF kernel with scalar bridge width:
        sigma(t) = sigma_base * sqrt(t(1-t))
    """
    def __init__(
        self,
        sigma_base: nn.Module,
        eta: float = 1e-5,
        event_ndim: int = 1,
        min_sigma: float = 1e-6,
        use_prefactor: bool = False,
    ):
        super().__init__(
            event_ndim=event_ndim,
            min_sigma=min_sigma,
            use_prefactor=use_prefactor,
        )
        self.sigma_base = sigma_base
        self.eta = eta

    def sigma(
        self,
        xt: torch.Tensor,
        mu_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        del xt
        sigma = self.sigma_base() * bridge_scale(t, eta=self.eta)
        return pad_batch_like_candidates(sigma, mu_t, self.event_ndim)


class TimeScaleRBFKernel(BaseRBFKernel):
    """
    RBF kernel with learned time width:
        sigma(t) = s(t) * sqrt(t(1-t))
    s(t) contains the whole width, no extra scalar sigma.
    """
    def __init__(
        self,
        time_scale: nn.Module,
        eta: float = 1e-5,
        event_ndim: int = 1,
        min_sigma: float = 1e-6,
        use_prefactor: bool = False,
    ):
        super().__init__(
            event_ndim=event_ndim,
            min_sigma=min_sigma,
            use_prefactor=use_prefactor,
        )
        self.time_scale = time_scale
        self.eta = eta

    def sigma(
        self,
        xt: torch.Tensor,
        mu_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        sigma = self.time_scale(t) * bridge_scale(t, eta=self.eta)
        return pad_batch_like_candidates(sigma, mu_t, self.event_ndim)


class PointwiseRBFKernel(BaseRBFKernel):
    """
    RBF kernel with pointwise width:

        sigma(t, x_t) = f(t, x_t) * sqrt(t(1-t))
    """
    def __init__(
        self,
        tx_sigma_model: nn.Module,
        eta: float = 1e-5,
        event_ndim: int = 1,
        min_sigma: float = 1e-6,
        use_prefactor: bool = False,
    ):
        super().__init__(
            event_ndim=event_ndim,
            min_sigma=min_sigma,
            use_prefactor=use_prefactor,
        )
        self.tx_sigma_model = tx_sigma_model
        self.eta = eta

    def sigma(
        self,
        xt: torch.Tensor,
        mu_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        xt_batch = extract_xt_batch(xt, self.event_ndim)
        sigma_batch = self.tx_sigma_model(t, xt_batch)
        sigma_batch = sigma_batch * bridge_scale(t, eta=self.eta)
        return pad_batch_like_candidates(sigma_batch, mu_t, self.event_ndim)


class LearnedEnergyKernel(nn.Module):
    """
    Learned exponential kernel:
        logw = -E(t, x_t, d)
    where d is one of:
        norm, squared_norm, vector.
    """
    def __init__(
        self,
        x_dim: int,
        time_emb_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 2,
        distance_input: Literal["norm", "squared_norm", "vector"] = "norm",
        event_ndim: int = 1,
        min_value: float = 1e-4,
        use_layernorm: bool = True,
    ):
        super().__init__()

        if distance_input not in {"norm", "squared_norm", "vector"}:
            raise ValueError(f"Unknown distance_input={distance_input}")
        if event_ndim < 1:
            raise ValueError(f"event_ndim must be >= 1, got {event_ndim}")

        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.distance_input = distance_input
        self.event_ndim = event_ndim
        self.min_value = min_value

        if distance_input == "vector":
            input_dim = time_emb_dim + x_dim + x_dim
        else:
            input_dim = time_emb_dim + x_dim + 1

        layers: list[nn.Module] = []
        dim = input_dim

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.SiLU())
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            dim = hidden_dim

        self.final = nn.Linear(dim, 1)
        layers.append(self.final)
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        xt: torch.Tensor,
        mu_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        diff = xt - mu_t
        dist2 = squared_event_norm(diff, event_ndim=self.event_ndim)

        xt_batch = extract_xt_batch(xt, self.event_ndim)
        xt_flat = xt_batch.reshape(xt_batch.shape[0], -1)

        t_emb = self.time_emb(t)

        # Expand t and xt to candidate dimensions.
        candidate_shape = dist2.shape[1:]
        for _ in candidate_shape:
            t_emb = t_emb[:, None, ...]
            xt_flat = xt_flat[:, None, ...]

        t_emb = t_emb.expand(*dist2.shape, t_emb.shape[-1])
        xt_flat = xt_flat.expand(*dist2.shape, xt_flat.shape[-1])

        if self.distance_input == "norm":
            d = torch.sqrt(dist2.clamp_min(0.0))[..., None]

        elif self.distance_input == "squared_norm":
            d = dist2[..., None]

        elif self.distance_input == "vector":
            d = diff.reshape(*dist2.shape, -1)

        else:
            raise RuntimeError(f"Invalid distance_input={self.distance_input}")

        h = torch.cat([t_emb, xt_flat, d], dim=-1)
        raw = self.net(h).squeeze(-1)

        energy = self.min_value + F.softplus(raw)
        return -energy
