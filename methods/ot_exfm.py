from typing import Optional, Tuple, Union
import warnings
import torch
import torch.nn as nn

from .optimal_transport import OTPlanSampler


def pad_t_like_x(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Pad time tensor t to match the shape of x."""
    if isinstance(t, (float, int)):
        return t
    return t.view(-1, *([1] * (x.dim() - 1)))


class ExplicitOptimalTransportFlowMatcher(nn.Module):
    """
    Explicit Flow Matching with minibatch OT coupling.

    Pipeline:
        1. sample OT-coupled pairs (x0, x1) from minibatch
        2. sample x_t from Gaussian bridge:
               x_t ~ N((1-t)x0 + t x1, sigma^2 (t+eta)(1-t+eta) I)
        3. compute explicit flow by weighted average over the OT-coupled minibatch pairs
           using a single sum over candidate pairs

    Weights:
        log w_{b,k} = - ||x_t^b - mu_t^k||^2 / (2 sigma_t(b,k)^2)
                      [- D log sigma_t(b,k) if use_full_gaussian_prefactor]
    """
    def __init__(
        self,
        sigma: Union[float, int] = 0.4,
        ot_method: str = "exact",
        eta: float = 1e-5,
        use_full_gaussian_prefactor: bool = False,
    ):
        super().__init__()
        if sigma <= 0:
            raise ValueError(f"Sigma must be strictly positive, got {sigma}.")
        elif sigma < 1e-3:
            warnings.warn("Small sigma values may lead to numerical instability.")

        self.sigma = float(sigma)
        self.eta = eta
        self.use_full_gaussian_prefactor = use_full_gaussian_prefactor

        reg = None
        if ot_method != "exact":
            reg = 2 * self.sigma ** 2

        if reg is None:
            self.ot_sampler = OTPlanSampler(method=ot_method)
        else:
            self.ot_sampler = OTPlanSampler(method=ot_method, reg=reg)

    def compute_mu_t(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        t_pad = pad_t_like_x(t, x0)
        return (1.0 - t_pad) * x0 + t_pad * x1

    def bridge_scale(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((t + self.eta) * (1.0 - t + self.eta))

    def compute_sigma_t(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Constant-in-space bridge std:
            sigma_t = sigma * sqrt((t+eta)(1-t+eta))
        """
        del x0, x1
        return self.sigma * self.bridge_scale(t)

    def sample_xt(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        epsilon: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if epsilon is None:
            epsilon = torch.randn_like(x0)
        mu_t = self.compute_mu_t(x0, x1, t)           # [B, D]
        sigma_t = self.compute_sigma_t(x0, x1, t)     # [B]
        return mu_t + pad_t_like_x(sigma_t, x0) * epsilon

    def compute_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        """
        u = (x1 - x0) + ((1 - 2t) / (2(t+eta)(1-t+eta))) * (xt - mu_t)

        Broadcastable shapes:
            x0: [1, K, D] or [B, D]
            x1: [1, K, D] or [B, D]
            t : [B]
            xt: [B, 1, D] or [B, D]

        Returns:
            [B, K, D] or [B, D]
        """
        t_pad = pad_t_like_x(t, xt)
        mu_t = (1.0 - t_pad) * x0 + t_pad * x1
        denom = 2.0 * (t_pad + self.eta) * (1.0 - t_pad + self.eta)
        return (x1 - x0) + ((1.0 - 2.0 * t_pad) / denom) * (xt - mu_t)

    def _compute_log_weight(
        self,
        xt: torch.Tensor,       # [B,1,D]
        mu_t: torch.Tensor,     # [B,K,D]
        sigma_t: torch.Tensor,  # [B] or [B,K]
    ) -> torch.Tensor:
        D = xt.shape[-1]
        dist2 = ((xt - mu_t) ** 2).sum(dim=-1)  # [B,K]
        logw = -dist2 / (2.0 * sigma_t ** 2)
        if self.use_full_gaussian_prefactor:
            logw = logw - D * torch.log(sigma_t)
        return logw

    def compute_explicit_flow(
        self,
        xt: torch.Tensor,      # [B,D]
        t: torch.Tensor,       # [B]
        x0_ref: torch.Tensor,  # [K,D]
        x1_ref: torch.Tensor,  # [K,D]
    ) -> torch.Tensor:
        assert xt.ndim == 2
        assert x0_ref.ndim == 2 and x1_ref.ndim == 2
        assert t.ndim == 1
        assert xt.shape[0] == t.shape[0]
        assert x0_ref.shape == x1_ref.shape
        assert xt.shape[1] == x0_ref.shape[1]

        xt_exp = xt[:, None, :]         # [B,1,D]
        x0_exp = x0_ref[None, :, :]     # [1,K,D]
        x1_exp = x1_ref[None, :, :]     # [1,K,D]
        t_exp = t[:, None, None]        # [B,1,1]

        mu_t = (1.0 - t_exp) * x0_exp + t_exp * x1_exp   # [B,K,D]

        sigma_t = self.compute_sigma_t(x0_exp, x1_exp, t)  # [B]
        sigma_t = sigma_t[:, None]                         # [B,1], broadcast to [B,K]

        logw = self._compute_log_weight(
            xt=xt_exp,
            mu_t=mu_t,
            sigma_t=sigma_t,
        )  # [B,K]

        v_cond = self.compute_conditional_flow(
            x0=x0_exp,
            x1=x1_exp,
            t=t,
            xt=xt_exp,
        )  # [B,K,D]

        w = torch.softmax(logw, dim=1)               # [B,K]
        ut = (w[..., None] * v_cond).sum(dim=1)      # [B,D]
        return ut

    def sample_location_and_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        epsilon: Optional[torch.Tensor] = None,
        return_noise: bool = False,
    ) -> Tuple:
        """
        Sample xt and compute explicit flow with minibatch OT coupling.

        Args:
            x0: source minibatch [B, D]
            x1: target minibatch [B, D]
            t: time samples [B] (optional)
            epsilon: noise [B, D] (optional)
            return_noise: whether to return sampled noise

        Returns:
            t: [B]
            xt: [B,D]
            ut: [B,D]
            epsilon: [B,D] (optional)
        """
        x0_ot, x1_ot = self.ot_sampler.sample_plan(x0, x1)

        B = x0_ot.shape[0]
        if t is None:
            t = torch.rand(B, device=x0_ot.device, dtype=x0_ot.dtype)
        if epsilon is None:
            epsilon = torch.randn_like(x0_ot)

        xt = self.sample_xt(
            x0=x0_ot,
            x1=x1_ot,
            t=t,
            epsilon=epsilon,
        )

        ut = self.compute_explicit_flow(
            xt=xt,
            t=t,
            x0_ref=x0_ot,
            x1_ref=x1_ot,
        )

        if return_noise:
            return t, xt, ut, epsilon
        return t, xt, ut
