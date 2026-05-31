"""
Conditional Flow Matching (CFM) method implementation.

Reference:
    Improving and Generalizing Flow-Based Generative Models
    with minibatch optimal transport, Preprint, Tong et al.
"""

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


class ConditionalFlowMatcher(nn.Module):
    """
    Basic Conditional Flow Matching implementation.

    Probability path: N(t * x1 + (1 - t) * x0, sigma)
    Conditional vector field: u(x1|x0) = x1 - x0
    """

    def __init__(self, sigma: float = 0.1):
        super().__init__()
        self.sigma = sigma

    def compute_mu_t(
            self,
            x0: torch.Tensor,
            x1: torch.Tensor,
            t: torch.Tensor
    ) -> torch.Tensor:
        """Compute mean of the probability path: mu_t = t * x1 + (1 - t) * x0"""
        t_pad = pad_t_like_x(t, x0)
        return t_pad * x1 + (1.0 - t_pad) * x0

    def compute_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Compute standard deviation (constant for basic CFM)."""
        del t
        return self.sigma

    def sample_xt(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        epsilon: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample from the probability path: xt = mu_t + sigma * epsilon"""
        if epsilon is None:
            epsilon = torch.randn_like(x0)

        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t_pad = pad_t_like_x(sigma_t, x0)

        return mu_t + sigma_t_pad * epsilon

    def compute_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        """Compute conditional vector field: u(x1|x0) = x1 - x0"""
        del t, xt
        return x1 - x0

    def sample_location_and_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        return_noise: bool = False,
    ) -> Tuple:
        """
        Sample location xt and compute conditional flow ut.

        Args:
            x0: source samples [B, D]
            x1: target samples [B, D]
            t: time samples [B] (optional, sampled uniformly if None)
            return_noise: whether to return noise epsilon

        Returns:
            t: time samples [B]
            xt: samples from probability path [B, D]
            ut: conditional vector field [B, D]
            epsilon (optional): noise sample
        """
        B = x0.shape[0]

        if t is None:
            t = torch.rand(B, device=x0.device, dtype=x0.dtype)

        epsilon = torch.randn_like(x0)
        xt = self.sample_xt(x0, x1, t, epsilon)
        ut = self.compute_conditional_flow(x0, x1, t, xt)

        if return_noise:
            return t, xt, ut, epsilon
        return t, xt, ut


class ExactOptimalTransportConditionalFlowMatcher(ConditionalFlowMatcher):
    """
    OT-CFM: Conditional Flow Matching with exact optimal transport coupling.

    Overrides sample_location_and_conditional_flow to sample (x0, x1) pairs
    according to the exact OT plan before computing the conditional flow.
    """

    def __init__(self, sigma: Union[float, int] = 0.0):
        super().__init__(sigma)
        self.ot_sampler = OTPlanSampler(method="exact")

    def sample_location_and_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        return_noise: bool = False,
    ) -> Tuple:
        """
        Sample xt and compute ut with respect to the minibatch OT plan.

        Args:
            x0: source minibatch [B, D]
            x1: target minibatch [B, D]
            t: time samples [B] (optional)
            return_noise: whether to return noise epsilon

        Returns:
            t, xt, ut, (epsilon if return_noise)
        """
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)


class SchrodingerBridgeConditionalFlowMatcher(ConditionalFlowMatcher):
    """
    SB-CFM: Schrödinger Bridge Conditional Flow Matching.

    Uses entropic OT coupling with time-dependent sigma:
        sigma_t = sigma * sqrt(t * (1 - t))
    and modified conditional flow field.
    """

    def __init__(self, sigma: Union[float, int] = 1.0, ot_method: str = "exact"):
        if sigma <= 0:
            raise ValueError(f"Sigma must be strictly positive, got {sigma}.")
        elif sigma < 1e-3:
            warnings.warn("Small sigma values may lead to numerical instability.")

        super().__init__(sigma)
        self.ot_method = ot_method
        self.ot_sampler = OTPlanSampler(method=ot_method, reg=2*self.sigma**2)

    def compute_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Compute time-dependent sigma: sigma_t = sigma * sqrt(t * (1 - t))"""
        return self.sigma * torch.sqrt(t * (1 - t))

    def compute_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute SB conditional vector field:
        ut = (1 - 2t) / (2t(1-t)) * (xt - mu_t) + (x1 - x0)
        """
        t_pad = pad_t_like_x(t, x0)
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t_prime_over_sigma_t = (1 - 2 * t_pad) / (2 * t_pad * (1 - t_pad) + 1e-8)
        return sigma_t_prime_over_sigma_t * (xt - mu_t) + x1 - x0

    def sample_location_and_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        return_noise: bool = False,
    ) -> Tuple:
        """
        Sample xt and compute ut with entropic OT coupling.

        Args:
            x0: source minibatch [B, D]
            x1: target minibatch [B, D]
            t: time samples [B] (optional)
            return_noise: whether to return noise epsilon

        Returns:
            t, xt, ut, (epsilon if return_noise)
        """
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
