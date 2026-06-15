"""
Explicit Flow Matching implementation.

weight_kernel: used to compute log-weights of candidate trajectories
"""

from typing import Optional

import torch
import torch.nn as nn


def pad_t_like_x(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Pad time tensor t to match the shape of x."""
    if isinstance(t, (float, int)):
        return t
    return t.view(-1, *([1] * (x.dim() - 1)))


class ExplicitFlowMatcher(nn.Module):
    """
    Explicit Flow Matcher.

    weight_kernel:
        module(xt, mu_t, t) -> logw
    """
    def __init__(
        self,
        weight_kernel: nn.Module,
        sampling_sigma: float = 0.5,
        eta: float = 1e-5,
        chunk_n0: Optional[int] = None,
        chunk_n1: Optional[int] = None,
    ):
        super().__init__()

        self.sampling_sigma = sampling_sigma
        self.eta = eta

        self.weight_kernel = weight_kernel

        self.chunk_n0 = chunk_n0
        self.chunk_n1 = chunk_n1

    def compute_mu_t(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        t_pad = pad_t_like_x(t, x0)
        return (1.0 - t_pad) * x0 + t_pad * x1

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
        bridge_scale = torch.sqrt((t + self.eta) * (1.0 - t + self.eta))
        return self.sampling_sigma * bridge_scale

    def sample_xt(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        epsilon: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if epsilon is None:
            epsilon = torch.randn_like(x0)

        mu_t = self.compute_mu_t(x0, x1, t)         # [B, D]
        sigma_t = self.compute_sigma_t(x0, x1, t)

        return mu_t + pad_t_like_x(sigma_t, x0) * epsilon

    def compute_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Conditional flow for Gaussian bridge with variance proportional to t(1-t):
            u_t = x1 - x0 + ((1 - 2t) / (2(t+eta)(1-t+eta))) * (x_t - mu_t)

        Broadcastable shapes:
            x0: [1,n0,1,D] or [B,D]
            x1: [1,1,n1,D] or [B,D]
            t : [B]
            xt: [B,1,1,D] or [B,D]
        """
        t_pad = pad_t_like_x(t, xt)

        mu_t = (1.0 - t_pad) * x0 + t_pad * x1
        denom = 2.0 * (t_pad + self.eta) * (1.0 - t_pad + self.eta)

        return (x1 - x0) + ((1.0 - 2.0 * t_pad) / denom) * (xt - mu_t)

    def compute_explicit_flow(
        self,
        xt: torch.Tensor,      # [B,D]
        t: torch.Tensor,       # [B]
        x0_ref: torch.Tensor,  # [N0,D]
        x1_ref: torch.Tensor,  # [N1,D]
        chunk_n0: Optional[int] = None,
        chunk_n1: Optional[int] = None,
    ) -> torch.Tensor:
        assert xt.ndim == 2
        assert x0_ref.ndim == 2 and x1_ref.ndim == 2
        assert t.ndim == 1
        assert xt.shape[0] == t.shape[0]
        assert xt.shape[1] == x0_ref.shape[1] == x1_ref.shape[1]

        B, D = xt.shape
        N0 = x0_ref.shape[0]
        N1 = x1_ref.shape[0]

        chunk_n0 = chunk_n0 or self.chunk_n0 or N0
        chunk_n1 = chunk_n1 or self.chunk_n1 or N1

        xt_exp = xt[:, None, None, :]     # [B,1,1,D]
        t_exp = t[:, None, None, None]    # [B,1,1,1]

        running_max = torch.full(
            (B, 1, 1),
            -torch.inf,
            device=xt.device,
            dtype=xt.dtype,
        )
        running_den = torch.zeros(B, 1, device=xt.device, dtype=xt.dtype)
        running_num = torch.zeros(B, D, device=xt.device, dtype=xt.dtype)

        for i0 in range(0, N0, chunk_n0):
            x0_chunk = x0_ref[i0:i0 + chunk_n0]      # [n0,D]
            x0_exp = x0_chunk[None, :, None, :]      # [1,n0,1,D]

            for j0 in range(0, N1, chunk_n1):
                x1_chunk = x1_ref[j0:j0 + chunk_n1]  # [n1,D]
                x1_exp = x1_chunk[None, None, :, :]  # [1,1,n1,D]

                mu_t = (1.0 - t_exp) * x0_exp + t_exp * x1_exp      # [B,n0,n1,D]

                logw = self.weight_kernel(
                    xt=xt_exp,
                    mu_t=mu_t,
                    t=t,
                )

                v_cond = self.compute_conditional_flow(
                    x0=x0_exp,
                    x1=x1_exp,
                    t=t,
                    xt=xt_exp,
                )  # [B,n0,n1,D]

                chunk_max = logw.amax(dim=(1, 2), keepdim=True)     # [B,1,1]
                new_max = torch.maximum(running_max, chunk_max)     # [B,1,1]

                old_rescale = torch.exp(running_max - new_max)      # [B,1,1]
                old_rescale_flat = old_rescale.view(B, 1)           # [B,1]

                w = torch.exp(logw - new_max)                       # [B,n0,n1]

                chunk_den = w.sum(dim=(1, 2))                       # [B]
                chunk_num = (w[..., None] * v_cond).sum(dim=(1, 2)) # [B,D]

                running_den = running_den * old_rescale_flat + chunk_den[:, None]
                running_num = running_num * old_rescale_flat + chunk_num
                running_max = new_max

        return running_num / running_den.clamp_min(1e-12)

    def sample_location_and_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        x0_ref: torch.Tensor,
        x1_ref: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        epsilon: Optional[torch.Tensor] = None,
        return_noise: bool = False,
        chunk_n0: Optional[int] = None,
        chunk_n1: Optional[int] = None,
    ):
        B = x0.shape[0]

        if t is None:
            t = torch.rand(B, device=x0.device, dtype=x0.dtype)
        if epsilon is None:
            epsilon = torch.randn_like(x0)

        xt = self.sample_xt(x0=x0, x1=x1, t=t, epsilon=epsilon)

        ut = self.compute_explicit_flow(
            xt=xt,
            t=t,
            x0_ref=x0_ref,
            x1_ref=x1_ref,
            chunk_n0=chunk_n0,
            chunk_n1=chunk_n1,
        )

        if return_noise:
            return t, xt, ut, epsilon
        return t, xt, ut
