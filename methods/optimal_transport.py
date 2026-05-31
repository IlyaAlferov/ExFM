"""
Optimal Transport plan samplers for OT-CFM methods.

Provides different OT solvers for computing coupling plans between source and target minibatches.
"""

from functools import partial
from typing import Union
import warnings

import numpy as np
import ot as pot
import torch


class OTPlanSampler:
    """
    Sampler for optimal transport plans between minibatches.

    Supports multiple OT solvers:
    - exact: exact EMD solution
    - sinkhorn: entropic regularized OT
    - unbalanced: unbalanced Sinkhorn-Knopp
    - partial: partial OT
    """

    def __init__(
        self,
        method: str,
        reg: float = 0.05,
        reg_m: float = 1.0,
        normalize_cost: bool = False,
        num_threads: Union[int, str] = 1,
        warn: bool = True,
    ):
        """
        Initialize OT plan sampler.

        Args:
            method: OT solver method ("exact", "sinkhorn", "unbalanced", "partial")
            reg: regularization parameter for Sinkhorn-based solvers
            reg_m: regularization weight for unbalanced solver
            normalize_cost: normalize cost matrix (helps stabilize Sinkhorn)
            num_threads: number of threads for exact solver (or "max")
            warn: if True, warn on convergence issues
        """
        if method == "exact":
            self.ot_fn = partial(pot.emd, numThreads=num_threads)
        elif method == "sinkhorn":
            self.ot_fn = partial(pot.sinkhorn, reg=reg)
        elif method == "unbalanced":
            self.ot_fn = partial(pot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m)
        elif method == "partial":
            self.ot_fn = partial(pot.partial.entropic_partial_wasserstein, reg=reg)
        else:
            raise ValueError(f"Unknown method: {method}")

        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        self.warn = warn

    def get_map(self, x0: torch.Tensor, x1: torch.Tensor) -> np.ndarray:
        """
        Compute OT plan between source and target minibatches.

        Args:
            x0: source minibatch [B, D]
            x1: target minibatch [B, D]

        Returns:
            OT plan matrix [B, B]
        """
        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])

        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)

        M = torch.cdist(x0, x1) ** 2
        if self.normalize_cost:
            M = M / M.max()

        p = self.ot_fn(a, b, M.detach().cpu().numpy())

        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)

        if np.abs(p.sum()) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            p = np.ones_like(p) / p.size

        return p

    def sample_map(self, pi: np.ndarray, batch_size: int, replace: bool = True):
        """
        Draw samples from OT plan.

        Args:
            pi: OT plan matrix [B, B]
            batch_size: number of samples to draw
            replace: sampling with or without replacement

        Returns:
            (i_s, i_j): indices of source and target samples
        """
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(
            pi.shape[0] * pi.shape[1], p=p, size=batch_size, replace=replace
        )
        return np.divmod(choices, pi.shape[1])

    def sample_plan(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        replace: bool = True,
    ):
        """
        Compute OT plan and draw coupled samples.

        Args:
            x0: source minibatch [B, D]
            x1: target minibatch [B, D]
            replace: sampling with or without replacement

        Returns:
            x0[i], x1[j]: coupled samples according to OT plan
        """
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        return x0[i], x1[j]

    def sample_plan_with_scipy(self, x0: torch.Tensor, x1: torch.Tensor):
        """
        Compute OT plan using scipy linear sum assignment.

        Advantages:
        - Reduced variance compared to sampling
        - Preserves order of x1
        - Preserves entire batch if sizes match

        Args:
            x0: source minibatch [B, D]
            x1: target minibatch [B, D]

        Returns:
            pi_x0, pi_x1: coupled samples
        """
        import scipy

        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)

        M = torch.cdist(x0.detach(), x1.detach()) ** 2
        if self.normalize_cost:
            M = M / M.max()

        _, j = scipy.optimize.linear_sum_assignment(M.cpu().numpy())
        return x0, x1[j]

    def sample_trajectory(self, X: np.ndarray):
        """
        Compute OT trajectories between multiple time points.

        Args:
            X: samples at different times [B, times, D]

        Returns:
            OT-sampled trajectories [B, times, D]
        """
        times = X.shape[1]
        pis = []
        for t in range(times - 1):
            pis.append(self.get_map(X[:, t], X[:, t + 1]))

        indices = [np.arange(X.shape[0])]
        for pi in pis:
            j = []
            for i in indices[-1]:
                j.append(np.random.choice(pi.shape[1], p=pi[i] / pi[i].sum()))
            indices.append(np.array(j))

        to_return = []
        for t in range(times):
            to_return.append(X[:, t][indices[t]])
        return np.stack(to_return, axis=1)
