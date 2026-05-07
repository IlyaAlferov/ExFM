import numpy as np
import torch
import ot
import dcor
import matplotlib.pyplot as plt
from torchdiffeq import odeint


def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def integrate_trajectories(
    model,
    x_init: torch.Tensor,
    t_steps: int = 100,
    method: str = "rk4",
    device: str = None,
):
    """
    Integrates trajectories dx/dt = v(x,t).

    Args:
        model: velocity model with signature model(x, t)
        x_init: [B, D]
        t_steps: number of time discretization points
        method: ODE solver method for torchdiffeq
        device: optional device override

    Returns:
        traj: [T, B, D]
    """
    if device is None:
        device = x_init.device

    model.eval()
    t_span = torch.linspace(0.0, 1.0, t_steps, device=device)

    class ODEfunc(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, t, x):
            t_batch = torch.full(
                (x.shape[0],),
                fill_value=t,
                device=x.device,
                dtype=x.dtype,
            )
            return self.model(x, t_batch)

    ode_rhs = ODEfunc(model)

    with torch.no_grad():
        traj = odeint(
            ode_rhs,
            x_init.to(device),
            t_span,
            method=method,
        )
    return traj


def trajectory_curvature(trajectories):
    """
    trajectories: [T, N, D]
    Returns per-trajectory mean turning angle.
    """
    # trajectories = to_numpy(trajectories)
    disp = trajectories[1:] - trajectories[:-1]  # [T-1, N, D]
    dirs = disp / (np.linalg.norm(disp, axis=-1, keepdims=True) + 1e-8)

    cos_angles = (dirs[:-1] * dirs[1:]).sum(axis=-1)  # [T-2, N]
    angles = np.arccos(np.clip(cos_angles, -1.0, 1.0))
    return angles.mean(axis=0)  # [N]


def trajectory_straight_line_deviation(trajectories):
    """
    trajectories: [T, N, D]

    Returns:
        mean_dev: [N]
        max_dev:  [N]
    """
    # trajectories = to_numpy(trajectories)
    start = trajectories[0]
    end = trajectories[-1]

    t = np.linspace(0.0, 1.0, len(trajectories))
    line = start[None, :, :] + (end - start)[None, :, :] * t[:, None, None]
    deviations = np.linalg.norm(trajectories - line, axis=-1)

    mean_dev = deviations.mean(axis=0)
    max_dev = deviations.max(axis=0)
    return mean_dev, max_dev


def trajectory_path_efficiency(trajectories):
    """
    trajectories: [T, N, D]

    Returns:
        efficiency: [N]
            displacement / path_length, in (0, 1]
    """
    # trajectories = to_numpy(trajectories)

    segment_disp = trajectories[1:] - trajectories[:-1]              # [T-1, N, D]
    path_length = np.linalg.norm(segment_disp, axis=-1).sum(axis=0)  # [N]

    displacement = np.linalg.norm(
        trajectories[-1] - trajectories[0], axis=-1
    )  # [N]

    efficiency = displacement / (path_length + 1e-8)
    return efficiency


def mean_paired_l2_distance(X, Y):
    """
    Mean Euclidean distance between corresponding samples.

    X, Y: [N, D]
    """
    # X = to_numpy(X)
    # Y = to_numpy(Y)
    return np.linalg.norm(X - Y, axis=1).mean()


def empirical_w2_distance(X, Y, return_squared: bool = False):
    """
    Empirical Wasserstein-2 distance between two point clouds with uniform weights.

    Returns:
        sqrt(emd2) by default
        or emd2 if return_squared=True
    """
    # X = to_numpy(X)
    # Y = to_numpy(Y)

    a = np.ones(len(X)) / len(X)
    b = np.ones(len(Y)) / len(Y)
    M = ot.dist(X, Y)
    emd2 = ot.emd2(a, b, M, numItermax=50000)

    if return_squared:
        return emd2
    return np.sqrt(emd2)


def empirical_energy_distance(X, Y):
    # X = to_numpy(X)
    # Y = to_numpy(Y)
    return dcor.energy_distance(X, Y)


def summarize_trajectory_metrics(traj, x_target=None):
    """
    Summarizes geometry metrics for trajectories and optional terminal matching metrics.

    Args:
        traj: [T, N, D]
        x_target: optional [N, D] target cloud

    Returns:
        dict
    """
    traj_np = to_numpy(traj)
    x0 = traj_np[0]
    xT = traj_np[-1]

    curv = trajectory_curvature(traj_np)
    dev_mean, dev_max = trajectory_straight_line_deviation(traj_np)
    eff = trajectory_path_efficiency(traj_np)

    metrics = {
        "trajectory_curvature_mean": float(curv.mean()),
        "mean_straight_line_deviation": float(dev_mean.mean()),
        "mean_max_straight_line_deviation": float(dev_max.mean()),
        "path_efficiency_mean": float(eff.mean()),
        "source_terminal_paired_l2_mean": float(mean_paired_l2_distance(x0, xT)),
        "source_terminal_w2": float(empirical_w2_distance(x0, xT)),
    }

    if x_target is not None:
        x_target = to_numpy(x_target)
        metrics.update({
            "terminal_target_w2": float(empirical_w2_distance(xT, x_target)),
            "terminal_target_energy_distance": float(empirical_energy_distance(xT, x_target)),
        })

    return metrics


def plot_trajectories_vs_straight_lines(traj, x_ref=None):
    """
    Plot ODE trajectories and corresponding straight-line interpolations.

    Args:
        traj:
            [T, B, 2]
        x_ref:
            optional [N, 2] reference point cloud
    """
    traj = to_numpy(traj)
    x_ref = to_numpy(x_ref)

    T, N, D = traj.shape
    assert D == 2, "Function supports only 2D trajectories."

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # ODE trajectories
    ax = axes[0]
    for i in range(N):
        ax.plot(traj[:, i, 0], traj[:, i, 1], alpha=0.2, color="olive")
    ax.scatter(traj[0, :, 0], traj[0, :, 1], s=10, alpha=0.8, c="black", label="x(0)")
    ax.scatter(traj[-1, :, 0], traj[-1, :, 1], s=6, alpha=1.0, c="blue", label="x(1)")
    ax.set_title("ODE trajectories")
    ax.legend()

    # Straight lines
    ax = axes[1]
    for i in range(N):
        x0 = traj[0, i]
        x1 = traj[-1, i]
        ax.plot([x0[0], x1[0]], [x0[1], x1[1]], alpha=0.2, color="olive")
    ax.scatter(traj[0, :, 0], traj[0, :, 1], s=10, alpha=0.8, c="black", label="x(0)")
    ax.scatter(traj[-1, :, 0], traj[-1, :, 1], s=6, alpha=1.0, c="blue", label="x(1)")
    if x_ref is not None:
        ax.scatter(x_ref[:, 0], x_ref[:, 1], s=5, alpha=0.2, color="red", label="reference")
    ax.set_title("Straight lines")
    ax.legend()

    plt.tight_layout()
    # plt.show()
    return fig
