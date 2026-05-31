import numpy as np
import torch
import ot
import dcor
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from losses import compute_velocity_and_acceleration_components


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


def trajectory_angle_turn(trajectories):
    """
    trajectories: [T, N, D]
    Returns per-trajectory mean turning angle.
    """
    # trajectories = to_numpy(trajectories)
    disp = trajectories[1:] - trajectories[:-1]  # [T-1, N, D]
    dirs = disp / (np.linalg.norm(disp, axis=-1, keepdims=True) + 1e-8)

    cos_angles = (dirs[:-1] * dirs[1:]).sum(axis=-1)  # [T-2, N]
    angles = np.arccos(np.clip(cos_angles, -1.0, 1.0))
    return angles.mean(axis=0), angles.max(axis=0)  # [N]


def trajectory_curvature(
    model,
    traj,
    chunk_size: int = 1024,
):
    """
    Computes curvature metrics along validation trajectories.

    Args:
        model: velocity field model, model(x, t) -> [B, D]
        traj: [T, N, D] integrated trajectories
        t_start: start time of integration
        t_end: end time of integration
        chunk_size: chunk size for curvature computation

    Returns:
        dict with curvature metrics
    """
    device = traj.device
    T, N, D = traj.shape

    t_grid = torch.linspace(0.0, 1.0, T, device=device)   # [T]
    t_points = t_grid[:, None].expand(T, N)                     # [T, N]

    x_flat = traj.reshape(T * N, D)
    t_flat = t_points.reshape(T * N)

    curvature_chunks = []

    for i in range(0, x_flat.size(0), chunk_size):
        xb = x_flat[i:i + chunk_size].detach()
        tb = t_flat[i:i + chunk_size].detach()

        with torch.enable_grad():
            _, _, _, curvature = compute_velocity_and_acceleration_components(
                model=model,
                x=xb,
                t=tb,
            )

        curvature_chunks.append(curvature.detach())

    curvature_flat = torch.cat(curvature_chunks, dim=0)     # [T*N]
    curvature_traj = curvature_flat.view(T, N)              # [T, N]

    curv_mean_per_traj = curvature_traj.mean(dim=0)         # [N]
    curv_med_per_traj = curvature_traj.median(dim=0).values    # [N]

    metrics = {
        "trajectory_curvature_mean": curv_mean_per_traj.mean().item(),
        "trajectory_curvature_median": curv_med_per_traj.mean().item(),
    }

    return metrics


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
    xT = traj_np[-1]

    angle_turn_mean, angle_turn_max = trajectory_angle_turn(traj_np)
    dev_mean, dev_max = trajectory_straight_line_deviation(traj_np)
    eff = trajectory_path_efficiency(traj_np)

    metrics = {}
    if x_target is not None:
        x_target = to_numpy(x_target)
        metrics.update({
            "terminal_target_w2": float(empirical_w2_distance(xT, x_target)),
            "terminal_target_energy_distance": float(empirical_energy_distance(xT, x_target)),
        })
    metrics.update({
        # "source_terminal_paired_l2_mean": float(mean_paired_l2_distance(x0, xT)),
        # "source_terminal_w2": float(empirical_w2_distance(x0, xT)),
        "mean_straight_line_deviation": float(dev_mean.mean()),
        # "mean_max_straight_line_deviation": float(dev_max.mean()),
        "path_efficiency_mean": float(eff.mean()),
        "trajectory_angle_turn_mean": float(angle_turn_mean.mean()),
        "trajectory_angle_turn_max": float(angle_turn_max.mean()),
    })

    return metrics


def plot_trajectories_vs_straight_lines(
    traj, x_ref=None, max_trajectories: int = 1000
):
    """
    Plot ODE trajectories and corresponding straight-line interpolations.

    Args:
        traj: [T, B, 2]
        x_ref: optional [N, 2] reference point cloud
        max_trajectories: maximum number of trajectories to plot (randomly sampled)
    """
    traj = to_numpy(traj)
    x_ref = to_numpy(x_ref)

    T, N, D = traj.shape
    assert D == 2, "Function supports only 2D trajectories."

    # Subsample trajectories if too many
    if N > max_trajectories:
        indices = np.random.choice(N, max_trajectories, replace=False)
        traj = traj[:, indices, :]
        N = max_trajectories

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
    # for i in range(N):
    #     x0 = traj[0, i]
    #     x1 = traj[-1, i]
    #     ax.plot([x0[0], x1[0]], [x0[1], x1[1]], alpha=0.2, color="olive")
    # ax.scatter(traj[0, :, 0], traj[0, :, 1], s=10, alpha=0.8, c="black", label="x(0)")
    ax.scatter(traj[-1, :, 0], traj[-1, :, 1], s=6, alpha=1.0, c="blue", label="x(1)")
    if x_ref is not None:
        ax.scatter(x_ref[:, 0], x_ref[:, 1], s=5, alpha=0.2, color="red", label="reference")
    ax.set_title("Straight lines")
    ax.legend()

    plt.tight_layout()
    # plt.show()
    return fig


def plot_temporal_sigma_profile(
    time_multipliear,
    device,
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_points: int = 100,
    title: str = r"Temporal sigma profile: f(t) t(1-t)",
):
    """
    Plots temporal profile f(t) * t * (1-t).

    Args:
        bridge_scale_model: TimeSigmaMultiplier-like module, maps t:[B] -> [B]
        device: torch device
        t_start: left bound
        t_end: right bound
        num_points: number of evaluation points
        title: plot title

    Returns:
        fig: matplotlib figure
    """
    with torch.no_grad():
        t = torch.linspace(t_start, t_end, num_points, device=device)
        f_t = time_multipliear(t)
        base_t = t * (1.0 - t)
        sigma_t = f_t * base_t

    t_np = t.cpu().numpy()
    f_np = f_t.cpu().numpy()
    base_np = base_t.cpu().numpy()
    sigma_np = sigma_t.cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_np, sigma_np, label=r"$f(t)\,t(1-t)$", linewidth=2)
    ax.plot(t_np, f_np, label=r"$f(t)$", linestyle="--", alpha=0.8)
    ax.plot(t_np, base_np, label=r"$t(1-t)$", linestyle=":", alpha=0.8)

    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig
