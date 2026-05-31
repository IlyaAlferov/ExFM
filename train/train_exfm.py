"""Training functions for Explicit Flow Matching (EXFM) method."""

import time
import random
import math
import copy
import numpy as np
from dataclasses import asdict

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from clearml import Task

from configs.exfm import ExperimentConfig
from data import create_datasets, create_dataloaders
from models import VelocityMLP, SpatialSigmaModel, TimeSigmaMultiplier
from methods.exfm import ExplicitFlowMatcher
from losses import normal_acceleration_penalty_loss, velocity_consistency_loss
from metrics import (
    integrate_trajectories,
    summarize_trajectory_metrics,
    plot_trajectories_vs_straight_lines,
    trajectory_curvature,
    plot_temporal_sigma_profile,
)


# =========================
# Utilities
# =========================

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Training
# =========================

def train_exfm_single(
    cfg: ExperimentConfig,
    enable_clearml: bool | None = None,
) -> dict:
    """
    Train a single EXFM experiment.

    Args:
        cfg: Experiment configuration
        enable_clearml: override ClearML setting (default: use cfg.clearml.use)

    Returns:
        Dictionary with training results and metrics
    """
    set_seed(cfg.train.seed)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # ClearML
    # -----------------------------
    task = None
    logger = None
    use_clearml = cfg.clearml.use if enable_clearml is None else enable_clearml
    if use_clearml:
        task = Task.init(
            project_name=cfg.clearml.project_name,
            task_name=cfg.clearml.task_name,
            auto_connect_frameworks={'pytorch': False}
        )
        task.connect(asdict(cfg))
        logger = task.get_logger()

    # -----------------------------
    # Data
    # -----------------------------
    x0_train, x1_train, x0_val, x1_val = create_datasets(
        source=cfg.data.source,
        target=cfg.data.target,
        train_size=cfg.data.train_size,
        val_size=cfg.data.val_size,
    )

    x0_val = x0_val.to(device)
    x1_val = x1_val.to(device)

    reference_loader = create_dataloaders(
        x0_train=x0_train,
        x1_train=x1_train,
        batch_size=cfg.data.local_batch_size,
        shuffle=cfg.data.shuffle_reference,
        num_workers=cfg.train.num_workers,
    )

    # -----------------------------
    # Models
    # -----------------------------
    model = VelocityMLP(
        x_dim=cfg.velocity.x_dim,
        hidden_dim=cfg.velocity.hidden_dim,
        num_layers=cfg.velocity.num_layers,
        time_conditioning=cfg.velocity.time_conditioning,
        time_emb_dim=cfg.velocity.time_emb_dim,
        use_layernorm=cfg.velocity.use_layernorm,
    ).to(device)

    sigma_model = SpatialSigmaModel(
        mode=cfg.sigma.mode,
        init_sigma=cfg.sigma.init_sigma,
        min_sigma=cfg.sigma.min_sigma,
    ).to(device)

    time_multiplier = TimeSigmaMultiplier(
        time_emb_dim=cfg.time.time_emb_dim,
        hidden_dim=cfg.time.hidden_dim,
        num_layers=cfg.time.num_layers,
        init_value=cfg.time.init_value,
        min_value=cfg.time.min_value,
        max_value=cfg.time.max_value,
        use_sinusoidal=cfg.time.use_sinusoidal,
        mode="additive"  #
    ).to(device)

    FM = ExplicitFlowMatcher(
        sigma_model=sigma_model,
        time_model=time_multiplier if cfg.time.use_multiplier else None,
        eta=cfg.flow.eta,
        min_sigma=cfg.flow.min_sigma,
        chunk_n0=cfg.flow.chunk_n0,
        chunk_n1=cfg.flow.chunk_n1,
        use_full_gaussian_prefactor=cfg.flow.use_full_gaussian_prefactor,
        implementation=cfg.flow.implementation,
    ).to(device)

    # -----------------------------
    # Optimizer
    # -----------------------------
    # -----------------------------
# Optimizer
# -----------------------------
# Separate parameter groups for different learning rates
    optimizer_params = [
        {
            "params": list(model.parameters()) + list(sigma_model.parameters()),
            "lr": cfg.train.lr,
            "weight_decay": cfg.train.weight_decay,
        }
    ]

    if cfg.time.use_multiplier:
        time_lr = 5e-5  # getattr(cfg.time, "lr", cfg.train.lr)
        optimizer_params.append(
            {
                "params": time_multiplier.parameters(),
                "lr": time_lr,
                "weight_decay": cfg.train.weight_decay,
            }
        )

    optimizer = torch.optim.AdamW(optimizer_params)

    # params_to_optimize = list(model.parameters()) + list(sigma_model.parameters())
    # if cfg.time.use_multiplier:
    #     params_to_optimize.extend(list(time_multiplier.parameters()))

    # optimizer = torch.optim.AdamW(
    #     params_to_optimize,
    #     lr=cfg.train.lr,
    #     weight_decay=cfg.train.weight_decay,
    # )

    start_time = time.time()

    # Storage for metrics
    history = {
        "train/avg_loss": [],
        "train/avg_fm": [],
        "train/avg_accel": [],
        "train/avg_cons": [],
    }

    # -----------------------------
    # Training Loop
    # -----------------------------
    for epoch in range(1, cfg.train.num_epochs + 1):
        model.train()
        sigma_model.train()
        if cfg.time.use_multiplier:
            time_multiplier.train()

        epoch_loss_sum = 0.0
        epoch_fm_sum = 0.0
        epoch_accel_sum = 0.0
        epoch_cons_sum = 0.0
        epoch_num_steps = 0

        for x0_ref, x1_ref in reference_loader:
            x0_ref = x0_ref.to(device)
            x1_ref = x1_ref.to(device)

            idx0 = torch.randperm(x0_ref.size(0), device=device)
            idx1 = torch.randperm(x1_ref.size(0), device=device)

            for i in range(0, x0_ref.size(0), cfg.data.local_batch_size):
                local_idx0 = idx0[i:i + cfg.data.local_batch_size]
                local_idx1 = idx1[i:i + cfg.data.local_batch_size]

                x0 = x0_ref[local_idx0]
                x1 = x1_ref[local_idx1]

                optimizer.zero_grad()

                t, xt, ut = FM.sample_location_and_conditional_flow(
                    x0=x0,
                    x1=x1,
                    x0_ref=x0_ref,
                    x1_ref=x1_ref,
                    t=None,
                    epsilon=None,
                    return_noise=False,
                    chunk_n0=cfg.flow.chunk_n0,
                    chunk_n1=cfg.flow.chunk_n1,
                    implementation=cfg.flow.implementation,
                )

                v_pred = model(xt, t)
                fm_loss = F.mse_loss(v_pred, ut)

                total_loss = cfg.loss.fm_weight * fm_loss

                accel_loss = torch.tensor(0.0, device=device)
                if cfg.loss.accel_weight > 0:
                    accel_loss = normal_acceleration_penalty_loss(model, xt, t)
                    total_loss = total_loss + cfg.loss.accel_weight * accel_loss

                cons_loss = torch.tensor(0.0, device=device)
                if cfg.loss.consistency_weight > 0:
                    cons_loss = velocity_consistency_loss(
                        model, xt, t,
                        epsilon=cfg.loss.consistency_epsilon,
                    )
                    total_loss = total_loss + cfg.loss.consistency_weight * cons_loss

                total_loss.backward()

                # Gradient clipping
                if cfg.train.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(sigma_model.parameters()),
                        cfg.train.grad_clip_norm,
                    )

                optimizer.step()

                epoch_num_steps += 1
                epoch_loss_sum += total_loss.item()
                epoch_fm_sum += fm_loss.item()
                epoch_accel_sum += accel_loss.item()
                epoch_cons_sum += cons_loss.item()

        avg_loss = epoch_loss_sum / max(epoch_num_steps, 1)
        avg_fm = epoch_fm_sum / max(epoch_num_steps, 1)
        avg_accel = epoch_accel_sum / max(epoch_num_steps, 1)
        avg_cons = epoch_cons_sum / max(epoch_num_steps, 1)

        history["train/avg_loss"].append(avg_loss)
        history["train/avg_fm"].append(avg_fm)
        history["train/avg_accel"].append(avg_accel)
        history["train/avg_cons"].append(avg_cons)

        print(
            f"\n[Epoch {epoch:03d}] "
            f"avg_loss={avg_loss:.6f} "
            f"avg_fm={avg_fm:.6f} "
            f"avg_accel={avg_accel:.6f} "
            f"avg_cons={avg_cons:.6f}"
        )

        if logger is not None:
            logger.report_scalar("epoch", "avg_loss", avg_loss, epoch)
            logger.report_scalar("epoch", "avg_fm_loss", avg_fm, epoch)
            logger.report_scalar("epoch", "avg_normal_accel_loss", avg_accel, epoch)
            logger.report_scalar("epoch", "avg_vel_consistency_loss", avg_cons, epoch)

            # Log spatial sigma value
            with torch.no_grad():
                if sigma_model.mode == "scalar":
                    current_sigma = F.softplus(sigma_model.sigma_param) + sigma_model.min_sigma
                    logger.report_scalar("sigma", "spatial_sigma_value", current_sigma.item(), epoch)
                elif sigma_model.mode == "constant":
                    logger.report_scalar("sigma", "spatial_sigma_value", sigma_model.sigma_const.item(), epoch)

        # -----------------------------
        # Evaluation on fixed validation set
        # -----------------------------
        if epoch % cfg.train.eval_every_epochs == 0:
            model.eval()
            if cfg.sigma.mode != 'constant':
                sigma_model.eval()
            if time_multiplier is not None:
                time_multiplier.eval()

            with torch.no_grad():
                traj = integrate_trajectories(
                    model=model,
                    x_init=x0_val,
                    t_steps=cfg.train.ode_steps,
                    method=cfg.train.ode_method,
                    device=device,
                )

            # Basic trajectory metrics
            metrics = summarize_trajectory_metrics(traj, x_target=x1_val)

            # Curvature metrics
            curvature_metrics = trajectory_curvature(
                model=model,
                traj=traj,
                chunk_size=1024,
            )
            metrics.update(curvature_metrics)

            print(f"[Eval @ epoch {epoch:03d}]")
            for k, v in metrics.items():
                print(f"  {k}: {v:.6f}")
                history.setdefault(f"val/{k}", []).append(float(v))

            if logger is not None:
                for k, v in metrics.items():
                    logger.report_scalar("val", k, float(v), epoch)

                fig = plot_trajectories_vs_straight_lines(traj, x_ref=x1_val)
                logger.report_matplotlib_figure(
                    title="val_trajectories",
                    series=f"trajectory_plot_ep_{epoch}",
                    iteration=0,
                    figure=fig,
                )
                plt.close(fig)

                if cfg.time.use_multiplier:
                    fig = plot_temporal_sigma_profile(time_multiplier, device)
                    logger.report_matplotlib_figure(
                        title="time_multiplier",
                        series=f"time_multiplier_plot_ep_{epoch}",
                        iteration=0,
                        figure=fig,
                    )
                    plt.close(fig)

    total_time = time.time() - start_time

    if task is not None:
        task.close()

    return {
        "history": history,
        "total_time": total_time,
        "seed": cfg.train.seed,
    }


def train_exfm_multi(cfg: ExperimentConfig, num_runs: int) -> dict:
    """
    Run multiple independent trainings and aggregate metrics/history.

    Behavior:
        - uses cfg.train.seed as base seed
        - for run i uses seed = cfg.train.seed + i
        - disables per-run ClearML logging
        - optionally logs only aggregated results to ClearML if cfg.clearml.use is True

    Args:
        cfg: Experiment configuration
        num_runs: number of independent runs

    Returns:
        Dictionary with aggregated results (history_mean, history_std, history_ci95, total_time_mean)
    """
    if num_runs < 1:
        raise ValueError("num_runs must be >= 1")

    run_results = []
    seeds = []

    # -----------------------------
    # Multiple runs
    # -----------------------------
    for run_idx in range(num_runs):
        run_cfg = copy.deepcopy(cfg)
        run_cfg.train.seed = cfg.train.seed + run_idx
        seeds.append(run_cfg.train.seed)

        print(f"\n========== Run {run_idx + 1}/{num_runs} | seed={run_cfg.train.seed} ==========")
        result = train_exfm_single(run_cfg, enable_clearml=False)
        run_results.append(result)

    total_times = [float(r.get("total_time", 0.0)) for r in run_results]
    total_time_mean = np.mean(total_times)

    # -----------------------------
    # Aggregate history
    # -----------------------------
    history_keys = set()
    for r in run_results:
        history_keys.update(r.get("history", {}).keys())
    history_keys = sorted(history_keys)

    history_mean = {}
    history_std = {}
    history_ci95 = {}

    for key in history_keys:
        curves = []
        lengths = []

        for r in run_results:
            hist = r.get("history", {})
            if key in hist:
                curve = np.asarray(hist[key], dtype=float)
                curves.append(curve)
                lengths.append(len(curve))

        if len(curves) == 0:
            continue

        # Require same length to aggregate epoch-wise
        if len(set(lengths)) != 1:
            print(f"[WARN] Skipping history key '{key}' due to inconsistent lengths: {lengths}")
            continue

        curves = np.stack(curves, axis=0)  # [num_runs, T]
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0, ddof=1) if curves.shape[0] > 1 else np.zeros_like(mean_curve)
        sem_curve = std_curve / math.sqrt(curves.shape[0]) if curves.shape[0] > 0 else np.zeros_like(mean_curve)
        ci95_curve = 1.96 * sem_curve if curves.shape[0] > 1 else np.zeros_like(mean_curve)

        history_mean[key] = mean_curve.tolist()
        history_std[key] = std_curve.tolist()
        history_ci95[key] = ci95_curve.tolist()

    # -----------------------------
    # Print aggregate summary
    # -----------------------------
    print("\n========== Aggregate results ==========")
    print(f"num_runs: {num_runs}")
    print(f"seeds: {seeds}")
    print(f"time_mean: {total_time_mean:.4f} sec")

    # -----------------------------
    # Optional ClearML aggregate logging
    # -----------------------------
    aggregate_task = None
    aggregate_logger = None

    if cfg.clearml.use:
        aggregate_task = Task.init(
            project_name=cfg.clearml.project_name,
            task_name=f"{cfg.clearml.task_name}",
            auto_connect_frameworks={"pytorch": False},
        )

        aggregate_cfg = copy.deepcopy(cfg)
        aggregate_cfg.clearml.use = False  # avoid confusing nested intent in logged config
        aggregate_logger = aggregate_task.get_logger()

        payload = {
            "base_cfg": asdict(aggregate_cfg),
            "num_runs": num_runs,
            "seeds": seeds,
            "total_time_mean": total_time_mean,
        }
        aggregate_task.connect(payload)

        aggregate_logger.report_scalar("time", "mean", total_time_mean, 0)

        # log aggregated curves with CI
        for key, mean_curve in history_mean.items():
            std_curve = history_std[key]
            ci_curve = history_ci95[key]

            for i, v in enumerate(mean_curve, start=1):
                aggregate_logger.report_scalar(key, "mean", float(v), i)
            for i, v in enumerate(std_curve, start=1):
                aggregate_logger.report_scalar(key, "std", float(v), i)
            for i, v in enumerate(ci_curve, start=1):
                aggregate_logger.report_scalar(key, "ci95_half", float(v), i)

            # local matplotlib figure for mean ± CI
            x = np.arange(1, len(mean_curve) + 1)
            mean_arr = np.asarray(mean_curve, dtype=float)
            std_arr = np.asarray(std_curve, dtype=float)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x, mean_arr, label="mean")
            ax.fill_between(x, mean_arr - std_arr, mean_arr + std_arr, alpha=0.25, label="mean ± std")
            ax.set_title(key)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(key)
            ax.legend()
            ax.grid(True, alpha=0.3)

            aggregate_logger.report_matplotlib_figure(
                title="aggregate_history",
                series=key.replace("/", "_"),
                iteration=0,
                figure=fig,
            )
            plt.close(fig)

        aggregate_task.close()

    return {
        "history_mean": history_mean,
        "history_std": history_std,
        "history_ci95": history_ci95,
        "total_time_mean": total_time_mean,
        "seeds": seeds,
        "num_runs": num_runs,
    }
