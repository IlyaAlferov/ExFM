"""Single-run training for CFM methods."""

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

from configs.cfm import ExperimentConfig
from data import create_datasets, create_dataloaders
from models import VelocityMLP
from methods.cfm import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
)
from methods.ot_exfm import ExplicitOptimalTransportFlowMatcher
from metrics import (
    integrate_trajectories,
    summarize_trajectory_metrics,
    plot_trajectories_vs_straight_lines,
    trajectory_curvature,
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


def create_flow_matcher(cfg):
    """Create flow matcher based on config."""
    if cfg.cfm.method == "basic":
        return ConditionalFlowMatcher(sigma=cfg.cfm.sigma)
    elif cfg.cfm.method == "exact_ot":
        return ExactOptimalTransportConditionalFlowMatcher(
            sigma=cfg.cfm.sigma,
        )
    elif cfg.cfm.method == "schrodinger":
        return SchrodingerBridgeConditionalFlowMatcher(
            sigma=cfg.cfm.sigma,
        )
    elif cfg.cfm.method == "ot-exfm":
        return ExplicitOptimalTransportFlowMatcher(
            sigma=cfg.cfm.sigma,
            ot_method="exact",
        )
    else:
        raise ValueError(f"Unknown CFM method: {cfg.cfm.method}")


# =========================
# Training
# =========================

def train_cfm_single(
    cfg: ExperimentConfig,
    enable_clearml: bool | None = None,
) -> dict:
    """
    Train a single CFM experiment.

    Args:
        cfg: Experiment configuration

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
    # Model
    # -----------------------------
    model = VelocityMLP(
        x_dim=cfg.velocity.x_dim,
        hidden_dim=cfg.velocity.hidden_dim,
        num_layers=cfg.velocity.num_layers,
        time_conditioning=cfg.velocity.time_conditioning,
        time_emb_dim=cfg.velocity.time_emb_dim,
        use_layernorm=cfg.velocity.use_layernorm,
    ).to(device)

    # -----------------------------
    # Flow Matcher
    # -----------------------------
    FM = create_flow_matcher(cfg)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    start_time = time.time()

    # Storage for metrics
    history = {
        "train/avg_fm": []
    }

    # -----------------------------
    # Training Loop
    # -----------------------------
    for epoch in range(1, cfg.train.num_epochs + 1):
        model.train()

        epoch_fm_sum = 0.0
        epoch_num_steps = 0

        for x0_batch, x1_batch in reference_loader:
            x0_batch = x0_batch.to(device)
            x1_batch = x1_batch.to(device)

            idx0 = torch.randperm(x0_batch.size(0), device=device)
            idx1 = torch.randperm(x1_batch.size(0), device=device)

            x0 = x0_batch[idx0]
            x1 = x1_batch[idx1]

            # Sample location and conditional flow
            t, xt, ut = FM.sample_location_and_conditional_flow(
                x0=x0,
                x1=x1,
                t=None,
                return_noise=False,
            )

            optimizer.zero_grad()

            # Predict velocity
            v_pred = model(xt, t)
            fm_loss = F.mse_loss(v_pred, ut)

            fm_loss.backward()

            # Gradient clipping
            grad_norm = None
            if cfg.train.grad_clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.train.grad_clip_norm,
                ).item()

            optimizer.step()

            epoch_num_steps += 1
            epoch_fm_sum += fm_loss.item()

        avg_fm = epoch_fm_sum / max(epoch_num_steps, 1)
        history["train/avg_fm"].append(avg_fm)
        print(
            f"\n[Epoch {epoch:03d}] "
            f"avg_fm={avg_fm:.6f} "
        )
        if logger is not None:
            logger.report_scalar("epoch", "avg_fm_loss", avg_fm, epoch)

        # -----------------------------
        # Evaluation on fixed validation set
        # -----------------------------
        if epoch % cfg.train.eval_every_epochs == 0:
            model.eval()

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

    total_time = time.time() - start_time

    if task is not None:
        task.close()

    return {
        "history": history,
        "total_time": total_time,
        "seed": cfg.train.seed,
    }


def train_cfm_multi(cfg: ExperimentConfig, num_runs: int) -> dict:
    """
    Run multiple independent trainings and aggregate metrics/history.

    Behavior:
        - uses cfg.train.seed as base seed
        - for run i uses seed = cfg.train.seed + i
        - disables per-run ClearML logging
        - optionally logs only aggregated results to ClearML if cfg.clearml.use is True
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
        result = train_cfm_single(run_cfg, enable_clearml=False)
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
    # Optional ClearML aggregate logging
    # -----------------------------
    aggregate_task = None
    aggregate_logger = None

    if cfg.clearml.use:
        aggregate_task = Task.init(
            project_name=cfg.clearml.project_name,
            task_name=cfg.clearml.task_name,
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
            ci_arr = np.asarray(ci_curve, dtype=float)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x, mean_arr, label="mean")
            ax.fill_between(x, mean_arr - ci_arr, mean_arr + ci_arr, alpha=0.25, label="95% CI")
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
