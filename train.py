import os
import time
import random
import numpy as np
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from clearml import Task

from data import sample_8gaussians, sample_moons
from models import VelocityMLP, SpatialSigmaModel
from exfm import ExplicitFlowMatcher
from losses import acceleration_penalty_loss, consistency_loss
from metrics import (
    integrate_trajectories, summarize_trajectory_metrics,
    plot_trajectories_vs_straight_lines, to_numpy
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_experiment(cfg):
    set_seed(cfg.train.seed)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # ClearML
    # -----------------------------
    task = None
    logger = None
    if cfg.clearml.use:
        task = Task.init(
            project_name=cfg.clearml.project_name,
            task_name=cfg.clearml.task_name or cfg.name,
            auto_connect_frameworks={'pytorch': False}
        )
        task.connect(asdict(cfg))
        logger = task.get_logger()

    # -----------------------------
    # Data
    # -----------------------------
    if cfg.data.source == "gaussians":
        x0_train = sample_8gaussians(cfg.data.train_size)
        x0_val = sample_8gaussians(cfg.data.val_size)
    elif cfg.data.source == "moons":
        x0_train = sample_moons(cfg.data.train_size).float()
        x0_val = sample_moons(cfg.data.val_size).float()
    else:
        raise ValueError(f"Unknown source dataset: {cfg.data.source}")

    if cfg.data.target == "gaussians":
        x1_train = sample_8gaussians(cfg.data.train_size)
        x1_val = sample_8gaussians(cfg.data.val_size)
    elif cfg.data.target == "moons":
        x1_train = sample_moons(cfg.data.train_size).float()
        x1_val = sample_moons(cfg.data.val_size).float()
    else:
        raise ValueError(f"Unknown target dataset: {cfg.data.target}")

    x0_val = x0_val.to(device)
    x1_val = x1_val.to(device)

    reference_dataset = TensorDataset(x0_train, x1_train)
    reference_loader = DataLoader(
        reference_dataset,
        batch_size=cfg.data.reference_batch_size,
        shuffle=cfg.data.shuffle_reference,
        num_workers=cfg.train.num_workers,
        drop_last=False,
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
        x_dim=cfg.velocity.x_dim,
        init_sigma=cfg.sigma.init_sigma,
        hidden_dim=cfg.sigma.hidden_dim,
        num_layers=cfg.sigma.num_layers,
        min_sigma=cfg.sigma.min_sigma,
        use_layernorm=cfg.sigma.use_layernorm,
    ).to(device)

    FM = ExplicitFlowMatcher(
        sigma_model=sigma_model,
        eta=cfg.flow.eta,
        min_sigma=cfg.flow.min_sigma,
        chunk_n0=cfg.flow.chunk_n0,
        chunk_n1=cfg.flow.chunk_n1,
        use_full_gaussian_prefactor=cfg.flow.use_full_gaussian_prefactor,
        implementation=cfg.flow.implementation,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(sigma_model.parameters()),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    global_step = 0

    start_time = time.time()

    # -----------------------------
    # Training
    # -----------------------------
    for epoch in range(1, cfg.train.num_epochs + 1):
        model.train()
        sigma_model.train()

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
                    accel_loss = acceleration_penalty_loss(model, xt, t)
                    total_loss = total_loss + cfg.loss.accel_weight * accel_loss

                cons_loss = torch.tensor(0.0, device=device)
                if cfg.loss.consistency_weight > 0:
                    cons_loss = consistency_loss(
                        model,
                        xt,
                        t,
                        epsilon=cfg.loss.consistency_epsilon,
                    )
                    total_loss = total_loss + cfg.loss.consistency_weight * cons_loss

                total_loss.backward()

                grad_norm = None
                if cfg.train.grad_clip_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(sigma_model.parameters()),
                        cfg.train.grad_clip_norm,
                    ).item()

                optimizer.step()

                global_step += 1
                epoch_num_steps += 1

                epoch_loss_sum += total_loss.item()
                epoch_fm_sum += fm_loss.item()
                epoch_accel_sum += accel_loss.item()
                epoch_cons_sum += cons_loss.item()

        avg_loss = epoch_loss_sum / max(epoch_num_steps, 1)
        avg_fm = epoch_fm_sum / max(epoch_num_steps, 1)
        avg_accel = epoch_accel_sum / max(epoch_num_steps, 1)
        avg_cons = epoch_cons_sum / max(epoch_num_steps, 1)

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
            logger.report_scalar("epoch", "avg_accel_loss", avg_accel, epoch)
            logger.report_scalar("epoch", "avg_consistency_loss", avg_cons, epoch)

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

            metrics = summarize_trajectory_metrics(traj, x_target=x1_val)

            print(f"[Eval @ epoch {epoch:03d}]")
            for k, v in metrics.items():
                print(f"  {k}: {v:.6f}")

            if logger is not None:
                for k, v in metrics.items():
                    logger.report_scalar("val", k, float(v), epoch)

                fig = plot_trajectories_vs_straight_lines(traj, x_ref=x1_val)
                logger.report_matplotlib_figure(
                    title="val_trajectories",
                    series="trajectory_plot",
                    iteration=epoch,
                    figure=fig,
                )
                plt.close(fig)

    if task is not None:
        task.close()
