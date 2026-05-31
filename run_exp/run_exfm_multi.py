"""Multi-run training for EXFM method."""

from configs.exfm import ExperimentConfig
from train import train_exfm_multi


if __name__ == "__main__":
    cfg = ExperimentConfig()

    # -------------------------
    # Velocity model
    # -------------------------
    cfg.velocity.hidden_dim = 256
    cfg.velocity.num_layers = 4

    # -------------------------
    # Sigma model
    # -------------------------
    cfg.sigma.init_sigma = 0.4

    # -------------------------
    # Time model
    # -------------------------
    cfg.time.use_multiplier = True

    # -------------------------
    # Losses
    # -------------------------
    cfg.loss.fm_weight = 1.0
    # cfg.loss.accel_weight = 1.0

    # -------------------------
    # Training
    # -------------------------
    cfg.train.num_epochs = 80
    cfg.train.lr = 3e-4

    # -------------------------
    # ClearML
    # -------------------------
    cfg.clearml.project_name = "Research"
    cfg.clearml.task_name = "exfm-06a-time"

    # -------------------------
    # Number of runs
    # -------------------------
    num_runs = 9

    # -------------------------
    # Run
    # -------------------------
    train_exfm_multi(cfg, num_runs=num_runs)
    print("\n=== Multi-run Training Complete ===")
