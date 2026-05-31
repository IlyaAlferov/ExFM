"""Single-run training for EXFM method."""

from configs.exfm import ExperimentConfig
from train import train_exfm_single


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
    cfg.loss.accel_weight = 0.0

    # -------------------------
    # Training
    # -------------------------
    cfg.train.num_epochs = 80
    cfg.train.lr = 3e-4

    # -------------------------
    # ClearML
    # -------------------------
    cfg.clearml.use = True
    cfg.clearml.project_name = "ExFM-2moons"
    cfg.clearml.task_name = "03-time_add"

    # -------------------------
    # Run
    # -------------------------
    results = train_exfm_single(cfg)
    print("\n=== Training Complete ===")
    print(f"Total time: {results['total_time']:.2f} sec")
