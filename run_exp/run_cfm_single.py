"""Single-run training for CFM method."""

from configs.cfm import ExperimentConfig
from train import train_cfm_single


if __name__ == "__main__":
    cfg = ExperimentConfig()

    # -------------------------
    # Velocity model
    # -------------------------
    # cfg.velocity.hidden_dim = 128
    # cfg.velocity.num_layers = 4

    # -------------------------
    # CFM method
    # -------------------------
    cfg.cfm.method = "basic"  # "basic" | "exact_ot" | "schrodinger"
    cfg.cfm.sigma = 0.1

    # -------------------------
    # Training
    # -------------------------
    cfg.train.num_epochs = 80

    # -------------------------
    # ClearML
    # -------------------------
    cfg.clearml.use = True
    cfg.clearml.task_name = "cfm_basic_sigma_0.1"

    # -------------------------
    # Run
    # -------------------------
    results = train_cfm_single(cfg)
    print("\n=== Training Complete ===")
    print(f"Total time: {results['total_time']:.2f} sec")
