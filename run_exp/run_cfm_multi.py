"""Multi-run training for CFM method."""

from configs.cfm import ExperimentConfig
from train import train_cfm_multi


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
    cfg.train.num_epochs = 60

    # -------------------------
    # ClearML
    # -------------------------
    cfg.clearml.use = True
    cfg.clearml.project_name = "Research"
    cfg.clearml.task_name = "cfm_test"

    # -------------------------
    # Number of runs
    # -------------------------
    num_runs = 5

    # -------------------------
    # Run
    # -------------------------
    train_cfm_multi(cfg, num_runs=num_runs)
    print("\n=== Multi-run Training Complete ===")
