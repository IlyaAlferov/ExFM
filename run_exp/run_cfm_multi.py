"""Multi-run training for CFM method."""

from configs.cfm import ExperimentConfig
from train import train_cfm_multi


if __name__ == "__main__":
    cfg = ExperimentConfig()

    # -------------------------
    # Velocity model
    # -------------------------

    # -------------------------
    # CFM method
    # -------------------------
    cfg.cfm.method = "ot-exfm"  # "basic" | "exact_ot" | "schrodinger" | "ot-exfm"
    cfg.cfm.sigma = 0.4

    # -------------------------
    # Training
    # -------------------------
    cfg.train.num_epochs = 80
    cfg.train.lr = 1e-4

    # -------------------------
    # ClearML
    # -------------------------
    cfg.clearml.use = True
    cfg.clearml.project_name = "Research"
    cfg.clearml.task_name = "cfm_otexfm-c_sigma_0.4-c_lr_1e4"

    # -------------------------
    # Number of runs
    # -------------------------
    num_runs = 9

    # -------------------------
    # Run
    # -------------------------
    train_cfm_multi(cfg, num_runs=num_runs)
    print("\n=== Multi-run Training Complete ===")
