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
    # Kernel: TimeRBF (constant sigma)
    # -------------------------
    cfg.kernel.type = "time_rbf"
    cfg.kernel.time_rbf.base_sigma.mode = "constant"
    cfg.kernel.time_rbf.base_sigma.init_sigma = 0.4
    cfg.kernel.time_rbf.base_sigma.min_sigma = 1e-4

    # RBF kernel parameters
    cfg.kernel.time_rbf.eta = 1e-5
    cfg.kernel.time_rbf.event_ndim = 1
    cfg.kernel.time_rbf.min_sigma = 1e-6
    cfg.kernel.time_rbf.use_prefactor = False

    # -------------------------
    # Flow Matcher
    # -------------------------
    cfg.flow.sigma = 0.4

    # -------------------------
    # Losses
    # -------------------------
    cfg.loss.regularization_type = "full_acceleration"
    cfg.loss.regularization_weight = 0.01

    # -------------------------
    # Training
    # -------------------------
    cfg.train.num_epochs = 80
    cfg.train.lr = 3e-4

    # -------------------------
    # ClearML
    # -------------------------
    cfg.clearml.project_name = "Research"
    cfg.clearml.task_name = "exfm-time_rbf-fullacc"

    # -------------------------
    # Number of runs
    # -------------------------
    num_runs = 9

    # -------------------------
    # Run
    # -------------------------
    train_exfm_multi(cfg, num_runs=num_runs)
    print("\n=== Multi-run Training Complete ===")
