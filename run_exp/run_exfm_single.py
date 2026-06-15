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
    # Kernel: TimeScaleRBF (sigma зависит от t через MLP)
    # -------------------------
    cfg.kernel.type = "time_scale_rbf"
    cfg.kernel.time_scale_rbf.scale_type = "mlp"
    
    # TimeMLPScale configuration
    cfg.kernel.time_scale_rbf.mlp.mode = "positive"
    cfg.kernel.time_scale_rbf.mlp.time_emb_dim = 32
    cfg.kernel.time_scale_rbf.mlp.hidden_dim = 64
    cfg.kernel.time_scale_rbf.mlp.num_layers = 3
    cfg.kernel.time_scale_rbf.mlp.init_value = 1.0
    cfg.kernel.time_scale_rbf.mlp.min_value = 1e-4
    cfg.kernel.time_scale_rbf.mlp.max_value = 10.0
    cfg.kernel.time_scale_rbf.mlp.use_sinusoidal = True
    cfg.kernel.time_scale_rbf.mlp.use_layernorm = True
    
    # RBF kernel parameters
    cfg.kernel.time_scale_rbf.eta = 1e-5
    cfg.kernel.time_scale_rbf.event_ndim = 1
    cfg.kernel.time_scale_rbf.min_sigma = 1e-6
    cfg.kernel.time_scale_rbf.use_prefactor = False

    # -------------------------
    # Flow Matcher
    # -------------------------
    cfg.flow.sigma = 0.4

    # -------------------------
    # Losses
    # -------------------------
    cfg.loss.regularization_type = "none"
    cfg.loss.regularization_weight = 0.0

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
    cfg.clearml.task_name = "time_scale_rbf_mlp"

    # -------------------------
    # Run
    # -------------------------
    results = train_exfm_single(cfg)
    print("\n=== Training Complete ===")
    print(f"Total time: {results['total_time']:.2f} sec")
