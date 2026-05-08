from dataclasses import dataclass, field
from typing import Literal, Optional

from train import train_experiment


# =========================
# Configs
# =========================

@dataclass
class DataConfig:
    source: Literal["gaussians", "moons"] = "gaussians"
    target: Literal["gaussians", "moons"] = "moons"

    train_size: int = 20000
    val_size: int = 2048

    reference_batch_size: int = 1024
    local_batch_size: int = 256
    shuffle_reference: bool = True


@dataclass
class VelocityModelConfig:
    x_dim: int = 2
    hidden_dim: int = 128
    num_layers: int = 4
    time_conditioning: Literal["scalar", "sinusoidal"] = "sinusoidal"
    time_emb_dim: int = 32
    use_layernorm: bool = False


@dataclass
class SigmaModelConfig:
    mode: Literal["constant", "scalar", "pair_mlp"] = "scalar"
    init_sigma: float = 0.2
    hidden_dim: int = 64
    num_layers: int = 3
    min_sigma: float = 1e-4
    use_layernorm: bool = False


@dataclass
class FlowMatcherConfig:
    eta: float = 1e-5
    min_sigma: float = 1e-6
    chunk_n0: Optional[int] = None
    chunk_n1: Optional[int] = None
    use_full_gaussian_prefactor: bool = False
    implementation: Literal["vcond", "compact"] = "vcond"


@dataclass
class LossConfig:
    fm_weight: float = 1.0
    accel_weight: float = 0.0
    consistency_weight: float = 0.0
    consistency_epsilon: float = 0.01


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda"

    num_epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 1e-6
    grad_clip_norm: Optional[float] = 1.0

    num_workers: int = 0
    eval_every_epochs: int = 1

    ode_steps: int = 100
    ode_method: str = "rk4"


@dataclass
class ClearMLConfig:
    use: bool = True
    project_name: str = "Explicit-Flow-Matching"
    task_name: str = "baseline_exfm"


@dataclass
class ExperimentConfig:
    name: str = "baseline_exfm"

    data: DataConfig = field(default_factory=DataConfig)
    velocity: VelocityModelConfig = field(default_factory=VelocityModelConfig)
    sigma: SigmaModelConfig = field(default_factory=SigmaModelConfig)
    flow: FlowMatcherConfig = field(default_factory=FlowMatcherConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    clearml: ClearMLConfig = field(default_factory=ClearMLConfig)


# =========================
# Main
# =========================

if __name__ == "__main__":
    cfg = ExperimentConfig()

    # -------------------------
    # Experiment name
    # -------------------------
    cfg.name = "00_baseline_const_sigma"
    cfg.clearml.task_name = cfg.name

    # -------------------------
    # Data
    # -------------------------


    # -------------------------
    # Velocity model
    # -------------------------


    # -------------------------
    # Sigma model
    # -------------------------
    cfg.sigma.mode = "constant"

    # -------------------------
    # Flow matcher
    # -------------------------


    # -------------------------
    # Losses
    # -------------------------


    # -------------------------
    # Training
    # -------------------------


    # -------------------------
    # ClearML
    # -------------------------
    cfg.clearml.use = True
    cfg.clearml.task_name = "00_baseline_const_sigma"

    # -------------------------
    # Run
    # -------------------------
    train_experiment(cfg)
