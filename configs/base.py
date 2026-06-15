"""
Base configuration classes shared across all Flow Matching methods.
"""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class DataConfig:
    """Configuration for data generation and loading."""
    source: Literal["gaussians", "moons"] = "gaussians"
    target: Literal["gaussians", "moons"] = "moons"

    train_size: int = 20000
    val_size: int = 2048

    reference_batch_size: int = 1024
    local_batch_size: int = 256
    shuffle_reference: bool = True


@dataclass
class TrainConfig:
    """Configuration for training process."""
    seed: int = 42
    device: str = "cuda"

    num_epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 1e-6
    grad_clip_norm: Optional[float] = 1.0

    num_workers: int = 0
    eval_every_epochs: int = 1

    ode_steps: int = 100
    ode_method: str = "rk4"


@dataclass
class ClearMLConfig:
    """Configuration for ClearML experiment tracking."""
    use: bool = True
    project_name: str = "Explicit-Flow-Matching"
    task_name: str = "baseline"


@dataclass
class VelocityModelConfig:
    """Configuration for velocity field model."""
    x_dim: int = 2
    hidden_dim: int = 256
    num_layers: int = 4
    time_conditioning: Literal["scalar", "sinusoidal"] = "sinusoidal"
    time_emb_dim: int = 32
    use_layernorm: bool = True
