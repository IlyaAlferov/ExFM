"""
Configuration classes for Explicit Flow Matching (EXFM) method.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

from .base import (
    DataConfig, TrainConfig, ClearMLConfig, VelocityModelConfig
)


@dataclass
class LossConfig:
    """Configuration for loss function weights."""
    fm_weight: float = 1.0
    accel_weight: float = 0.0
    consistency_weight: float = 0.0
    consistency_epsilon: float = 0.01


@dataclass
class SigmaModelConfig:
    """Configuration for spatial sigma model."""
    mode: Literal["constant", "scalar"] = "scalar"
    init_sigma: float = 0.2
    min_sigma: float = 1e-4


@dataclass
class TimeModelConfig:
    """Configuration for temporal sigma multiplier."""
    use_multiplier: bool = False
    time_emb_dim: int = 32
    hidden_dim: int = 64
    num_layers: int = 3
    init_value: float = 1.0
    min_value: float = 1.0e-4
    max_value: float = 10.0
    use_sinusoidal: bool = True


@dataclass
class FlowMatcherConfig:
    """Configuration for Explicit Flow Matcher."""
    eta: float = 1e-5
    min_sigma: float = 1e-6
    chunk_n0: Optional[int] = None
    chunk_n1: Optional[int] = None
    use_full_gaussian_prefactor: bool = False
    implementation: Literal["vcond", "compact"] = "vcond"


@dataclass
class ExperimentConfig:
    """Complete configuration for EXFM experiment."""

    data: DataConfig = field(default_factory=DataConfig)
    velocity: VelocityModelConfig = field(default_factory=VelocityModelConfig)
    sigma: SigmaModelConfig = field(default_factory=SigmaModelConfig)
    time: TimeModelConfig = field(default_factory=TimeModelConfig)
    flow: FlowMatcherConfig = field(default_factory=FlowMatcherConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    clearml: ClearMLConfig = field(default_factory=ClearMLConfig)
