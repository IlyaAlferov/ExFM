"""
Configuration classes for Explicit Flow Matching (EXFM) method.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

from .base import (
    DataConfig, TrainConfig, ClearMLConfig, VelocityModelConfig,
)
from .sigma import WeightKernelConfig


@dataclass
class LossConfig:
    """Configuration for loss function weights."""
    regularization_type: Literal[
        "none", "normal_acceleration", "full_acceleration", "curvature",
        "velocity_consistency", "direction_consistency"
    ] = "none"
    regularization_weight: float = 0.0
    consistency_epsilon: float = 0.01


@dataclass
class FlowMatcherConfig:
    """Configuration for Explicit Flow Matcher."""
    sigma: float = 0.4
    eta: float = 1e-5
    chunk_n0: Optional[int] = None
    chunk_n1: Optional[int] = None


@dataclass
class ExperimentConfig:
    """Complete configuration for EXFM experiment."""

    data: DataConfig = field(default_factory=DataConfig)
    velocity: VelocityModelConfig = field(default_factory=VelocityModelConfig)

    # New unified sigma/kernel system:
    kernel: WeightKernelConfig = field(default_factory=WeightKernelConfig)

    flow: FlowMatcherConfig = field(default_factory=FlowMatcherConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    clearml: ClearMLConfig = field(default_factory=ClearMLConfig)
