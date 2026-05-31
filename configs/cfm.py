"""
Configuration classes for Conditional Flow Matching (CFM) method.
"""

from dataclasses import dataclass, field
from .base import DataConfig, TrainConfig, ClearMLConfig, VelocityModelConfig


@dataclass
class CFMConfig:
    """
    Configuration for basic Conditional Flow Matching.

    sigma: fixed standard deviation for the probability path
           N(t * x1 + (1 - t) * x0, sigma)
    """
    sigma: float = 0.1


@dataclass
class ExperimentConfig:
    """Complete configuration for CFM experiment."""

    data: DataConfig = field(default_factory=DataConfig)
    velocity: VelocityModelConfig = field(default_factory=VelocityModelConfig)
    cfm: CFMConfig = field(default_factory=CFMConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    clearml: ClearMLConfig = field(default_factory=ClearMLConfig)
