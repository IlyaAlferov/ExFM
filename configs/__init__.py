"""
Configuration modules for Flow Matching experiments.
"""

from .base import DataConfig, TrainConfig, ClearMLConfig
from .cfm import VelocityModelConfig as CFMVelocityModelConfig
from .cfm import CFMConfig, ExperimentConfig as CFMExperimentConfig
from .exfm import (
    VelocityModelConfig as EXFMVelocityModelConfig,
    SigmaModelConfig,
    TimeModelConfig,
    FlowMatcherConfig,
    ExperimentConfig as EXFMExperimentConfig,
)

__all__ = [
    # Base configs
    "DataConfig",
    "TrainConfig",
    "ClearMLConfig",
    "LossConfig",
    # CFM configs
    "CFMVelocityModelConfig",
    "CFMConfig",
    "CFMExperimentConfig",
    # EXFM configs
    "EXFMVelocityModelConfig",
    "SigmaModelConfig",
    "TimeModelConfig",
    "FlowMatcherConfig",
    "EXFMExperimentConfig",
]
