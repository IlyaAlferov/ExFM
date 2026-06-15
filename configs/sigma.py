from dataclasses import dataclass, field
from typing import Literal, Optional

import torch.nn as nn

from models.sigma import (
    BaseSigmaModel,
    TimeMLPScale,
    ChebyshevTimeScale,
    TimeXMLPScale,
    TimeRBFKernel,
    TimeScaleRBFKernel,
    PointwiseRBFKernel,
    LearnedEnergyKernel,
)


@dataclass
class SigmaBaseConfig:
    mode: Literal["constant", "scalar"] = "scalar"
    init_sigma: float = 0.4
    min_sigma: float = 1e-4


@dataclass
class TimeRBFKernelConfig:
    base_sigma: SigmaBaseConfig = field(default_factory=SigmaBaseConfig)

    eta: float = 1e-5
    event_ndim: int = 1
    min_sigma: float = 1e-6
    use_prefactor: bool = False


@dataclass
class TimeMLPScaleConfig:
    mode: Literal["positive", "residual"] = "positive"

    time_emb_dim: int = 32
    hidden_dim: int = 32
    num_layers: int = 2

    min_value: float = 1e-4
    max_value: Optional[float] = 10.0

    use_sinusoidal: bool = True
    use_layernorm: bool = True


@dataclass
class ChebyshevTimeScaleConfig:
    degree: int = 4
    init_value: float = 1.0
    min_value: float = 1e-4
    max_value: Optional[float] = 10.0


@dataclass
class TimeScaleRBFKernelConfig:
    scale_type: Literal["mlp", "chebyshev"] = "mlp"

    mlp: TimeMLPScaleConfig = field(default_factory=TimeMLPScaleConfig)
    chebyshev: ChebyshevTimeScaleConfig = field(default_factory=ChebyshevTimeScaleConfig)

    eta: float = 1e-5
    event_ndim: int = 1
    min_sigma: float = 1e-6
    use_prefactor: bool = False


@dataclass
class TXScalarModelConfig:
    x_dim: int = 2
    time_emb_dim: int = 32
    hidden_dim: int = 64
    num_layers: int = 2

    init_value: float = 1.0
    min_value: float = 1e-4
    max_value: Optional[float] = 10.0

    use_layernorm: bool = True


@dataclass
class PointwiseRBFKernelConfig:
    tx_model: TXScalarModelConfig = field(default_factory=TXScalarModelConfig)

    eta: float = 1e-5
    event_ndim: int = 1
    min_sigma: float = 1e-6
    use_prefactor: bool = False


@dataclass
class LearnedEnergyKernelConfig:
    x_dim: int = 2
    time_emb_dim: int = 32
    hidden_dim: int = 64
    num_layers: int = 2

    distance_input: Literal["norm", "squared_norm", "vector"] = "norm"
    event_ndim: int = 1
    min_value: float = 1e-4
    use_layernorm: bool = True


@dataclass
class WeightKernelConfig:
    type: Literal[
        "time_rbf",
        "time_scale_rbf",
        "pointwise_rbf",
        "learned_energy",
    ] = "time_rbf"

    time_rbf: TimeRBFKernelConfig = field(default_factory=TimeRBFKernelConfig)
    time_scale_rbf: TimeScaleRBFKernelConfig = field(default_factory=TimeScaleRBFKernelConfig)
    pointwise_rbf: PointwiseRBFKernelConfig = field(default_factory=PointwiseRBFKernelConfig)
    learned_energy: LearnedEnergyKernelConfig = field(default_factory=LearnedEnergyKernelConfig)


def build_weight_kernel(cfg: WeightKernelConfig) -> nn.Module:
    if cfg.type == "time_rbf":
        k = cfg.time_rbf
        s = k.base_sigma

        sigma_base = BaseSigmaModel(
            mode=s.mode,
            init_sigma=s.init_sigma,
            min_sigma=s.min_sigma,
        )

        return TimeRBFKernel(
            sigma_base=sigma_base,
            eta=k.eta,
            event_ndim=k.event_ndim,
            min_sigma=k.min_sigma,
            use_prefactor=k.use_prefactor,
        )

    if cfg.type == "time_scale_rbf":
        k = cfg.time_scale_rbf

        if k.scale_type == "mlp":
            m = k.mlp
            time_scale = TimeMLPScale(
                mode=m.mode,
                time_emb_dim=m.time_emb_dim,
                hidden_dim=m.hidden_dim,
                num_layers=m.num_layers,
                min_value=m.min_value,
                max_value=m.max_value,
                use_sinusoidal=m.use_sinusoidal,
                use_layernorm=m.use_layernorm,
            )

        elif k.scale_type == "chebyshev":
            c = k.chebyshev
            time_scale = ChebyshevTimeScale(
                degree=c.degree,
                init_value=c.init_value,
                min_value=c.min_value,
                max_value=c.max_value,
            )

        else:
            raise ValueError(f"Unknown scale_type: {k.scale_type}")

        return TimeScaleRBFKernel(
            time_scale=time_scale,
            eta=k.eta,
            event_ndim=k.event_ndim,
            min_sigma=k.min_sigma,
            use_prefactor=k.use_prefactor,
        )

    if cfg.type == "pointwise_rbf":
        k = cfg.pointwise_rbf
        tx = k.tx_model

        tx_model = TimeXMLPScale(
            x_dim=tx.x_dim,
            time_emb_dim=tx.time_emb_dim,
            hidden_dim=tx.hidden_dim,
            num_layers=tx.num_layers,
            init_value=tx.init_value,
            min_value=tx.min_value,
            max_value=tx.max_value,
            use_layernorm=tx.use_layernorm,
        )

        return PointwiseRBFKernel(
            tx_sigma_model=tx_model,
            eta=k.eta,
            event_ndim=k.event_ndim,
            min_sigma=k.min_sigma,
            use_prefactor=k.use_prefactor,
        )

    if cfg.type == "learned_energy":
        k = cfg.learned_energy

        return LearnedEnergyKernel(
            x_dim=k.x_dim,
            time_emb_dim=k.time_emb_dim,
            hidden_dim=k.hidden_dim,
            num_layers=k.num_layers,
            distance_input=k.distance_input,
            event_ndim=k.event_ndim,
            min_value=k.min_value,
            use_layernorm=k.use_layernorm,
        )

    raise ValueError(f"Unknown weight kernel type: {cfg.type}")
