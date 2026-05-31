import math
import random

from typing import Literal, Tuple

import numpy as np
import torch
from torchdyn.datasets import generate_moons
from torch.utils.data import TensorDataset, DataLoader


def eight_normal_sample(n, dim, scale=1, var=1):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), math.sqrt(var) * torch.eye(dim)
    )
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(8), n, replacement=True)
    data = []
    for i in range(n):
        data.append(centers[multi[i]] + noise[i])
    data = torch.stack(data)
    return data


def sample_8gaussians(n):
    return eight_normal_sample(n, 2, scale=5, var=0.1).float()


def sample_moons(n):
    x0, _ = generate_moons(n, noise=0.2)
    return x0 * 3 - 1


def create_datasets(
    source: Literal["gaussians", "moons"],
    target: Literal["gaussians", "moons"],
    train_size: int,
    val_size: int,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create source and target datasets for training and validation.

    Args:
        source: source distribution type
        target: target distribution type
        train_size: number of training samples
        val_size: number of validation samples
        device: target device for tensors

    Returns:
        Tuple of (x0_train, x1_train, x0_val, x1_val)
    """
    # Source distribution
    if source == "gaussians":
        x0_train = sample_8gaussians(train_size)
        x0_val = sample_8gaussians(val_size)
    elif source == "moons":
        x0_train = sample_moons(train_size).float()
        x0_val = sample_moons(val_size).float()
    else:
        raise ValueError(f"Unknown source distribution: {source}")

    # Target distribution
    if target == "gaussians":
        x1_train = sample_8gaussians(train_size)
        x1_val = sample_8gaussians(val_size)
    elif target == "moons":
        x1_train = sample_moons(train_size).float()
        x1_val = sample_moons(val_size).float()
    else:
        raise ValueError(f"Unknown target distribution: {target}")

    # Move validation data to device
    x0_val = x0_val.to(device)
    x1_val = x1_val.to(device)

    return x0_train, x1_train, x0_val, x1_val


def create_dataloaders(
    x0_train: torch.Tensor,
    x1_train: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create DataLoader for training pairs (x0, x1).

    Args:
        x0_train: source training samples [N, D]
        x1_train: target training samples [N, D]
        batch_size: batch size for training
        shuffle: whether to shuffle data
        num_workers: number of data loading workers
        drop_last: whether to drop last incomplete batch

    Returns:
        DataLoader yielding (x0, x1) pairs
    """
    dataset = TensorDataset(x0_train, x1_train)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    return loader
