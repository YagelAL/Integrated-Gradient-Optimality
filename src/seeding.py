"""Deterministic seeding for reproducible runs."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SeedingState:
    """Record of the seeds + flags applied."""

    seed: int
    deterministic: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    pythonhashseed: str
    cublas_workspace_config: str | None


def seed_everything(seed: int, *, deterministic: bool = True) -> SeedingState:
    """Seed Python, NumPy, and PyTorch (CPU + all CUDA devices).

    Args:
        seed: Non-negative seed value. All RNGs are derived from this single seed.
        deterministic: When True, also enables deterministic CUDA kernels via
            ``torch.use_deterministic_algorithms(True)`` and ``cudnn.deterministic = True``,
            and sets ``CUBLAS_WORKSPACE_CONFIG`` for deterministic matmul.

    Returns:
        A :class:`SeedingState` recording the seed and determinism flags.

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    pythonhashseed = str(seed)
    os.environ["PYTHONHASHSEED"] = pythonhashseed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cublas_workspace_config: str | None = None
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        cublas_workspace_config = ":4096:8"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace_config
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    return SeedingState(
        seed=seed,
        deterministic=deterministic,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        pythonhashseed=pythonhashseed,
        cublas_workspace_config=cublas_workspace_config,
    )


__all__ = ["SeedingState", "seed_everything"]