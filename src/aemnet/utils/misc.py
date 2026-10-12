"""Reproducibility and device helpers."""
from __future__ import annotations

import random

import numpy as np


def seed_all(seed: int) -> np.random.Generator:
    """Seed python/numpy/torch and return a numpy Generator for the simulator."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return np.random.default_rng(seed)


def get_device(prefer_cuda: bool = True) -> str:
    try:
        import torch

        if prefer_cuda and torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"
