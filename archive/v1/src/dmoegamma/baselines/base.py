"""Baseline interface: every estimator maps a sample X (n, k) to a FitResult."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..params import Theta


@dataclass
class FitResult:
    theta: Theta
    resp: np.ndarray | None = None   # (n, g) responsibilities, if available
    loglik: float = float("nan")     # achieved mixture log-likelihood on X
    converged: bool = True
    runtime: float = 0.0             # seconds
    extra: dict = field(default_factory=dict)


class Baseline(ABC):
    name: str = "abstract"

    @abstractmethod
    def _fit(self, X: np.ndarray, g: int | None, rng: np.random.Generator) -> FitResult:
        ...

    def fit(self, X: np.ndarray, g: int | None = None, seed: int = 0) -> FitResult:
        """g=None means the method selects the model order itself (if able)."""
        t0 = time.perf_counter()
        result = self._fit(np.asarray(X, dtype=np.float64), g, np.random.default_rng(seed))
        result.runtime = time.perf_counter() - t0
        return result
