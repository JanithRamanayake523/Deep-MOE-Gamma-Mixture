"""Adapter exposing a trained Deep-MoE network through the Baseline interface,
so the benchmark runner treats neural and classical estimators uniformly."""
from __future__ import annotations

import numpy as np

from ..simulation.mixture import logpdf as mixture_logpdf
from .base import Baseline, FitResult


class NeuralEstimatorBaseline(Baseline):
    name = "deep_moe"

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device

    def _fit(self, X: np.ndarray, g: int | None, rng: np.random.Generator) -> FitResult:
        theta, resp = self.model.estimate(X, g=g, device=self.device)
        return FitResult(theta=theta, resp=resp,
                         loglik=float(mixture_logpdf(X, theta).sum()),
                         converged=True)  # no iterative fitting: no failure mode
