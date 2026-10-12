"""Prior p(theta) over gamma-mixture parameters and stratified sampling.

Design (RESEARCH_METHODOLOGY.md section 5.4): wide, weakly-informative ranges
covering both the singular (alpha < 1) and near-Gaussian regimes, Dirichlet(1)
weights, and optional overlap stratification so training coverage matches the
evaluation design.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..params import Theta
from .mixture import bayes_error


@dataclass
class PriorConfig:
    k: int = 2
    g_min: int = 1
    g_max: int = 4
    alpha_range: tuple[float, float] = (0.1, 30.0)
    beta_range: tuple[float, float] = (0.1, 30.0)
    dirichlet_concentration: float = 1.0
    n_min: int = 25
    n_max: int = 5000
    overlap_bins: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.05), (0.05, 0.20), (0.20, 0.50)]
    )


class ThetaPrior:
    """Samples theta ~ p(theta); supports overlap- and shape-stratified draws."""

    def __init__(self, config: PriorConfig):
        self.cfg = config

    def _loguniform(self, rng: np.random.Generator, lo: float, hi: float, size) -> np.ndarray:
        return np.exp(rng.uniform(np.log(lo), np.log(hi), size=size))

    def sample_g(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.cfg.g_min, self.cfg.g_max + 1))

    def sample_n(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.cfg.n_min, self.cfg.n_max + 1))

    def sample_theta(
        self,
        rng: np.random.Generator,
        g: int | None = None,
        shape_regime: str | None = None,
    ) -> Theta:
        """Draw one theta.

        shape_regime: None (mixed), "large" (all alpha >= 1), or "small"
        (at least one alpha < 1) -- used by the factorial study.
        """
        cfg = self.cfg
        if g is None:
            g = self.sample_g(rng)
        if shape_regime in (None, "mixed"):
            alpha = self._loguniform(rng, *cfg.alpha_range, size=(g, cfg.k))
        elif shape_regime == "large":
            alpha = self._loguniform(rng, 1.0, cfg.alpha_range[1], size=(g, cfg.k))
        elif shape_regime == "small":
            alpha = self._loguniform(rng, *cfg.alpha_range, size=(g, cfg.k))
            # force at least one singular shape
            i, p = rng.integers(g), rng.integers(cfg.k)
            alpha[i, p] = self._loguniform(rng, cfg.alpha_range[0], 1.0, size=())
        else:
            raise ValueError(f"unknown shape_regime: {shape_regime}")
        beta = self._loguniform(rng, *cfg.beta_range, size=(g, cfg.k))
        pi = rng.dirichlet(np.full(g, cfg.dirichlet_concentration))
        return Theta(pi, alpha, beta)

    def sample_stratified(
        self,
        rng: np.random.Generator,
        overlap_bin: tuple[float, float] | None = None,
        g: int | None = None,
        shape_regime: str | None = None,
        max_tries: int = 200,
        n_mc: int = 4000,
    ) -> Theta:
        """Rejection-sample theta until its estimated Bayes error falls inside
        `overlap_bin` (lo, hi). Falls back to an unstratified draw after
        `max_tries` rejections (never infinite-loops)."""
        if overlap_bin is None:
            return self.sample_theta(rng, g=g, shape_regime=shape_regime)
        lo, hi = overlap_bin
        if g is None:
            g = self.sample_g(rng)
        if g == 1:  # overlap undefined for a single component
            return self.sample_theta(rng, g=g, shape_regime=shape_regime)
        for _ in range(max_tries):
            theta = self.sample_theta(rng, g=g, shape_regime=shape_regime)
            err = bayes_error(theta, n_mc=n_mc, rng=rng)
            if lo <= err < hi:
                return theta
        return self.sample_theta(rng, g=g, shape_regime=shape_regime)
