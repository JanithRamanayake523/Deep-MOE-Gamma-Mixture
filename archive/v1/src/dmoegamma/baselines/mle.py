"""B3 -- direct numerical MLE on the observed-data log-likelihood.

BFGS with random restarts and best-of selection: the strong, honest baseline
the base paper did not compare against (RESEARCH_METHODOLOGY.md 6.3).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..params import Theta, pack_unconstrained, unpack_unconstrained
from ..simulation.mixture import logpdf as mixture_logpdf, responsibilities
from .base import Baseline, FitResult
from .ecm import weighted_gamma_mle
from .wh_mclust import _kmeans

EPS = 1e-12


class DirectMLEBaseline(Baseline):
    name = "mle"

    def __init__(self, n_restarts: int = 10, max_iter: int = 500):
        self.n_restarts = n_restarts
        self.max_iter = max_iter

    def _moment_init(self, X: np.ndarray, g: int, rng: np.random.Generator) -> Theta:
        """K-means partition + per-cluster univariate gamma MLE as a sane start."""
        assign = _kmeans(np.log(np.maximum(X, EPS)), g, rng)
        k = X.shape[1]
        alpha = np.ones((g, k))
        beta = np.ones((g, k))
        pi = np.bincount(assign, minlength=g) / X.shape[0] + EPS
        pi /= pi.sum()
        for i in range(g):
            pts = X[assign == i]
            if len(pts) < 3:
                pts = X
            for p in range(k):
                alpha[i, p], beta[i, p] = weighted_gamma_mle(pts[:, p], np.ones(len(pts)))
        return Theta(pi, alpha, beta)

    def _fit(self, X: np.ndarray, g: int | None, rng: np.random.Generator) -> FitResult:
        if g is None:
            raise ValueError("Direct MLE baseline requires a known g")
        k = X.shape[1]

        def objective(u: np.ndarray) -> float:
            try:
                theta = unpack_unconstrained(u, g, k)
            except (ValueError, FloatingPointError):
                return 1e12
            ll = mixture_logpdf(X, theta)
            if not np.all(np.isfinite(ll)):
                return 1e12
            return -float(ll.mean())

        inits = [pack_unconstrained(self._moment_init(X, g, rng))]
        for _ in range(self.n_restarts - 1):
            jitter = rng.normal(0.0, 1.0, size=inits[0].shape)
            inits.append(inits[0] + jitter)

        best = None
        for u0 in inits:
            res = minimize(objective, u0, method="BFGS",
                           options={"maxiter": self.max_iter, "gtol": 1e-5})
            if best is None or res.fun < best.fun:
                best = res
        theta = unpack_unconstrained(best.x, g, k)
        return FitResult(theta=theta, resp=responsibilities(X, theta),
                         loglik=float(mixture_logpdf(X, theta).sum()),
                         converged=bool(best.success))
