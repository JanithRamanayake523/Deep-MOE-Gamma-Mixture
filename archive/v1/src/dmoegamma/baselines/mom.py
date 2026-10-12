"""B4 -- method of moments: the fast classical floor.

Matches the first 3g-1 raw moments of each marginal (mixture marginal moment:
E[X_p^r] = sum_i pi_i * Gamma(a_ip + r)/Gamma(a_ip) * b_ip^r) by least squares
over the full unconstrained parameter vector. Heuristic but defensible as a
floor; documented as such in the evaluation.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.special import gammaln

from ..params import pack_unconstrained, unpack_unconstrained
from ..simulation.mixture import logpdf as mixture_logpdf
from .base import Baseline, FitResult

EPS = 1e-12


def _theoretical_moments(theta, orders: np.ndarray) -> np.ndarray:
    """(k, len(orders)) theoretical marginal raw moments."""
    k = theta.k
    out = np.empty((k, len(orders)))
    for p in range(k):
        a, b = theta.alpha[:, p], theta.beta[:, p]  # (g,)
        for j, r in enumerate(orders):
            out[p, j] = (theta.pi * np.exp(gammaln(a + r) - gammaln(a)) * b**r).sum()
    return out


class MethodOfMomentsBaseline(Baseline):
    name = "mom"

    def _fit(self, X: np.ndarray, g: int | None, rng: np.random.Generator) -> FitResult:
        if g is None:
            raise ValueError("MoM baseline requires a known g")
        n, k = X.shape
        orders = np.arange(1, 3 * g, dtype=float)
        emp = np.stack([(X**r).mean(0) for r in orders], axis=1)  # (k, 3g-1)

        # crude start: overall mean/var as a one-component fit, replicated
        mean, var = X.mean(0), X.var(0) + EPS
        b0 = var / mean
        a0 = mean / b0
        u0 = np.concatenate(
            [np.zeros(g - 1),
             np.log(np.tile(a0, (g, 1)) * rng.uniform(0.5, 1.5, (g, k))).ravel(),
             np.log(np.tile(b0, (g, 1)) * rng.uniform(0.5, 1.5, (g, k))).ravel()]
        )
        theta0 = unpack_unconstrained(u0, g, k)

        def residuals(u: np.ndarray) -> np.ndarray:
            theta = unpack_unconstrained(u, g, k)
            theo = _theoretical_moments(theta, orders)
            return ((theo - emp) / (np.abs(emp) + EPS)).ravel()

        res = least_squares(residuals, pack_unconstrained(theta0), method="lm", max_nfev=5000)
        theta = unpack_unconstrained(res.x, g, k)
        return FitResult(theta=theta, resp=None, loglik=float(mixture_logpdf(X, theta).sum()),
                         converged=bool(res.success))
