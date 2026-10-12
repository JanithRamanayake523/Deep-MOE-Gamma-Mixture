"""B5 -- Bayesian MCMC (Wiper et al., 2001 style), dependency-free.

Random-walk Metropolis-Hastings on the unconstrained parameterization with a
weak Gaussian prior; the posterior mean is the point estimate. Used as a
gold-standard accuracy reference on small-n subsets only.
"""
from __future__ import annotations

import numpy as np

from ..params import pack_unconstrained, unpack_unconstrained
from ..simulation.mixture import logpdf as mixture_logpdf, responsibilities
from .base import Baseline, FitResult
from .mle import DirectMLEBaseline

PRIOR_SD = 5.0  # weak N(0, PRIOR_SD^2) prior on each unconstrained coordinate


class MCMCSBaseline(Baseline):
    name = "mcmc"

    def __init__(self, n_iter: int = 20000, burn_in: int = 5000, step: float = 0.05):
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.step = step

    def _fit(self, X: np.ndarray, g: int | None, rng: np.random.Generator) -> FitResult:
        if g is None:
            raise ValueError("MCMC baseline requires a known g")
        k = X.shape[1]
        init = DirectMLEBaseline(n_restarts=3, max_iter=200)._fit(X, g, rng).theta
        u = pack_unconstrained(init)
        d = u.size

        def log_target(v: np.ndarray) -> float:
            try:
                theta = unpack_unconstrained(v, g, k)
            except (ValueError, FloatingPointError):
                return -np.inf
            ll = mixture_logpdf(X, theta)
            if not np.all(np.isfinite(ll)):
                return -np.inf
            return float(ll.sum()) - 0.5 * float((v**2).sum()) / PRIOR_SD**2

        cur, cur_lp = u.copy(), log_target(u)
        acc, samples = 0, []
        step = self.step
        for it in range(self.n_iter):
            prop = cur + step * rng.normal(size=d)
            prop_lp = log_target(prop)
            if np.log(rng.uniform()) < prop_lp - cur_lp:
                cur, cur_lp, acc = prop, prop_lp, acc + 1
            if it >= self.burn_in:
                samples.append(cur.copy())
            # crude adaptation during burn-in targeting ~30% acceptance
            if it < self.burn_in and (it + 1) % 200 == 0:
                rate = acc / (it + 1)
                step *= 1.1 if rate > 0.35 else (0.9 if rate < 0.2 else 1.0)
        theta = unpack_unconstrained(np.mean(samples, axis=0), g, k)
        return FitResult(theta=theta, resp=responsibilities(X, theta),
                         loglik=float(mixture_logpdf(X, theta).sum()),
                         converged=True,
                         extra={"acceptance": acc / self.n_iter})
