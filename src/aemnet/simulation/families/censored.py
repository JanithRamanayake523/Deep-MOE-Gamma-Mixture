"""Right-censored mixtures (survival / reliability). Wraps any independent-
marginal base family that implements `log_sf_marginals` (gamma, Weibull).

Censoring mechanism (dataset level, applied identically to every component):
C_p ~ Exponential(scale s_p), with s_p solved per dataset so that the censoring
fraction P(C_p < X_p) matches a target drawn from `censor_frac_range`. Because
s_p is set relative to the data, the mechanism commutes with rescaling, so the
scale-canonicalisation stays exact. The event indicator is an extra input feature.
"""
from __future__ import annotations

import numpy as np
import torch

from .base import Family


def _solve_censor_scale(x: np.ndarray, target: float) -> float:
    """Find s with mean(1 - exp(-x/s)) = target (monotone in s) by bisection in log s."""
    x = np.maximum(x, 1e-300)
    lo, hi = np.log(x.min()) - 10, np.log(x.max()) + 20
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        frac = np.mean(-np.expm1(-x / np.exp(mid)))
        if frac > target:
            lo = mid   # too much censoring: larger scale
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))


class CensoredFamily(Family):
    independent_marginals = True

    def __init__(self, base: Family, censor_frac_range=(0.1, 0.4)):
        if not base.independent_marginals:
            raise ValueError("censoring wrapper requires an independent-marginal base family")
        super().__init__(base.k, censor_frac_range=tuple(censor_frac_range))
        self.base = base
        self.censor_frac_range = tuple(censor_frac_range)
        self.name = f"censored_{base.name}"
        self.equivariance = base.equivariance
        self.positive_support = base.positive_support
        self.mstep_maxiter = base.mstep_maxiter

    def spec(self) -> dict:
        return {"name": "censored", "base": self.base.spec(), "censor_frac_range": list(self.censor_frac_range)}

    def param_shapes(self):
        return self.base.param_shapes()

    @property
    def d(self) -> int:
        return self.base.d

    def decode(self, u):
        return self.base.decode(u)

    def encode(self, params):
        return self.base.encode(params)

    def u_bounds(self):
        return self.base.u_bounds()

    def log_prob_marginals(self, X, params):
        return self.base.log_prob_marginals(X, params)

    def log_prob(self, X, events, params):
        ev = events.unsqueeze(2)                                  # (B,n,1,k)
        lp = self.base.log_prob_marginals(X, params)
        ls = self.base.log_sf_marginals(X, params)
        return (ev * lp + (1.0 - ev) * ls).sum(-1)

    def sample_component(self, rng, comp, m):
        return self.base.sample_component(rng, comp, m)

    def sample_dataset(self, rng, mp, n):
        X, _, labels = self.base.sample_dataset(rng, mp, n)
        events = np.ones_like(X)
        for p in range(self.k):
            target = rng.uniform(*self.censor_frac_range)
            s = _solve_censor_scale(X[:, p], target)
            C = rng.exponential(s, size=n)
            events[:, p] = (X[:, p] <= C).astype(np.float64)
            X[:, p] = np.minimum(X[:, p], C)
        return X, events, labels

    def sample_prior(self, rng, g, **kw):
        return self.base.sample_prior(rng, g, **kw)

    def canonicalize(self, params, loc, scale):
        return self.base.canonicalize(params, loc, scale)

    def uncanonicalize(self, params, loc, scale):
        return self.base.uncanonicalize(params, loc, scale)

    def features(self, X, events):
        return torch.cat([self.base.features(X, events), events], -1)

    @property
    def feature_dim(self) -> int:
        return self.base.feature_dim + self.k

    def suff_stats(self, X, events):
        return self.features(X, events)

    @property
    def suff_dim(self) -> int:
        return self.feature_dim

    def order_key(self, params):
        return self.base.order_key(params)

    def init_features_np(self, X):
        return self.base.init_features_np(X)

    def moment_init(self, X, events, w):
        # naive (ignores censoring); the numerical M-step then corrects it
        return self.base.moment_init(X, events, w)
