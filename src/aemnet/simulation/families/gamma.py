"""Independent-marginal (multivariate) gamma components -- the flagship family
and the model of the original base paper (Vaidyanathan & Vani Lakshmi, 2017).

    f(z; alpha, beta) = prod_p Gam(z_p; shape alpha_p, scale beta_p)

Exponential family with T(z) = (z, log z) per marginal and mean parameters
mu = (alpha*beta, digamma(alpha) + log beta); the inverse link needs the root of
log(alpha) - digamma(alpha) = s, which has no closed form (Thm 1(b)).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.special import digamma, polygamma

from .base import EPS, Family

U_CLAMP = 12.0


def _loguniform(rng, lo, hi, size):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size=size))


def weighted_gamma_mle(x: np.ndarray, w: np.ndarray, iters: int = 100, tol: float = 1e-10):
    """Weighted MLE for Gamma(shape a, scale b): Minka (2002) start + Newton on the digamma score
    (the ECM M-step of Young et al., 2019)."""
    wsum = w.sum()
    if wsum < EPS:
        return 1.0, max(float(x.mean()), 1e-3)
    x = np.maximum(x, EPS)
    xbar = float((w * x).sum() / wsum)
    log_xbar = float((w * np.log(x)).sum() / wsum)
    s = np.log(max(xbar, EPS)) - log_xbar
    if s <= 0:  # degenerate (all x equal): huge shape fits
        return 1e4, xbar / 1e4
    # Minka's closed-form approximation as init
    a = (3.0 - s + np.sqrt((s - 3.0) ** 2 + 24.0 * s)) / (12.0 * s)
    for _ in range(iters):
        f = np.log(a) - digamma(a) - s
        fp = 1.0 / a - polygamma(1, a)
        step = f / fp
        a_new = a - step
        if a_new <= 0 or not np.isfinite(a_new):
            a_new = a / 2.0
        if abs(a_new - a) < tol * a:
            a = a_new
            break
        a = a_new
    return float(a), float(xbar / a)


def inverse_digamma_gap(s: torch.Tensor, iters: int = 8) -> torch.Tensor:
    """Solve log(a) - digamma(a) = s (s > 0) for a > 0, differentiably.

    Minka's (2002) closed-form start followed by Newton steps on log(a)."""
    s = s.clamp_min(1e-8)
    a = (3.0 - s + torch.sqrt((s - 3.0) ** 2 + 24.0 * s)) / (12.0 * s)
    for _ in range(iters):
        f = torch.log(a) - torch.digamma(a) - s
        fp = 1.0 / a - torch.polygamma(1, a)
        a = (a - f / fp).clamp_min(1e-6)
    return a


class GammaFamily(Family):
    name = "gamma"
    equivariance = "scale"
    exponential_family = True
    positive_support = True

    def __init__(self, k: int, alpha_range=(0.2, 50.0), beta_range=(0.05, 20.0), **kw):
        super().__init__(k, alpha_range=tuple(alpha_range), beta_range=tuple(beta_range), **kw)
        self.alpha_range = tuple(alpha_range)
        self.beta_range = tuple(beta_range)

    def param_shapes(self):
        return {"alpha": (self.k,), "beta": (self.k,)}

    @property
    def d(self) -> int:
        return 2 * self.k

    def decode(self, u):
        u = u.clamp(-U_CLAMP, U_CLAMP)
        return {"alpha": u[..., : self.k].exp(), "beta": u[..., self.k:].exp()}

    def encode(self, params):
        return torch.cat([params["alpha"].clamp_min(EPS).log(), params["beta"].clamp_min(EPS).log()], -1)

    def _broadcast(self, X, params):
        a = params["alpha"].unsqueeze(1)          # (B, 1, S, k)
        b = params["beta"].unsqueeze(1)
        Xc = X.clamp_min(EPS).unsqueeze(2)        # (B, n, 1, k)
        return Xc, a, b

    def log_prob_marginals(self, X, params):
        Xc, a, b = self._broadcast(X, params)
        return (a - 1.0) * Xc.log() - Xc / b - torch.lgamma(a) - a * b.log()

    def log_sf_marginals(self, X, params):
        # torch has no gradient of gammaincc w.r.t. the shape: shape enters detached.
        Xc, a, b = self._broadcast(X, params)
        a_e, x_e = torch.broadcast_tensors(a.detach(), Xc / b)
        return torch.special.gammaincc(a_e, x_e).clamp_min(1e-300).log()

    def sample_component(self, rng, comp, m):
        return np.stack([rng.gamma(comp["alpha"][p], comp["beta"][p], size=m) for p in range(self.k)], 1)

    def sample_prior(self, rng, g, shape_regime: str | None = None, **kw):
        alpha = _loguniform(rng, *self.alpha_range, size=(g, self.k))
        if shape_regime == "large":
            alpha = _loguniform(rng, 1.0, self.alpha_range[1], size=(g, self.k))
        elif shape_regime == "small":
            i, p = rng.integers(g), rng.integers(self.k)
            alpha[i, p] = _loguniform(rng, self.alpha_range[0], 1.0, size=())
        beta = _loguniform(rng, *self.beta_range, size=(g, self.k))
        return {"alpha": alpha, "beta": beta}

    def canonicalize(self, params, loc, scale):
        return {"alpha": params["alpha"], "beta": params["beta"] / scale.unsqueeze(1)}

    def uncanonicalize(self, params, loc, scale):
        return {"alpha": params["alpha"], "beta": params["beta"] * scale.unsqueeze(1)}

    def suff_stats(self, X, events):
        Xc = X.clamp_min(EPS)
        return torch.cat([Xc, Xc.log()], dim=-1)

    @property
    def suff_dim(self) -> int:
        return 2 * self.k

    def inverse_link(self, mu):
        m1, m2 = mu[..., : self.k].clamp_min(EPS), mu[..., self.k:]
        alpha = inverse_digamma_gap(m1.log() - m2)
        return {"alpha": alpha, "beta": m1 / alpha}

    def order_key(self, params):
        return (params["alpha"][..., 0] * params["beta"][..., 0]).clamp_min(EPS).log()

    def moment_init(self, X, events, w):
        w = w / max(w.sum(), EPS)
        mean = (w[:, None] * X).sum(0)
        var = (w[:, None] * (X - mean) ** 2).sum(0)
        alpha = np.clip(mean ** 2 / np.maximum(var, EPS), 0.05, 1e3)
        return {"alpha": alpha, "beta": np.maximum(mean / alpha, EPS)}

    def mstep(self, X, events, w, init=None):
        if not np.all(events == 1):
            return super().mstep(X, events, w, init)
        ab = [weighted_gamma_mle(X[:, p], w) for p in range(self.k)]
        return {"alpha": np.array([a for a, _ in ab]), "beta": np.array([b for _, b in ab])}
