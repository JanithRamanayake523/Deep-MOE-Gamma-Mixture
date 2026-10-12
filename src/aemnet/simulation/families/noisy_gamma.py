"""Gamma components observed with additive Gaussian measurement error -- the
genuinely intractable member of the gamma family (E4).

    y_p = x_p + eps_p,   x_p ~ Gamma(alpha_p, beta_p),   eps_p ~ N(0, sigma_p^2)  (independent marginals)
    f(y) = int_0^inf Gam(x; alpha, beta) phi_sigma(y - x) dx      -- no closed form

Natural for sensor / gauge data (rain gauges, flow meters, intensities) where the
quantity is gamma but each reading carries instrument noise. The noise sd is a
per-component, per-marginal *scale* parameter, so scale equivariance holds.

There is no torch likelihood (AEM-Net's E-block uses the learned score only, as in
the likelihood-free setting). A numpy density is provided for evaluation and for
the EM / ABC baselines by panelled Gauss-Legendre quadrature in x: panel breakpoints
at gamma quantiles {1e-12, .001, .1, .5, .9, .999, 1-1e-12} and at y + sigma*{-8,-2,0,2,8},
16 nodes per panel; for alpha < 1 every panel is integrated in w = x^alpha, which
removes the x^(alpha-1) singularity at 0. Accurate to ~1e-6 relative in bulk and tails
(tests/v2 compare it with direct adaptive integration).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.special import gammaincinv, gammaln, logsumexp

from ...mixture_params import MixtureParams
from .base import EPS, Family
from .gamma import U_CLAMP, _loguniform

_T, _W = np.polynomial.legendre.leggauss(16)
_Q = np.array([1e-12, 1e-3, 0.1, 0.5, 0.9, 0.999, 1 - 1e-12])
_K = np.array([-8.0, -2.0, 0.0, 2.0, 8.0])
_LOG2PI = np.log(2 * np.pi)


def noisy_gamma_logpdf(y: np.ndarray, alpha: float, beta: float, sigma: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    n = y.shape[0]
    qs = gammaincinv(alpha, _Q) * beta
    pts = np.concatenate([np.broadcast_to(qs, (n, len(qs))), y[:, None] + sigma * _K, np.zeros((n, 1))], 1)
    pts = np.sort(np.maximum(pts, 0.0), axis=1)                            # (n, P+1), pts[:, 0] = 0
    lgam = gammaln(alpha) + alpha * np.log(beta)

    def log_normal(x):
        return -0.5 * ((y[:, None, None] - x) / sigma) ** 2 - np.log(sigma) - 0.5 * _LOG2PI

    lo, hi = pts[:, :-1], pts[:, 1:]                                         # (n, P) panels, first = [0, .]
    if alpha < 1.0:
        # integrate in w = x^alpha on every panel: Gam(x) dx = e^{-x/beta} / (alpha Gamma(alpha) beta^alpha) dw,
        # which removes the x^(alpha-1) singularity at 0
        wlo, whi = lo ** alpha, hi ** alpha
        half = 0.5 * (whi - wlo)
        w = wlo[:, :, None] + half[:, :, None] * (_T + 1.0)[None, None, :]
        x = np.maximum(w, 0.0) ** (1.0 / alpha)
        with np.errstate(divide="ignore"):
            lw = np.log(half)[:, :, None] + np.log(_W)[None, None, :]
        terms = -np.log(alpha) - x / beta - lgam + log_normal(x) + lw
    else:
        half = 0.5 * (hi - lo)
        x = lo[:, :, None] + half[:, :, None] * (_T + 1.0)[None, None, :]
        xs = np.maximum(x, 1e-300)
        with np.errstate(divide="ignore"):
            lw = np.log(half)[:, :, None] + np.log(_W)[None, None, :]
        terms = (alpha - 1.0) * np.log(xs) - xs / beta - lgam + log_normal(xs) + lw
    return logsumexp(terms.reshape(n, -1), axis=1)


class NoisyGammaFamily(Family):
    name = "noisy_gamma"
    equivariance = "scale"
    has_torch_logprob = False
    positive_support = False

    def __init__(self, k: int, alpha_range=(0.5, 30.0), beta_range=(0.05, 20.0), noise_ratio_range=(0.05, 1.0), **kw):
        super().__init__(k, alpha_range=tuple(alpha_range), beta_range=tuple(beta_range),
                         noise_ratio_range=tuple(noise_ratio_range), **kw)
        self.alpha_range, self.beta_range = tuple(alpha_range), tuple(beta_range)
        self.noise_ratio_range = tuple(noise_ratio_range)

    def param_shapes(self):
        return {"alpha": (self.k,), "beta": (self.k,), "sigma": (self.k,)}

    @property
    def d(self) -> int:
        return 3 * self.k

    def decode(self, u):
        u = u.clamp(-U_CLAMP, U_CLAMP)
        k = self.k
        return {"alpha": u[..., :k].exp(), "beta": u[..., k:2 * k].exp(), "sigma": u[..., 2 * k:].exp()}

    def encode(self, params):
        return torch.cat([params[n].clamp_min(EPS).log() for n in ("alpha", "beta", "sigma")], -1)

    def component_logpdf_np(self, X, events, mp: MixtureParams) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.zeros((X.shape[0], mp.g))
        for i in range(mp.g):
            c = mp.component(i)
            for p in range(self.k):
                out[:, i] += noisy_gamma_logpdf(X[:, p], c["alpha"][p], c["beta"][p], c["sigma"][p])
        return out

    def sample_component(self, rng, comp, m):
        x = np.stack([rng.gamma(comp["alpha"][p], comp["beta"][p], size=m) for p in range(self.k)], 1)
        return x + rng.standard_normal((m, self.k)) * comp["sigma"]

    def sample_prior(self, rng, g, **kw):
        alpha = _loguniform(rng, *self.alpha_range, size=(g, self.k))
        beta = _loguniform(rng, *self.beta_range, size=(g, self.k))
        ratio = _loguniform(rng, *self.noise_ratio_range, size=(g, self.k))
        return {"alpha": alpha, "beta": beta, "sigma": ratio * np.sqrt(alpha) * beta}

    def canonicalize(self, params, loc, scale):
        s = scale.unsqueeze(1)
        return {"alpha": params["alpha"], "beta": params["beta"] / s, "sigma": params["sigma"] / s}

    def uncanonicalize(self, params, loc, scale):
        s = scale.unsqueeze(1)
        return {"alpha": params["alpha"], "beta": params["beta"] * s, "sigma": params["sigma"] * s}

    def order_key(self, params):
        return (params["alpha"][..., 0] * params["beta"][..., 0]).clamp_min(EPS).log()

    def init_features_np(self, X):
        return np.asarray(X, dtype=np.float64)

    def moment_init(self, X, events, w):
        w = w / max(w.sum(), EPS)
        mean = np.maximum((w[:, None] * X).sum(0), 1e-3)
        var = np.maximum((w[:, None] * (X - mean) ** 2).sum(0), 1e-8)
        sigma = 0.3 * np.sqrt(var)
        gvar = np.maximum(var - sigma ** 2, 1e-8)
        alpha = np.clip(mean ** 2 / gvar, 0.3, 1e3)
        return {"alpha": alpha, "beta": np.maximum(gvar / mean, EPS), "sigma": sigma}
