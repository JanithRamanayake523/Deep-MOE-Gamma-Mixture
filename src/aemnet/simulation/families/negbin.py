"""Independent-marginal negative-binomial components (counts), parameterised by
mean m and dispersion r (Var = m + m^2/r). Discrete: no canonicalisation group."""
from __future__ import annotations

import numpy as np
import torch

from .base import EPS, Family
from .gamma import U_CLAMP, _loguniform


class NegBinFamily(Family):
    name = "negbin"
    equivariance = "none"
    discrete = True

    def __init__(self, k: int, mean_range=(0.5, 100.0), disp_range=(0.3, 30.0), **kw):
        super().__init__(k, mean_range=tuple(mean_range), disp_range=tuple(disp_range), **kw)
        self.mean_range, self.disp_range = tuple(mean_range), tuple(disp_range)

    def param_shapes(self):
        return {"m": (self.k,), "r": (self.k,)}

    @property
    def d(self) -> int:
        return 2 * self.k

    def decode(self, u):
        u = u.clamp(-U_CLAMP, U_CLAMP)
        return {"m": u[..., : self.k].exp(), "r": u[..., self.k:].exp()}

    def encode(self, params):
        return torch.cat([params["m"].clamp_min(EPS).log(), params["r"].clamp_min(EPS).log()], -1)

    def log_prob_marginals(self, X, params):
        x = X.clamp_min(0).unsqueeze(2)
        m, r = params["m"].unsqueeze(1), params["r"].unsqueeze(1)
        return (torch.lgamma(x + r) - torch.lgamma(r) - torch.lgamma(x + 1)
                + r * (r.log() - (r + m).log()) + x * (m.log() - (r + m).log()))

    def sample_component(self, rng, comp, m_):
        m, r = comp["m"], comp["r"]
        return np.stack([rng.negative_binomial(r[p], r[p] / (r[p] + m[p]), size=m_) for p in range(self.k)],
                        1).astype(np.float64)

    def sample_prior(self, rng, g, **kw):
        return {"m": _loguniform(rng, *self.mean_range, size=(g, self.k)),
                "r": _loguniform(rng, *self.disp_range, size=(g, self.k))}

    def features(self, X, events):
        Xc = X.clamp_min(0)
        return torch.cat([torch.log1p(Xc), Xc.sqrt(), (Xc == 0).to(X.dtype)], -1)

    def order_key(self, params):
        return params["m"][..., 0].clamp_min(EPS).log()

    def moment_init(self, X, events, w):
        w = w / max(w.sum(), EPS)
        mean = np.maximum((w[:, None] * X).sum(0), 1e-3)
        var = (w[:, None] * (X - mean) ** 2).sum(0)
        r = np.clip(mean ** 2 / np.maximum(var - mean, 1e-3), 0.05, 1e3)
        return {"m": mean, "r": r}
