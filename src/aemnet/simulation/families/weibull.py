"""Independent-marginal Weibull components (shape kappa, scale lambda).
No closed-form M-step (the shape solves a transcendental equation)."""
from __future__ import annotations

import numpy as np
import torch

from .base import EPS, Family
from .gamma import U_CLAMP, _loguniform


def weighted_weibull_mle(x: np.ndarray, w: np.ndarray, iters: int = 60):
    """Weighted Weibull MLE via Newton on the profile score in kappa."""
    x = np.maximum(x, EPS)
    w = w / max(w.sum(), EPS)
    lx = np.log(x)
    mlx = float((w * lx).sum())
    sd = float(np.sqrt(max((w * (lx - mlx) ** 2).sum(), EPS)))
    kap = 1.2 / sd  # moment-type start (Gumbel on log scale)
    for _ in range(iters):
        xk = np.exp(kap * (lx - lx.max()))  # scaled to avoid overflow; ratios invariant
        s0 = (w * xk).sum()
        s1 = (w * xk * lx).sum()
        s2 = (w * xk * lx ** 2).sum()
        f = s1 / s0 - 1.0 / kap - mlx
        fp = (s2 / s0 - (s1 / s0) ** 2) + 1.0 / kap ** 2
        step = f / fp
        new = kap - step
        if not np.isfinite(new) or new <= 0:
            new = kap / 2
        if abs(new - kap) < 1e-10 * kap:
            kap = new
            break
        kap = new
    lam = float(np.exp(lx.max()) * ((w * np.exp(kap * (lx - lx.max()))).sum()) ** (1.0 / kap))
    return float(kap), lam


class WeibullFamily(Family):
    name = "weibull"
    equivariance = "scale"
    positive_support = True

    def __init__(self, k: int, shape_range=(0.5, 8.0), scale_range=(0.05, 20.0), **kw):
        super().__init__(k, shape_range=tuple(shape_range), scale_range=tuple(scale_range), **kw)
        self.shape_range, self.scale_range = tuple(shape_range), tuple(scale_range)

    def param_shapes(self):
        return {"kappa": (self.k,), "lam": (self.k,)}

    @property
    def d(self) -> int:
        return 2 * self.k

    def decode(self, u):
        u = u.clamp(-U_CLAMP, U_CLAMP)
        return {"kappa": u[..., : self.k].exp(), "lam": u[..., self.k:].exp()}

    def encode(self, params):
        return torch.cat([params["kappa"].clamp_min(EPS).log(), params["lam"].clamp_min(EPS).log()], -1)

    def _z(self, X, params):
        kap = params["kappa"].unsqueeze(1)
        lam = params["lam"].unsqueeze(1)
        Xc = X.clamp_min(EPS).unsqueeze(2)
        return Xc, kap, lam

    def log_prob_marginals(self, X, params):
        Xc, kap, lam = self._z(X, params)
        lz = Xc.log() - lam.log()
        return kap.log() - lam.log() + (kap - 1.0) * lz - (kap * lz).clamp(max=60).exp()

    def log_sf_marginals(self, X, params):
        Xc, kap, lam = self._z(X, params)
        return -(kap * (Xc.log() - lam.log())).clamp(max=60).exp()

    def sample_component(self, rng, comp, m):
        return np.stack([comp["lam"][p] * rng.weibull(comp["kappa"][p], size=m) for p in range(self.k)], 1)

    def sample_prior(self, rng, g, **kw):
        return {"kappa": _loguniform(rng, *self.shape_range, size=(g, self.k)),
                "lam": _loguniform(rng, *self.scale_range, size=(g, self.k))}

    def canonicalize(self, params, loc, scale):
        return {"kappa": params["kappa"], "lam": params["lam"] / scale.unsqueeze(1)}

    def uncanonicalize(self, params, loc, scale):
        return {"kappa": params["kappa"], "lam": params["lam"] * scale.unsqueeze(1)}

    def order_key(self, params):
        k0, l0 = params["kappa"][..., 0], params["lam"][..., 0]
        return l0.clamp_min(EPS).log() + torch.lgamma(1.0 + 1.0 / k0)

    def moment_init(self, X, events, w):
        ka = [weighted_weibull_mle(X[:, p], w, iters=5) for p in range(self.k)]
        return {"kappa": np.array([a for a, _ in ka]), "lam": np.array([b for _, b in ka])}

    def mstep(self, X, events, w, init=None):
        if not np.all(events == 1):
            return super().mstep(X, events, w, init)
        ka = [weighted_weibull_mle(X[:, p], w) for p in range(self.k)]
        return {"kappa": np.array([a for a, _ in ka]), "lam": np.array([b for _, b in ka])}
