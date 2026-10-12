"""Independent-marginal g-and-k components -- a genuinely likelihood-free family
(quantile-defined; the density needs numerical inversion), the standard SBI
benchmark. Used for E4 with the E-block likelihood term switched off.

    Q(z) = A + B (1 + c tanh(g z / 2)) (1 + z^2)^kk z,   z ~ N(0, 1),  c = 0.8
    f(x) = phi(z) / Q'(z),   z = Q^{-1}(x)   (numpy only, vectorised bisection)
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import norm

from ...mixture_params import MixtureParams
from .base import EPS, Family

C_GK = 0.8
Z_MAX = 12.0


def gk_quantile(z, A, B, g, kk):
    return A + B * (1.0 + C_GK * np.tanh(g * z / 2.0)) * (1.0 + z ** 2) ** kk * z


def gk_dquantile(z, B, g, kk):
    th = np.tanh(g * z / 2.0)
    h = 1.0 + C_GK * th
    dh = C_GK * (g / 2.0) * (1.0 - th ** 2)
    p = (1.0 + z ** 2) ** kk * z
    dp = (1.0 + z ** 2) ** (kk - 1.0) * (1.0 + (1.0 + 2.0 * kk) * z ** 2)
    return B * (dh * p + h * dp)


def gk_logpdf(x: np.ndarray, A, B, g, kk, iters: int = 70) -> np.ndarray:
    lo = np.full_like(x, -Z_MAX)
    hi = np.full_like(x, Z_MAX)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        above = gk_quantile(mid, A, B, g, kk) > x
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    z = 0.5 * (lo + hi)
    dq = np.maximum(gk_dquantile(z, B, g, kk), 1e-300)
    return norm.logpdf(z) - np.log(dq)


class GAndKFamily(Family):
    name = "gandk"
    equivariance = "location_scale"
    has_torch_logprob = False

    def __init__(self, k: int, A_range=5.0, B_range=(0.3, 3.0), g_range=3.0, k_range=(0.05, 1.0), **kw):
        super().__init__(k, A_range=float(A_range), B_range=tuple(B_range), g_range=float(g_range),
                         k_range=tuple(k_range), **kw)
        self.A_range, self.B_range, self.g_range, self.k_range = float(A_range), tuple(B_range), float(g_range), tuple(k_range)

    def param_shapes(self):
        return {"A": (self.k,), "B": (self.k,), "g": (self.k,), "kk": (self.k,)}

    @property
    def d(self) -> int:
        return 4 * self.k

    def decode(self, u):
        k = self.k
        return {"A": u[..., :k], "B": u[..., k:2 * k].clamp(-10, 10).exp(),
                "g": u[..., 2 * k:3 * k].clamp(-6, 6), "kk": u[..., 3 * k:].clamp(-8, 1.5).exp()}

    def encode(self, params):
        return torch.cat([params["A"], params["B"].clamp_min(EPS).log(), params["g"],
                          params["kk"].clamp_min(EPS).log()], -1)

    def u_bounds(self):
        return (-8.0, 8.0)

    def component_logpdf_np(self, X, events, mp: MixtureParams) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.zeros((X.shape[0], mp.g))
        for i in range(mp.g):
            c = mp.component(i)
            for p in range(self.k):
                out[:, i] += gk_logpdf(X[:, p], c["A"][p], c["B"][p], c["g"][p], c["kk"][p])
        return out

    def sample_component(self, rng, comp, m):
        z = rng.standard_normal((m, self.k))
        return gk_quantile(z, comp["A"], comp["B"], comp["g"], comp["kk"])

    def sample_prior(self, rng, g, **kw):
        k = self.k
        return {"A": rng.uniform(-self.A_range, self.A_range, size=(g, k)),
                "B": np.exp(rng.uniform(*np.log(self.B_range), size=(g, k))),
                "g": rng.uniform(-self.g_range, self.g_range, size=(g, k)),
                "kk": np.exp(rng.uniform(*np.log(self.k_range), size=(g, k)))}

    def canonicalize(self, params, loc, scale):
        s = scale.unsqueeze(1)
        return {**params, "A": (params["A"] - loc.unsqueeze(1)) / s, "B": params["B"] / s}

    def uncanonicalize(self, params, loc, scale):
        s = scale.unsqueeze(1)
        return {**params, "A": params["A"] * s + loc.unsqueeze(1), "B": params["B"] * s}

    def order_key(self, params):
        return params["A"][..., 0]

    def moment_init(self, X, events, w):
        wn = w / max(w.sum(), EPS)
        order = np.argsort(X, axis=0)
        A, B = np.empty(self.k), np.empty(self.k)
        for p in range(self.k):
            xs, ws = X[order[:, p], p], np.cumsum(wn[order[:, p]])
            q = lambda t: xs[min(np.searchsorted(ws, t), len(xs) - 1)]  # noqa: E731
            A[p] = q(0.5)
            B[p] = max((q(0.75) - q(0.25)) / 1.349, 1e-3)
        return {"A": A, "B": B, "g": np.zeros(self.k), "kk": np.full(self.k, 0.2)}
