"""Independent-marginal Student-t components with a component-level degrees of
freedom nu (heavy tails; the M-step needs an inner ECM loop, no closed form)."""
from __future__ import annotations

import math

import numpy as np
import torch

from .base import EPS, Family


class StudentTFamily(Family):
    name = "student_t"
    equivariance = "location_scale"

    def __init__(self, k: int, mu_range=5.0, sigma_range=(0.2, 3.0), nu_range=(1.5, 30.0), **kw):
        super().__init__(k, mu_range=float(mu_range), sigma_range=tuple(sigma_range),
                         nu_range=tuple(nu_range), **kw)
        self.mu_range, self.sigma_range, self.nu_range = float(mu_range), tuple(sigma_range), tuple(nu_range)

    def param_shapes(self):
        return {"mu": (self.k,), "sigma": (self.k,), "nu": (1,)}

    @property
    def d(self) -> int:
        return 2 * self.k + 1

    def decode(self, u):
        k = self.k
        return {"mu": u[..., :k], "sigma": u[..., k:2 * k].clamp(-10, 10).exp(),
                "nu": 1.0 + u[..., 2 * k:].clamp(-6, 6).exp()}

    def encode(self, params):
        return torch.cat([params["mu"], params["sigma"].clamp_min(EPS).log(),
                          (params["nu"] - 1.0).clamp_min(EPS).log()], -1)

    def log_prob_marginals(self, X, params):
        mu, sig, nu = (params[n].unsqueeze(1) for n in ("mu", "sigma", "nu"))
        z = (X.unsqueeze(2) - mu) / sig
        return (torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - 0.5 * torch.log(nu * math.pi)
                - sig.log() - (nu + 1) / 2 * torch.log1p(z ** 2 / nu))

    def sample_component(self, rng, comp, m):
        return comp["mu"] + comp["sigma"] * rng.standard_t(comp["nu"][0], size=(m, self.k))

    def sample_prior(self, rng, g, **kw):
        k = self.k
        return {"mu": rng.uniform(-self.mu_range, self.mu_range, size=(g, k)),
                "sigma": np.exp(rng.uniform(*np.log(self.sigma_range), size=(g, k))),
                "nu": 1.0 + np.exp(rng.uniform(np.log(self.nu_range[0] - 1), np.log(self.nu_range[1] - 1),
                                               size=(g, 1)))}

    def canonicalize(self, params, loc, scale):
        s = scale.unsqueeze(1)
        return {"mu": (params["mu"] - loc.unsqueeze(1)) / s, "sigma": params["sigma"] / s, "nu": params["nu"]}

    def uncanonicalize(self, params, loc, scale):
        s = scale.unsqueeze(1)
        return {"mu": params["mu"] * s + loc.unsqueeze(1), "sigma": params["sigma"] * s, "nu": params["nu"]}

    def order_key(self, params):
        return params["mu"][..., 0]

    def moment_init(self, X, events, w):
        w = w / max(w.sum(), EPS)
        mu = (w[:, None] * X).sum(0)
        # weighted median absolute deviation is more robust for heavy tails
        sd = np.sqrt(np.maximum((w[:, None] * (X - mu) ** 2).sum(0), EPS))
        return {"mu": mu, "sigma": np.maximum(0.7 * sd, EPS), "nu": np.array([5.0])}
