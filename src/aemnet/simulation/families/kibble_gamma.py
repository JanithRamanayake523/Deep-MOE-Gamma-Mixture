"""Kibble-Moran bivariate gamma components (k = 2): common shape alpha, scales
(beta_1, beta_2), correlation rho in (0, 1).

The density involves a modified Bessel function I_{alpha-1}; we use its exact
negative-binomial mixture representation
    f(x, y) = sum_m c_m Gam(x; alpha+m, (1-rho) beta_1) Gam(y; alpha+m, (1-rho) beta_2),
    c_m = NB(m; alpha, 1-rho) = Gamma(alpha+m)/(Gamma(alpha) m!) rho^m (1-rho)^alpha,
truncated at `series_terms` terms (ample for the default prior, where the NB
mean alpha*rho/(1-rho) <= 45). The same representation gives an exact sampler.
"""
from __future__ import annotations

import numpy as np
import torch

from .base import EPS, Family
from .gamma import GammaFamily, U_CLAMP, _loguniform


class KibbleGammaFamily(Family):
    name = "kibble_gamma"
    equivariance = "scale"
    positive_support = True
    independent_marginals = False
    mstep_maxiter = 40

    def __init__(self, k: int = 2, alpha_range=(0.3, 8.0), beta_range=(0.05, 20.0), rho_range=(0.05, 0.85),
                 series_terms: int = 256, chunk: int = 32, **kw):
        if k != 2:
            raise ValueError("Kibble-Moran bivariate gamma requires k = 2")
        super().__init__(k, alpha_range=tuple(alpha_range), beta_range=tuple(beta_range),
                         rho_range=tuple(rho_range), series_terms=int(series_terms), **kw)
        self.alpha_range, self.beta_range, self.rho_range = tuple(alpha_range), tuple(beta_range), tuple(rho_range)
        self.series_terms, self.chunk = int(series_terms), int(chunk)

    def param_shapes(self):
        return {"alpha": (1,), "beta": (2,), "rho": (1,)}

    @property
    def d(self) -> int:
        return 4

    def decode(self, u):
        uc = u[..., :3].clamp(-U_CLAMP, U_CLAMP)
        rho = torch.sigmoid(u[..., 3:]).clamp(1e-4, 0.97)
        return {"alpha": uc[..., :1].exp(), "beta": uc[..., 1:3].exp(), "rho": rho}

    def encode(self, params):
        rho = params["rho"].clamp(1e-6, 1 - 1e-6)
        return torch.cat([params["alpha"].clamp_min(EPS).log(), params["beta"].clamp_min(EPS).log(),
                          torch.log(rho) - torch.log1p(-rho)], -1)

    def log_prob(self, X, events, params):
        a = params["alpha"].unsqueeze(1)                     # (B,1,S,1)
        b = params["beta"].unsqueeze(1)                      # (B,1,S,2)
        rho = params["rho"].unsqueeze(1)                     # (B,1,S,1)
        Xc = X.clamp_min(EPS).unsqueeze(2)                   # (B,n,1,2)
        th = (1.0 - rho) * b                                 # (B,1,S,2)
        slx = Xc.log().sum(-1, keepdim=True)                 # log x + log y
        sx = (Xc / th).sum(-1, keepdim=True)
        slth = th.log().sum(-1, keepdim=True)
        base = -torch.lgamma(a) + a * torch.log1p(-rho) - sx      # (B,n,S,1)
        acc = None
        for start in range(0, self.series_terms, self.chunk):
            m = torch.arange(start, min(start + self.chunk, self.series_terms), dtype=X.dtype, device=X.device)
            am = a + m                                            # (B,1,S,C)
            terms = (base - torch.lgamma(m + 1.0) + m * rho.log() + (am - 1.0) * slx
                     - torch.lgamma(am) - am * slth)
            lse = torch.logsumexp(terms, dim=-1)
            acc = lse if acc is None else torch.logaddexp(acc, lse)
        return acc

    def sample_component(self, rng, comp, m):
        a, rho = comp["alpha"][0], comp["rho"][0]
        N = rng.negative_binomial(a, 1.0 - rho, size=m)
        th = (1.0 - rho) * comp["beta"]
        return np.stack([rng.gamma(a + N, th[0]), rng.gamma(a + N, th[1])], 1)

    def sample_prior(self, rng, g, **kw):
        return {"alpha": _loguniform(rng, *self.alpha_range, size=(g, 1)),
                "beta": _loguniform(rng, *self.beta_range, size=(g, 2)),
                "rho": rng.uniform(*self.rho_range, size=(g, 1))}

    def canonicalize(self, params, loc, scale):
        return {**params, "beta": params["beta"] / scale.unsqueeze(1)}

    def uncanonicalize(self, params, loc, scale):
        return {**params, "beta": params["beta"] * scale.unsqueeze(1)}

    def suff_stats(self, X, events):
        Xc = X.clamp_min(EPS)
        lx = Xc.log()
        return torch.cat([Xc, lx, (Xc[..., :1] * Xc[..., 1:])], -1)

    @property
    def suff_dim(self) -> int:
        return 5

    def order_key(self, params):
        return (params["alpha"][..., 0] * params["beta"][..., 0]).clamp_min(EPS).log()

    def moment_init(self, X, events, w):
        base = GammaFamily(2).moment_init(X, events, w)
        alpha = float(np.exp(np.log(base["alpha"]).mean()))
        wn = w / max(w.sum(), EPS)
        mean = (wn[:, None] * X).sum(0)
        D = X - mean
        cov = (wn * D[:, 0] * D[:, 1]).sum()
        sd = np.sqrt(np.maximum((wn[:, None] * D ** 2).sum(0), EPS))
        rho = float(np.clip(cov / (sd[0] * sd[1]), 0.02, 0.9))
        return {"alpha": np.array([alpha]), "beta": np.maximum(mean / alpha, EPS), "rho": np.array([rho])}
