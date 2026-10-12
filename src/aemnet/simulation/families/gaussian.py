"""Full-covariance Gaussian components -- the comparison family on which TGMM,
Set Transformer and Cluster-PFN operate. Closed-form M-step and inverse link."""
from __future__ import annotations

import math

import numpy as np
import torch

from .base import EPS, Family


def _unit_lower_to_corr_chol(M: torch.Tensor) -> torch.Tensor:
    """Unit-diagonal lower-triangular M -> Cholesky factor of a correlation matrix
    (rows normalised to unit norm)."""
    return M / M.norm(dim=-1, keepdim=True)


class GaussianFamily(Family):
    name = "gaussian"
    equivariance = "location_scale"
    exponential_family = True
    independent_marginals = False

    def __init__(self, k: int, mu_range=5.0, sigma_range=(0.2, 3.0), corr_scale=0.7, **kw):
        super().__init__(k, mu_range=float(mu_range), sigma_range=tuple(sigma_range),
                         corr_scale=float(corr_scale), **kw)
        self.mu_range, self.sigma_range, self.corr_scale = float(mu_range), tuple(sigma_range), float(corr_scale)
        r, c = torch.tril_indices(k, k, offset=-1)
        self._r, self._c = r, c
        iu, ju = torch.triu_indices(k, k)
        self._iu, self._ju = iu, ju

    def param_shapes(self):
        return {"mu": (self.k,), "L": (self.k, self.k)}

    @property
    def d(self) -> int:
        return 2 * self.k + self.k * (self.k - 1) // 2

    def decode(self, u):
        k = self.k
        mu = u[..., :k]
        ld = u[..., k:2 * k].clamp(-10, 10)
        L = u.new_zeros(*u.shape[:-1], k, k)
        if k > 1:
            L[..., self._r, self._c] = u[..., 2 * k:]
        L = L + torch.diag_embed(ld.exp())
        return {"mu": mu, "L": L}

    def encode(self, params):
        L = params["L"]
        diag = torch.diagonal(L, dim1=-2, dim2=-1).clamp_min(EPS).log()
        parts = [params["mu"], diag]
        if self.k > 1:
            parts.append(L[..., self._r, self._c])
        return torch.cat(parts, -1)

    def log_prob(self, X, events, params):
        mu, L = params["mu"], params["L"]                       # (B,S,k), (B,S,k,k)
        eye = torch.eye(self.k, dtype=L.dtype, device=L.device).expand_as(L)
        Linv = torch.linalg.solve_triangular(L, eye, upper=False)
        diff = X.unsqueeze(2) - mu.unsqueeze(1)                 # (B,n,S,k)
        z = torch.einsum("bsij,bnsj->bnsi", Linv, diff)
        logdet = torch.diagonal(L, dim1=-2, dim2=-1).clamp_min(EPS).log().sum(-1)  # (B,S)
        return -0.5 * (z ** 2).sum(-1) - logdet.unsqueeze(1) - 0.5 * self.k * math.log(2 * math.pi)

    def sample_component(self, rng, comp, m):
        return comp["mu"] + rng.standard_normal((m, self.k)) @ comp["L"].T

    def sample_prior(self, rng, g, **kw):
        k = self.k
        mu = rng.uniform(-self.mu_range, self.mu_range, size=(g, k))
        sig = np.exp(rng.uniform(*np.log(self.sigma_range), size=(g, k)))
        L = np.empty((g, k, k))
        for i in range(g):
            M = np.eye(k)
            M[np.tril_indices(k, -1)] = rng.normal(0, self.corr_scale, size=k * (k - 1) // 2)
            Lc = M / np.linalg.norm(M, axis=1, keepdims=True)
            L[i] = sig[i][:, None] * Lc
        return {"mu": mu, "L": L}

    def canonicalize(self, params, loc, scale):
        s = scale.unsqueeze(1)
        return {"mu": (params["mu"] - loc.unsqueeze(1)) / s, "L": params["L"] / s.unsqueeze(-1)}

    def uncanonicalize(self, params, loc, scale):
        s = scale.unsqueeze(1)
        return {"mu": params["mu"] * s + loc.unsqueeze(1), "L": params["L"] * s.unsqueeze(-1)}

    def features(self, X, events):
        return torch.cat([X, X ** 2, torch.sign(X) * torch.log1p(X.abs())], -1)

    def suff_stats(self, X, events):
        outer = X.unsqueeze(-1) * X.unsqueeze(-2)
        return torch.cat([X, outer[..., self._iu, self._ju]], -1)

    @property
    def suff_dim(self) -> int:
        return self.k + self.k * (self.k + 1) // 2

    def inverse_link(self, mu):
        k = self.k
        m = mu[..., :k]
        second = mu.new_zeros(*mu.shape[:-1], k, k)
        second[..., self._iu, self._ju] = mu[..., k:]
        second = second + second.transpose(-1, -2) - torch.diag_embed(torch.diagonal(second, dim1=-2, dim2=-1))
        cov = second - m.unsqueeze(-1) * m.unsqueeze(-2)
        cov = cov + 1e-6 * torch.eye(k, dtype=mu.dtype, device=mu.device)
        return {"mu": m, "L": torch.linalg.cholesky(cov)}

    def order_key(self, params):
        return params["mu"][..., 0]

    def moment_init(self, X, events, w):
        return self.mstep(X, events, w)

    def mstep(self, X, events, w, init=None):
        ws = max(float(w.sum()), EPS)
        mu = (w[:, None] * X).sum(0) / ws
        D = X - mu
        cov = (w[:, None, None] * D[:, :, None] * D[:, None, :]).sum(0) / ws
        cov = cov + 1e-6 * np.eye(self.k)
        return {"mu": mu, "L": np.linalg.cholesky(cov)}
