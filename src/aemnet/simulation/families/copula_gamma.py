"""Gaussian-copula multivariate gamma components (dependent marginals) -- the
extension the base paper names as future work and WH-MCLUST cannot handle.

Component density: prod_p Gam(z_p; a_p, b_p) * c_R(q), q_p = Phi^{-1}(F_{a_p,b_p}(z_p)),
    log c_R(q) = -log|L| - 1/2 (||L^{-1} q||^2 - ||q||^2),  R = L L^T.
The correlation Cholesky factor is parameterised by an unconstrained unit-lower
matrix M with L = rownormalise(M) (so u contains M's strictly-lower entries).

Note: torch provides no gradient of the regularised incomplete gamma w.r.t. the
shape, so the shape enters the copula term detached (the gamma-marginal part
still carries the full shape gradient).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm

from .base import EPS, Family
from .gamma import GammaFamily, U_CLAMP, _loguniform
from .gaussian import _unit_lower_to_corr_chol


class CopulaGammaFamily(Family):
    name = "copula_gamma"
    equivariance = "scale"
    positive_support = True
    independent_marginals = False
    mstep_maxiter = 40

    def __init__(self, k: int, alpha_range=(0.3, 30.0), beta_range=(0.05, 20.0), corr_scale=0.8, **kw):
        super().__init__(k, alpha_range=tuple(alpha_range), beta_range=tuple(beta_range),
                         corr_scale=float(corr_scale), **kw)
        self.alpha_range, self.beta_range, self.corr_scale = tuple(alpha_range), tuple(beta_range), float(corr_scale)
        self._gamma = GammaFamily(k, alpha_range, beta_range)
        self._r, self._c = torch.tril_indices(k, k, offset=-1)

    def param_shapes(self):
        return {"alpha": (self.k,), "beta": (self.k,), "Lc": (self.k, self.k)}

    @property
    def d(self) -> int:
        return 2 * self.k + self.k * (self.k - 1) // 2

    def decode(self, u):
        k = self.k
        uc = u[..., : 2 * k].clamp(-U_CLAMP, U_CLAMP)
        M = torch.diag_embed(u.new_ones(*u.shape[:-1], k))
        if k > 1:
            M = M.clone()
            M[..., self._r, self._c] = u[..., 2 * k:].clamp(-20, 20)
        return {"alpha": uc[..., :k].exp(), "beta": uc[..., k:].exp(), "Lc": _unit_lower_to_corr_chol(M)}

    def encode(self, params):
        parts = [params["alpha"].clamp_min(EPS).log(), params["beta"].clamp_min(EPS).log()]
        if self.k > 1:
            Lc = params["Lc"]
            diag = torch.diagonal(Lc, dim1=-2, dim2=-1).clamp_min(EPS)
            M = Lc / diag.unsqueeze(-1)
            parts.append(M[..., self._r, self._c])
        return torch.cat(parts, -1)

    def log_prob(self, X, events, params):
        marg = self._gamma.log_prob_marginals(X, params).sum(-1)       # (B,n,S)
        if self.k == 1:
            return marg
        a = params["alpha"].unsqueeze(1).detach()
        b = params["beta"].unsqueeze(1)
        Xc = X.clamp_min(EPS).unsqueeze(2)
        a_e, x_e = torch.broadcast_tensors(a, Xc / b)
        p = torch.special.gammainc(a_e, x_e).clamp(1e-7, 1 - 1e-7)
        q = torch.special.ndtri(p)                                      # (B,n,S,k)
        Lc = params["Lc"]
        eye = torch.eye(self.k, dtype=Lc.dtype, device=Lc.device).expand_as(Lc)
        Linv = torch.linalg.solve_triangular(Lc, eye, upper=False)
        z = torch.einsum("bsij,bnsj->bnsi", Linv, q)
        logdet = torch.diagonal(Lc, dim1=-2, dim2=-1).clamp_min(EPS).log().sum(-1).unsqueeze(1)
        return marg - logdet - 0.5 * ((z ** 2).sum(-1) - (q ** 2).sum(-1))

    def sample_component(self, rng, comp, m):
        Zn = rng.standard_normal((m, self.k)) @ comp["Lc"].T
        P = np.clip(norm.cdf(Zn), 1e-12, 1 - 1e-12)
        return np.stack([gamma_dist.ppf(P[:, p], a=comp["alpha"][p], scale=comp["beta"][p])
                         for p in range(self.k)], 1)

    def sample_prior(self, rng, g, **kw):
        k = self.k
        alpha = _loguniform(rng, *self.alpha_range, size=(g, k))
        beta = _loguniform(rng, *self.beta_range, size=(g, k))
        Lc = np.empty((g, k, k))
        for i in range(g):
            M = np.eye(k)
            M[np.tril_indices(k, -1)] = rng.normal(0, self.corr_scale, size=k * (k - 1) // 2)
            Lc[i] = M / np.linalg.norm(M, axis=1, keepdims=True)
        return {"alpha": alpha, "beta": beta, "Lc": Lc}

    def canonicalize(self, params, loc, scale):
        return {**params, "beta": params["beta"] / scale.unsqueeze(1)}

    def uncanonicalize(self, params, loc, scale):
        return {**params, "beta": params["beta"] * scale.unsqueeze(1)}

    def suff_stats(self, X, events):
        Xc = X.clamp_min(EPS)
        lx = Xc.log()
        feats = [Xc, lx]
        if self.k > 1:
            outer = lx.unsqueeze(-1) * lx.unsqueeze(-2)
            feats.append(outer[..., self._r, self._c])
        return torch.cat(feats, -1)

    @property
    def suff_dim(self) -> int:
        return 2 * self.k + self.k * (self.k - 1) // 2

    def order_key(self, params):
        return (params["alpha"][..., 0] * params["beta"][..., 0]).clamp_min(EPS).log()

    def moment_init(self, X, events, w):
        base = self._gamma.moment_init(X, events, w)
        wn = w / max(w.sum(), EPS)
        lx = np.log(np.maximum(X, EPS))
        mu = (wn[:, None] * lx).sum(0)
        C = (wn[:, None, None] * (lx - mu)[:, :, None] * (lx - mu)[:, None, :]).sum(0)
        sd = np.sqrt(np.maximum(np.diag(C), EPS))
        R = C / np.outer(sd, sd)
        R = 0.95 * R + 0.05 * np.eye(self.k)
        try:
            Lc = np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            Lc = np.eye(self.k)
        Lc = Lc / np.linalg.norm(Lc, axis=1, keepdims=True)
        return {**base, "Lc": Lc}
