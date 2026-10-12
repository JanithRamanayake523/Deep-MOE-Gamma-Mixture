"""ABC-SMC / population Monte Carlo (Beaumont et al., 2009) for any family -- the
likelihood-free classical baseline of E4.

Works on canonicalised data. Summaries: per-marginal quantiles of a support-
appropriate transform, pairwise correlations of that transform, and (counts)
the zero fraction; distances standardised by the MAD of round-0 summaries.
Prior: the family prior in u (+ Dirichlet(1) weights). The importance-weight
prior density is approximated by a Gaussian fitted to prior draws in u
(documented approximation; exact for none of our priors, but monotone and smooth).
Point estimate = quotient Fréchet mean of the final weighted-resampled population.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal

from ..evaluation.relabel import quotient_frechet_mean
from ..simulation.families.base import Family
from ..simulation.families.stats import prior_u_stats
from .generic import GenericBaseline, GenericFit
from .mle import unpack_v

QUANTILES = np.linspace(0.05, 0.95, 10)


def summaries(family: Family, X: np.ndarray) -> np.ndarray:
    T = family.init_features_np(X)
    parts = [np.quantile(T, QUANTILES, axis=0).ravel(order="F")]
    if family.k > 1:
        C = np.corrcoef(T, rowvar=False)
        parts.append(C[np.triu_indices(family.k, 1)])
    if family.discrete:
        parts.append((X == 0).mean(0))
    return np.nan_to_num(np.concatenate(parts))


class ABCSMCBaseline(GenericBaseline):
    name = "abc_smc"

    def __init__(self, family: Family, n_particles: int = 1000, n_rounds: int = 6, keep: float = 0.5,
                 max_sim_n: int = 1000):
        super().__init__(family)
        self.n_particles, self.n_rounds, self.keep, self.max_sim_n = n_particles, n_rounds, keep, max_sim_n

    def _prior_v(self, rng, g):
        fam = self.family
        pi = rng.dirichlet(np.ones(g))
        u = fam.encode_np(fam.sample_prior(rng, g)).ravel()
        return np.concatenate([np.log(pi[:-1] / pi[-1]), u])

    def _sim_summary(self, v, g, n, rng):
        fam = self.family
        mp = unpack_v(fam, v, g)
        X, _, _ = fam.sample_dataset(rng, mp, n)
        return summaries(fam, X)

    def _fit(self, X, events, g, rng):
        fam = self.family
        loc, scale = fam.canonical_loc_scale_np(X)
        Xc = (X - loc) / scale
        n = min(X.shape[0], self.max_sim_n)
        s_obs = summaries(fam, Xc)
        N = self.n_particles
        # prior density approximation in v-space
        u_mean, u_std = prior_u_stats(fam)
        pv = np.stack([self._prior_v(rng, g) for _ in range(2000)])
        prior_pdf = multivariate_normal(pv.mean(0), np.cov(pv, rowvar=False) + 1e-6 * np.eye(pv.shape[1]),
                                        allow_singular=True)
        # round 0
        V = np.stack([self._prior_v(rng, g) for _ in range(N)])
        S = np.stack([self._sim_summary(v, g, n, rng) for v in V])
        mad = np.maximum(np.median(np.abs(S - np.median(S, 0)), 0), 1e-8)
        dist = np.sqrt((((S - s_obs) / mad) ** 2).sum(1))
        eps = np.quantile(dist, self.keep)
        keep = dist <= eps
        V, W = V[keep], np.full(keep.sum(), 1.0 / keep.sum())
        for _ in range(1, self.n_rounds):
            cov = 2.0 * np.cov(V.T, aweights=W) + 1e-8 * np.eye(V.shape[1])
            newV, newD = [], []
            while len(newV) < N:
                base = V[rng.choice(len(V), p=W)]
                v = rng.multivariate_normal(base, cov)
                d = np.sqrt((((self._sim_summary(v, g, n, rng) - s_obs) / mad) ** 2).sum())
                newV.append(v)
                newD.append(d)
            newV, newD = np.asarray(newV), np.asarray(newD)
            eps = np.quantile(newD, self.keep)
            sel = newD <= eps
            Vs = newV[sel]
            kern = multivariate_normal(np.zeros(V.shape[1]), cov, allow_singular=True)
            denom = np.array([(W * kern.pdf(V - v)).sum() for v in Vs])
            Wn = prior_pdf.pdf(Vs) / np.maximum(denom, 1e-300)
            V, W = Vs, Wn / Wn.sum()
        idx = rng.choice(len(V), size=min(len(V), 500), p=W)
        samples_c = [unpack_v(fam, V[i], g) for i in idx]
        est_c, aligned_c = quotient_frechet_mean(fam, samples_c, u_std)
        est = fam.uncanonicalize_np(est_c, loc, scale)
        return GenericFit(est, fam.responsibilities_np(X, events, est),
                          extra={"samples": [fam.uncanonicalize_np(s, loc, scale) for s in aligned_c],
                                 "final_eps": float(eps)})
