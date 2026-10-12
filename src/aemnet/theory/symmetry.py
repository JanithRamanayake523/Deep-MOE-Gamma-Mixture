"""Label-symmetry illustrations (Thm 2, E3).

Given posterior samples (e.g. from MCMC) of an exchangeable-prior mixture:
    naive_posterior_mean   -> collapses: every component ~ the same (Thm 2a)
    sorted_posterior_mean  -> posterior mean of order statistics (Thm 2b, biased under overlap)
    evaluation.relabel.quotient_frechet_mean -> the quotient Bayes estimator (Thm 3)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ..mixture_params import MixtureParams
from ..simulation.families.base import Family
from ..simulation.families.stats import prior_u_stats


def naive_posterior_mean(family: Family, samples: list[MixtureParams]) -> MixtureParams:
    """Coordinate-wise mean with the labels as sampled (no relabelling). Under label
    switching in the sampler (or averaging over all g! relabellings) this is symmetric."""
    U = np.stack([family.encode_np(s.comps) for s in samples])
    pi = np.stack([s.pi for s in samples]).mean(0)
    return MixtureParams(pi / pi.sum(), family.decode_np(U.mean(0)))


def symmetrize(samples: list[MixtureParams], rng=None) -> list[MixtureParams]:
    """Apply a uniformly random label permutation to each sample (what an ideal,
    perfectly mixing sampler of an exchangeable posterior produces)."""
    rng = np.random.default_rng(rng)
    return [s.permuted(rng.permutation(s.g)) for s in samples]


def sorted_posterior_mean(family: Family, samples: list[MixtureParams]) -> MixtureParams:
    """Relabel each sample by the family's order key (identifiability constraint), then average."""
    import torch

    out = []
    for s in samples:
        with torch.no_grad():
            key = family.order_key(family.to_torch(s.comps))[0].numpy()
        out.append(s.permuted(np.argsort(key, kind="stable")))
    return naive_posterior_mean(family, out)


def collapse_index(family: Family, mp: MixtureParams, u_std=None) -> float:
    """Mean pairwise distance between components in standardised coordinates (0 = collapsed)."""
    u_std = prior_u_stats(family)[1] if u_std is None else u_std
    Z = family.encode_np(mp.comps) / u_std
    if mp.g < 2:
        return 0.0
    D = np.sqrt(((Z[:, None] - Z[None]) ** 2).mean(-1))
    return float(D[np.triu_indices(mp.g, 1)].mean())


def ordering_bias_two_normals(delta: float, s: float) -> float:
    """Thm 2b closed form. If the two component locations have (approximately)
    independent N(m_i, s^2) posteriors with |m_1 - m_2| = delta, the sort-then-average
    estimator of the gap is E[max] - E[min] = delta (2 Phi(delta/tau) - 1) + 2 tau phi(delta/tau),
    tau = s sqrt(2); the returned bias (>= 0) is that minus delta, largest at delta = 0."""
    tau = s * np.sqrt(2.0)
    gap = delta * (2 * norm.cdf(delta / tau) - 1) + 2 * tau * norm.pdf(delta / tau)
    return float(gap - abs(delta))
