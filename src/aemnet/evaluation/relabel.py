"""Label-switching resolution for sample-based estimators (MCMC, ABC, flows).

quotient_frechet_mean
    The quotient-space Bayes estimator of Thm 3: iterate (i) Hungarian-align every
    sample to the current estimate in standardised coordinates, (ii) average. Its
    fixed points are exactly the fixed points of Stephens' (2000) relabelling
    algorithm under the squared quotient loss, i.e. the target a network trained
    with L_Q converges to. Used as the E3 reference and as the MCMC/ABC point estimate.

stephens_relabel
    Classical Stephens (2000) KL relabelling on allocation probabilities p_ij.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..mixture_params import MixtureParams
from ..simulation.families.base import Family


def _coords(family: Family, mp: MixtureParams, u_std: np.ndarray) -> np.ndarray:
    return family.encode_np(mp.comps) / u_std                              # (g, d)


def align_to_pivot(family: Family, samples: list[MixtureParams], pivot: MixtureParams,
                   u_std: np.ndarray, w_pi: float = 1.0) -> tuple[list[MixtureParams], np.ndarray]:
    """Permute every sample to best match `pivot`. Returns (aligned samples, perms (M,g))."""
    zp = _coords(family, pivot, u_std)
    aligned, perms = [], []
    for s in samples:
        zs = _coords(family, s, u_std)
        C = ((zp[:, None, :] - zs[None, :, :]) ** 2).mean(-1) + w_pi * (pivot.pi[:, None] - s.pi[None, :]) ** 2
        _, cols = linear_sum_assignment(C)
        aligned.append(s.permuted(cols))
        perms.append(cols)
    return aligned, np.asarray(perms)


def _mean_in_coords(family: Family, samples: list[MixtureParams]) -> MixtureParams:
    U = np.stack([family.encode_np(s.comps) for s in samples])            # (M,g,d)
    pi = np.stack([s.pi for s in samples]).mean(0)
    return MixtureParams(pi / pi.sum(), family.decode_np(U.mean(0)))


def quotient_frechet_mean(family: Family, samples: list[MixtureParams], u_std: np.ndarray | None = None,
                          n_iter: int = 100, init: MixtureParams | None = None,
                          w_pi: float = 1.0) -> tuple[MixtureParams, list[MixtureParams]]:
    """Fréchet mean on Theta_g / S_g under the standardised squared quotient loss."""
    if u_std is None:
        from ..simulation.families.stats import prior_u_stats

        u_std = prior_u_stats(family)[1]
    est = init if init is not None else samples[len(samples) // 2]
    prev = None
    aligned = samples
    for _ in range(n_iter):
        aligned, perms = align_to_pivot(family, samples, est, u_std, w_pi)
        est = _mean_in_coords(family, aligned)
        if prev is not None and np.array_equal(perms, prev):
            break
        prev = perms
    return est, aligned


def stephens_relabel(P: np.ndarray, n_iter: int = 100) -> np.ndarray:
    """Stephens (2000) KL relabelling. P (M, n, g) allocation probabilities per draw.
    Returns perms (M, g) such that P[m][:, perms[m]] is consistently labelled."""
    M, n, g = P.shape
    perms = np.tile(np.arange(g), (M, 1))
    for _ in range(n_iter):
        Q = np.mean([P[m][:, perms[m]] for m in range(M)], axis=0)       # (n, g)
        logQ = np.log(np.maximum(Q, 1e-300))
        new = np.empty_like(perms)
        for m in range(M):
            Pm = np.maximum(P[m], 1e-300)
            # cost[i, j] = KL contribution of assigning draw label j to reference label i
            cost = -(Pm[:, None, :] * (logQ[:, :, None] - np.log(Pm)[:, None, :])).sum(0)  # (g_ref, g_draw)
            _, cols = linear_sum_assignment(cost)
            new[m] = cols
        if np.array_equal(new, perms):
            break
        perms = new
    return perms
