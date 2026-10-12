"""Real-data diagnostics (E6/E7): is a real dataset inside the simulator's reach?

prior_predictive_check compares canonicalised summary statistics of real
datasets with those of simulated datasets via an RBF-kernel MMD and a
permutation p-value. A small p-value flags a simulation gap -> use hybrid
refinement and/or `inference.adaptation`.
"""
from __future__ import annotations

import numpy as np

from ..simulation.families.base import Family
from ..simulation.simulator import MixtureSimulator


def canonical_summaries(family: Family, X: np.ndarray) -> np.ndarray:
    from ..baselines.abc_smc import summaries  # lazy: avoids an import cycle

    X = np.asarray(X, dtype=np.float64)
    X = X[:, None] if X.ndim == 1 else X
    loc, scale = family.canonical_loc_scale_np(X)
    return summaries(family, (X - loc) / scale)


def mmd_rbf(A: np.ndarray, B: np.ndarray, bandwidth: float | None = None) -> float:
    Z = np.concatenate([A, B])
    D2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    if bandwidth is None:
        bandwidth = np.sqrt(0.5 * np.median(D2[D2 > 0])) if np.any(D2 > 0) else 1.0
    K = np.exp(-D2 / (2 * bandwidth ** 2))
    n = len(A)
    Kaa, Kbb, Kab = K[:n, :n], K[n:, n:], K[:n, n:]
    m = len(B)
    return float((Kaa.sum() - np.trace(Kaa)) / (n * (n - 1)) + (Kbb.sum() - np.trace(Kbb)) / (m * (m - 1))
                 - 2 * Kab.mean())


def prior_predictive_check(family: Family, real_sets: list, simulator: MixtureSimulator, n_sim: int = 300,
                           n_perm: int = 500, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    R = np.stack([canonical_summaries(family, X) for X in real_sets])
    sims = []
    for _ in range(n_sim):
        n = int(np.asarray(real_sets[rng.integers(len(real_sets))]).shape[0])
        sims.append(canonical_summaries(family, simulator.sample_dataset(rng, n=n)["X"]))
    S = np.stack(sims)
    mu, sd = S.mean(0), S.std(0) + 1e-8
    R, S = (R - mu) / sd, (S - mu) / sd
    stat = mmd_rbf(R, S)
    Z = np.concatenate([R, S])
    null = []
    for _ in range(n_perm):
        p = rng.permutation(len(Z))
        null.append(mmd_rbf(Z[p[: len(R)]], Z[p[len(R):]]))
    pval = float((np.sum(np.asarray(null) >= stat) + 1) / (n_perm + 1))
    return {"mmd": stat, "p_value": pval, "n_real": len(R), "n_sim": len(S)}
