"""Empirical check of Thm 4's Assumption A (local EM contraction).

Perturb the (local) MLE in standardised canonical coordinates at several radii,
apply one EM step, and report the contraction ratio
    kappa_hat(r) = d_Q(M(theta), theta_hat) / d_Q(theta, theta_hat),
d_Q being the root quotient distance. kappa_hat < 1 up to radius r estimates the
basin radius/rate used in Thm 4 (and the K_n = log n / (2 log 1/kappa) schedule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluation.metrics import quotient_alignment
from ..inference.refine import em_step
from ..mixture_params import MixtureParams
from ..simulation.families.base import Family
from ..simulation.families.stats import prior_u_stats


def _perturb(family: Family, mp: MixtureParams, direction: np.ndarray, r: float, u_std) -> MixtureParams:
    g, d = mp.g, family.d
    u = family.encode_np(mp.comps) + r * direction[: g * d].reshape(g, d) * u_std
    logpi = np.log(mp.pi) + r * direction[g * d:]
    pi = np.exp(logpi - logpi.max())
    return MixtureParams(pi / pi.sum(), family.decode_np(u))


def estimate_em_contraction(family: Family, X, events, mle: MixtureParams, radii=(0.05, 0.1, 0.2, 0.4, 0.8),
                            n_dirs: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    u_std = prior_u_stats(family)[1]
    ev = np.ones_like(X) if events is None else events
    rows = []
    for r in radii:
        for _ in range(n_dirs):
            v = rng.standard_normal(mle.g * family.d + mle.g)
            v /= np.linalg.norm(v)
            theta = _perturb(family, mle, v, r, u_std)
            d0, _ = quotient_alignment(family, X, mle, theta, u_std)
            new, _ = em_step(family, X, ev, theta)
            d1, _ = quotient_alignment(family, X, mle, new, u_std)
            rows.append({"radius": r, "ratio": float(np.sqrt(d1 / max(d0, 1e-300)))})
    df = pd.DataFrame(rows)
    return df.groupby("radius")["ratio"].agg(["median", "max", "mean"]).reset_index()
