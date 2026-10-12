"""Prior statistics of a family's unconstrained coordinates (cached).

Used to standardise evaluation metrics (quotient risk in z-units) and as the
weakly-informative prior scale of the MCMC / ABC baselines, so every method is
scored in the same coordinates the network is trained in."""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from .base import Family


@lru_cache(maxsize=64)
def _prior_u_stats(spec_json: str, n: int, seed: int):
    from . import family_from_spec

    fam = family_from_spec(json.loads(spec_json))
    rng = np.random.default_rng(seed)
    U = fam.encode_np(fam.sample_prior(rng, n))
    U = U[np.all(np.isfinite(U), 1)]
    return U.mean(0), np.maximum(U.std(0), 1e-3)


def prior_u_stats(family: Family, n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    mean, std = _prior_u_stats(json.dumps(family.spec(), sort_keys=True, default=list), n, seed)
    return mean.copy(), std.copy()
