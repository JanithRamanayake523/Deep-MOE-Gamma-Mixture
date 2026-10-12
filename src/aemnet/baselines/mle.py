"""Direct numerical MLE for any family: L-BFGS-B on the full observed-data
log-likelihood in v = [alr(pi), u_1, ..., u_g], restarted from k-means++ starts."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..mixture_params import MixtureParams
from ..simulation.families.base import Family
from .generic import GenericBaseline, GenericFit, init_from_labels, kmeanspp


def pack_v(family: Family, mp: MixtureParams) -> np.ndarray:
    alr = np.log(np.maximum(mp.pi[:-1], 1e-300) / max(mp.pi[-1], 1e-300))
    return np.concatenate([alr, family.encode_np(mp.comps).ravel()])


def unpack_v(family: Family, v: np.ndarray, g: int) -> MixtureParams:
    a = np.concatenate([v[: g - 1], [0.0]])
    pi = np.exp(a - a.max())
    return MixtureParams(pi / pi.sum(), family.decode_np(v[g - 1:].reshape(g, family.d)))


class DirectMLEBaseline(GenericBaseline):
    name = "mle"

    def __init__(self, family: Family, n_restarts: int = 10, max_iter: int = 500):
        super().__init__(family)
        self.n_restarts, self.max_iter = n_restarts, max_iter

    def _fit(self, X, events, g, rng):
        fam = self.family
        lo, hi = fam.u_bounds()
        bounds = [(-15.0, 15.0)] * (g - 1) + [(lo, hi)] * (g * fam.d)

        def negll(v):
            val = -float(fam.mixture_logpdf_np(X, events, unpack_v(fam, v, g)).sum())
            return val if np.isfinite(val) else 1e300

        F = fam.init_features_np(X)
        best = None
        for _ in range(self.n_restarts):
            try:
                v0 = np.clip(pack_v(fam, init_from_labels(fam, X, events, kmeanspp(F, g, rng), g)),
                             [b[0] for b in bounds], [b[1] for b in bounds])
                res = minimize(negll, v0, method="L-BFGS-B", bounds=bounds, options={"maxiter": self.max_iter})
            except (np.linalg.LinAlgError, ValueError):
                continue
            if best is None or res.fun < best.fun:
                best = res
        if best is None:
            raise RuntimeError("all MLE restarts failed")
        mp = unpack_v(fam, best.x, g)
        return GenericFit(mp, fam.responsibilities_np(X, events, mp), -float(best.fun), bool(best.success),
                          extra={"nit": int(best.nit)})
