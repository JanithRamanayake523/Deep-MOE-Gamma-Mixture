"""EM / ECM for any family (k-means++ starts, restarts, optional BIC order selection).

M-steps come from `family.mstep`: closed form (Gaussian), Newton (gamma --
Young et al. 2019 ECM; Weibull) or numerical weighted MLE (everything else).
This is the "EM/ECM with k-means++ and 10 restarts" baseline of E1/E2.
"""
from __future__ import annotations

import numpy as np

from ..inference.refine import em_refine
from ..simulation.families.base import Family
from .generic import GenericBaseline, GenericFit, init_from_labels, kmeanspp, n_free_params


class EMBaseline(GenericBaseline):
    name = "em"
    supports_unknown_g = True

    def __init__(self, family: Family, n_init: int = 10, max_iter: int = 500, tol: float = 1e-8, g_max: int = 6):
        super().__init__(family)
        self.n_init, self.max_iter, self.tol, self.g_max = n_init, max_iter, tol, g_max

    def _fit_known(self, X, events, g, rng) -> GenericFit:
        F = self.family.init_features_np(X)
        best = None
        total_steps = 0
        for _ in range(self.n_init):
            try:
                mp0 = init_from_labels(self.family, X, events, kmeanspp(F, g, rng), g)
                mp, info = em_refine(self.family, X, events, mp0, max_steps=self.max_iter, tol=self.tol)
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                continue
            ll = info["loglik_trace"][-1]
            total_steps += info["steps"]
            if best is None or ll > best.loglik:
                best = GenericFit(mp, None, ll, info["converged"], extra={"em_steps": info["steps"]})
        if best is None:
            raise RuntimeError("all EM restarts failed")
        best.resp = self.family.responsibilities_np(X, events, best.params)
        best.extra["em_steps_total"] = total_steps      # cost across all restarts (E2: AEM start vs k-means++)
        return best

    def _fit(self, X, events, g, rng):
        if g is not None:
            return self._fit_known(X, events, g, rng)
        n = X.shape[0]
        fits, bics = [], []
        for gg in range(1, self.g_max + 1):
            f = self._fit_known(X, events, gg, rng)
            fits.append(f)
            bics.append(-2 * f.loglik + n_free_params(self.family, gg) * np.log(n))
        i = int(np.argmin(bics))
        fits[i].extra["bic"] = bics
        return fits[i]
