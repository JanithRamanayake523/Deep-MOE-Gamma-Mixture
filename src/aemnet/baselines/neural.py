"""Neural estimators behind the v2 baseline interface.

mode="amortized" : one forward pass (AEM-Net, NBE-DeepSets, Set Transformer, NPE...)
mode="hybrid"    : amortised start + EM, ceil(log2 n) steps by default (Thm 4)
mode="learned"   : extra weight-tied AEM iterations (likelihood-free refinement)
"""
from __future__ import annotations

from ..inference.estimator import AmortizedEstimator
from .generic import GenericBaseline, GenericFit


class NeuralBaseline(GenericBaseline):
    supports_unknown_g = True

    def __init__(self, estimator: AmortizedEstimator, name: str = "aem", mode: str = "amortized",
                 extra_T: int = 4, max_em_steps: int | None = None):
        super().__init__(estimator.family)
        if mode not in ("amortized", "hybrid", "learned"):
            raise ValueError(f"unknown mode '{mode}'")
        self.est, self.mode, self.extra_T, self.max_em_steps = estimator, mode, extra_T, max_em_steps
        self.name = name if mode == "amortized" else f"{name}+{mode}"

    @classmethod
    def from_checkpoint(cls, path, device: str = "cpu", **kw) -> "NeuralBaseline":
        return cls(AmortizedEstimator.from_checkpoint(path, device), **kw)

    def _fit(self, X, events, g, rng):
        if self.mode == "hybrid":
            mp, info = self.est.hybrid(X, events, g, max_steps=self.max_em_steps)
            return GenericFit(mp, self.family.responsibilities_np(X, events, mp), info["loglik_trace"][-1],
                              extra={"em_steps": info["steps"]})
        T = self.est.model.cfg.T + self.extra_T if self.mode == "learned" else None
        mp, resp = self.est.estimate(X, events, g, T=T)
        return GenericFit(mp, resp)
