"""Family-agnostic baseline interface (v2) and shared helpers.

Every estimator -- classical, sample-based or neural -- maps (X, events, g) to a
`GenericFit` holding a `MixtureParams` in the original data scale. `g=None` asks
the method to choose the number of components itself (if it can).
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..mixture_params import MixtureParams
from ..simulation.families.base import Family


@dataclass
class GenericFit:
    params: MixtureParams
    resp: np.ndarray | None = None
    loglik: float = float("nan")
    converged: bool = True
    runtime: float = 0.0
    extra: dict = field(default_factory=dict)


class GenericBaseline(ABC):
    name: str = "abstract"
    supports_unknown_g: bool = False

    def __init__(self, family: Family):
        self.family = family

    @abstractmethod
    def _fit(self, X: np.ndarray, events: np.ndarray, g: int | None, rng: np.random.Generator) -> GenericFit:
        ...

    def fit(self, X, events=None, g: int | None = None, seed: int = 0) -> GenericFit:
        X = np.asarray(X, dtype=np.float64)
        X = X[:, None] if X.ndim == 1 else X
        events = np.ones_like(X) if events is None else np.asarray(events, dtype=np.float64)
        if g is None and not self.supports_unknown_g:
            raise ValueError(f"{self.name} requires a known g")
        t0 = time.perf_counter()
        res = self._fit(X, events, g, np.random.default_rng(seed))
        res.runtime = time.perf_counter() - t0
        if not np.isfinite(res.loglik):
            res.loglik = float(self.family.mixture_logpdf_np(X, events, res.params).sum())
        return res


def kmeanspp(F: np.ndarray, g: int, rng: np.random.Generator, iters: int = 30) -> np.ndarray:
    """k-means++ seeding + Lloyd iterations on standardised features. Returns labels (n,)."""
    F = (F - F.mean(0)) / np.maximum(F.std(0), 1e-12)
    n = F.shape[0]
    centers = [F[rng.integers(n)]]
    for _ in range(1, g):
        d2 = np.min(((F[:, None, :] - np.asarray(centers)[None]) ** 2).sum(-1), axis=1)
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
        centers.append(F[rng.choice(n, p=p)])
    C = np.asarray(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        labels_new = ((F[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        if np.array_equal(labels_new, labels) and _ > 0:
            break
        labels = labels_new
        for j in range(g):
            if np.any(labels == j):
                C[j] = F[labels == j].mean(0)
            else:
                C[j] = F[rng.integers(n)]
    return labels


def init_from_labels(family: Family, X, events, labels, g: int, smoothing: float = 0.1) -> MixtureParams:
    """Soft one-hot responsibilities from hard labels -> weighted M-step per component."""
    R = np.full((X.shape[0], g), smoothing / g)
    R[np.arange(X.shape[0]), labels] += 1.0 - smoothing
    pi = R.mean(0)
    comps = [family.mstep(X, events, R[:, i]) for i in range(g)]
    return MixtureParams(pi / pi.sum(), {k: np.stack([c[k] for c in comps]) for k in comps[0]})


def n_free_params(family: Family, g: int) -> int:
    return g * family.d + g - 1

