"""Family-agnostic mixture parameter container (v2).

A mixture of g components from a family F is described by
    pi    : (g,)            mixing weights on the simplex
    comps : {name: (g, *shape)}  per-component parameters, constrained space,
                                 names/shapes declared by `Family.param_shapes()`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MixtureParams:
    pi: np.ndarray
    comps: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.pi = np.asarray(self.pi, dtype=np.float64).reshape(-1)
        self.comps = {k: np.asarray(v, dtype=np.float64) for k, v in self.comps.items()}
        for name, v in self.comps.items():
            if v.shape[0] != self.pi.shape[0]:
                raise ValueError(f"component parameter '{name}' has leading dim {v.shape[0]}, "
                                 f"expected g={self.pi.shape[0]}")
        if np.any(self.pi < -1e-12) or not np.isclose(self.pi.sum(), 1.0, atol=1e-5):
            raise ValueError(f"pi must lie on the simplex, got {self.pi}")

    @property
    def g(self) -> int:
        return int(self.pi.shape[0])

    def component(self, i: int) -> dict[str, np.ndarray]:
        return {k: v[i] for k, v in self.comps.items()}

    def permuted(self, perm) -> "MixtureParams":
        """new[i] = old[perm[i]]."""
        perm = np.asarray(perm)
        return MixtureParams(self.pi[perm], {k: v[perm] for k, v in self.comps.items()})

    def subset(self, idx) -> "MixtureParams":
        """Keep components `idx`, renormalising the weights."""
        idx = np.asarray(idx)
        pi = self.pi[idx]
        pi = pi / pi.sum()
        return MixtureParams(pi, {k: v[idx] for k, v in self.comps.items()})

    def copy(self) -> "MixtureParams":
        return MixtureParams(self.pi.copy(), {k: v.copy() for k, v in self.comps.items()})

    def __repr__(self) -> str:  # compact, readable
        body = ", ".join(f"{k}={np.array2string(v, precision=3)}" for k, v in self.comps.items())
        return f"MixtureParams(g={self.g}, pi={np.array2string(self.pi, precision=3)}, {body})"
