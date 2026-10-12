"""Parameter containers and (un)constrained packing for gamma mixture models.

A theta consists of
    pi    : (g,)    mixing proportions on the simplex
    alpha : (g, k)  component-wise shapes, one per marginal
    beta  : (g, k)  component-wise scales (mean of component i, marginal p is
                    alpha[i, p] * beta[i, p]; variance is alpha[i, p] * beta[i, p]^2)

Free dimension: 2*g*k + g - 1  (Vaidyanathan & Vani Lakshmi, 2017, eq. (2)).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Theta:
    """A single multivariate gamma-mixture parameter set (numpy)."""

    pi: np.ndarray    # (g,)
    alpha: np.ndarray # (g, k)
    beta: np.ndarray  # (g, k)

    def __post_init__(self) -> None:
        self.pi = np.asarray(self.pi, dtype=np.float64).reshape(-1)
        self.alpha = np.asarray(self.alpha, dtype=np.float64)
        self.beta = np.asarray(self.beta, dtype=np.float64)
        g = self.pi.shape[0]
        if self.alpha.shape != (g, self.alpha.shape[1]) or self.alpha.ndim != 2:
            raise ValueError("alpha must have shape (g, k)")
        if self.beta.shape != self.alpha.shape:
            raise ValueError("beta must have the same shape as alpha")
        if np.any(self.alpha <= 0) or np.any(self.beta <= 0):
            raise ValueError("alpha and beta must be strictly positive")
        if np.any(self.pi < 0) or not np.isclose(self.pi.sum(), 1.0, atol=1e-6):
            raise ValueError("pi must lie on the simplex")

    @property
    def g(self) -> int:
        return int(self.pi.shape[0])

    @property
    def k(self) -> int:
        return int(self.alpha.shape[1])

    @property
    def dim(self) -> int:
        """Number of free parameters: 2*g*k + g - 1."""
        return 2 * self.g * self.k + self.g - 1

    def means(self) -> np.ndarray:
        """Component means, shape (g, k)."""
        return self.alpha * self.beta

    def variances(self) -> np.ndarray:
        """Component variances, shape (g, k)."""
        return self.alpha * self.beta**2

    def sorted_by_mean(self) -> "Theta":
        """Return a copy with components ordered by first-marginal mean.

        Useful as a canonical representative of the label-equivalence class
        when reporting point estimates (does NOT replace Hungarian alignment
        in evaluation).
        """
        order = np.argsort(self.means()[:, 0])
        return Theta(self.pi[order], self.alpha[order], self.beta[order])

    def permuted(self, perm: np.ndarray) -> "Theta":
        """Return a copy with components reordered: new[i] = old[perm[i]]."""
        perm = np.asarray(perm)
        return Theta(self.pi[perm], self.alpha[perm], self.beta[perm])


def pack(theta: Theta) -> np.ndarray:
    """Pack to a constrained vector of length 2*g*k + g - 1.

    Layout: [pi_1 .. pi_{g-1}, log(alpha).ravel(), log(beta).ravel()].
    Logs keep shapes/scales on an unconstrained-ish, scale-balanced footing.
    """
    return np.concatenate(
        [theta.pi[:-1], np.log(theta.alpha).ravel(), np.log(theta.beta).ravel()]
    )


def unpack(vec: np.ndarray, g: int, k: int) -> Theta:
    """Inverse of `pack`."""
    vec = np.asarray(vec, dtype=np.float64)
    expected = 2 * g * k + g - 1
    if vec.size != expected:
        raise ValueError(f"expected vector of length {expected}, got {vec.size}")
    pi = np.concatenate([vec[: g - 1], [1.0 - vec[: g - 1].sum()]])
    alpha = np.exp(vec[g - 1 : g - 1 + g * k]).reshape(g, k)
    beta = np.exp(vec[g - 1 + g * k :]).reshape(g, k)
    return Theta(pi, alpha, beta)


def pack_unconstrained(theta: Theta) -> np.ndarray:
    """Pack to a fully unconstrained vector (for scipy optimizers / MCMC).

    Layout: [logit_pi_1 .. logit_pi_{g-1}, log(alpha).ravel(), log(beta).ravel()]
    where logit_pi uses the additive-log-ratio transform against component g.
    """
    alr = np.log(theta.pi[:-1] / theta.pi[-1])
    return np.concatenate([alr, np.log(theta.alpha).ravel(), np.log(theta.beta).ravel()])


def unpack_unconstrained(vec: np.ndarray, g: int, k: int) -> Theta:
    """Inverse of `pack_unconstrained` (softmax alr, exp logs)."""
    vec = np.asarray(vec, dtype=np.float64)
    alr = np.concatenate([vec[: g - 1], [0.0]])
    alr = alr - alr.max()
    pi = np.exp(alr)
    pi = pi / pi.sum()
    alpha = np.exp(vec[g - 1 : g - 1 + g * k]).reshape(g, k)
    beta = np.exp(vec[g - 1 + g * k :]).reshape(g, k)
    return Theta(pi, alpha, beta)


def padded_batch(thetas: list[Theta], g_max: int, device=None) -> dict:
    """Pad a list of Theta to a torch batch over g_max component slots.

    Returns a dict with
        pi        : (B, g_max)      padded with 0
        alpha     : (B, g_max, k)   padded with 1 (harmless positive filler)
        beta      : (B, g_max, k)   padded with 1
        comp_mask : (B, g_max)      True for real components
        g         : (B,)            true number of components
    """
    import torch

    B = len(thetas)
    k = thetas[0].k
    pi = torch.zeros(B, g_max, dtype=torch.float32)
    alpha = torch.ones(B, g_max, k, dtype=torch.float32)
    beta = torch.ones(B, g_max, k, dtype=torch.float32)
    comp_mask = torch.zeros(B, g_max, dtype=torch.bool)
    g = torch.zeros(B, dtype=torch.long)
    for b, th in enumerate(thetas):
        pi[b, : th.g] = torch.from_numpy(th.pi).float()
        alpha[b, : th.g] = torch.from_numpy(th.alpha).float()
        beta[b, : th.g] = torch.from_numpy(th.beta).float()
        comp_mask[b, : th.g] = True
        g[b] = th.g
    out = {"pi": pi, "alpha": alpha, "beta": beta, "comp_mask": comp_mask, "g": g}
    if device is not None:
        out = {k_: v.to(device) for k_, v in out.items()}
    return out
