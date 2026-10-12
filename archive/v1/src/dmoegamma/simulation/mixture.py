"""Multivariate gamma mixture with independent marginals: sampler and density.

Model (base paper, eq. (2)):
    f(z; theta) = sum_i pi_i * prod_p Gam(z_p; alpha_ip, beta_ip)
with shape alpha > 0 and scale beta > 0 (mean = alpha * beta).
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from ..params import Theta

EPS = 1e-12


def sample(
    theta: Theta,
    n: int | None = None,
    counts: list[int] | np.ndarray | None = None,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate from the mixture.

    Parameters
    ----------
    theta : Theta
    n : total sample size; labels drawn i.i.d. from pi. Ignored if `counts` given.
    counts : exact per-component sample sizes (paper's replication design: the
        base paper simulates exactly n_i observations from component i).
    rng : numpy Generator or seed.

    Returns
    -------
    X : (n_total, k) observations
    labels : (n_total,) true component indices in {0, ..., g-1}
    """
    rng = np.random.default_rng(rng)
    if counts is not None:
        counts = np.asarray(counts, dtype=int)
        if counts.shape[0] != theta.g:
            raise ValueError("counts must have one entry per component")
        labels = np.repeat(np.arange(theta.g), counts)
    else:
        if n is None:
            raise ValueError("either n or counts must be given")
        labels = rng.choice(theta.g, size=n, p=theta.pi)
    n_total = labels.shape[0]
    X = np.empty((n_total, theta.k), dtype=np.float64)
    for i in range(theta.g):
        idx = labels == i
        m = int(idx.sum())
        if m == 0:
            continue
        for p in range(theta.k):
            X[idx, p] = rng.gamma(shape=theta.alpha[i, p], scale=theta.beta[i, p], size=m)
    return X, labels


def component_logpdf(X: np.ndarray, theta: Theta) -> np.ndarray:
    """Log of each component's joint density (independent marginals).

    Returns (n, g) with entry [j, i] = log f_i(z_j; alpha_i, beta_i).
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    logX = np.log(np.maximum(X, EPS))
    out = np.empty((n, theta.g), dtype=np.float64)
    for i in range(theta.g):
        a, b = theta.alpha[i], theta.beta[i]  # (k,)
        lp = (a - 1.0) * logX - X / b - gammaln(a) - a * np.log(b)
        out[:, i] = lp.sum(axis=1)
    return out


def logpdf(X: np.ndarray, theta: Theta) -> np.ndarray:
    """Mixture log-density per observation, shape (n,)."""
    lp = component_logpdf(X, theta) + np.log(np.maximum(theta.pi, EPS))
    m = lp.max(axis=1, keepdims=True)
    return (m[:, 0] + np.log(np.exp(lp - m).sum(axis=1)))


def responsibilities(X: np.ndarray, theta: Theta) -> np.ndarray:
    """Posterior component probabilities r_{ji}, shape (n, g)."""
    lp = component_logpdf(X, theta) + np.log(np.maximum(theta.pi, EPS))
    m = lp.max(axis=1, keepdims=True)
    e = np.exp(lp - m)
    return e / e.sum(axis=1, keepdims=True)


def bayes_error(
    theta: Theta,
    n_mc: int = 20000,
    rng: np.random.Generator | int | None = None,
) -> float:
    """Monte Carlo estimate of the Bayes (optimal) misclassification rate.

    Used to quantify component overlap for stratified training and for the
    factorial study's overlap cells.
    """
    X, labels = sample(theta, n=n_mc, rng=rng)
    pred = responsibilities(X, theta).argmax(axis=1)
    return float(np.mean(pred != labels))


# ---------------------------------------------------------------------------
# Torch versions (padded batches), used by the composite loss.
# ---------------------------------------------------------------------------

def component_logpdf_torch(X, alpha, beta):
    """Per-component log-density on a padded batch.

    X : (B, n, k); alpha, beta : (B, g, k).
    Returns (B, n, g).
    """
    import torch

    Xc = X.clamp_min(EPS)
    logX = Xc.log()
    a = alpha.unsqueeze(1)  # (B, 1, g, k)
    b = beta.unsqueeze(1)
    Xe = Xc.unsqueeze(2)    # (B, n, 1, k)
    lp = (a - 1.0) * logX.unsqueeze(2) - Xe / b - torch.lgamma(a) - a * b.log()
    return lp.sum(dim=-1)   # (B, n, g)


def mixture_logpdf_torch(X, pi, alpha, beta, comp_mask):
    """Mixture log-density per observation on a padded batch.

    X : (B, n, k); pi : (B, g); alpha, beta : (B, g, k); comp_mask : (B, g).
    Returns (B, n). Padded components contribute -inf.
    """
    import torch

    lp = component_logpdf_torch(X, alpha, beta)
    lp = lp + pi.clamp_min(EPS).log().unsqueeze(1)  # (B, 1, g)
    lp = lp.masked_fill(~comp_mask.unsqueeze(1), float("-inf"))
    return torch.logsumexp(lp, dim=-1)              # (B, n)
