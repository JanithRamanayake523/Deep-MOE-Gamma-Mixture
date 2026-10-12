"""Metrics. All per-parameter quantities are computed AFTER Hungarian component
alignment so label switching cannot contaminate the reporting
(RESEARCH_METHODOLOGY.md 6.4)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..losses.matching import hungarian_match
from ..params import Theta
from ..simulation.mixture import logpdf as mixture_logpdf, sample


def aligned_errors(true: Theta, est: Theta) -> dict[str, np.ndarray]:
    """Signed estimation errors per parameter after optimal alignment.

    Returns dict with keys "pi" (g,), "alpha" (g, k), "beta" (g, k).
    Requires est.g == true.g (the replication studies satisfy this; model-order
    errors are tracked separately via g accuracy).
    """
    perm = hungarian_match(true, est)
    al = est.permuted(perm)
    return {"pi": al.pi - true.pi, "alpha": al.alpha - true.alpha, "beta": al.beta - true.beta,
            "perm": perm}


def summarize_errors(error_rows: list[dict], true: Theta, method: str) -> pd.DataFrame:
    """AVG / Bias / SD / MSE per parameter, matching the base paper's Tables 3-4.

    error_rows: list of aligned_errors() outputs plus the estimate itself is
    reconstructed as true + error. Returns a long-format DataFrame with columns
    [method, param, component, marginal, true, avg, bias, sd, mse].
    """
    rows = []
    E_alpha = np.stack([r["alpha"] for r in error_rows])  # (R, g, k)
    E_beta = np.stack([r["beta"] for r in error_rows])
    E_pi = np.stack([r["pi"] for r in error_rows])        # (R, g)
    spec = [("alpha", true.alpha, E_alpha), ("beta", true.beta, E_beta)]
    for name, tv, E in spec:
        for i in range(true.g):
            for p in range(true.k):
                est = tv[i, p] + E[:, i, p]
                rows.append(dict(method=method, param=f"{name}{i+1}{p+1}", component=i + 1,
                                 marginal=p + 1, true=tv[i, p], avg=float(est.mean()),
                                 bias=float(E[:, i, p].mean()), sd=float(est.std(ddof=1)),
                                 mse=float((E[:, i, p] ** 2).mean())))
    for i in range(true.g):
        est = true.pi[i] + E_pi[:, i]
        rows.append(dict(method=method, param=f"pi{i+1}", component=i + 1, marginal=0,
                         true=true.pi[i], avg=float(est.mean()), bias=float(E_pi[:, i].mean()),
                         sd=float(est.std(ddof=1)), mse=float((E_pi[:, i] ** 2).mean())))
    return pd.DataFrame(rows)


def misclassification_rate(true_labels: np.ndarray, resp: np.ndarray, perm: np.ndarray) -> float:
    """MAP responsibility vs. true label, after alignment."""
    pred = resp[:, perm].argmax(axis=1)
    return float(np.mean(pred != true_labels))


def test_loglik(est: Theta, true: Theta, n: int = 10000,
                rng: np.random.Generator | int | None = None) -> float:
    """Mean log-likelihood of the estimate on an independent same-theta sample
    (label-free accuracy measure)."""
    X_test, _ = sample(true, n=n, rng=rng)
    return float(mixture_logpdf(X_test, est).mean())


def interval_coverage(samples: list[Theta], true: Theta, nominal: float = 0.95) -> dict:
    """Empirical coverage of percentile intervals from a bootstrap/ensemble
    distribution of estimates (Study 4). Alignment applied per sample."""
    aligned = []
    for s in samples:
        if s.g != true.g:
            continue
        aligned.append(s.permuted(hungarian_match(true, s)))
    if not aligned:
        return {"nominal": nominal, "coverage": float("nan"), "n": 0}
    q_lo, q_hi = (1 - nominal) / 2 * 100, (1 + nominal) / 2 * 100
    A = np.stack([s.alpha for s in aligned])
    B = np.stack([s.beta for s in aligned])
    P = np.stack([s.pi for s in aligned])
    cov_alpha = (np.percentile(A, q_lo, 0) <= true.alpha) & (true.alpha <= np.percentile(A, q_hi, 0))
    cov_beta = (np.percentile(B, q_lo, 0) <= true.beta) & (true.beta <= np.percentile(B, q_hi, 0))
    cov_pi = (np.percentile(P, q_lo, 0) <= true.pi) & (true.pi <= np.percentile(P, q_hi, 0))
    return {"nominal": nominal,
            "coverage": float(np.concatenate([cov_alpha.ravel(), cov_beta.ravel(), cov_pi.ravel()]).mean()),
            "n": len(aligned)}
