"""B2 -- EM/ECM for gamma mixtures with independent marginals.

E-step: standard responsibilities. M-step: pi = mean responsibility; each
(component, marginal) pair gets a weighted univariate gamma MLE solved by
Newton iteration on the digamma score equation (Young et al., 2019,
`mixtools::gammamixEM` generalized to independent multivariate marginals).
"""
from __future__ import annotations

import numpy as np
from scipy.special import digamma, polygamma

from ..params import Theta
from ..simulation.mixture import component_logpdf, logpdf as mixture_logpdf
from .base import Baseline, FitResult
from .wh_mclust import _kmeans

EPS = 1e-12


def weighted_gamma_mle(x: np.ndarray, w: np.ndarray, iters: int = 100, tol: float = 1e-10):
    """Weighted MLE for Gamma(shape a, scale b) via Newton on digamma score."""
    wsum = w.sum()
    if wsum < EPS:
        return 1.0, max(float(x.mean()), 1e-3)
    x = np.maximum(x, EPS)
    xbar = float((w * x).sum() / wsum)
    log_xbar = float((w * np.log(x)).sum() / wsum)
    s = np.log(max(xbar, EPS)) - log_xbar
    if s <= 0:  # degenerate (all x equal): huge shape fits
        return 1e4, xbar / 1e4
    # Minka's closed-form approximation as init
    a = (3.0 - s + np.sqrt((s - 3.0) ** 2 + 24.0 * s)) / (12.0 * s)
    for _ in range(iters):
        f = np.log(a) - digamma(a) - s
        fp = 1.0 / a - polygamma(1, a)
        step = f / fp
        a_new = a - step
        if a_new <= 0 or not np.isfinite(a_new):
            a_new = a / 2.0
        if abs(a_new - a) < tol * a:
            a = a_new
            break
        a = a_new
    return float(a), float(xbar / a)


class ECMBaseline(Baseline):
    name = "ecm"

    def __init__(self, n_init: int = 5, max_iter: int = 500, tol: float = 1e-8):
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol

    def _fit(self, X: np.ndarray, g: int | None, rng: np.random.Generator) -> FitResult:
        if g is None:
            raise ValueError("ECM baseline requires a known g (no built-in model selection)")
        n, k = X.shape
        best: FitResult | None = None
        for _ in range(self.n_init):
            assign = _kmeans(np.log(np.maximum(X, EPS)), g, rng)
            resp = np.eye(g)[assign] + 0.05 * rng.uniform(size=(n, g))
            resp /= resp.sum(1, keepdims=True)
            prev_ll = -np.inf
            converged = False
            for _ in range(self.max_iter):
                # M-step
                pi = resp.mean(0)
                alpha = np.empty((g, k))
                beta = np.empty((g, k))
                for i in range(g):
                    for p in range(k):
                        alpha[i, p], beta[i, p] = weighted_gamma_mle(X[:, p], resp[:, i])
                theta = Theta(pi, alpha, beta)
                # E-step
                lp = component_logpdf(X, theta) + np.log(np.maximum(pi, EPS))
                m = lp.max(1, keepdims=True)
                e = np.exp(lp - m)
                resp = e / e.sum(1, keepdims=True)
                ll = float((m[:, 0] + np.log(e.sum(1))).sum())
                if abs(ll - prev_ll) < self.tol * (1 + abs(ll)):
                    converged = True
                    break
                prev_ll = ll
            result = FitResult(theta=theta, resp=resp, loglik=float(mixture_logpdf(X, theta).sum()),
                               converged=converged)
            if best is None or result.loglik > best.loglik:
                best = result
        return best
