"""The base paper's pipeline (Vaidyanathan & Vani Lakshmi, 2017), re-implemented
dependency-free in Python; gamma family only. Kept as the historical comparator.

Wilson-Hilferty cube-root transform -> Gaussian mixture (EM + BIC model-order
selection, standing in for MCLUST) -> CI-based 2-D grid-search back-mapping to
(alpha, beta) per component/marginal (Vaidyanathan & Vani Lakshmi, 2017,
section 2.2; gamma_ci = 0.001 as published).

WH relations (Krishnamoorthy et al., 2008): if X ~ Gamma(a, b) (scale b), then
    X^(1/3) ~ approximately N(mu, sigma^2) with
    mu(a, b)    = (a b)^(1/3) * (1 - 1/(9 a))
    sigma2(a, b)= (a b)^(2/3) / (9 a)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ..mixture_params import MixtureParams
from ..simulation.families.base import Family
from .generic import GenericBaseline, GenericFit

EPS = 1e-12


def wh_mu(alpha, beta):
    return (alpha * beta) ** (1.0 / 3.0) * (1.0 - 1.0 / (9.0 * alpha))


def wh_sigma2(alpha, beta):
    return (alpha * beta) ** (2.0 / 3.0) / (9.0 * alpha)


def _kmeans(Y: np.ndarray, g: int, rng: np.random.Generator, iters: int = 50) -> np.ndarray:
    idx = rng.choice(Y.shape[0], size=g, replace=False)
    centers = Y[idx].copy()
    assign = np.zeros(Y.shape[0], dtype=int)
    for _ in range(iters):
        d = ((Y[:, None, :] - centers[None]) ** 2).sum(-1)
        new_assign = d.argmin(axis=1)
        if np.array_equal(new_assign, assign):
            break
        assign = new_assign
        for i in range(g):
            pts = Y[assign == i]
            if len(pts):
                centers[i] = pts.mean(axis=0)
    return assign


def _gaussian_em(
    Y: np.ndarray, g: int, rng: np.random.Generator,
    n_init: int = 5, max_iter: int = 500, tol: float = 1e-7, reg: float = 1e-6,
):
    """Full-covariance Gaussian-mixture EM (MCLUST stand-in). Returns the best
    (pi, mu, cov, resp, loglik) over n_init k-means-initialized runs."""
    n, k = Y.shape
    best = None
    for _ in range(n_init):
        assign = _kmeans(Y, g, rng)
        pi = np.bincount(assign, minlength=g) / n + EPS
        pi /= pi.sum()
        mu = np.stack([Y[assign == i].mean(0) if (assign == i).any() else Y[rng.integers(n)]
                       for i in range(g)])
        cov = np.stack([np.cov(Y[assign == i].T) + reg * np.eye(k) if (assign == i).sum() > k
                        else np.eye(k) for i in range(g)]).reshape(g, k, k)
        prev_ll = -np.inf
        for _ in range(max_iter):
            lp = np.stack([_mvn_logpdf(Y, mu[i], cov[i]) for i in range(g)], axis=1)
            lp += np.log(pi + EPS)
            m = lp.max(1, keepdims=True)
            e = np.exp(lp - m)
            resp = e / e.sum(1, keepdims=True)
            ll = float((m[:, 0] + np.log(e.sum(1))).sum())
            Nk = resp.sum(0) + EPS
            pi = Nk / n
            mu = (resp.T @ Y) / Nk[:, None]
            for i in range(g):
                d = Y - mu[i]
                cov[i] = (resp[:, i, None] * d).T @ d / Nk[i] + reg * np.eye(k)
            if abs(ll - prev_ll) < tol * (1 + abs(ll)):
                break
            prev_ll = ll
        if best is None or ll > best[4]:
            best = (pi, mu, cov, resp, ll)
    return best


def _mvn_logpdf(Y, mu, cov):
    k = Y.shape[1]
    sign, logdet = np.linalg.slogdet(cov)
    sol = np.linalg.solve(cov, (Y - mu).T).T
    return -0.5 * (k * np.log(2 * np.pi) + logdet + ((Y - mu) * sol).sum(1))


def _select_g_bic(Y: np.ndarray, g_max: int, rng: np.random.Generator) -> int:
    n, k = Y.shape
    best_g, best_bic = 1, np.inf
    for g in range(1, g_max + 1):
        _, _, _, _, ll = _gaussian_em(Y, g, rng)
        n_params = g * k + g * k * (k + 1) // 2 + (g - 1)
        bic = -2 * ll + n_params * np.log(n)
        if bic < best_bic:
            best_bic, best_g = bic, g
    return best_g


def _back_map_marginal(
    y: np.ndarray, n_i: float, gamma_ci: float = 0.001,
    alpha_grid: np.ndarray | None = None, beta_grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """CI-based 2-D grid search for one component/marginal (paper's steps 1-3).

    y : WH-transformed observations (MAP-assigned to this component).
    """
    if alpha_grid is None:
        alpha_grid = np.exp(np.linspace(np.log(0.05), np.log(60.0), 200))
    if beta_grid is None:
        beta_grid = np.exp(np.linspace(np.log(0.05), np.log(60.0), 200))
    mu_hat = float(y.mean())
    sd_hat = float(y.var(ddof=1) ** 0.5)
    z = norm.ppf(1.0 - gamma_ci / 2.0)
    half = z * sd_hat / max(np.sqrt(n_i), EPS)
    lo, hi = mu_hat - half, mu_hat + half

    A, B = np.meshgrid(alpha_grid, beta_grid, indexing="ij")
    MU = wh_mu(A, B)
    in_ci = (MU >= lo) & (MU <= hi)
    if not in_ci.any():  # fall back: unconstrained WH inversion at mu_hat
        in_ci = np.ones_like(in_ci, dtype=bool)
    S2 = wh_sigma2(A, B)
    # normal log-likelihood of the transformed marginal data under N(MU, S2)
    ll = -0.5 * (np.log(2 * np.pi * S2) + ((y[:, None, None] - MU) ** 2) / S2).sum(0)
    ll = np.where(in_ci, ll, -np.inf)
    i, j = np.unravel_index(np.argmax(ll), ll.shape)
    return float(alpha_grid[i]), float(beta_grid[j])


class WHMclustBaseline(GenericBaseline):
    name = "wh_mclust"
    supports_unknown_g = True

    def __init__(self, family: Family, gamma_ci: float = 0.001, g_max: int = 6):
        if family.name != "gamma":
            raise ValueError("WH-MCLUST is defined for the (independent-marginal) gamma family only")
        super().__init__(family)
        self.gamma_ci = gamma_ci
        self.g_max = g_max

    def _fit(self, X: np.ndarray, events: np.ndarray, g: int | None, rng: np.random.Generator) -> GenericFit:
        Y = np.cbrt(np.maximum(X, EPS))
        if g is None:
            g = _select_g_bic(Y, self.g_max, rng)
        pi, mu, cov, resp, _ = _gaussian_em(Y, g, rng)
        assign = resp.argmax(1)
        n_i = resp.sum(0)
        alpha = np.empty((g, X.shape[1]))
        beta = np.empty((g, X.shape[1]))
        for i in range(g):
            Yi = Y[assign == i]
            if Yi.shape[0] < 3:
                Yi = Y  # degenerate component: fall back to pooled data
            for p in range(X.shape[1]):
                alpha[i, p], beta[i, p] = _back_map_marginal(Yi[:, p], n_i[i], self.gamma_ci)
        mp = MixtureParams(pi / pi.sum(), {"alpha": alpha, "beta": beta})
        return GenericFit(mp, resp, converged=True, extra={"g_selected": g})
