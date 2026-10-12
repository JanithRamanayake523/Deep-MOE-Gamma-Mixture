"""Family-agnostic metrics (RESEARCH_METHODOLOGY.md 7, E1).

All parameter errors are computed after Hungarian alignment in the *same*
standardised canonical coordinates the network is trained in:
    z = (encode(canonicalize(theta, loc(D), scale(D))) - 0) / u_std(prior)
so a method's quotient risk is comparable across data scales and families.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..mixture_params import MixtureParams
from ..simulation.families.base import Family
from ..simulation.families.stats import prior_u_stats


def canonical_z(family: Family, mp: MixtureParams, loc, scale, u_std) -> np.ndarray:
    return family.encode_np(family.canonicalize_np(mp, loc, scale).comps) / u_std


def quotient_alignment(family: Family, X, true: MixtureParams, est: MixtureParams, u_std=None,
                       w_pi: float = 1.0):
    """Returns (quotient_risk, perm) where perm[i] = est component matched to true i.
    If est has more components than truth, unmatched est weights are penalised
    (pi^2, as in training); if fewer, the risk is undefined (nan)."""
    u_std = prior_u_stats(family)[1] if u_std is None else u_std
    if est.g < true.g:
        return float("nan"), None
    loc, scale = family.canonical_loc_scale_np(X)
    zt, ze = canonical_z(family, true, loc, scale, u_std), canonical_z(family, est, loc, scale, u_std)
    C = ((zt[:, None] - ze[None]) ** 2).mean(-1) + w_pi * (true.pi[:, None] - est.pi[None]) ** 2
    rows, cols = linear_sum_assignment(C)
    risk = float(C[rows, cols].mean())
    unmatched = np.setdiff1d(np.arange(est.g), cols)
    risk += float((est.pi[unmatched] ** 2).sum())
    return risk, cols


def aligned_param_errors(family: Family, true: MixtureParams, est: MixtureParams, perm) -> dict:
    """Signed errors per named parameter (constrained scale) and per z-coordinate."""
    al = est.permuted(perm)
    out = {f"err_{k}": al.comps[k] - true.comps[k] for k in true.comps}
    out["err_pi"] = al.pi - true.pi
    return out


def rmse_by_param(family: Family, true: MixtureParams, est: MixtureParams, perm) -> dict:
    """RMSE of log-parameters for positive params, raw otherwise (per named parameter)."""
    al = est.permuted(perm)
    rows = {}
    for k in true.comps:
        t, e = true.comps[k], al.comps[k]
        if np.all(t > 0) and np.all(e > 0) and k not in ("Lc", "L"):
            rows[f"rmse_log_{k}"] = float(np.sqrt(np.mean((np.log(e) - np.log(t)) ** 2)))
        else:
            rows[f"rmse_{k}"] = float(np.sqrt(np.mean((e - t) ** 2)))
    rows["mae_pi"] = float(np.mean(np.abs(al.pi - true.pi)))
    return rows


def heldout_loglik(family: Family, est: MixtureParams, true: MixtureParams, n: int = 5000, rng=None) -> float:
    rng = np.random.default_rng(rng)
    X, ev, _ = family.sample_dataset(rng, true, n)
    return float(np.mean(family.mixture_logpdf_np(X, ev, est)))


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    C = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(C, (ai, bi), 1)
    comb = lambda x: x * (x - 1) / 2.0  # noqa: E731
    sum_c = comb(C).sum()
    sa, sb = comb(C.sum(1)).sum(), comb(C.sum(0)).sum()
    n = comb(len(a))
    expected = sa * sb / n if n > 0 else 0.0
    denom = 0.5 * (sa + sb) - expected
    return float((sum_c - expected) / denom) if denom != 0 else 1.0


def evaluate_fit(family: Family, X, events, labels, true: MixtureParams, est: MixtureParams,
                 resp=None, u_std=None, n_test: int = 5000, rng=None) -> dict:
    row = {"g_true": true.g, "g_est": est.g, "g_correct": int(est.g == true.g)}
    risk, perm = quotient_alignment(family, X, true, est, u_std)
    row["quotient_risk"] = risk
    if perm is not None and est.g == true.g:
        row.update(rmse_by_param(family, true, est, perm))
    if resp is None:  # every family provides a numpy density
        resp = family.responsibilities_np(X, events, est)
    if resp is not None and labels is not None:
        pred = resp.argmax(1)
        row["ari"] = adjusted_rand_index(labels, pred)
        if perm is not None:
            inv = {int(c): i for i, c in enumerate(perm)}
            mapped = np.array([inv.get(int(p), -1) for p in pred])
            row["misclassification"] = float(np.mean(mapped != labels))
    if n_test > 0:
        row["heldout_loglik"] = heldout_loglik(family, est, true, n_test, rng)
        row["heldout_loglik_true"] = heldout_loglik(family, true, true, n_test, rng)
    return row
