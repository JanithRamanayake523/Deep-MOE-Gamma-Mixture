"""Posterior calibration on the quotient space (E5; Prop 2).

Samples and the truth are both aligned to the same pivot (the point estimate),
then per-coordinate summaries are computed in standardised canonical z-units
plus the mixing weights. Implements coverage of central credible intervals,
simulation-based calibration ranks (Talts et al., 2018), expected-coverage
curves (Hermans et al., 2022) and a classifier two-sample test (C2ST).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import chisquare

from ..mixture_params import MixtureParams
from ..simulation.families.base import Family
from ..simulation.families.stats import prior_u_stats
from .relabel import align_to_pivot


def aligned_matrix(family: Family, mps: list[MixtureParams], pivot: MixtureParams, X, u_std=None) -> np.ndarray:
    """(M, g*(d+1)) canonical z-coordinates + weights of each sample aligned to pivot."""
    u_std = prior_u_stats(family)[1] if u_std is None else u_std
    loc, scale = family.canonical_loc_scale_np(X)
    aligned, _ = align_to_pivot(family, mps, pivot, u_std)
    rows = []
    for s in aligned:
        z = family.encode_np(family.canonicalize_np(s, loc, scale).comps) / u_std
        rows.append(np.concatenate([z.ravel(), s.pi]))
    return np.asarray(rows)


def coverage(family: Family, X, true: MixtureParams, samples: list[MixtureParams], pivot: MixtureParams,
             levels=(0.5, 0.9, 0.95), u_std=None) -> dict:
    S = aligned_matrix(family, samples, pivot, X, u_std)
    t = aligned_matrix(family, [true], pivot, X, u_std)[0]
    out = {}
    for lv in levels:
        lo, hi = np.quantile(S, (1 - lv) / 2, 0), np.quantile(S, (1 + lv) / 2, 0)
        out[f"cov_{lv}"] = float(np.mean((lo <= t) & (t <= hi)))
    return out


def sbc_ranks(family: Family, X, true: MixtureParams, samples: list[MixtureParams], pivot: MixtureParams,
              u_std=None) -> np.ndarray:
    S = aligned_matrix(family, samples, pivot, X, u_std)
    t = aligned_matrix(family, [true], pivot, X, u_std)[0]
    return (S < t).sum(0)


def sbc_uniformity(ranks: np.ndarray, n_samples: int, n_bins: int = 20) -> np.ndarray:
    """Chi-square p-value per coordinate for rank uniformity. ranks (R, D)."""
    edges = np.linspace(0, n_samples + 1, n_bins + 1)
    return np.array([chisquare(np.histogram(ranks[:, j], edges)[0]).pvalue for j in range(ranks.shape[1])])


def expected_coverage_curve(cases: list[tuple], family: Family, levels=np.linspace(0.05, 0.95, 19), u_std=None):
    """cases: list of (X, true, samples, pivot). Returns (levels, empirical coverage)."""
    emp = np.zeros(len(levels))
    for X, true, samples, pivot in cases:
        S = aligned_matrix(family, samples, pivot, X, u_std)
        t = aligned_matrix(family, [true], pivot, X, u_std)[0]
        for i, lv in enumerate(levels):
            lo, hi = np.quantile(S, (1 - lv) / 2, 0), np.quantile(S, (1 + lv) / 2, 0)
            emp[i] += np.mean((lo <= t) & (t <= hi))
    return np.asarray(levels), emp / max(len(cases), 1)


def c2st(A: np.ndarray, B: np.ndarray, n_folds: int = 5, epochs: int = 300, hidden: int = 64, seed: int = 0) -> float:
    """Classifier two-sample test accuracy (0.5 = indistinguishable)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    Xall = np.concatenate([A, B]).astype(np.float32)
    mu, sd = Xall.mean(0), Xall.std(0) + 1e-8
    Xall = (Xall - mu) / sd
    y = np.concatenate([np.zeros(len(A)), np.ones(len(B))]).astype(np.float32)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, n_folds)
    accs = []
    for f in range(n_folds):
        te = folds[f]
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != f])
        net = torch.nn.Sequential(torch.nn.Linear(Xall.shape[1], hidden), torch.nn.ReLU(),
                                  torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        Xt, yt = torch.from_numpy(Xall[tr]), torch.from_numpy(y[tr])
        for _ in range(epochs):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt).squeeze(-1), yt)
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = (net(torch.from_numpy(Xall[te])).squeeze(-1) > 0).numpy()
        accs.append(np.mean(pred == y[te]))
    return float(np.mean(accs))
