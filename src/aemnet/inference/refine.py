"""Refinement from an amortised initialisation (C3, Thm 4; RESEARCH_METHODOLOGY.md 5).

    em_refine        exact EM / ECM steps using `family.mstep` (tractable families).
                     Thm 4: K_n = O(log n) steps from a consistent amortised start
                     give an asymptotically efficient estimator.
    gradient_refine  Adam on the observed-data log-likelihood in (alr pi, u) space
                     (tractable torch likelihoods; useful when the M-step is costly).
    default_em_steps ceil(log2 n) -- the Thm 4 schedule.
Learned refinement (extra weight-tied AEM iterations) lives in
`AmortizedEstimator.estimate(..., T=...)`.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..mixture_params import MixtureParams
from ..simulation.families.base import Family


def default_em_steps(n: int) -> int:
    return max(1, int(math.ceil(math.log2(max(n, 2)))))


def em_step(family: Family, X, events, mp: MixtureParams, min_weight: float = 1e-8):
    r = family.responsibilities_np(X, events, mp)
    pi = np.maximum(r.mean(0), min_weight)
    pi = pi / pi.sum()
    comps = [family.mstep(X, events, r[:, i], init=mp.component(i)) for i in range(mp.g)]
    stacked = {name: np.stack([c[name] for c in comps]) for name in comps[0]}
    return MixtureParams(pi, stacked), r


def em_refine(family: Family, X: np.ndarray, events: np.ndarray | None, mp: MixtureParams,
              max_steps: int | None = None, tol: float = 1e-8) -> tuple[MixtureParams, dict]:
    """Run EM from `mp`. Returns (params, info) with the log-likelihood trace.
    Guarantees monotone non-decreasing log-likelihood by rejecting a step that
    decreases it (can happen with inexact numerical M-steps)."""
    X = np.asarray(X, dtype=np.float64)
    events = np.ones_like(X) if events is None else np.asarray(events, dtype=np.float64)
    max_steps = default_em_steps(X.shape[0]) if max_steps is None else max_steps
    ll = float(family.mixture_logpdf_np(X, events, mp).sum())
    trace = [ll]
    converged = False
    for _ in range(max_steps):
        try:
            new, _ = em_step(family, X, events, mp)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            break
        ll_new = float(family.mixture_logpdf_np(X, events, new).sum())
        if not np.isfinite(ll_new) or ll_new < ll - 1e-9 * (1 + abs(ll)):
            break
        mp = new
        trace.append(ll_new)
        if abs(ll_new - ll) < tol * (1 + abs(ll)):
            converged = True
            ll = ll_new
            break
        ll = ll_new
    return mp, {"loglik_trace": trace, "steps": len(trace) - 1, "converged": converged}


def gradient_refine(family: Family, X: np.ndarray, events: np.ndarray | None, mp: MixtureParams,
                    steps: int = 200, lr: float = 0.01) -> tuple[MixtureParams, dict]:
    if not family.has_torch_logprob:
        raise ValueError(f"{family.name}: no torch likelihood for gradient refinement")
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float64))[None]
    Et = torch.ones_like(Xt) if events is None else torch.as_tensor(np.asarray(events, dtype=np.float64))[None]
    u = torch.as_tensor(family.encode_np(mp.comps))[None].clone().requires_grad_(True)
    logit = torch.as_tensor(np.log(np.maximum(mp.pi, 1e-12)))[None].clone().requires_grad_(True)
    opt = torch.optim.Adam([u, logit], lr=lr)
    trace = []
    for _ in range(steps):
        lp = family.log_prob(Xt, Et, family.decode(u)) + torch.log_softmax(logit, -1).unsqueeze(1)
        nll = -torch.logsumexp(lp, -1).sum()
        opt.zero_grad()
        nll.backward()
        opt.step()
        trace.append(-float(nll.detach()))
    with torch.no_grad():
        comps = {k: v[0].numpy() for k, v in family.decode(u).items()}
        pi = torch.softmax(logit, -1)[0].numpy()
    return MixtureParams(pi, comps), {"loglik_trace": trace}
