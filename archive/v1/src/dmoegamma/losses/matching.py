"""Label-switching handling: optimal component alignment (Hungarian matching).

Mixture parameters are set-valued: permuting component labels leaves the model
unchanged. All per-parameter losses and metrics are therefore computed after
aligning estimated components to the truth via the linear-sum assignment that
minimizes total standardized log-parameter L1 distance (DETR-style matching
loss; Carion et al., 2020; RESEARCH_METHODOLOGY.md 5.3).

Convention: perm[i] = index of the *estimated* component matched to true
component i, so that `pred.permuted(perm)` aligns pred to the true order.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..params import Theta

# Standardization scales for the cost: log-shapes/log-scales vary over
# log(30/0.1) ~ 5.7 in the prior; weights are O(1). Equal footing:
LOG_SCALE = np.log(30.0 / 0.1)


def _signature(alpha: np.ndarray, beta: np.ndarray, pi: float) -> np.ndarray:
    """Standardized per-component signature used in the assignment cost."""
    return np.concatenate([np.log(alpha) / LOG_SCALE, np.log(beta) / LOG_SCALE, [pi]])


def component_cost(true: Theta, pred: Theta) -> np.ndarray:
    """(g, g) cost matrix: C[i, j] = L1 distance between true comp i and
    estimated comp j in standardized log-parameter space."""
    g = true.g
    C = np.empty((g, g))
    for i in range(g):
        s_true = _signature(true.alpha[i], true.beta[i], true.pi[i])
        for j in range(g):
            s_pred = _signature(pred.alpha[j], pred.beta[j], pred.pi[j])
            C[i, j] = np.abs(s_true - s_pred).sum()
    return C


def hungarian_match(true: Theta, pred: Theta) -> np.ndarray:
    """Optimal assignment perm (length g): perm[i] = estimated component
    matched to true component i."""
    if pred.g != true.g:
        raise ValueError("hungarian_match requires equal numbers of components")
    _, col_ind = linear_sum_assignment(component_cost(true, pred))
    return col_ind


def batch_permutations(batch: dict, pred: dict) -> "torch.Tensor":
    """Hungarian permutation per dataset in a padded batch.

    batch/pred carry pi (B, g_max), alpha/beta (B, g_max, k), comp_mask (B, g_max).
    Returns LongTensor (B, g_max); identity on padded slots.
    """
    import torch

    B, g_max = batch["pi"].shape
    perms = torch.arange(g_max, dtype=torch.long).unsqueeze(0).repeat(B, 1)
    pi_t = batch["pi"].detach().cpu().numpy()
    a_t = batch["alpha"].detach().cpu().numpy()
    b_t = batch["beta"].detach().cpu().numpy()
    pi_p = pred["pi"].detach().cpu().numpy()
    a_p = pred["alpha"].detach().cpu().numpy()
    b_p = pred["beta"].detach().cpu().numpy()
    g_true = batch["g"].detach().cpu().numpy()
    for b in range(B):
        g = int(g_true[b])
        true = Theta(pi_t[b, :g], a_t[b, :g], b_t[b, :g])
        est = Theta(pi_p[b, :g] / pi_p[b, :g].sum(), a_p[b, :g], b_p[b, :g])
        perms[b, :g] = torch.from_numpy(hungarian_match(true, est))
    return perms.to(batch["pi"].device)
