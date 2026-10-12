"""Composite training loss (RESEARCH_METHODOLOGY.md 5.3).

    L = lambda_1 * L_Hungarian + lambda_2 * L_NLL + lambda_3 * L_aux

  - L_Hungarian : parameter MSE after optimal component alignment (on
    log alpha, log beta, and pi), killing label-switching interference.
  - L_NLL       : negative log-likelihood of the *training sample* under the
    predicted parameters -- label-invariant by construction, ties training to
    the ML principle, doubles as the early-stopping metric.
  - L_aux       : responsibility cross-entropy against simulator-known true
    labels (a luxury classical EM lacks), Switch-style expert load balancing,
    and (optional) model-order cross-entropy.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..simulation.mixture import mixture_logpdf_torch
from ..utils.config import LossWeights
from .matching import batch_permutations

EPS = 1e-12


class CompositeLoss(nn.Module):
    def __init__(self, weights: LossWeights, g_max: int):
        super().__init__()
        self.w = weights
        self.g_max = g_max

    def forward(self, pred: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        w = self.w
        comp_mask = batch["comp_mask"]                     # (B, g_max)
        obs_mask = batch["obs_mask"]                       # (B, n)
        perms = batch_permutations(batch, pred)            # (B, g_max)

        # --- L_Hungarian: align predicted components to true order ----------
        idx_a = perms.unsqueeze(-1).expand(-1, -1, pred["alpha"].shape[-1])
        alpha_al = torch.gather(pred["alpha"], 1, idx_a)
        beta_al = torch.gather(pred["beta"], 1, idx_a)
        pi_al = torch.gather(pred["pi"], 1, perms)
        m = comp_mask.unsqueeze(-1).float()
        l_alpha = ((alpha_al.clamp_min(EPS).log() - batch["alpha"].clamp_min(EPS).log()) ** 2 * m).sum() / m.sum()
        l_beta = ((beta_al.clamp_min(EPS).log() - batch["beta"].clamp_min(EPS).log()) ** 2 * m).sum() / m.sum()
        l_pi = ((pi_al - batch["pi"]) ** 2 * comp_mask.float()).sum() / comp_mask.float().sum()
        l_param = l_alpha + l_beta + l_pi

        # --- L_NLL: label-invariant likelihood anchor -----------------------
        logp = mixture_logpdf_torch(batch["X"], pred["pi"], pred["alpha"], pred["beta"], comp_mask)
        l_nll = -(logp * obs_mask.float()).sum() / obs_mask.float().sum()

        # --- L_aux ----------------------------------------------------------
        stats = {}
        l_aux = pred["pi"].new_zeros(())

        if w.resp > 0:
            # align responsibility columns to true component order
            idx_r = perms.unsqueeze(1).expand(-1, pred["resp"].shape[1], -1)  # (B, n, g_max)
            resp_al = torch.gather(pred["resp"], 2, idx_r)
            log_resp = resp_al.clamp_min(EPS).log()
            labels = batch["labels"]                                          # (B, n), -1 padded
            valid = labels >= 0
            l_resp = F.nll_loss(
                log_resp.reshape(-1, self.g_max),
                labels.clamp_min(0).reshape(-1),
                reduction="none",
            ).reshape_as(labels)
            l_resp = (l_resp * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
            l_aux = l_aux + w.resp * l_resp
            stats["resp"] = float(l_resp.detach())

        if w.load_balance > 0:
            # Switch-style: g_max * sum_i f_i * P_i over valid comps/obs
            labels = batch["labels"]
            valid = labels >= 0
            resp = pred["resp"]
            lb = pred["pi"].new_zeros(())
            for b in range(pred["pi"].shape[0]):
                g = int(batch["g"][b])
                r = resp[b][valid[b]][:, :g]                                  # (n_b, g)
                f = torch.zeros(g, device=r.device)
                assign = r.argmax(dim=-1)
                for i in range(g):
                    f[i] = (assign == i).float().mean()
                P = r.mean(dim=0)
                lb = lb + g * (f * P).sum()
            lb = lb / pred["pi"].shape[0]
            l_aux = l_aux + w.load_balance * lb
            stats["load_balance"] = float(lb.detach())

        if w.g > 0 and pred["g_logits"] is not None:
            l_g = F.cross_entropy(pred["g_logits"], batch["g"] - 1)  # index i <-> g = i+1
            l_aux = l_aux + w.g * l_g
            stats["g_ce"] = float(l_g.detach())

        loss = w.param * l_param + w.nll * l_nll + l_aux
        stats.update(
            loss=float(loss.detach()),
            param=float(l_param.detach()),
            nll=float(l_nll.detach()),
        )
        return loss, stats
