"""Symmetry-aware training objective (RESEARCH_METHODOLOGY.md 4.2 and 6.1).

Quotient loss on standardised canonical coordinates z = (u - u_mean) / u_std:

    L_Q(theta, theta^) = min_sigma (1/g) sum_i [ ||z_i - z^_sigma(i)||^2 / d
                                               + w_pi (pi_i - pi^_sigma(i))^2 ]
                         + w_un * sum_{unmatched slots s} pi^_s^2

The minimising sigma is found by the Hungarian algorithm (rectangular when the
true g < S slots). The assignment cost *is* the loss, so this is exactly the
quotient-space squared loss whose Bayes estimator is characterised in Thm 3.

Matching modes (E3 / Thm 2):
    quotient : Hungarian (the method)
    naive    : true component i <-> slot i  (label-switching collapse, Thm 2a)
    sorted   : both sides ordered by `family.order_key` (ordering-constraint bias, Thm 2b)
    none_nll : train on the label-invariant NLL only (Set-Transformer style); L_Q is
               still computed for monitoring.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ..simulation.families.base import Family
from ..simulation.simulator import target_params


def match_components(cost: torch.Tensor, g: torch.Tensor, mode: str = "quotient",
                     key_true: torch.Tensor | None = None, key_pred: torch.Tensor | None = None) -> torch.Tensor:
    """cost (B,G,S). Returns assign (B,G) long: slot index for each true component, -1 for padding."""
    B, G, S = cost.shape
    C = cost.detach().float().cpu().numpy()
    gs = g.detach().cpu().numpy()
    kt = None if key_true is None else key_true.detach().float().cpu().numpy()
    kp = None if key_pred is None else key_pred.detach().float().cpu().numpy()
    assign = np.full((B, G), -1, dtype=np.int64)
    for b in range(B):
        gb = int(gs[b])
        if gb == 0:
            continue
        if mode == "quotient":
            rows, cols = linear_sum_assignment(C[b, :gb, :])
            assign[b, rows] = cols
        elif mode == "naive":
            assign[b, :gb] = np.arange(gb)
        elif mode == "sorted":
            ot = np.argsort(kt[b, :gb], kind="stable")
            op = np.argsort(kp[b, :gb], kind="stable")
            assign[b, ot] = op
        else:
            raise ValueError(f"unknown match mode '{mode}'")
    return torch.from_numpy(assign).to(cost.device)


def gather_assigned(x: torch.Tensor, assign: torch.Tensor) -> torch.Tensor:
    """x (B,S,...) -> (B,G,...) picking the slot assigned to each true component."""
    idx = assign.clamp_min(0)
    while idx.dim() < x.dim():
        idx = idx.unsqueeze(-1)
    return torch.gather(x, 1, idx.expand(*assign.shape, *x.shape[2:]))


def matched_slot_mask(assign: torch.Tensor, S: int) -> torch.Tensor:
    B = assign.shape[0]
    m = torch.zeros(B, S + 1, dtype=torch.bool, device=assign.device)
    m.scatter_(1, torch.where(assign >= 0, assign, torch.full_like(assign, S)), True)
    return m[:, :S]


class QuotientLoss(nn.Module):
    def __init__(self, family: Family, cfg):
        super().__init__()
        self.family = family
        self.cfg = cfg
        self.aux_scale = 1.0   # set by the trainer: auxiliary terms fade to 0 at the end (pure L_Q)

    # ------------------------------------------------------------------ targets
    def true_coordinates(self, out: dict, batch: dict, model) -> torch.Tensor:
        tp = target_params(batch, self.family)
        tp = {k: v.to(out["z"].dtype) for k, v in tp.items()}
        canon = self.family.canonicalize(tp, out["loc"], out["scale"])
        return model.u_to_z(self.family.encode(canon))

    def pair_cost(self, z_true, pi_true, z_pred, pi_pred) -> torch.Tensor:
        dz = ((z_true.unsqueeze(2) - z_pred.unsqueeze(1)) ** 2).mean(-1)          # (B,G,S)
        dp = (pi_true.unsqueeze(2) - pi_pred.unsqueeze(1)) ** 2
        return dz + self.cfg.w_pi * dp

    def _match(self, cost, batch, z_true, z_pred, model):
        mode = "quotient" if self.cfg.match == "none_nll" else self.cfg.match
        kt = kp = None
        if mode == "sorted":
            kt = self.family.order_key(model.decode_z(z_true))
            kp = self.family.order_key(model.decode_z(z_pred))
        return match_components(cost, batch["g"], mode, kt, kp)

    def step_loss(self, z_true, batch, z_pred, pi_pred, model):
        cmask = batch["comp_mask"]
        cost = self.pair_cost(z_true, batch["tp_pi"].to(pi_pred.dtype), z_pred, pi_pred)
        assign = self._match(cost, batch, z_true, z_pred, model)
        matched = torch.gather(cost, 2, assign.clamp_min(0).unsqueeze(-1)).squeeze(-1)   # (B,G)
        cm = cmask.to(matched.dtype)
        per_ds = (matched * cm).sum(1) / cm.sum(1).clamp_min(1.0)
        unmatched = ~matched_slot_mask(assign, pi_pred.shape[1])
        pen = (pi_pred ** 2 * unmatched.to(pi_pred.dtype)).sum(1)
        return per_ds.mean(), pen.mean(), assign

    # ------------------------------------------------------------------ forward
    def forward(self, out: dict, batch: dict, model) -> tuple[torch.Tensor, dict, torch.Tensor]:
        cfg = self.cfg
        z_true = self.true_coordinates(out, batch, model)
        steps = out["steps"]
        w_ds = cfg.deep_supervision * self.aux_scale
        use_steps = steps if w_ds > 0 else steps[-1:]
        losses = []
        assign = None
        l_final = pen_final = None
        for i, st in enumerate(use_steps):
            l_q, pen, a = self.step_loss(z_true, batch, st["z"], st["pi"], model)
            losses.append(l_q + cfg.unmatched_pi * pen)
            if i == len(use_steps) - 1:
                l_final, pen_final, assign = l_q, pen, a
        l_param = losses[-1]
        if len(losses) > 1:
            l_param = l_param + w_ds * torch.stack(losses[:-1]).mean()

        stats = {"quotient": float(l_final.detach()), "unmatched_pi": float(pen_final.detach()),
                 "aux_scale": float(self.aux_scale)}
        total = l_param if cfg.match != "none_nll" else l_param.new_zeros(())

        # responsibility supervision with Hungarian-aligned labels
        resp = out.get("resp")
        w_resp = cfg.resp * self.aux_scale
        if w_resp > 0 and resp is not None and resp.requires_grad:
            labels = batch["labels"]
            valid = labels >= 0
            slot_of = torch.gather(assign.clamp_min(0), 1, labels.clamp_min(0))       # (B,n)
            p = torch.gather(resp, 2, slot_of.unsqueeze(-1)).squeeze(-1)
            ce = -(p.clamp_min(1e-8).log() * valid.to(p.dtype)).sum() / valid.sum().clamp_min(1)
            total = total + w_resp * ce
            stats["resp_ce"] = float(ce.detach())

        # existence of each slot (in the pure phase the model detaches the head's input,
        # so this trains the head only and leaves the backbone's L_Q target untouched)
        if cfg.exist > 0 and out.get("exist_logits") is not None:
            tgt = matched_slot_mask(assign, out["pi"].shape[1]).to(out["exist_logits"].dtype)
            bce = F.binary_cross_entropy_with_logits(out["exist_logits"], tgt)
            total = total + cfg.exist * bce
            stats["exist_bce"] = float(bce.detach())

        # label-invariant NLL (ablation "amortised-MLE target", or none_nll mode)
        w_nll = cfg.nll if cfg.match != "none_nll" else max(cfg.nll, 1.0)
        if w_nll > 0 and self.family.has_torch_logprob:
            lp = self.family.log_prob(out["Xc"], batch["events"], out["params_canon"])
            lp = lp + out["pi"].clamp_min(1e-8).log().unsqueeze(1)
            ll = torch.logsumexp(lp, -1)
            m = batch["obs_mask"].to(ll.dtype)
            nll = -(ll * m).sum() / m.sum()
            total = total + w_nll * nll
            stats["nll"] = float(nll.detach())

        # flow-matching posterior on pivot-aligned targets
        if cfg.flow > 0 and getattr(model, "flow", None) is not None:
            y1 = model.flow_inputs(out["z"].detach(), out["pi"].detach()).clone()
            y_true = model.flow_inputs(z_true, batch["tp_pi"].to(z_true.dtype))
            B, G = assign.shape
            for_b = torch.arange(B, device=y1.device).unsqueeze(1).expand(B, G)
            ok = assign >= 0
            y1[for_b[ok], assign[ok]] = y_true[ok].to(y1.dtype)
            l_flow = model.flow_loss(out, y1, matched_slot_mask(assign, y1.shape[1]))
            total = total + cfg.flow * l_flow
            stats["flow"] = float(l_flow.detach())

        stats["loss"] = float(total.detach())
        return total, stats, assign
