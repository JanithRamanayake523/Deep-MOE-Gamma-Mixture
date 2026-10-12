"""Constructive check of Thm 1 (exact EM embedding).

For a regular exponential family f(z; eta) = h(z) exp(<eta, T(z)> - A(eta)), one EM
iteration equals one AEM block whose learned parts are set to
    E-block : compatibility weights q = k = 0, log-likelihood scale a = 1, tau = 1
    M-block : psi = T (raw sufficient statistics only), gated update u = 1, U_s = m_s
    Expert  : (grad A)^{-1}  (`family.inverse_link`)
`ExactEMBlock` builds exactly these settings out of the *same* EBlock / MBlock.pool
code used by AEM-Net, so a unit test comparing it with a classical EM step
(tests/test_em_embedding.py) verifies the theorem's construction.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from ..simulation.families.base import Family
from .aem import EBlock, MBlock


class ExactEMBlock(nn.Module):
    def __init__(self, family: Family, d_model: int = 8):
        super().__init__()
        if not family.exponential_family:
            raise ValueError(f"{family.name} is not an exponential family with an implemented inverse link")
        self.family = family
        cfg = SimpleNamespace(d_model=d_model, use_loglik=True, loglik_scale_init=1.0, learn_loglik_scale=False,
                              temperature_init=1.0, detach_loglik=False, loglik_floor=-1e30)
        self.eblock = EBlock(cfg, family)
        with torch.no_grad():
            self.eblock.q.weight.zero_()
            self.eblock.k.weight.zero_()

    def forward(self, X, events, obs_mask, params, pi):
        B, n, _ = X.shape
        H = self.eblock.q.in_features
        rho = X.new_zeros(B, n, H)
        slots = X.new_zeros(B, pi.shape[1], H)
        r = self.eblock(rho, slots, params, pi, X, events, obs_mask)       # exact E-step
        m, _, pi_new = MBlock.pool(r, self.family.suff_stats(X, events), obs_mask)  # U_s = m_s
        return self.family.inverse_link(m), pi_new, r                      # Expert = (grad A)^{-1}
