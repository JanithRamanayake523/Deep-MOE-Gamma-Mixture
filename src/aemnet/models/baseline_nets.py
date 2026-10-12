"""Neural baselines trained with the same simulator, loss and trainer as AEM-Net.

NBEDeepSetsNet     -- neural Bayes estimator with a DeepSets encoder and an MLP
                      head emitting all S components at once (Sainsbury-Dale et al., 2024).
SetTransformerNet  -- ISAB encoder + PMA onto S seeds + slot self-attention + shared
                      per-seed head; the "amortised clustering" model of Lee et al. (2019).
                      With `model.flow_head=true` and `loss.match=sorted` it becomes the
                      ordering-constrained NPE / ABMM-style baseline (Kucharsky & Burkner).
Neither has EM structure; both report responsibilities computed from their
predicted parameters (when the likelihood is tractable) for clustering metrics.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..simulation.families.base import Family
from .aem import PointEncoder
from .base_net import AmortizedMixtureNet
from .layers import PMA, SAB, masked_mean, mlp


class NBEDeepSetsNet(AmortizedMixtureNet):
    def __init__(self, family: Family, cfg, n_slots: int):
        super().__init__(family, cfg, n_slots)
        H = cfg.d_model
        self.rho = mlp(family.feature_dim, H, H, 2)
        self.phi = mlp(2 * H + 1, H, H, 2)
        d = family.d
        self.head = nn.Linear(H, n_slots * (d + 2))
        nn.init.zeros_(self.head.bias)
        self.head.weight.data.mul_(0.1)
        self.slot_embed = nn.Parameter(torch.randn(1, n_slots, H) * 0.1)

    def forward(self, X, events, obs_mask, T=None):
        Xc, loc, scale = self.canonical(X, obs_mask)
        r = self.rho(self.family.features(Xc, events))
        mean = masked_mean(r, obs_mask)
        std = (masked_mean(r ** 2, obs_mask) - mean ** 2).clamp_min(0).sqrt()
        logn = obs_mask.sum(1, keepdim=True).clamp_min(1).to(X.dtype).log()
        h = self.phi(torch.cat([mean, std, logn], -1))
        d, S = self.family.d, self.S
        o = self.head(h).view(-1, S, d + 2)
        z, pi_logit, ex = o[..., :d], o[..., d], o[..., d + 1]
        if self.exist_detach:
            ex = self.head(h.detach()).view(-1, S, d + 2)[..., d + 1]
        pi = torch.softmax(pi_logit, -1)
        slots = h.unsqueeze(1) + self.slot_embed
        out = {"z": z, "pi": pi, "steps": [{"z": z, "pi": pi}], "slots": slots, "ctx": h,
               "loc": loc, "scale": scale, "Xc": Xc,
               "exist_logits": ex if self.cfg.existence_head else None,
               "resp": self.responsibilities_from_params(Xc, events, obs_mask, z, pi)}
        return self.finalize(out)


class SetTransformerNet(AmortizedMixtureNet):
    def __init__(self, family: Family, cfg, n_slots: int):
        super().__init__(family, cfg, n_slots)
        H = cfg.d_model
        self.encoder = PointEncoder(family.feature_dim, cfg)
        self.pma = PMA(H, cfg.n_heads, n_slots, cfg.ff_mult)
        self.sabs = nn.ModuleList([SAB(H, cfg.n_heads, cfg.ff_mult) for _ in range(2)])
        self.head = mlp(H, cfg.expert_hidden, family.d + 2, cfg.expert_layers)
        nn.init.zeros_(self.head[-1].bias)
        self.head[-1].weight.data.mul_(0.1)

    def forward(self, X, events, obs_mask, T=None):
        Xc, loc, scale = self.canonical(X, obs_mask)
        rho = self.encoder(self.family.features(Xc, events), obs_mask)
        slots = self.pma(rho, obs_mask)
        for sab in self.sabs:
            slots = sab(slots)
        d = self.family.d
        o = self.head(slots)
        z, pi = o[..., :d], torch.softmax(o[..., d], -1)
        out = {"z": z, "pi": pi, "steps": [{"z": z, "pi": pi}], "slots": slots,
               "ctx": masked_mean(rho, obs_mask), "loc": loc, "scale": scale, "Xc": Xc,
               "exist_logits": ((self.head(slots.detach())[..., d + 1] if self.exist_detach else o[..., d + 1])
                                if self.cfg.existence_head else None),
               "resp": self.responsibilities_from_params(Xc, events, obs_mask, z, pi)}
        return self.finalize(out)
