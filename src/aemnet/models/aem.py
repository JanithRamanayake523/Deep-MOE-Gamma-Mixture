"""AEM-Net: Amortized Expectation-Maximization Network (RESEARCH_METHODOLOGY.md 3).

    0. canonicalise        x~ = (x - loc(D)) / scale(D)            (exact scale equivariance)
    1. point encoder       rho_j = ISAB^L(MLP(features(x~_j)))
    2. slot init           S^0 = PMA_S(rho)  (distinct learned seeds break slot symmetry)
    3. T unrolled blocks   E-block: r_js = softmax_s([a log f(x~_j; theta_s) + log pi_s
                                                     + <q(rho_j), k(S_s)>/sqrt(H)] / tau)
                           M-block: m_s = sum_j r_js psi_j / N_s  (responsibility pooling of
                                    learned + raw sufficient statistics), pi_s = N_s / n,
                                    S_s <- gated update, theta_s = Expert(S_s)  (ONE shared expert)
    4. heads               existence e_s, optional flow posterior

The gate outputs per-observation responsibilities and each expert consumes only
its own responsibility-weighted data: a genuine (unrolled) mixture of experts in
the sense of Jordan & Jacobs (1994). Contrast with the v1 design (archive/v1),
whose experts only saw a global embedding -- reproduced as an ablation via `pooling="global", shared_expert=False`.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..simulation.families.base import Family
from .base_net import AmortizedMixtureNet
from .layers import ISAB, PMA, SAB, masked_mean, mlp


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


class PointEncoder(nn.Module):
    """features -> per-point embeddings rho (B,n,H); ISAB interactions or plain DeepSets."""

    def __init__(self, in_dim: int, cfg):
        super().__init__()
        H = cfg.d_model
        self.embed = mlp(in_dim, H, H, 1)
        self.kind = cfg.encoder
        if cfg.encoder == "isab":
            self.layers = nn.ModuleList([ISAB(H, cfg.n_heads, cfg.n_inducing, cfg.ff_mult)
                                         for _ in range(cfg.n_enc_layers)])
        elif cfg.encoder == "deepsets":
            self.layers = nn.ModuleList([nn.Sequential(nn.LayerNorm(H), mlp(H, 2 * H, H, 1))
                                         for _ in range(cfg.n_enc_layers)])
        else:
            raise ValueError(f"unknown encoder '{cfg.encoder}'")

    def forward(self, feats, mask):
        h = self.embed(feats)
        for layer in self.layers:
            h = layer(h, mask) if self.kind == "isab" else h + layer(h)
        return h * mask.unsqueeze(-1).to(h.dtype)


class SlotInit(nn.Module):
    def __init__(self, cfg, n_slots: int):
        super().__init__()
        H = cfg.d_model
        self.mode = cfg.slot_init
        self.pma = PMA(H, cfg.n_heads, n_slots, cfg.ff_mult)
        if self.mode == "random":
            self.mu = nn.Parameter(torch.zeros(1, 1, H))
            self.log_sigma = nn.Parameter(torch.full((1, 1, H), -1.0))
        self.n_slots = n_slots

    def forward(self, rho, mask):
        if self.mode == "learned":
            return self.pma(rho, mask)
        B, H = rho.shape[0], rho.shape[-1]
        seeds = self.mu + self.log_sigma.exp() * torch.randn(B, self.n_slots, H, device=rho.device, dtype=rho.dtype)
        return self.pma(rho, mask, seeds=seeds)


class EBlock(nn.Module):
    """Learned E-step. With a=1, zero compatibility and tau=1 it is exactly the EM E-step."""

    def __init__(self, cfg, family: Family):
        super().__init__()
        H = cfg.d_model
        self.family = family
        self.q = nn.Linear(H, H, bias=False)
        self.k = nn.Linear(H, H, bias=False)
        self.use_ll = bool(cfg.use_loglik and family.has_torch_logprob)
        self.ll_scale = nn.Parameter(torch.tensor(float(cfg.loglik_scale_init)),
                                     requires_grad=bool(cfg.learn_loglik_scale))
        self.log_tau = nn.Parameter(torch.tensor(math.log(cfg.temperature_init)))
        self.detach_ll = cfg.detach_loglik
        self.floor = cfg.loglik_floor
        self.scale = H ** -0.5

    def forward(self, rho, slots, params_prev, pi_prev, Xc, events, mask):
        logits = torch.einsum("bnh,bsh->bns", self.q(rho), self.k(slots)) * self.scale
        logits = logits + pi_prev.clamp_min(1e-8).log().unsqueeze(1)
        if self.use_ll:
            ll = self.family.log_prob(Xc, events, params_prev)
            if self.detach_ll:
                ll = ll.detach()
            ll = torch.nan_to_num(ll, nan=self.floor, neginf=self.floor).clamp_min(self.floor)
            logits = logits + self.ll_scale * ll
        r = torch.softmax(logits / self.log_tau.exp(), dim=-1)
        return r * mask.unsqueeze(-1).to(r.dtype)


class MBlock(nn.Module):
    """Learned M-step: responsibility-weighted pooling -> gated slot update."""

    def __init__(self, cfg, family: Family):
        super().__init__()
        H = cfg.d_model
        self.pooling = cfg.pooling
        self.use_ss = cfg.use_suffstats
        self.psi = mlp(H, H, H, 1)
        P = H + (family.suff_dim if self.use_ss else 0)
        in_dim = 2 * P + 1 + 2 * H
        self.upd = mlp(in_dim, H, H, 1)
        self.gate = mlp(in_dim, H, H, 1)
        self.res = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, 2 * H), nn.GELU(), nn.Linear(2 * H, H))
        self.slot_attn = SAB(H, cfg.n_heads, cfg.ff_mult) if cfg.slot_attention else None

    @staticmethod
    def pool(r, feats, mask):
        """Responsibility-weighted pooling: m_s = sum_j r_js f_j / N_s, its variance,
        and pi_s = N_s / n. With feats = T(z) this is exactly the EM M-step's
        sufficient-statistic average (Thm 1)."""
        n_obs = mask.sum(1, keepdim=True).clamp_min(1).to(r.dtype)          # (B,1)
        N = r.sum(1)                                                          # (B,S)
        Nc = N.clamp_min(1e-6).unsqueeze(-1)
        m = torch.einsum("bns,bnp->bsp", r, feats) / Nc
        v = (torch.einsum("bns,bnp->bsp", r, feats ** 2) / Nc - m ** 2).clamp_min(0)
        return m, v, N / n_obs

    def forward(self, rho, ss, r, slots, ctx, mask):
        feats = self.psi(rho)
        if self.use_ss:
            feats = torch.cat([feats, ss], -1)
        m, v, pi = self.pool(r, feats, mask)
        if self.pooling == "responsibility":
            pass
        elif self.pooling == "global":
            gm = masked_mean(feats, mask)
            gv = (masked_mean(feats ** 2, mask) - gm ** 2).clamp_min(0)
            m = gm.unsqueeze(1).expand(-1, slots.shape[1], -1)
            v = gv.unsqueeze(1).expand(-1, slots.shape[1], -1)
        else:
            raise ValueError(f"unknown pooling '{self.pooling}'")
        S = slots.shape[1]
        inp = torch.cat([symlog(m), symlog(v), (pi + 1e-6).log().unsqueeze(-1), slots,
                         ctx.unsqueeze(1).expand(-1, S, -1)], -1)
        g = torch.sigmoid(self.gate(inp))
        slots = (1 - g) * slots + g * self.upd(inp)
        slots = slots + self.res(slots)
        if self.slot_attn is not None:
            slots = self.slot_attn(slots)
        return slots, pi


class Experts(nn.Module):
    """Slot state -> standardised parameters z. One shared network (default) or one per slot."""

    def __init__(self, cfg, d: int, n_slots: int):
        super().__init__()
        self.shared = cfg.shared_expert
        make = lambda: mlp(cfg.d_model, cfg.expert_hidden, d, cfg.expert_layers)  # noqa: E731
        self.nets = nn.ModuleList([make()] if self.shared else [make() for _ in range(n_slots)])
        for net in self.nets:
            nn.init.zeros_(net[-1].bias)
            net[-1].weight.data.mul_(0.1)

    def forward(self, slots):
        if self.shared:
            return self.nets[0](slots)
        return torch.stack([net(slots[:, s]) for s, net in enumerate(self.nets)], 1)


class AEMNet(AmortizedMixtureNet):
    def __init__(self, family: Family, cfg, n_slots: int):
        super().__init__(family, cfg, n_slots)
        H = cfg.d_model
        self.encoder = PointEncoder(family.feature_dim, cfg)
        self.slot_init = SlotInit(cfg, n_slots)
        self.pi0 = nn.Linear(H, 1)
        n_blocks = 1 if cfg.tie_iterations else max(cfg.T, 1)
        self.eblocks = nn.ModuleList([EBlock(cfg, family) for _ in range(n_blocks)])
        self.mblocks = nn.ModuleList([MBlock(cfg, family) for _ in range(n_blocks)])
        self.experts = Experts(cfg, family.d, n_slots)
        self.exist = mlp(H, H, 1, 1) if cfg.existence_head else None

    def _block(self, t: int):
        i = min(t, len(self.eblocks) - 1)
        return self.eblocks[i], self.mblocks[i]

    def forward(self, X, events, obs_mask, T: int | None = None) -> dict:
        T = self.cfg.T if T is None else T
        Xc, loc, scale = self.canonical(X, obs_mask)
        rho = self.encoder(self.family.features(Xc, events), obs_mask)
        ss = self.family.suff_stats(Xc, events) if self.cfg.use_suffstats else None
        ctx = masked_mean(rho, obs_mask)
        slots = self.slot_init(rho, obs_mask)
        z = self.experts(slots)
        pi = torch.softmax(self.pi0(slots).squeeze(-1), -1)
        steps = [{"z": z, "pi": pi}]
        for t in range(T):
            eb, mb = self._block(t)
            r = eb(rho, slots, self.decode_z(z), pi, Xc, events, obs_mask)
            slots, pi = mb(rho, ss, r, slots, ctx, obs_mask)
            z = self.experts(slots)
            steps.append({"z": z, "pi": pi})
        eb, _ = self._block(max(T - 1, 0))
        resp = eb(rho, slots, self.decode_z(z), pi, Xc, events, obs_mask)   # E-step at final params
        out = {"z": z, "pi": pi, "steps": steps, "resp": resp, "slots": slots, "ctx": ctx,
               "loc": loc, "scale": scale, "Xc": Xc,
               "exist_logits": (self.exist(slots.detach() if self.exist_detach else slots).squeeze(-1)
                                if self.exist is not None else None)}
        return self.finalize(out)
