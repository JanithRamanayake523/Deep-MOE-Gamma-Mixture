"""Shared machinery for every amortised mixture estimator (AEM-Net and the
neural baselines): canonicalisation, standardised output coordinates, decoding
to constrained/original-scale parameters, and the optional posterior flow head.

Output dict contract (consumed by losses/quotient.py and inference/):
    z        (B,S,d)  standardised canonical coordinates, u = u_mean + u_std * z
    pi       (B,S)    mixing weights (sum to 1 over slots)
    steps    list of {"z", "pi"} for deep supervision (last == final)
    resp     (B,n,S) or None
    exist_logits (B,S) or None
    slots    (B,S,H)  per-slot states (flow conditioning)
    loc, scale (B,k)  canonicalisation statistics
    Xc       (B,n,k)  canonical data
    u, params_canon, params   filled by `finalize`
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..simulation.families.base import Family
from .flow import SlotFlowHead

LOG_PI_FLOOR = -12.0


class AmortizedMixtureNet(nn.Module):
    def __init__(self, family: Family, cfg, n_slots: int):
        super().__init__()
        self.family = family
        self.cfg = cfg
        self.S = int(n_slots)
        self.exist_detach = False   # True in the pure-L_Q phase: existence head trains on detached features
        d = family.d
        self.register_buffer("u_mean", torch.zeros(d))
        self.register_buffer("u_std", torch.ones(d))
        self.flow = (SlotFlowHead(d + 1, cfg.d_model, cfg.flow_hidden, cfg.flow_layers, cfg.n_heads)
                     if cfg.flow_head else None)

    # ------------------------------------------------------------------ stats
    def set_u_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.u_mean.copy_(torch.as_tensor(mean, dtype=self.u_mean.dtype))
        self.u_std.copy_(torch.as_tensor(std, dtype=self.u_std.dtype).clamp_min(1e-3))

    # ------------------------------------------------------------------ canonicalisation
    def canonical(self, X, obs_mask):
        if self.cfg.canonicalize:
            loc, scale = self.family.canonical_loc_scale(X.detach(), obs_mask)
        else:
            loc = X.new_zeros(X.shape[0], X.shape[-1])
            scale = X.new_ones(X.shape[0], X.shape[-1])
        Xc = (X - loc.unsqueeze(1)) / scale.unsqueeze(1)
        # keep padded rows finite and inside the support
        fill = 1.0 if self.family.positive_support else 0.0
        Xc = torch.where(obs_mask.unsqueeze(-1), Xc, torch.full_like(Xc, fill))
        return Xc, loc, scale

    # ------------------------------------------------------------------ decoding
    def z_to_u(self, z):
        return self.u_mean + self.u_std * z

    def u_to_z(self, u):
        return (u - self.u_mean) / self.u_std

    def decode_z(self, z):
        return self.family.decode(self.z_to_u(z))

    def finalize(self, out: dict) -> dict:
        out["u"] = self.z_to_u(out["z"])
        out["params_canon"] = self.family.decode(out["u"])
        out["params"] = self.family.uncanonicalize(out["params_canon"], out["loc"], out["scale"])
        return out

    # ------------------------------------------------------------------ posterior head
    def flow_inputs(self, z, pi):
        return torch.cat([z, pi.clamp_min(1e-8).log().clamp_min(LOG_PI_FLOOR).unsqueeze(-1)], -1)

    def flow_cond(self, out):
        c = out["slots"]
        return c.detach() if self.cfg.flow_detach else c

    def flow_loss(self, out, y1, loss_mask):
        if self.flow is None:
            return out["z"].new_zeros(())
        return self.flow.loss(y1, self.flow_cond(out), loss_mask)

    @torch.no_grad()
    def sample_posterior(self, out, n_samples: int, steps: int | None = None):
        """Returns z (N,B,S,d) and pi (N,B,S) samples in canonical coordinates."""
        if self.flow is None:
            raise RuntimeError("model was built without a flow head (model.flow_head=false)")
        y = self.flow.sample(self.flow_cond(out), n_samples, steps or self.cfg.flow_steps)
        z = y[..., :-1]
        pi = torch.softmax(y[..., -1], dim=-1)
        return z, pi

    def responsibilities_from_params(self, Xc, events, obs_mask, z, pi):
        """Exact posterior responsibilities under the predicted params (no grad), if tractable."""
        if not self.family.has_torch_logprob:
            return None
        with torch.no_grad():
            lp = self.family.log_prob(Xc, events, self.decode_z(z)) + pi.clamp_min(1e-8).log().unsqueeze(1)
            r = torch.softmax(lp, -1)
        return r * obs_mask.unsqueeze(-1).to(r.dtype)
