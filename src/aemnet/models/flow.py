"""Conditional flow-matching posterior head over the quotient space (C4, Prop 2).

Targets are pivot-aligned: each true component is placed in the slot it was
Hungarian-matched to (i.e. aligned to the network's own point estimate), so the
flow models a posterior in fixed slot coordinates. The velocity field is a small
set transformer over slots, so the *joint* posterior over all components is
modelled (not a mean-field product).

    y_s = [z_s (standardised u), log pi_s]         per slot
    y_t = (1 - t) y_0 + t y_1,  y_0 ~ N(0, I),     target velocity y_1 - y_0
    (rectified flow / conditional flow matching; Lipman et al., 2023)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .layers import SAB

TIME_FEATS = 16


def time_embedding(t: torch.Tensor, n: int = TIME_FEATS) -> torch.Tensor:
    """t (B,) in [0, 1] -> (B, n) sinusoidal features."""
    freqs = torch.exp(torch.linspace(0, math.log(1000.0), n // 2, device=t.device, dtype=t.dtype))
    ang = t.unsqueeze(-1) * freqs * math.pi
    return torch.cat([ang.sin(), ang.cos()], -1)


class SlotFlowHead(nn.Module):
    def __init__(self, dim_y: int, dim_cond: int, hidden: int = 128, n_layers: int = 3, heads: int = 4):
        super().__init__()
        self.dim_y = dim_y
        self.inp = nn.Linear(dim_y + dim_cond + TIME_FEATS, hidden)
        self.blocks = nn.ModuleList([SAB(hidden, heads) for _ in range(n_layers)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, dim_y))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def velocity(self, y, t, cond):
        """y (B,S,dy), t (B,), cond (B,S,Hc) -> (B,S,dy)."""
        te = time_embedding(t).unsqueeze(1).expand(-1, y.shape[1], -1)
        h = self.inp(torch.cat([y, cond, te], -1))
        for blk in self.blocks:
            h = blk(h)
        return self.out(h)

    def loss(self, y1, cond, loss_mask):
        """Flow-matching loss; loss_mask (B,S) marks slots that carry a target."""
        y0 = torch.randn_like(y1)
        t = torch.rand(y1.shape[0], device=y1.device, dtype=y1.dtype)
        tt = t.view(-1, 1, 1)
        yt = (1 - tt) * y0 + tt * y1
        v = self.velocity(yt, t, cond)
        err = ((v - (y1 - y0)) ** 2).mean(-1)
        m = loss_mask.to(err.dtype)
        return (err * m).sum() / m.sum().clamp_min(1.0)

    @torch.no_grad()
    def sample(self, cond, n_samples: int, steps: int = 32):
        """Heun integration from t=0 to 1. Returns (n_samples, B, S, dy)."""
        B, S, Hc = cond.shape
        c = cond.unsqueeze(0).expand(n_samples, -1, -1, -1).reshape(n_samples * B, S, Hc)
        y = torch.randn(n_samples * B, S, self.dim_y, device=cond.device, dtype=cond.dtype)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((y.shape[0],), i * dt, device=y.device, dtype=y.dtype)
            v1 = self.velocity(y, t, c)
            y_pred = y + dt * v1
            v2 = self.velocity(y_pred, t + dt, c)
            y = y + 0.5 * dt * (v1 + v2)
        return y.view(n_samples, B, S, self.dim_y)

    def log_prob(self, y1, cond, steps: int = 32, hutchinson: bool = False):
        """log q(y1 | cond) by integrating the instantaneous change of variables
        backwards (Euler), with the exact divergence (or a Hutchinson estimate).
        Differentiable w.r.t. the flow parameters. Returns (B,).
        Runs attention through the math SDPA kernel: the fused kernels have no
        double-backward, which the divergence term needs when this is trained on."""
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.MATH):
            return self._log_prob(y1, cond, steps, hutchinson)

    def _log_prob(self, y1, cond, steps, hutchinson):
        B, S, dy = y1.shape
        y = y1
        div_int = y1.new_zeros(B)
        dt = 1.0 / steps
        for i in range(steps, 0, -1):
            t = torch.full((B,), i * dt, device=y.device, dtype=y.dtype)
            with torch.enable_grad():
                y_req = y if y.requires_grad else y.detach().requires_grad_(True)
                v = self.velocity(y_req, t, cond)
                if hutchinson:
                    eps = torch.randint_like(y_req, 2) * 2 - 1
                    (g,) = torch.autograd.grad((v * eps).sum(), y_req, create_graph=True)
                    div = (g * eps).flatten(1).sum(-1)
                else:
                    vf = v.flatten(1)
                    div = y1.new_zeros(B)
                    for j in range(vf.shape[1]):
                        (g,) = torch.autograd.grad(vf[:, j].sum(), y_req, create_graph=True, retain_graph=True)
                        div = div + g.flatten(1)[:, j]
            div_int = div_int + dt * div
            y = y_req - dt * v
        log_p0 = -0.5 * (y ** 2).flatten(1).sum(-1) - 0.5 * S * dy * math.log(2 * math.pi)
        return log_p0 - div_int
