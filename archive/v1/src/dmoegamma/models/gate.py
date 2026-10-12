"""Block B: gating network -- the "E-step" of the architecture.

Produces
  - pi_hat : (B, g_max) mixing-weight estimates (softmax), and
  - resp   : (B, n, g_max) per-observation soft assignments, interpretable as
             learned responsibilities r_{ji}, directly comparable to EM
             responsibilities (Jordan & Jacobs, 1994).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.config import ModelConfig
from .encoder import build_mlp


class GateNet(nn.Module):
    def __init__(self, cfg: ModelConfig, g_max: int):
        super().__init__()
        self.pi_head = build_mlp(cfg.enc_out, cfg.head_hidden, g_max, cfg.head_layers)
        # responsibility head sees both the per-observation embedding and the
        # global sample embedding: r_ji = softmax(gate([rho(z_j), h]))
        self.resp_head = build_mlp(2 * cfg.enc_out, cfg.head_hidden, g_max, cfg.head_layers)

    def forward(self, h: torch.Tensor, rho: torch.Tensor):
        pi = torch.softmax(self.pi_head(h), dim=-1)                    # (B, g_max)
        h_exp = h.unsqueeze(1).expand(-1, rho.shape[1], -1)            # (B, n, H)
        resp = torch.softmax(self.resp_head(torch.cat([rho, h_exp], dim=-1)), dim=-1)
        return pi, resp
