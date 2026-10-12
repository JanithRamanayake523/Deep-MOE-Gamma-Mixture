"""Block C: expert networks -- the "M-step" of the architecture.

G_max parallel expert MLPs; expert i estimates component i's parameter vector
(alpha_i, beta_i) in R_+^{2k} via softplus outputs. Trailing experts are masked
when g < g_max (handled by comp_mask in the loss, not here).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.config import ModelConfig
from .encoder import build_mlp


class ExpertHeads(nn.Module):
    def __init__(self, cfg: ModelConfig, g_max: int, k: int):
        super().__init__()
        self.g_max = g_max
        self.k = k
        self.experts = nn.ModuleList(
            [build_mlp(cfg.enc_out, cfg.head_hidden, 2 * k, cfg.head_layers) for _ in range(g_max)]
        )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = torch.stack([expert(h) for expert in self.experts], dim=1)  # (B, g_max, 2k)
        alpha = F.softplus(raw[..., : self.k]) + 1e-4
        beta = F.softplus(raw[..., self.k :]) + 1e-4
        return alpha, beta
