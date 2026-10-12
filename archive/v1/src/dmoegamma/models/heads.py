"""Block D (optional): model-order head, predicting g in {1, ..., g_max}.

At test time experts beyond g_hat are masked (RESEARCH_METHODOLOGY.md 5.1,
Block D; evaluated under RQ3 against BIC-based selection).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.config import ModelConfig
from .encoder import build_mlp


class ModelOrderHead(nn.Module):
    def __init__(self, cfg: ModelConfig, g_max: int):
        super().__init__()
        self.head = build_mlp(cfg.enc_out, cfg.head_hidden, g_max, cfg.head_layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Logits (B, g_max); class index i corresponds to g = i + 1."""
        return self.head(h)
