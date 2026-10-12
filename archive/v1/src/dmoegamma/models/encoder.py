"""Block A: permutation-invariant sample encoder (DeepSets).

Each observation z_j is embedded by an MLP rho(.), pooled by a permutation-
invariant reduction, optionally fused with classical summary statistics, and
mapped to a sample embedding h by a second MLP phi.

Domain-knowledge feature engineering (RESEARCH_METHODOLOGY.md 5.1): inputs are
augmented with Wilson-Hilferty cube-root features z^(1/3) and log z. These are
*features, not an approximation crutch* -- the network is free to ignore them.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.config import ModelConfig

_EPS = 1e-8


def build_mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int) -> nn.Sequential:
    """Plain MLP with ReLU activations; `n_layers` counts hidden layers."""
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(max(n_layers, 0)):
        layers += [nn.Linear(dim, hidden), nn.ReLU()]
        dim = hidden
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class DeepSetsEncoder(nn.Module):
    """Maps a padded set X : (B, n, k) to (h, rho) with h : (B, enc_out)."""

    N_STATS_PER_DIM = 8  # mean, std, min, max, q25, q50, q75, mean(log x)

    def __init__(self, cfg: ModelConfig, k: int):
        super().__init__()
        if cfg.pooling not in ("mean", "sum", "meansum"):
            raise ValueError(f"unknown pooling: {cfg.pooling}")
        self.cfg = cfg
        self.k = k
        self.in_feats = 3 * k if cfg.use_wh_features else k
        self.rho = build_mlp(self.in_feats, cfg.enc_hidden, cfg.enc_out, cfg.enc_layers)
        self.n_stats = self.N_STATS_PER_DIM * k if cfg.use_summary_stats else 0
        pooled_dim = cfg.enc_out * (2 if cfg.pooling == "meansum" else 1)
        self.phi = build_mlp(pooled_dim + self.n_stats, cfg.enc_hidden, cfg.enc_out, 2)

    def _features(self, X: torch.Tensor) -> torch.Tensor:
        if not self.cfg.use_wh_features:
            return X
        Xc = X.clamp_min(_EPS)
        return torch.cat([Xc, Xc.pow(1.0 / 3.0), Xc.log()], dim=-1)

    def _summary_stats(self, X: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        """Classical per-marginal summary statistics, masked. Returns (B, 8k)."""
        m = obs_mask.unsqueeze(-1).float()                    # (B, n, 1)
        count = m.sum(dim=1).clamp_min(1.0)                   # (B, 1)
        Xc = X.clamp_min(_EPS)
        mean = (Xc * m).sum(1) / count                        # (B, k)
        var = ((Xc - mean.unsqueeze(1)) ** 2 * m).sum(1) / count
        std = var.clamp_min(0.0).sqrt()
        logmean = (Xc.log() * m).sum(1) / count
        Xnan = Xc.masked_fill(~obs_mask.unsqueeze(-1), float("nan"))
        q = torch.tensor([0.25, 0.5, 0.75], device=X.device, dtype=X.dtype)
        qs = torch.nanquantile(Xnan, q, dim=1)                # (3, B, k)
        qs = qs.permute(1, 2, 0).flatten(1)                   # (B, 3k)
        valid = obs_mask.unsqueeze(-1)
        mn = Xc.masked_fill(~valid, float("inf")).min(dim=1).values
        mx = Xc.masked_fill(~valid, float("-inf")).max(dim=1).values
        return torch.cat([mean, std, mn, mx, qs, logmean], dim=-1)

    def forward(self, X: torch.Tensor, obs_mask: torch.Tensor):
        rho = self.rho(self._features(X))                     # (B, n, enc_out)
        rho = rho * obs_mask.unsqueeze(-1).float()
        n_obs = obs_mask.sum(1, keepdim=True).clamp_min(1).float()
        if self.cfg.pooling == "mean":
            pooled = rho.sum(1) / n_obs
        elif self.cfg.pooling == "sum":
            pooled = rho.sum(1)
        else:  # meansum
            pooled = torch.cat([rho.sum(1) / n_obs, rho.sum(1)], dim=-1)
        parts = [pooled]
        if self.cfg.use_summary_stats:
            parts.append(self._summary_stats(X, obs_mask))
        h = self.phi(torch.cat(parts, dim=-1))
        return h, rho
