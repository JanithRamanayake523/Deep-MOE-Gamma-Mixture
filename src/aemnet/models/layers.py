"""Set-attention building blocks (Lee et al., 2019, Set Transformer), pre-norm.

Masks follow the convention mask[b, j] = True for a *valid* element.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def mlp(in_dim: int, hidden: int, out_dim: int, n_hidden: int = 2, act=nn.GELU, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(max(n_hidden, 0)):
        layers += [nn.Linear(d, hidden), act()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x (B, n, H), mask (B, n) -> (B, H)."""
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp_min(1.0)


class MAB(nn.Module):
    """Multihead attention block: H = Q + Att(LN Q, LN K); out = H + FF(LN H)."""

    def __init__(self, dim: int, heads: int, dim_kv: int | None = None, ff_mult: int = 2, dropout: float = 0.0):
        super().__init__()
        dim_kv = dim_kv or dim
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim_kv)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True, kdim=dim_kv, vdim=dim_kv)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, ff_mult * dim), nn.GELU(),
                                nn.Linear(ff_mult * dim, dim))

    def forward(self, Q: torch.Tensor, K: torch.Tensor, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        kv = self.kv_norm(K)
        pad = None if key_mask is None else ~key_mask
        h = Q + self.attn(self.q_norm(Q), kv, kv, key_padding_mask=pad, need_weights=False)[0]
        return h + self.ff(h)


class SAB(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int = 2):
        super().__init__()
        self.mab = MAB(dim, heads, ff_mult=ff_mult)

    def forward(self, X, mask=None):
        return self.mab(X, X, mask)


class ISAB(nn.Module):
    """Induced set attention: O(n m) instead of O(n^2)."""

    def __init__(self, dim: int, heads: int, n_inducing: int = 16, ff_mult: int = 2):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, n_inducing, dim) * dim ** -0.5)
        self.mab1 = MAB(dim, heads, ff_mult=ff_mult)
        self.mab2 = MAB(dim, heads, ff_mult=ff_mult)

    def forward(self, X, mask=None):
        H = self.mab1(self.inducing.expand(X.shape[0], -1, -1), X, mask)
        return self.mab2(X, H)


class PMA(nn.Module):
    """Pooling by multihead attention onto S learned seed vectors."""

    def __init__(self, dim: int, heads: int, n_seeds: int, ff_mult: int = 2):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, dim) * dim ** -0.5)
        self.mab = MAB(dim, heads, ff_mult=ff_mult)

    def forward(self, X, mask=None, seeds: torch.Tensor | None = None):
        S = self.seeds.expand(X.shape[0], -1, -1) if seeds is None else seeds
        return self.mab(S, X, mask)
