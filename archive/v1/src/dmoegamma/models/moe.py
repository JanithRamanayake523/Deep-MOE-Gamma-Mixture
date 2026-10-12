"""The assembled Deep-MoE estimator (Blocks A-D, RESEARCH_METHODOLOGY.md 5.1).

phi: (R_+^k)^n -> Theta, a permutation-invariant set function. Trained with the
composite label-switching-aware loss in dmoegamma.losses.composite.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..params import Theta
from ..utils.config import ModelConfig
from .encoder import DeepSetsEncoder
from .experts import ExpertHeads
from .gate import GateNet
from .heads import ModelOrderHead


class DeepMoEGammaEstimator(nn.Module):
    def __init__(self, cfg: ModelConfig, k: int, g_max: int):
        super().__init__()
        self.cfg = cfg
        self.k = k
        self.g_max = g_max
        self.encoder = DeepSetsEncoder(cfg, k)
        self.gate = GateNet(cfg, g_max)
        self.experts = ExpertHeads(cfg, g_max, k)
        self.g_head = ModelOrderHead(cfg, g_max) if cfg.use_g_head else None

    def forward(self, X: torch.Tensor, obs_mask: torch.Tensor) -> dict:
        """X : (B, n, k), obs_mask : (B, n) bool. Returns a dict with
            pi       : (B, g_max)      softmax mixing weights
            alpha    : (B, g_max, k)   softplus shapes
            beta     : (B, g_max, k)   softplus scales
            resp     : (B, n, g_max)   learned responsibilities
            g_logits : (B, g_max) or None (model-order head)
            h        : (B, enc_out)    sample embedding
        """
        h, rho = self.encoder(X, obs_mask)
        pi, resp = self.gate(h, rho)
        alpha, beta = self.experts(h)
        g_logits = self.g_head(h) if self.g_head is not None else None
        return {"pi": pi, "alpha": alpha, "beta": beta, "resp": resp,
                "g_logits": g_logits, "h": h}

    @torch.no_grad()
    def estimate(
        self, X: np.ndarray, g: int | None = None, device: str = "cpu"
    ) -> tuple[Theta, np.ndarray]:
        """Point estimate from a numpy sample (n, k).

        If g is None, uses the model-order head (requires use_g_head=True).
        Returns (theta_hat, responsibilities (n, g_used)).
        """
        self.eval()
        Xt = torch.as_tensor(X, dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.ones(1, Xt.shape[1], dtype=torch.bool, device=device)
        out = self(Xt.to(device), mask)
        if g is None:
            if out["g_logits"] is None:
                raise ValueError("g=None requires a model-order head")
            g = int(out["g_logits"].argmax(dim=-1).item()) + 1
        pi = out["pi"][0, :g].cpu().numpy().astype(np.float64)
        pi = pi / pi.sum()
        alpha = out["alpha"][0, :g].cpu().numpy().astype(np.float64)
        beta = out["beta"][0, :g].cpu().numpy().astype(np.float64)
        resp = out["resp"][0, :, :g].cpu().numpy()
        theta = Theta(pi, alpha, beta).sorted_by_mean()
        return theta, resp


def load_estimator(path: str, device: str = "cpu") -> "DeepMoEGammaEstimator":
    """Rebuild a trained estimator from a checkpoint written by Trainer."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**(ckpt["model_config"] or {}))
    sim_cfg = ckpt.get("sim_config") or {}
    model = DeepMoEGammaEstimator(
        model_cfg, k=int(sim_cfg.get("k", 2)), g_max=int(sim_cfg.get("g_max", 4))
    )
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()
