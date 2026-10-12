"""Online (on-the-fly) training-data generation.

Fresh (theta, D) pairs every step -- effectively an infinite training set, so
there is no overfitting to a fixed grid (RESEARCH_METHODOLOGY.md 5.4).

Batch dict keys (single source of truth, consumed by the loss and trainer):
    X         : (B, n_max, k) float32, zero-padded
    obs_mask  : (B, n_max) bool
    labels    : (B, n_max) int64, -1 padded
    pi/alpha/beta, comp_mask, g : from params.padded_batch
"""
from __future__ import annotations

import numpy as np
import torch

from ..params import padded_batch
from ..simulation.mixture import sample
from ..simulation.priors import ThetaPrior
from ..utils.config import SimConfig


class OnlineGenerator:
    def __init__(self, prior: ThetaPrior, cfg: SimConfig, batch_size: int, fixed_g: int | None = None):
        self.prior = prior
        self.cfg = cfg
        self.batch_size = batch_size
        self.fixed_g = fixed_g
        self._bin_cycle = 0

    def _next_overlap_bin(self):
        bins = self.cfg.overlap_bins
        if not self.cfg.overlap_stratification or not bins:
            return None
        b = bins[self._bin_cycle % len(bins)]
        self._bin_cycle += 1
        return b

    def sample_batch(self, rng: np.random.Generator) -> dict:
        Xs, labels_list, thetas, ns = [], [], [], []
        for _ in range(self.batch_size):
            g = self.fixed_g if self.fixed_g is not None else self.prior.sample_g(rng)
            theta = self.prior.sample_stratified(rng, overlap_bin=self._next_overlap_bin(), g=g)
            n = self.prior.sample_n(rng)
            X, labels = sample(theta, n=n, rng=rng)
            Xs.append(X)
            labels_list.append(labels)
            thetas.append(theta)
            ns.append(n)
        B = len(Xs)
        n_max = max(ns)
        k = self.cfg.k
        X_pad = torch.zeros(B, n_max, k, dtype=torch.float32)
        obs_mask = torch.zeros(B, n_max, dtype=torch.bool)
        labels_pad = torch.full((B, n_max), -1, dtype=torch.long)
        for b, (X, labels) in enumerate(zip(Xs, labels_list)):
            n = X.shape[0]
            X_pad[b, :n] = torch.from_numpy(X).float()
            obs_mask[b, :n] = True
            labels_pad[b, :n] = torch.from_numpy(labels)
        batch = padded_batch(thetas, self.cfg.g_max)
        batch.update({"X": X_pad, "obs_mask": obs_mask, "labels": labels_pad})
        return batch
