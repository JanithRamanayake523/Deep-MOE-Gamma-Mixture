"""Baselines that live in external code bases (thin adapters).

These are deliberately *not* re-implemented from paper descriptions -- a
reviewer would rightly object to a weakened re-implementation. Each adapter
loads the official code and fails loudly with setup instructions otherwise.

  TGMMBaseline            Chen, Wu & Fang (2025), arXiv:2505.11918
                          code: https://github.com/Rorschach1989/transformer-for-gmm
                          set TGMM_REPO=<path to clone> and TGMM_CKPT=<checkpoint>
  ClusterPFNBaseline      Bhaskaran & Viering (2025), arXiv:2510.24318 (order selection)
                          set CLUSTER_PFN_REPO / CLUSTER_PFN_CKPT once code is released
  VBGaussianBaseline      variational Bayes GMM via scikit-learn (Gaussian family only)

Ordering-constrained amortised posteriors (ABMM, Kucharsky & Burkner 2026) and
flow NPE are covered *inside* this code base with an identical encoder:
train `model.arch=set_transformer model.flow_head=true loss.match=sorted`
(configs/train/baselines/npe_sorted_gamma_k2.yaml) and wrap with NeuralBaseline.
"""
from __future__ import annotations

import importlib
import os
import sys

import numpy as np

from ..mixture_params import MixtureParams
from ..simulation.families.base import Family
from .generic import GenericBaseline, GenericFit


def _import_from_repo(env_var: str, module: str):
    repo = os.environ.get(env_var)
    if not repo or not os.path.isdir(repo):
        raise ImportError(f"set {env_var} to a local clone of the official repository")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return importlib.import_module(module)


class TGMMBaseline(GenericBaseline):
    """Adapter for the official TGMM code. The expected entry point is a callable
    `predict(X: np.ndarray, k: int) -> (weights, means, covariances)`; point
    TGMM_ENTRY at `module:function` if the repository exposes it differently."""

    name = "tgmm"

    def __init__(self, family: Family, entry: str | None = None):
        if family.name != "gaussian":
            raise ValueError("TGMM is defined for Gaussian mixtures")
        super().__init__(family)
        self.entry = entry or os.environ.get("TGMM_ENTRY", "inference:predict")
        mod_name, fn_name = self.entry.split(":")
        self._predict = getattr(_import_from_repo("TGMM_REPO", mod_name), fn_name)

    def _fit(self, X, events, g, rng):
        w, mu, cov = self._predict(X, g)
        L = np.linalg.cholesky(np.asarray(cov) + 1e-6 * np.eye(X.shape[1]))
        mp = MixtureParams(np.asarray(w) / np.sum(w), {"mu": np.asarray(mu), "L": L})
        return GenericFit(mp, self.family.responsibilities_np(X, events, mp))


class ClusterPFNBaseline(GenericBaseline):
    """Adapter for Cluster-PFN (number of clusters + assignments). Parameters are
    obtained by one M-step on its MAP assignments so it can be scored on E1 metrics."""

    name = "cluster_pfn"
    supports_unknown_g = True

    def __init__(self, family: Family, entry: str | None = None):
        super().__init__(family)
        self.entry = entry or os.environ.get("CLUSTER_PFN_ENTRY", "cluster_pfn:predict")
        mod_name, fn_name = self.entry.split(":")
        self._predict = getattr(_import_from_repo("CLUSTER_PFN_REPO", mod_name), fn_name)

    def _fit(self, X, events, g, rng):
        labels = np.asarray(self._predict(X, g))
        uniq = np.unique(labels)
        comps, pis = [], []
        for c in uniq:
            w = (labels == c).astype(float)
            comps.append(self.family.mstep(X, events, w))
            pis.append(w.mean())
        mp = MixtureParams(np.asarray(pis), {k: np.stack([c[k] for c in comps]) for k in comps[0]})
        return GenericFit(mp, self.family.responsibilities_np(X, events, mp), extra={"labels": labels})


class VBGaussianBaseline(GenericBaseline):
    name = "vb"
    supports_unknown_g = True

    def __init__(self, family: Family, g_max: int = 6, n_init: int = 5):
        if family.name != "gaussian":
            raise ValueError("VBGaussianBaseline is Gaussian-only")
        super().__init__(family)
        self.g_max, self.n_init = g_max, n_init

    def _fit(self, X, events, g, rng):
        try:
            from sklearn.mixture import BayesianGaussianMixture
        except ImportError as exc:  # pragma: no cover
            raise ImportError("VBGaussianBaseline needs scikit-learn") from exc
        m = BayesianGaussianMixture(n_components=g or self.g_max, n_init=self.n_init,
                                    random_state=int(rng.integers(2 ** 31)), max_iter=1000).fit(X)
        keep = np.arange(len(m.weights_)) if g else np.where(m.weights_ > 0.01)[0]
        L = np.linalg.cholesky(m.covariances_[keep] + 1e-9 * np.eye(X.shape[1]))
        mp = MixtureParams(m.weights_[keep] / m.weights_[keep].sum(), {"mu": m.means_[keep], "L": L})
        return GenericFit(mp, self.family.responsibilities_np(X, events, mp))
