"""User-facing amortised estimator: load a checkpoint, estimate one or many
datasets, draw quotient-space posterior samples, or run the hybrid
(amortised start + exact EM) estimator.

    est = AmortizedEstimator.from_checkpoint("runs/gamma_k2/best.pt")
    mp, resp = est.estimate(X)               # amortised, g chosen by the existence head
    mp, resp = est.estimate(X, g=2)          # known g
    mp, info = est.hybrid(X, g=2)            # + ceil(log2 n) EM steps (Thm 4)
    samples  = est.posterior(X, g=2, n_samples=1000)   # list[MixtureParams], pivot-aligned
"""
from __future__ import annotations

import numpy as np
import torch

from ..mixture_params import MixtureParams
from ..models.factory import build_model
from ..simulation.families import family_from_spec
from ..simulation.simulator import collate
from ..utils.config import load_experiment
from .refine import em_refine


def load_checkpoint(path, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    family = family_from_spec(ckpt["family_spec"])
    exp = load_experiment(raw=ckpt["experiment"])
    model = build_model(family, exp.model, ckpt["n_slots"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), family, exp


class AmortizedEstimator:
    def __init__(self, model, device: str = "cpu", exist_threshold: float = 0.5, min_pi: float = 0.01):
        self.model = model.to(device).eval()
        self.family = model.family
        self.device = device
        self.exist_threshold = exist_threshold
        self.min_pi = min_pi

    @classmethod
    def from_checkpoint(cls, path, device: str = "cpu", **kw) -> "AmortizedEstimator":
        model, _, _ = load_checkpoint(path, device)
        return cls(model, device, **kw)

    # ------------------------------------------------------------------ low level
    def _batch(self, Xs, events_list=None) -> dict:
        data = []
        for i, X in enumerate(Xs):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X[:, None]
            ev = np.ones_like(X) if events_list is None or events_list[i] is None else np.asarray(events_list[i])
            data.append({"X": X, "events": ev, "labels": None, "params": None})
        b = collate(data, self.family, self.model.S)
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in b.items()}

    @torch.no_grad()
    def forward(self, Xs, events_list=None, T: int | None = None) -> tuple[dict, dict]:
        batch = self._batch(Xs, events_list)
        kw = {} if T is None else {"T": T}
        out = self.model(batch["X"], batch["events"], batch["obs_mask"], **kw)
        return out, batch

    def select_slots(self, out: dict, b: int, g: int | None = None) -> np.ndarray:
        pi = out["pi"][b].float().cpu().numpy()
        ex = out.get("exist_logits")
        score = torch.sigmoid(ex[b]).float().cpu().numpy() if ex is not None else pi
        if g is not None:
            return np.sort(np.argsort(-score)[:g])
        if ex is not None:
            idx = np.where(score > self.exist_threshold)[0]
        else:
            idx = np.where(pi > self.min_pi)[0]
        return idx if len(idx) else np.array([int(np.argmax(score))])

    def to_mixture(self, out: dict, b: int, idx: np.ndarray) -> MixtureParams:
        pi = out["pi"][b, idx].double().cpu().numpy()
        pi = np.maximum(pi, 1e-12)
        comps = {k: v[b, idx].double().cpu().numpy() for k, v in out["params"].items()}
        return MixtureParams(pi / pi.sum(), comps)

    def _resp(self, out, b, idx, n):
        if out.get("resp") is None:
            return None
        r = out["resp"][b, :n][:, idx].double().cpu().numpy()
        return r / np.maximum(r.sum(1, keepdims=True), 1e-300)

    # ------------------------------------------------------------------ public API
    def estimate(self, X, events=None, g: int | None = None, T: int | None = None):
        out, _ = self.forward([X], None if events is None else [events], T)
        idx = self.select_slots(out, 0, g)
        return self.to_mixture(out, 0, idx), self._resp(out, 0, idx, np.asarray(X).shape[0])

    def estimate_many(self, Xs, events_list=None, g: int | None = None, batch_size: int = 256,
                      T: int | None = None) -> list[MixtureParams]:
        """Amortised inference over many datasets (sorted by size to limit padding)."""
        order = np.argsort([np.asarray(X).shape[0] for X in Xs])
        results: list = [None] * len(Xs)
        for start in range(0, len(Xs), batch_size):
            ids = order[start:start + batch_size]
            evs = None if events_list is None else [events_list[i] for i in ids]
            out, _ = self.forward([Xs[i] for i in ids], evs, T)
            for j, i in enumerate(ids):
                results[i] = self.to_mixture(out, j, self.select_slots(out, j, g))
        return results

    @torch.no_grad()
    def posterior(self, X, events=None, g: int | None = None, n_samples: int = 1000,
                  steps: int | None = None) -> list[MixtureParams]:
        """Posterior samples in the slots selected for the point estimate. Samples are
        pivot-aligned by construction (the flow was trained on targets aligned to the
        point estimate), so per-slot summaries are valid on the quotient space (Prop 2)."""
        out, _ = self.forward([X], None if events is None else [events])
        idx = self.select_slots(out, 0, g)
        z, pi = self.model.sample_posterior(out, n_samples, steps)          # (N,1,S,d), (N,1,S)
        params_c = self.family.decode(self.model.z_to_u(z[:, 0]))           # (N,S,...)
        loc = out["loc"].expand(n_samples, -1)
        scale = out["scale"].expand(n_samples, -1)
        params = self.family.uncanonicalize(params_c, loc, scale)
        pis = pi[:, 0, idx].double().cpu().numpy()
        pis = pis / pis.sum(1, keepdims=True)
        comps = {k: v[:, idx].double().cpu().numpy() for k, v in params.items()}
        return [MixtureParams(pis[s], {k: v[s] for k, v in comps.items()}) for s in range(n_samples)]

    def hybrid(self, X, events=None, g: int | None = None, max_steps: int | None = None, tol: float = 1e-8):
        """Amortised estimate followed by EM (Thm 4). Every family provides a numpy
        density (g-and-k via numerical inversion), so this runs for all of them; for
        likelihood-free use prefer learned refinement: estimate(..., T=larger)."""
        mp, _ = self.estimate(X, events, g)
        return em_refine(self.family, X, events, mp, max_steps=max_steps, tol=tol)
