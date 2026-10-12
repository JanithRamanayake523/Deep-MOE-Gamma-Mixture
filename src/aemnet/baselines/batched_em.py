"""Batched EM on the GPU -- the fair "many datasets" comparator for H7 / E7.

Running EM one dataset at a time in numpy would make the amortisation speed-up
look larger than it is, so this baseline runs EM for *many datasets and restarts at
once* in torch, for exponential families with an implemented inverse link (gamma,
Gaussian). One iteration is exactly `models.exact.ExactEMBlock` (Thm 1), so the
same code path serves the theory check and the baseline.

Initialisation per dataset: one quantile split on the first sufficient statistic
(sorted log x / x into g equal groups) + (n_restarts - 1) random Dirichlet soft
assignments; the restart with the highest log-likelihood is kept. `refine` starts
from given parameters instead (batched hybrid: AEM-Net start + batched EM).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ..mixture_params import MixtureParams
from ..models.aem import MBlock
from ..models.exact import ExactEMBlock
from ..simulation.families.base import Family
from ..simulation.simulator import collate
from .generic import GenericBaseline, GenericFit


class BatchedEM(GenericBaseline):
    name = "batched_em"

    def __init__(self, family: Family, n_restarts: int = 10, max_iter: int = 500, tol: float = 1e-8,
                 device: str | None = None, batch_size: int = 512, dtype=torch.float64):
        if not family.exponential_family:
            raise ValueError(f"BatchedEM needs an exponential family with inverse_link; got {family.name}")
        super().__init__(family)
        self.n_restarts, self.max_iter, self.tol = n_restarts, max_iter, tol
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size, self.dtype = batch_size, dtype
        self.block = ExactEMBlock(family).to(self.device, dtype)

    # ------------------------------------------------------------------ core
    def _loglik(self, X, ev, mask, params, pi):
        lp = self.family.log_prob(X, ev, params) + pi.clamp_min(1e-300).log().unsqueeze(1)
        return (torch.logsumexp(lp, -1) * mask.to(X.dtype)).sum(1)

    def _run(self, X, ev, mask, params, pi):
        """EM to convergence for a padded batch. Returns params, pi, loglik (B,), iterations (B,)."""
        B = X.shape[0]
        ll = self._loglik(X, ev, mask, params, pi)
        iters = torch.zeros(B, dtype=torch.long, device=X.device)
        active = torch.ones(B, dtype=torch.bool, device=X.device)
        for _ in range(self.max_iter):
            new_p, new_pi, _ = self.block(X, ev, mask, params, pi)
            new_pi = new_pi.clamp_min(1e-10)
            new_pi = new_pi / new_pi.sum(-1, keepdim=True)
            new_ll = self._loglik(X, ev, mask, new_p, new_pi)
            ok = torch.isfinite(new_ll) & active
            for k in params:
                sel = ok.view(-1, *([1] * (params[k].dim() - 1)))
                params[k] = torch.where(sel, new_p[k], params[k])
            pi = torch.where(ok.unsqueeze(-1), new_pi, pi)
            gain = new_ll - ll
            ll = torch.where(ok, new_ll, ll)
            iters = iters + active.long()
            active = active & ok & (gain.abs() > self.tol * (1 + ll.abs()))
            if not active.any():
                break
        return params, pi, ll, iters

    def _init_from_resp(self, X, ev, mask, R):
        ss = self.family.suff_stats(X, ev)
        m, _, pi = MBlock.pool(R, ss, mask)
        return self.family.inverse_link(m), pi.clamp_min(1e-6) / pi.clamp_min(1e-6).sum(-1, keepdim=True)

    def _batch(self, Xs):
        data = [{"X": np.asarray(x, float).reshape(len(x), -1), "events": None, "labels": None, "params": None}
                for x in Xs]
        for d in data:
            d["events"] = np.ones_like(d["X"])
        b = collate(data, self.family, 1)
        X = b["X"].to(self.device, self.dtype)
        # pad rows with a valid in-support value (masked out of every sum)
        fill = 1.0 if self.family.positive_support else 0.0
        X = torch.where(b["obs_mask"].to(self.device).unsqueeze(-1), X, torch.full_like(X, fill))
        return X, b["events"].to(self.device, self.dtype), b["obs_mask"].to(self.device)

    def _to_mp(self, params, pi, b) -> MixtureParams:
        p = pi[b].double().cpu().numpy()
        return MixtureParams(p / p.sum(), {k: v[b].double().cpu().numpy() for k, v in params.items()})

    # ------------------------------------------------------------------ public
    @torch.no_grad()
    def fit_many(self, Xs: list, g: int, seed: int = 0) -> tuple[list[MixtureParams], dict]:
        rng = np.random.default_rng(seed)                     # restarts are reproducible given the seed
        out, iters_all, t0 = [None] * len(Xs), [], time.perf_counter()
        order = np.argsort([len(x) for x in Xs])
        for s in range(0, len(Xs), self.batch_size):
            ids = order[s:s + self.batch_size]
            X, ev, mask = self._batch([Xs[i] for i in ids])
            B, n, _ = X.shape
            R_ = self.n_restarts
            Xr, evr, mr = (t.repeat_interleave(R_, 0) for t in (X, ev, mask))
            # restart 0: quantile split on the first sufficient statistic; others: random soft assignments
            key = self.family.suff_stats(X, ev)[..., -1 if self.family.positive_support else 0]
            key = key.masked_fill(~mask, float("inf"))
            rank = key.argsort(1).argsort(1).to(X.dtype)
            nobs = mask.sum(1, keepdim=True).to(X.dtype)
            lab = (rank / nobs * g).floor().clamp(max=g - 1).long()
            R0 = torch.nn.functional.one_hot(lab, g).to(X.dtype) * 0.9 + 0.1 / g
            Rr = torch.as_tensor(rng.dirichlet(np.ones(g), size=(B * R_, n)), device=self.device, dtype=X.dtype)
            Rr = Rr.view(B, R_, n, g)
            Rr[:, 0] = R0
            Rr = Rr.view(B * R_, n, g) * mr.unsqueeze(-1).to(X.dtype)
            params, pi = self._init_from_resp(Xr, evr, mr, Rr)
            params, pi, ll, iters = self._run(Xr, evr, mr, params, pi)
            best = ll.view(B, R_).argmax(1)
            for j, i in enumerate(ids):
                r = j * R_ + int(best[j])
                out[i] = self._to_mp(params, pi, r)
            iters_all.append(iters.view(B, R_).sum(1).cpu())
        return out, {"seconds": time.perf_counter() - t0, "iterations": torch.cat(iters_all).numpy()}

    @torch.no_grad()
    def refine(self, Xs: list, inits: list[MixtureParams]) -> tuple[list[MixtureParams], dict]:
        """Batched hybrid: EM from given starts (e.g. AEM-Net estimates), one start per dataset."""
        t0 = time.perf_counter()
        out, iters_all = [None] * len(Xs), []
        for s in range(0, len(Xs), self.batch_size):
            ids = list(range(s, min(s + self.batch_size, len(Xs))))
            X, ev, mask = self._batch([Xs[i] for i in ids])
            params = {k: torch.as_tensor(np.stack([inits[i].comps[k] for i in ids]), device=self.device,
                                         dtype=self.dtype) for k in inits[ids[0]].comps}
            pi = torch.as_tensor(np.stack([inits[i].pi for i in ids]), device=self.device, dtype=self.dtype)
            params, pi, _, iters = self._run(X, ev, mask, params, pi)
            for j, i in enumerate(ids):
                out[i] = self._to_mp(params, pi, j)
            iters_all.append(iters.cpu())
        return out, {"seconds": time.perf_counter() - t0, "iterations": torch.cat(iters_all).numpy()}

    def _fit(self, X, events, g, rng):
        (mp,), info = self.fit_many([X], g, seed=int(rng.integers(2 ** 31)))
        return GenericFit(mp, self.family.responsibilities_np(X, events, mp),
                          extra={"em_steps_total": int(info["iterations"][0])})
