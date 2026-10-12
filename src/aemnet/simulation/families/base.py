"""The `Family` interface: everything AEM-Net, the simulator, the baselines and
the evaluation code need to know about a component distribution.

Conventions (torch side)
------------------------
    X       : (B, n, k)  observations (padded entries arbitrary but finite)
    events  : (B, n, k)  1.0 = observed, 0.0 = right-censored at X
    params  : {name: (B, S, *shape)}  constrained component parameters
    u       : (B, S, d)  unconstrained component coordinates (`encode`/`decode`)
    log_prob(X, events, params) -> (B, n, S)

Canonicalisation (RESEARCH_METHODOLOGY.md 3.1)
----------------------------------------------
Each family declares its equivariance group:
    "scale"          : x_p -> c_p x_p         (gamma, Weibull, copula/Kibble gamma)
    "location_scale" : x_p -> l_p + c_p x_p   (Gaussian, Student-t, g-and-k)
    "none"           : no group               (counts)
Data are standardised as x~ = (x - loc) / scale with a robust statistic, the
network estimates canonical parameters, and `uncanonicalize` maps them back.
`canonicalize(params, loc, scale)` is the parameter action of x -> (x-loc)/scale.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from scipy.optimize import minimize

from ...mixture_params import MixtureParams

EPS = 1e-10
MAD_TO_SD = 1.4826


class Family(ABC):
    name: str = "abstract"
    equivariance: str = "none"          # "scale" | "location_scale" | "none"
    has_torch_logprob: bool = True      # False => likelihood-free family (E-block uses learned score only)
    exponential_family: bool = False    # True => suff_stats + inverse_link implemented (Thm 1)
    positive_support: bool = False
    discrete: bool = False
    independent_marginals: bool = True  # log_prob_marginals available
    mstep_maxiter: int = 60

    def __init__(self, k: int, **prior_kwargs):
        self.k = int(k)
        self.prior_kwargs = dict(prior_kwargs)

    # ------------------------------------------------------------------ spec
    @abstractmethod
    def param_shapes(self) -> dict[str, tuple[int, ...]]:
        """Per-component parameter shapes, e.g. {"alpha": (k,), "beta": (k,)}."""

    @property
    @abstractmethod
    def d(self) -> int:
        """Dimension of the unconstrained per-component coordinate u."""

    def spec(self) -> dict:
        """Serialisable description (stored in checkpoints)."""
        return {"name": self.name, "k": self.k, **self.prior_kwargs}

    # ------------------------------------------------------------------ coordinates
    @abstractmethod
    def decode(self, u: torch.Tensor) -> dict[str, torch.Tensor]:
        """u (..., d) -> constrained params {name: (..., *shape)}."""

    @abstractmethod
    def encode(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Inverse of `decode`."""

    def u_bounds(self) -> tuple[float, float]:
        return (-15.0, 15.0)

    # ------------------------------------------------------------------ density
    def log_prob_marginals(self, X, params) -> torch.Tensor:
        """(B, n, S, k) per-marginal log densities (independent-marginal families)."""
        raise NotImplementedError

    def log_sf_marginals(self, X, params) -> torch.Tensor:
        """(B, n, S, k) per-marginal log survival functions (for censoring)."""
        raise NotImplementedError

    def log_prob(self, X, events, params) -> torch.Tensor:
        if not self.has_torch_logprob:
            raise NotImplementedError(f"{self.name}: likelihood not available in torch")
        return self.log_prob_marginals(X, params).sum(-1)

    # ------------------------------------------------------------------ simulation
    @abstractmethod
    def sample_component(self, rng: np.random.Generator, comp: dict[str, np.ndarray], m: int) -> np.ndarray:
        """Draw m observations (m, k) from one component."""

    def sample_dataset(self, rng: np.random.Generator, mp: MixtureParams, n: int):
        """Returns X (n, k), events (n, k), labels (n,)."""
        labels = rng.choice(mp.g, size=n, p=mp.pi)
        X = np.empty((n, self.k), dtype=np.float64)
        for i in range(mp.g):
            idx = labels == i
            m = int(idx.sum())
            if m:
                X[idx] = self.sample_component(rng, mp.component(i), m)
        return X, np.ones_like(X), labels

    @abstractmethod
    def sample_prior(self, rng: np.random.Generator, g: int, **kw) -> dict[str, np.ndarray]:
        """Component parameters {name: (g, *shape)} from the prior (canonical-ish scale)."""

    # ------------------------------------------------------------------ canonicalisation
    def canonical_loc_scale(self, X: torch.Tensor, obs_mask: torch.Tensor):
        """Robust (loc, scale), each (B, k). Equivariant: (aX + b) -> (a loc + b, a scale)."""
        B, _, k = X.shape
        if self.equivariance == "none":
            return X.new_zeros(B, k), X.new_ones(B, k)
        Xn = X.masked_fill(~obs_mask.unsqueeze(-1), float("nan"))
        med = torch.nanquantile(Xn, 0.5, dim=1)
        if self.equivariance == "scale":
            return X.new_zeros(B, k), med.abs().clamp_min(EPS)
        mad = torch.nanquantile((Xn - med.unsqueeze(1)).abs(), 0.5, dim=1) * MAD_TO_SD
        return med, mad.clamp_min(EPS)

    def canonicalize(self, params, loc, scale):
        """Parameter action of x -> (x - loc)/scale. loc/scale: (B, k)."""
        return dict(params)

    def uncanonicalize(self, params, loc, scale):
        """Parameter action of x -> loc + scale * x."""
        return dict(params)

    # ------------------------------------------------------------------ features
    def features(self, X: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        """Per-point input features (B, n, F) on canonical data."""
        if self.positive_support:
            Xc = X.clamp_min(EPS)
            return torch.cat([Xc, Xc.log(), Xc.pow(1.0 / 3.0)], dim=-1)
        return torch.cat([X, torch.sign(X) * torch.log1p(X.abs()), X ** 2], dim=-1)

    @property
    def feature_dim(self) -> int:
        return 3 * self.k

    def suff_stats(self, X: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        """Known sufficient statistics T(z) (B, n, Fs); defaults to the features."""
        return self.features(X, events)

    @property
    def suff_dim(self) -> int:
        return self.feature_dim

    def inverse_link(self, mu: torch.Tensor) -> dict[str, torch.Tensor]:
        """(grad A)^{-1}: mean parameters -> constrained params. Exponential families only."""
        raise NotImplementedError

    def order_key(self, params) -> torch.Tensor:
        """Scalar per component used by ordering-constraint baselines (B, S)."""
        return self.encode(params)[..., 0]

    # ------------------------------------------------------------------ numpy helpers
    def to_torch(self, comps: dict[str, np.ndarray], dtype=torch.float64, device="cpu"):
        return {k: torch.as_tensor(v, dtype=dtype, device=device).unsqueeze(0) for k, v in comps.items()}

    def encode_np(self, comps: dict[str, np.ndarray]) -> np.ndarray:
        with torch.no_grad():
            return self.encode(self.to_torch(comps))[0].cpu().numpy()

    def decode_np(self, u: np.ndarray) -> dict[str, np.ndarray]:
        with torch.no_grad():
            p = self.decode(torch.as_tensor(np.atleast_2d(u), dtype=torch.float64).unsqueeze(0))
        return {k: v[0].cpu().numpy() for k, v in p.items()}

    def component_logpdf_np(self, X: np.ndarray, events: np.ndarray | None, mp: MixtureParams) -> np.ndarray:
        """(n, g) component log densities, float64."""
        X = np.asarray(X, dtype=np.float64)
        ev = np.ones_like(X) if events is None else np.asarray(events, dtype=np.float64)
        with torch.no_grad():
            lp = self.log_prob(torch.from_numpy(X)[None], torch.from_numpy(ev)[None], self.to_torch(mp.comps))
        return lp[0].cpu().numpy()

    def mixture_logpdf_np(self, X, events, mp: MixtureParams) -> np.ndarray:
        lp = self.component_logpdf_np(X, events, mp) + np.log(np.maximum(mp.pi, 1e-300))
        m = lp.max(1, keepdims=True)
        return m[:, 0] + np.log(np.exp(lp - m).sum(1))

    def responsibilities_np(self, X, events, mp: MixtureParams) -> np.ndarray:
        lp = self.component_logpdf_np(X, events, mp) + np.log(np.maximum(mp.pi, 1e-300))
        lp -= lp.max(1, keepdims=True)
        e = np.exp(lp)
        return e / e.sum(1, keepdims=True)

    def canonical_loc_scale_np(self, X: np.ndarray):
        with torch.no_grad():
            Xt = torch.as_tensor(X, dtype=torch.float64)[None]
            loc, scale = self.canonical_loc_scale(Xt, torch.ones(Xt.shape[:2], dtype=torch.bool))
        return loc[0].numpy(), scale[0].numpy()

    def _np_param_action(self, mp: MixtureParams, loc, scale, inverse: bool) -> MixtureParams:
        with torch.no_grad():
            p = self.to_torch(mp.comps)
            l = torch.as_tensor(loc, dtype=torch.float64)[None]
            s = torch.as_tensor(scale, dtype=torch.float64)[None]
            out = self.uncanonicalize(p, l, s) if inverse else self.canonicalize(p, l, s)
        return MixtureParams(mp.pi.copy(), {k: v[0].numpy() for k, v in out.items()})

    def canonicalize_np(self, mp, loc, scale) -> MixtureParams:
        return self._np_param_action(mp, loc, scale, inverse=False)

    def uncanonicalize_np(self, mp, loc, scale) -> MixtureParams:
        return self._np_param_action(mp, loc, scale, inverse=True)

    # ------------------------------------------------------------------ estimation helpers
    def init_features_np(self, X: np.ndarray) -> np.ndarray:
        """Space in which k-means++ initialisation of EM is run."""
        if self.positive_support:
            return np.log(np.maximum(X, EPS))
        if self.discrete:
            return np.log1p(np.maximum(X, 0))
        return np.asarray(X, dtype=np.float64)

    @abstractmethod
    def moment_init(self, X: np.ndarray, events: np.ndarray, w: np.ndarray) -> dict[str, np.ndarray]:
        """Crude weighted moment estimate of one component, {name: shape} (no g axis)."""

    def mstep(self, X: np.ndarray, events: np.ndarray, w: np.ndarray, init: dict | None = None) -> dict:
        """Weighted MLE for one component. Default: L-BFGS-B in u-space
        (finite-difference gradients, so it works for every family incl. censoring
        and copulas). Families with closed forms override this."""
        if init is None:
            init = self.moment_init(X, events, w)
        u0 = self.encode_np({k: np.asarray(v)[None] for k, v in init.items()})[0]
        u0 = np.clip(u0, *self.u_bounds())
        wsum = max(float(w.sum()), EPS)
        keep = w > 1e-8 * w.max() if w.size and w.max() > 0 else np.ones_like(w, dtype=bool)
        Xk, Ek, wk = X[keep], events[keep], w[keep]

        def negll(u):
            comps = self.decode_np(u[None])
            lp = self.component_logpdf_np(Xk, Ek, MixtureParams(np.ones(1), comps))[:, 0]
            val = -float((wk * lp).sum()) / wsum
            return val if np.isfinite(val) else 1e10

        lo, hi = self.u_bounds()
        res = minimize(negll, u0, method="L-BFGS-B", bounds=[(lo, hi)] * len(u0),
                       options={"maxiter": self.mstep_maxiter})
        u = res.x if np.isfinite(res.fun) and res.fun <= negll(u0) else u0
        return {k: v[0] for k, v in self.decode_np(u[None]).items()}
