"""Family-agnostic simulator and online training stream (RESEARCH_METHODOLOGY.md 6.2).

Datasets are drawn as
    g ~ U{g_min..g_max},  n ~ LogUniform(n_min, n_max),
    pi ~ Dirichlet(gamma 1),  gamma ~ LogUniform(dirichlet_conc_range),
    theta_1..g ~ family prior,  (optionally) accept only if the Monte-Carlo Bayes
    error falls in a randomly chosen overlap bin,
    D ~ mixture;  then a random location/scale transform is applied to (D, theta)
    (`scale_augment`), which AEM-Net's canonicalisation makes irrelevant but keeps
    the baselines honest about arbitrary data scales.

Batch dict (flat; tensors unless noted):
    X (B,n,k) float32, events (B,n,k), obs_mask (B,n) bool, labels (B,n) long (-1 pad),
    comp_mask (B,G) bool, g (B,), tp_pi (B,G), tp_<name> (B,G,*shape) true params
    (original data scale, padded by repeating component 0), params (list[MixtureParams]).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..mixture_params import MixtureParams
from .families.base import Family


@dataclass
class SimulatorConfig:
    g_min: int = 1
    g_max: int = 4
    n_min: int = 20
    n_max: int = 5000
    n_log_uniform: bool = True
    same_n_per_batch: bool = True
    dirichlet_conc_range: tuple[float, float] = (0.5, 3.0)
    min_weight: float = 0.02
    overlap_stratification: bool = True
    overlap_bins: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.05), (0.05, 0.20), (0.20, 0.50)])
    overlap_n_mc: int = 2000
    overlap_max_tries: int = 40
    scale_augment: bool = True
    log_scale_range: float = 4.6          # scale ~ exp(U(-r, r)) per marginal (x100 either way)
    loc_range: float = 5.0                # location-scale families: loc ~ U(-L, L) * scale
    shape_regime: str | None = None       # gamma-only: None | "large" | "small"
    param_bank: str | None = None         # path to a pre-computed overlap-labelled bank (simulation/bank.py)


def bayes_error(family: Family, mp: MixtureParams, n_mc: int = 2000, rng=None) -> float:
    """Monte-Carlo Bayes misclassification rate of the true classifier (overlap measure)."""
    if mp.g == 1:
        return 0.0
    rng = np.random.default_rng(rng)
    X, ev, labels = family.sample_dataset(rng, mp, n_mc)
    pred = family.responsibilities_np(X, ev, mp).argmax(1)
    return float(np.mean(pred != labels))


class MixtureSimulator:
    def __init__(self, family: Family, cfg: SimulatorConfig | None = None, split: str = "train"):
        self.family = family
        self.cfg = cfg or SimulatorConfig()
        self.split = split            # "train" | "val": which part of the parameter bank to draw from
        self._bank = None
        self._bank_checked = False

    def with_split(self, split: str) -> "MixtureSimulator":
        return MixtureSimulator(self.family, self.cfg, split)

    @property
    def bank(self):
        """Lazily loaded parameter bank (None if not configured / missing / incompatible)."""
        if not self._bank_checked:
            self._bank_checked = True
            path = self.cfg.param_bank
            if path:
                from pathlib import Path

                from .bank import ParameterBank

                if not Path(path).exists():
                    warnings.warn(f"parameter bank '{path}' not found: falling back to rejection sampling "
                                  f"(slow; build it with scripts/build_param_bank.py)")
                else:
                    bank = ParameterBank.load(path)
                    if bank.compatible(self.family, self.cfg):
                        self._bank = bank
                    else:
                        warnings.warn(f"parameter bank '{path}' was built for a different family/simulator "
                                      f"config: ignoring it")
        return self._bank

    def __getstate__(self):  # keep DataLoader worker pickles small; each worker reloads the bank
        st = dict(self.__dict__)
        st["_bank"], st["_bank_checked"] = None, False
        return st

    # ------------------------------------------------------------------ draws
    def sample_g(self, rng) -> int:
        return int(rng.integers(self.cfg.g_min, self.cfg.g_max + 1))

    def sample_n(self, rng) -> int:
        c = self.cfg
        if c.n_log_uniform:
            return int(round(np.exp(rng.uniform(np.log(c.n_min), np.log(c.n_max)))))
        return int(rng.integers(c.n_min, c.n_max + 1))

    def _raw_params(self, rng, g: int) -> MixtureParams:
        c = self.cfg
        for _ in range(100):
            conc = np.exp(rng.uniform(*np.log(c.dirichlet_conc_range)))
            pi = rng.dirichlet(np.full(g, conc))
            if pi.min() >= c.min_weight or g == 1:
                break
        pi = np.maximum(pi, 1e-6)
        pi = pi / pi.sum()
        kw = {"shape_regime": c.shape_regime} if c.shape_regime else {}
        return MixtureParams(pi, self.family.sample_prior(rng, g, **kw))

    def sample_params(self, rng, g: int | None = None, overlap_bin=None) -> tuple[MixtureParams, float]:
        """Returns (params, bayes_error or nan). Canonical (prior) scale."""
        c = self.cfg
        g = self.sample_g(rng) if g is None else g
        if overlap_bin is None and c.overlap_stratification and c.overlap_bins and g > 1:
            overlap_bin = c.overlap_bins[rng.integers(len(c.overlap_bins))]
        if overlap_bin is None or g == 1:
            return self._raw_params(rng, g), float("nan")
        bins = [tuple(b) for b in (c.overlap_bins or [])]
        if self.bank is not None and tuple(overlap_bin) in bins:
            b = bins.index(tuple(overlap_bin))
            if self.bank.has(g, b):
                return self.bank.draw(rng, g, b, self.split)
        lo, hi = overlap_bin
        mp = None
        for _ in range(c.overlap_max_tries):
            mp = self._raw_params(rng, g)
            err = bayes_error(self.family, mp, c.overlap_n_mc, rng)
            if lo <= err < hi:
                return mp, err
        return mp, float("nan")  # fall back (never loops forever)

    def augment(self, rng, X: np.ndarray, mp: MixtureParams):
        """Random data-scale transform applied consistently to data and parameters."""
        fam, c = self.family, self.cfg
        if not c.scale_augment or fam.equivariance == "none":
            return X, mp
        k = fam.k
        scale = np.exp(rng.uniform(-c.log_scale_range, c.log_scale_range, size=k))
        loc = np.zeros(k)
        if fam.equivariance == "location_scale":
            loc = rng.uniform(-c.loc_range, c.loc_range, size=k) * scale
        return loc + scale * X, fam.uncanonicalize_np(mp, loc, scale)

    def sample_dataset(self, rng, g: int | None = None, n: int | None = None, overlap_bin=None) -> dict:
        mp, err = self.sample_params(rng, g, overlap_bin)
        n = self.sample_n(rng) if n is None else n
        X, events, labels = self.family.sample_dataset(rng, mp, n)
        X, mp = self.augment(rng, X, mp)
        return {"X": X, "events": events, "labels": labels, "params": mp, "bayes_error": err}

    def sample_batch(self, rng, batch_size: int, g: int | None = None, n: int | None = None) -> dict:
        if n is None and self.cfg.same_n_per_batch:
            n = self.sample_n(rng)
        data = [self.sample_dataset(rng, g=g, n=n) for _ in range(batch_size)]
        return collate(data, self.family, self.cfg.g_max)


def collate(datasets: list[dict], family: Family, g_max: int) -> dict:
    """Pad a list of simulated datasets into a batch dict (see module docstring)."""
    B = len(datasets)
    k = family.k
    n_max = max(d["X"].shape[0] for d in datasets)
    X = torch.zeros(B, n_max, k)
    events = torch.ones(B, n_max, k)
    obs_mask = torch.zeros(B, n_max, dtype=torch.bool)
    labels = torch.full((B, n_max), -1, dtype=torch.long)
    comp_mask = torch.zeros(B, g_max, dtype=torch.bool)
    g = torch.zeros(B, dtype=torch.long)
    tp_pi = torch.zeros(B, g_max)
    shapes = family.param_shapes()
    tp = {name: torch.zeros(B, g_max, *shp) for name, shp in shapes.items()}
    for b, d in enumerate(datasets):
        n = d["X"].shape[0]
        X[b, :n] = torch.from_numpy(d["X"]).float()
        events[b, :n] = torch.from_numpy(np.asarray(d["events"])).float()
        obs_mask[b, :n] = True
        if d.get("labels") is not None:
            labels[b, :n] = torch.from_numpy(np.asarray(d["labels"])).long()
        mp: MixtureParams | None = d.get("params")
        if mp is None:
            continue
        gb = mp.g
        if gb > g_max:
            raise ValueError(f"dataset has g={gb} > g_max={g_max}")
        g[b] = gb
        comp_mask[b, :gb] = True
        tp_pi[b, :gb] = torch.from_numpy(mp.pi).float()
        for name in shapes:
            v = torch.from_numpy(mp.comps[name]).float()
            tp[name][b, :gb] = v
            tp[name][b, gb:] = v[0]
    batch = {"X": X, "events": events, "obs_mask": obs_mask, "labels": labels,
             "comp_mask": comp_mask, "g": g, "tp_pi": tp_pi,
             "params": [d.get("params") for d in datasets]}
    batch.update({f"tp_{name}": v for name, v in tp.items()})
    return batch


def target_params(batch: dict, family: Family) -> dict[str, torch.Tensor]:
    return {name: batch[f"tp_{name}"] for name in family.param_shapes()}


def batch_to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


class SimulationStream(IterableDataset):
    """Infinite stream of simulated batches; each DataLoader worker gets its own RNG."""

    def __init__(self, simulator: MixtureSimulator, batch_size: int, seed: int = 0, g: int | None = None):
        super().__init__()
        self.sim, self.batch_size, self.seed, self.g = simulator, batch_size, seed, g

    def __iter__(self):
        info = get_worker_info()
        wid = 0 if info is None else info.id
        rng = np.random.default_rng([self.seed, wid])
        while True:
            yield self.sim.sample_batch(rng, self.batch_size, g=self.g)


def _identity_collate(x):
    return x


def make_loader(simulator: MixtureSimulator, batch_size: int, seed: int, num_workers: int = 0,
                g: int | None = None):
    from torch.utils.data import DataLoader

    return DataLoader(SimulationStream(simulator, batch_size, seed, g), batch_size=None,
                      num_workers=num_workers, collate_fn=_identity_collate,
                      persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None)


def make_fixed_set(simulator: MixtureSimulator, n_batches: int, batch_size: int, seed: int,
                   g: int | None = None) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [simulator.sample_batch(rng, batch_size, g=g) for _ in range(n_batches)]
