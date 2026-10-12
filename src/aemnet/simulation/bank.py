"""Pre-computed, overlap-labelled parameter bank.

Overlap stratification by rejection costs one Monte-Carlo Bayes-error estimate
per *try* -- measured at ~2.8 s per batch of 64 for gamma k=2 (40x the cost of
unstratified simulation), which would leave the GPU idle. The bank instead
draws parameters once, computes each draw's Bayes error once, and files it into
whichever overlap bin it falls in (no draw is wasted on a specific bin). Training
then resamples parameters from the bank and simulates *fresh data* every time.

Each (g, bin) cell keeps a disjoint validation tail (`val_frac`) so validation
datasets never reuse training parameters.

Build once per family/simulator config:
    python scripts/build_param_bank.py --config configs/train/gamma_k2.yaml
"""
from __future__ import annotations

import json
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from ..mixture_params import MixtureParams


def _fingerprint(family, cfg) -> str:
    keys = ("dirichlet_conc_range", "min_weight", "overlap_bins", "overlap_n_mc", "shape_regime")
    d = asdict(cfg)
    return json.dumps({"family": family.spec(), **{k: d[k] for k in keys}}, sort_keys=True, default=list)


def _worker(args):
    spec, cfg_dict, g, n_draws, seed = args
    from dataclasses import replace

    from .families import family_from_spec
    from .simulator import MixtureSimulator, SimulatorConfig, bayes_error

    fam = family_from_spec(spec)
    cfg = replace(SimulatorConfig(), **cfg_dict)
    sim = MixtureSimulator(fam, cfg)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_draws):
        mp = sim._raw_params(rng, g)
        out.append((mp.pi, mp.comps, bayes_error(fam, mp, cfg.overlap_n_mc, rng)))
    return out


class ParameterBank:
    def __init__(self, cells: dict, bins: list, fingerprint: str, val_frac: float = 0.05):
        self.cells = cells            # {(g, b): {"pi": (N,g), "err": (N,), <name>: (N,g,*shape)}}
        self.bins = [tuple(b) for b in bins]
        self.fingerprint = fingerprint
        self.val_frac = val_frac

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, simulator, per_cell: int = 20000, g_values=None, workers: int = 1, seed: int = 0,
              chunk: int = 500, max_draw_factor: int = 200, verbose: bool = True) -> "ParameterBank":
        fam, cfg = simulator.family, simulator.cfg
        bins = [tuple(b) for b in cfg.overlap_bins]
        g_values = list(g_values or range(max(cfg.g_min, 2), cfg.g_max + 1))
        cfg_dict = {k: v for k, v in asdict(cfg).items() if k != "param_bank"}
        cells: dict = {}
        ctx = get_context("spawn")
        for g in g_values:
            filled = {b: [] for b in range(len(bins))}
            drawn, round_ = 0, 0
            pool = ctx.Pool(workers) if workers > 1 else None
            mapper = pool.map if pool is not None else map
            try:
                while min(len(v) for v in filled.values()) < per_cell and drawn < max_draw_factor * per_cell:
                    jobs = [(fam.spec(), cfg_dict, g, chunk, [seed, g, round_, w]) for w in range(max(workers, 1))]
                    round_ += 1
                    for res in mapper(_worker, jobs):
                        for pi, comps, err in res:
                            drawn += 1
                            for b, (lo, hi) in enumerate(bins):
                                if lo <= err < hi and len(filled[b]) < per_cell:
                                    filled[b].append((pi, comps, err))
                                    break
                    if verbose:
                        print(f"  g={g}: drawn {drawn}, filled {[len(v) for v in filled.values()]}")
            finally:
                if pool is not None:
                    pool.close()
                    pool.join()
            for b, items in filled.items():
                if not items:
                    continue
                cell = {"pi": np.stack([it[0] for it in items]), "err": np.array([it[2] for it in items])}
                for name in items[0][1]:
                    cell[name] = np.stack([it[1][name] for it in items])
                cells[(g, b)] = cell
        return cls(cells, bins, _fingerprint(fam, cfg))

    # ------------------------------------------------------------------ io
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"g{g}_b{b}__{k}": v for (g, b), cell in self.cells.items() for k, v in cell.items()}
        np.savez_compressed(path, __meta__=np.array(json.dumps(
            {"bins": self.bins, "fingerprint": self.fingerprint, "val_frac": self.val_frac})), **arrays)

    @classmethod
    def load(cls, path) -> "ParameterBank":
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["__meta__"]))
        cells: dict = {}
        for key in data.files:
            if key == "__meta__":
                continue
            head, name = key.split("__", 1)
            g, b = (int(x[1:]) for x in head.split("_"))
            cells.setdefault((g, b), {})[name] = data[key]
        return cls(cells, meta["bins"], meta["fingerprint"], meta.get("val_frac", 0.05))

    # ------------------------------------------------------------------ use
    def compatible(self, family, cfg) -> bool:
        return self.fingerprint == _fingerprint(family, cfg)

    def has(self, g: int, b: int) -> bool:
        return (g, b) in self.cells

    def size(self, g: int, b: int) -> int:
        return len(self.cells[(g, b)]["err"]) if self.has(g, b) else 0

    def draw(self, rng, g: int, b: int, split: str = "train") -> tuple[MixtureParams, float]:
        cell = self.cells[(g, b)]
        N = len(cell["err"])
        n_val = max(1, int(self.val_frac * N))
        i = int(rng.integers(N - n_val)) if split == "train" else int(N - n_val + rng.integers(n_val))
        comps = {k: v[i] for k, v in cell.items() if k not in ("pi", "err")}
        return MixtureParams(cell["pi"][i], comps), float(cell["err"][i])
