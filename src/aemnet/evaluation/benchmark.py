"""Paired benchmark runner for any family (E1, E2, E4, E6).

Every method sees the identical simulated dataset in every replication; a
failure (exception) is recorded as a row with ok=False, never silently dropped.
Rows are appended to a CSV as they are produced so long runs can be resumed or
inspected. Statistical comparison: evaluation.stats.wilcoxon_table(df, metric,
reference=<method name>) (keys: cell columns + rep).
"""
from __future__ import annotations

import itertools
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..simulation.families.base import Family
from ..simulation.families.stats import prior_u_stats
from ..simulation.simulator import MixtureSimulator, SimulatorConfig
from .metrics import evaluate_fit

OVERLAP_BINS = {"low": (0.0, 0.05), "medium": (0.05, 0.20), "high": (0.20, 0.50), "any": None}


def cells_from_grid(grid: dict) -> list[dict]:
    """Cartesian product of a dict of lists, e.g. {"g": [2, 3], "n": [100, 1000], "overlap": ["low"]}."""
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def simulate_cell(family: Family, sim_cfg: SimulatorConfig, cell: dict, rng):
    cfg = replace(sim_cfg, shape_regime=cell.get("shape_regime"), overlap_stratification=False)
    sim = MixtureSimulator(family, cfg)
    ov = OVERLAP_BINS[cell.get("overlap", "any")]
    mp, err = sim.sample_params(rng, g=cell["g"], overlap_bin=ov)
    X, ev, labels = family.sample_dataset(rng, mp, int(cell["n"]))
    X, mp = sim.augment(rng, X, mp)
    return X, ev, labels, mp, err


def run_benchmark(family: Family, sim_cfg: SimulatorConfig, cells: list[dict], methods: list, reps: int = 100,
                  seed: int = 0, known_g: bool = True, n_test: int = 5000, out_csv: str | Path | None = None,
                  verbose: bool = True, cell_ids: list[int] | None = None, rep_range: tuple[int, int] | None = None
                  ) -> pd.DataFrame:
    """cell_ids / rep_range let a shard run a subset with exactly the seeds of the
    full run (dataset (cell i, rep r) always uses rng [seed, i, r])."""
    u_std = prior_u_stats(family)[1]
    cell_ids = list(range(len(cells))) if cell_ids is None else list(cell_ids)
    rep_lo, rep_hi = (0, reps) if rep_range is None else rep_range
    rows: list[dict] = []
    out_csv = Path(out_csv) if out_csv else None
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
    header_written = out_csv.exists() if out_csv else False
    t0 = time.time()
    for ci, cell in zip(cell_ids, cells):
        cell_rows = []
        for rep in range(rep_lo, rep_hi):
            rng = np.random.default_rng([seed, ci, rep])
            X, ev, labels, true, err = simulate_cell(family, sim_cfg, cell, rng)
            for m in methods:
                row = {"family": family.name, **cell, "rep": rep, "method": m.name, "bayes_error": err}
                try:
                    fit = m.fit(X, ev, g=true.g if known_g else None, seed=seed + rep)
                    row.update(ok=True, converged=fit.converged, runtime=fit.runtime, loglik=fit.loglik)
                    # scalar diagnostics (EM iterations, acceptance rates, ...) -> x_<name> columns
                    row.update({f"x_{k}": v for k, v in fit.extra.items()
                                if isinstance(v, (int, float, np.integer, np.floating, bool))})
                    row.update(evaluate_fit(family, X, ev, labels, true, fit.params, fit.resp, u_std, n_test,
                                            np.random.default_rng([seed, ci, rep, 7])))
                except Exception as exc:  # noqa: BLE001 -- failures are results
                    row.update(ok=False, error=f"{type(exc).__name__}: {exc}")
                cell_rows.append(row)
        rows.extend(cell_rows)
        if out_csv:
            pd.DataFrame(cell_rows).to_csv(out_csv, mode="a", header=not header_written, index=False)
            header_written = True
        if verbose:
            print(f"[{family.name}] cell {ci} {cell} reps {rep_lo}-{rep_hi} done ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


def _shard_job(args):
    family_spec, sim_cfg, cell, ci, rep_range, method_specs, kw, shard_csv = args
    import torch

    from ..baselines import build_baseline
    from ..simulation.families import family_from_spec

    torch.set_num_threads(1)
    fam = family_from_spec(family_spec)
    methods = [build_baseline(dict(s), fam) for s in method_specs]
    return run_benchmark(fam, sim_cfg, [cell], methods, cell_ids=[ci], rep_range=rep_range, out_csv=shard_csv,
                         verbose=False, **kw)


def run_benchmark_parallel(family_spec: dict, sim_cfg: SimulatorConfig, cells: list[dict], method_specs: list[dict],
                           reps: int = 100, workers: int = 4, reps_per_shard: int = 25, shard_dir: str | Path | None = None,
                           **kw) -> pd.DataFrame:
    """Shard (cell, rep-block) jobs over processes. Results are identical to the serial
    run (same per-dataset seeds); each shard also writes its own CSV for resumability."""
    from multiprocessing import get_context

    jobs = []
    for ci, cell in enumerate(cells):
        for lo in range(0, reps, reps_per_shard):
            hi = min(reps, lo + reps_per_shard)
            csv = None if shard_dir is None else str(Path(shard_dir) / f"cell{ci:03d}_rep{lo:05d}-{hi:05d}.csv")
            if csv and Path(csv).exists():
                continue                                    # resume: shard already done
            jobs.append((family_spec, sim_cfg, cell, ci, (lo, hi), method_specs, kw, csv))
    if shard_dir is not None:
        Path(shard_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_shard_job, jobs), 1):
            print(f"shard {i}/{len(jobs)} done ({time.time() - t0:.0f}s)")
    if shard_dir is None:
        raise ValueError("run_benchmark_parallel needs shard_dir to collect results")
    return pd.concat([pd.read_csv(p) for p in sorted(Path(shard_dir).glob("*.csv"))], ignore_index=True)


def run_n_regime(family: Family, sim_cfg: SimulatorConfig, methods: list, ns=(30, 100, 300, 1000, 3000, 10000),
                 g: int = 2, overlap: str = "medium", reps: int = 200, **kw) -> pd.DataFrame:
    """E2 headline: risk vs n for amortised / hybrid / EM / MLE on identical data."""
    cells = [{"g": g, "n": n, "overlap": overlap} for n in ns]
    return run_benchmark(family, sim_cfg, cells, methods, reps=reps, **kw)


def summarize(df: pd.DataFrame, metric: str = "quotient_risk", by=("method", "g", "n", "overlap")) -> pd.DataFrame:
    by = [b for b in by if b in df.columns]
    ok = df[df["ok"] == True]  # noqa: E712
    agg = ok.groupby(by)[metric].agg(["mean", "median", "std", "count"]).reset_index()
    fail = df.groupby(by)["ok"].apply(lambda s: float((~s.astype(bool)).mean())).rename("failure_rate").reset_index()
    return agg.merge(fail, on=by, how="left")
