"""Paired benchmark runners for Study 1 (replication) and Study 2 (factorial).

All methods see identical simulated datasets per replication (paired design);
failures are recorded, not silently dropped (RESEARCH_METHODOLOGY.md 8).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..baselines.base import Baseline
from ..losses.matching import hungarian_match
from ..simulation.cases import PaperCase
from ..simulation.mixture import sample
from ..simulation.priors import PriorConfig, ThetaPrior
from .metrics import aligned_errors, misclassification_rate, summarize_errors, test_loglik

OVERLAP_BINS = {"low": (0.0, 0.05), "medium": (0.05, 0.20), "high": (0.20, 0.50)}


def _fit_one(method: Baseline, X: np.ndarray, g: int | None, seed: int) -> dict:
    """Fit one method on one dataset; never raises (failure is a result)."""
    try:
        res = method.fit(X, g=g, seed=seed)
        return {"ok": True, "result": res}
    except Exception as exc:  # noqa: BLE001 -- failure mode is part of the record
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def run_replication_study(
    case: PaperCase,
    methods: list[Baseline],
    n_runs: int = 500,
    seed: int = 0,
    out_dir: str | Path | None = None,
    use_estimated_g: bool = False,
) -> dict[str, pd.DataFrame]:
    """Study 1: reproduce a base-paper case, all methods on identical datasets.

    Returns dict of DataFrames: per-method summary (AVG/Bias/SD/MSE), plus
    "misclassification", "efficiency", "failures", and the raw "runs" table.
    """
    true = case.theta
    run_rows: list[dict] = []
    errors: dict[str, list] = {m.name: [] for m in methods}
    for r in range(n_runs):
        rng = np.random.default_rng(seed + r)
        X, labels = sample(true, counts=case.counts, rng=rng)
        for m in methods:
            g_arg = None if use_estimated_g else true.g
            fit = _fit_one(m, X, g_arg, seed + r)
            row = dict(run=r, method=m.name)
            if not fit["ok"]:
                row.update(ok=False, error=fit["error"])
            else:
                res = fit["result"]
                row.update(ok=True, converged=res.converged, runtime=res.runtime,
                           loglik=res.loglik, g_est=res.theta.g)
                if res.theta.g == true.g:
                    errs = aligned_errors(true, res.theta)
                    errors[m.name].append(errs)
                    row["rmse_logalpha"] = float(np.sqrt(np.mean(
                        (np.log(res.theta.permuted(errs["perm"]).alpha) - np.log(true.alpha)) ** 2)))
                    row["mse_pi"] = float(np.mean(errs["pi"] ** 2))
                    if res.resp is not None:
                        row["misclassification"] = misclassification_rate(labels, res.resp, errs["perm"])
                    row["test_loglik"] = test_loglik(res.theta, true, rng=rng)
                else:
                    row["misclassification"] = np.nan
            run_rows.append(row)
        if (r + 1) % 50 == 0:
            print(f"[{case.name}] {r + 1}/{n_runs} runs done")

    runs = pd.DataFrame(run_rows)
    tables: dict[str, pd.DataFrame] = {"runs": runs}
    for m in methods:
        if errors[m.name]:
            tables[f"summary_{m.name}"] = summarize_errors(errors[m.name], true, m.name)
    tables["misclassification"] = (
        runs.groupby("method")["misclassification"].mean().rename("misclassification").reset_index()
        .assign(published=case.published_misclassification)
    )
    tables["efficiency"] = runs[runs["ok"]].groupby("method").agg(
        mean_runtime_s=("runtime", "mean"), mean_test_loglik=("test_loglik", "mean"),
        mean_rmse_logalpha=("rmse_logalpha", "mean"), mean_mse_pi=("mse_pi", "mean"),
    ).reset_index()
    tables["failures"] = runs.groupby("method")["ok"].agg(
        n_runs="size", n_failures=lambda s: int((~s).sum()),
        failure_rate=lambda s: float((~s).mean())).reset_index()
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            df.to_csv(out / f"{case.name}_{name}.csv", index=False)
    return tables


def run_factorial_study(
    grid: dict,
    methods: list[Baseline],
    reps_per_cell: int = 500,
    seed: int = 31337,
    sim_overrides: dict | None = None,
    out_dir: str | Path | None = None,
    use_estimated_g: bool = False,
) -> pd.DataFrame:
    """Study 2: full factorial with randomized theta per replication.

    grid keys: g, k, n, overlap, shape_regime (lists). One long DataFrame out,
    one row per (cell, rep, method).
    """
    rows: list[dict] = []
    cells = [(g, k, n, ov, sr)
             for g in grid["g"] for k in grid["k"] for n in grid["n"]
             for ov in grid["overlap"] for sr in grid["shape_regime"]]
    for ci, (g, k, n, ov, sr) in enumerate(cells):
        prior = ThetaPrior(PriorConfig(k=k, g_min=g, g_max=g, **(sim_overrides or {})))
        for rep in range(reps_per_cell):
            rng = np.random.default_rng(seed + 100003 * ci + rep)
            theta = prior.sample_stratified(rng, overlap_bin=OVERLAP_BINS[ov], g=g, shape_regime=sr)
            X, _ = sample(theta, n=n, rng=rng)
            for m in methods:
                fit = _fit_one(m, X, None if use_estimated_g else g, seed + rep)
                row = dict(g=g, k=k, n=n, overlap=ov, shape_regime=sr, rep=rep, method=m.name)
                if not fit["ok"]:
                    row.update(ok=False, error=fit["error"])
                else:
                    res = fit["result"]
                    row.update(ok=True, converged=res.converged, runtime=res.runtime,
                               g_est=res.theta.g)
                    if res.theta.g == g:
                        perm = hungarian_match(theta, res.theta)
                        al = res.theta.permuted(perm)
                        row["rmse_logalpha"] = float(np.sqrt(np.mean(
                            (np.log(al.alpha) - np.log(theta.alpha)) ** 2)))
                        row["rmse_logbeta"] = float(np.sqrt(np.mean(
                            (np.log(al.beta) - np.log(theta.beta)) ** 2)))
                        row["mae_pi"] = float(np.mean(np.abs(al.pi - theta.pi)))
                        row["test_loglik"] = test_loglik(res.theta, theta, n=5000, rng=rng)
                rows.append(row)
        print(f"cell {ci + 1}/{len(cells)} (g={g}, k={k}, n={n}, {ov}, {sr}) done")
    df = pd.DataFrame(rows)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "factorial_runs.csv", index=False)
    return df
