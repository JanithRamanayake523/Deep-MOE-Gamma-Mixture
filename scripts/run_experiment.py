#!/usr/bin/env python
"""Run an evaluation experiment described by a YAML file (RESEARCH_METHODOLOGY.md 7).

    python scripts/run_experiment.py --config configs/experiments/e1_gamma_k2.yaml
    python scripts/run_experiment.py --config ... --set reps=5 out=results/debug

`experiment:` selects the handler:
    benchmark    E1 / E2 / E4 : paired benchmark over a grid of cells
    robustness   E6           : prior shift, contamination, misspecified data family
    symmetry     E3           : loss variants vs MCMC-derived reference estimators
    calibration  E5           : coverage / SBC / C2ST of the flow posterior
    contraction  Thm 4 check  : empirical EM contraction around the MLE
    real_data    E7           : held-out likelihood + runtime on real datasets
    ppc          E7           : prior-predictive MMD check of real vs simulated data
    replication  E9           : the base paper's four cases (fixed parameters and counts, its metrics)
Methods are baseline specs (see aemnet.baselines.build_baseline).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aemnet.baselines import AdaptiveMCMCBaseline, EMBaseline, build_baseline  # noqa: E402
from aemnet.evaluation.benchmark import cells_from_grid, run_benchmark, run_benchmark_parallel, simulate_cell, summarize  # noqa: E402
from aemnet.evaluation.calibration import c2st, coverage, sbc_ranks, sbc_uniformity, aligned_matrix  # noqa: E402
from aemnet.evaluation.diagnostics import prior_predictive_check  # noqa: E402
from aemnet.evaluation.metrics import evaluate_fit, quotient_alignment  # noqa: E402
from aemnet.evaluation.relabel import quotient_frechet_mean  # noqa: E402
from aemnet.evaluation.stats import wilcoxon_table  # noqa: E402
from aemnet.inference.estimator import AmortizedEstimator  # noqa: E402
from aemnet.simulation.families import get_family  # noqa: E402
from aemnet.simulation.families.stats import prior_u_stats  # noqa: E402
from aemnet.simulation.simulator import MixtureSimulator, SimulatorConfig  # noqa: E402
from aemnet.theory.contraction import estimate_em_contraction  # noqa: E402
from aemnet.theory.symmetry import collapse_index, naive_posterior_mean, sorted_posterior_mean, symmetrize  # noqa: E402
from aemnet.utils.config import _build, apply_overrides, load_yaml_with_base  # noqa: E402


# ---------------------------------------------------------------------------- helpers
def sim_config(cfg: dict) -> SimulatorConfig:
    return _build(SimulatorConfig, cfg.get("simulator"))


def methods_from(cfg: dict, family):
    return [build_baseline(spec, family) for spec in cfg["methods"]]


def contaminate(X: np.ndarray, frac: float, family, rng) -> np.ndarray:
    if frac <= 0:
        return X
    X = X.copy()
    idx = rng.choice(X.shape[0], size=max(1, int(frac * X.shape[0])), replace=False)
    if family.positive_support:
        X[idx] *= np.exp(rng.uniform(np.log(10), np.log(100), size=(len(idx), X.shape[1])))
    elif family.discrete:
        X[idx] = np.round(X[idx] * rng.uniform(10, 100, size=(len(idx), X.shape[1])))
    else:
        X[idx] += rng.normal(0, 10 * X.std(0) + 1e-8, size=(len(idx), X.shape[1]))
    return X


# ---------------------------------------------------------------------------- handlers
def exp_benchmark(cfg: dict, out: Path) -> None:
    """Set `workers: N` (N > 1) to shard the run over processes; shards land in out/shards
    and finished shards are skipped on restart (resumable)."""
    family = get_family(cfg["family"])
    kw = dict(seed=cfg.get("seed", 0), known_g=cfg.get("known_g", True), n_test=cfg.get("n_test", 5000))
    if cfg.get("workers", 1) > 1:
        df = run_benchmark_parallel(cfg["family"], sim_config(cfg), cells_from_grid(cfg["grid"]), cfg["methods"],
                                    reps=cfg.get("reps", 100), workers=cfg["workers"],
                                    reps_per_shard=cfg.get("reps_per_shard", 25), shard_dir=out / "shards", **kw)
        df.to_csv(out / "runs.csv", index=False)
    else:
        df = run_benchmark(family, sim_config(cfg), cells_from_grid(cfg["grid"]), methods_from(cfg, family),
                           reps=cfg.get("reps", 100), out_csv=out / "runs.csv", **kw)
    summarize(df, "quotient_risk").to_csv(out / "summary_quotient_risk.csv", index=False)
    if "heldout_loglik" in df:
        summarize(df, "heldout_loglik").to_csv(out / "summary_heldout_loglik.csv", index=False)
    ref = cfg.get("reference")
    if ref:
        tabs = [wilcoxon_table(df, m, reference=ref) for m in cfg.get("test_metrics", ["quotient_risk"])]
        pd.concat(tabs).to_csv(out / "wilcoxon.csv", index=False)


def exp_robustness(cfg: dict, out: Path) -> None:
    family = get_family(cfg["family"])                                   # model family
    data_family = get_family(cfg.get("data_family", cfg["family"]))      # possibly misspecified
    same = data_family.spec() == family.spec()
    methods = methods_from(cfg, family)
    base_sim = sim_config(cfg)
    rows = []
    for scen in cfg["scenarios"]:
        s_cfg = replace(base_sim, **{k: (tuple(v) if isinstance(v, list) else v)
                                     for k, v in scen.get("simulator", {}).items()})
        dfam = get_family(scen["data_family"]) if "data_family" in scen else data_family
        is_same = same and "data_family" not in scen
        for rep in range(cfg.get("reps", 100)):
            rng = np.random.default_rng([cfg.get("seed", 0), rep, zlib.crc32(scen["name"].encode())])
            X, ev, labels, true, _ = simulate_cell(dfam, s_cfg, {"g": scen.get("g", 2), "n": scen.get("n", 500),
                                                                 "overlap": scen.get("overlap", "any")}, rng)
            X = contaminate(X, scen.get("contamination", 0.0), dfam, rng)
            Xt, evt, _ = dfam.sample_dataset(rng, true, cfg.get("n_test", 5000))
            for m in methods:
                row = {"scenario": scen["name"], "rep": rep, "method": m.name}
                try:
                    fit = m.fit(X, ev, g=true.g, seed=rep)
                    row.update(ok=True, runtime=fit.runtime,
                               heldout_loglik=float(np.mean(family.mixture_logpdf_np(Xt, evt, fit.params))),
                               heldout_loglik_true=float(np.mean(dfam.mixture_logpdf_np(Xt, evt, true))))
                    if is_same:
                        row["quotient_risk"] = quotient_alignment(family, X, true, fit.params)[0]
                except Exception as exc:  # noqa: BLE001
                    row.update(ok=False, error=f"{type(exc).__name__}: {exc}")
                rows.append(row)
        print(f"scenario {scen['name']} done")
    df = pd.DataFrame(rows)
    df.to_csv(out / "runs.csv", index=False)
    df[df.ok == True].groupby(["scenario", "method"]).mean(numeric_only=True).to_csv(out / "summary.csv")  # noqa: E712


def exp_symmetry(cfg: dict, out: Path) -> None:
    family = get_family(cfg["family"])
    u_std = prior_u_stats(family)[1]
    ests = {name: AmortizedEstimator.from_checkpoint(p, cfg.get("device", "cpu"))
            for name, p in cfg["checkpoints"].items()}
    mcmc = AdaptiveMCMCBaseline(family, **cfg.get("mcmc", {}))
    rows = []
    for ci, cell in enumerate(cells_from_grid(cfg["grid"])):
        for rep in range(cfg.get("reps", 50)):
            rng = np.random.default_rng([cfg.get("seed", 0), ci, rep])
            X, ev, labels, true, err = simulate_cell(family, sim_config(cfg), cell, rng)
            samples = mcmc.fit(X, ev, g=true.g, seed=rep).extra["samples"]
            refs = {"quotient": quotient_frechet_mean(family, samples, u_std)[0],
                    "naive": naive_posterior_mean(family, symmetrize(samples, rng)),
                    "sorted": sorted_posterior_mean(family, samples)}
            for name, est in ests.items():
                mp, _ = est.estimate(X, ev, g=true.g)
                row = {**cell, "rep": rep, "variant": name, "bayes_error": err,
                       "risk_vs_truth": quotient_alignment(family, X, true, mp, u_std)[0],
                       "collapse_index": collapse_index(family, mp, u_std)}
                for rn, r in refs.items():
                    row[f"risk_vs_{rn}_ref"] = quotient_alignment(family, X, r, mp, u_std)[0]
                rows.append(row)
            for rn, r in refs.items():
                rows.append({**cell, "rep": rep, "variant": f"mcmc_{rn}", "bayes_error": err,
                             "risk_vs_truth": quotient_alignment(family, X, true, r, u_std)[0],
                             "collapse_index": collapse_index(family, r, u_std)})
        print(f"cell {cell} done")
    df = pd.DataFrame(rows)
    df.to_csv(out / "runs.csv", index=False)
    df.groupby(["variant"] + [k for k in cfg["grid"]]).mean(numeric_only=True).to_csv(out / "summary.csv")


def exp_calibration(cfg: dict, out: Path) -> None:
    family = get_family(cfg["family"])
    est = AmortizedEstimator.from_checkpoint(cfg["checkpoint"], cfg.get("device", "cpu"))
    n_samples = cfg.get("n_samples", 1000)
    mcmc = AdaptiveMCMCBaseline(family, **cfg.get("mcmc", {})) if cfg.get("c2st_reps", 0) else None
    rows, ranks = [], []
    for ci, cell in enumerate(cells_from_grid(cfg["grid"])):
        for rep in range(cfg.get("reps", 200)):
            rng = np.random.default_rng([cfg.get("seed", 0), ci, rep])
            X, ev, labels, true, _ = simulate_cell(family, sim_config(cfg), cell, rng)
            pivot, _ = est.estimate(X, ev, g=true.g)
            samples = est.posterior(X, ev, g=true.g, n_samples=n_samples)
            row = {**cell, "rep": rep, **coverage(family, X, true, samples, pivot)}
            ranks.append(sbc_ranks(family, X, true, samples, pivot))
            if mcmc is not None and rep < cfg["c2st_reps"]:
                ms = mcmc.fit(X, ev, g=true.g, seed=rep).extra["samples"]
                A = aligned_matrix(family, samples[: len(ms)], pivot, X)
                B = aligned_matrix(family, ms, pivot, X)
                row["c2st"] = c2st(A, B)
            rows.append(row)
        print(f"cell {cell} done")
    pd.DataFrame(rows).to_csv(out / "coverage.csv", index=False)
    R = np.asarray(ranks)
    np.save(out / "sbc_ranks.npy", R)
    pd.DataFrame({"coord": np.arange(R.shape[1]), "p_uniform": sbc_uniformity(R, n_samples)}).to_csv(
        out / "sbc_uniformity.csv", index=False)


def exp_contraction(cfg: dict, out: Path) -> None:
    family = get_family(cfg["family"])
    frames = []
    for ci, cell in enumerate(cells_from_grid(cfg["grid"])):
        for rep in range(cfg.get("reps", 20)):
            rng = np.random.default_rng([cfg.get("seed", 0), ci, rep])
            X, ev, _, true, _ = simulate_cell(family, sim_config(cfg), cell, rng)
            mle = EMBaseline(family, n_init=5).fit(X, ev, g=true.g, seed=rep).params
            df = estimate_em_contraction(family, X, ev, mle, radii=cfg.get("radii", (0.05, 0.1, 0.2, 0.4, 0.8)),
                                         n_dirs=cfg.get("n_dirs", 20), seed=rep)
            frames.append(df.assign(rep=rep, **cell))
        print(f"cell {cell} done")
    pd.concat(frames).to_csv(out / "contraction.csv", index=False)


def load_real(spec: dict) -> dict:
    spec = dict(spec)
    kind = spec.pop("kind")
    from aemnet import data as D

    if kind == "ghcn":
        return D.load_stations(**spec)
    if kind == "cas":
        return D.load_severity_csv(**spec)
    if kind == "cas_bivariate":
        return D.load_bivariate_csv(**spec)
    if kind == "allen_npz":
        return D.load_units_npz(**spec)
    if kind == "madawaska":
        return {"madawaska": D.load_madawaska(**spec)}
    if kind == "sar":
        patches, pos = D.patches_from_array(D.load_geotiff(spec.pop("path")), **spec)
        return {f"{r}_{c}": p for p, (r, c) in zip(patches, pos)}
    raise KeyError(f"unknown real-data kind '{kind}'")


def exp_real_data(cfg: dict, out: Path) -> None:
    family = get_family(cfg["family"])
    datasets = load_real(cfg["data"])
    print(f"{len(datasets)} real datasets")
    methods = methods_from(cfg, family)
    g = cfg.get("g")
    rng = np.random.default_rng(cfg.get("seed", 0))
    rows = []
    for key, X in datasets.items():
        X = np.asarray(X, float)
        perm = rng.permutation(len(X))
        n_te = int(cfg.get("test_frac", 0.2) * len(X))
        Xte, Xtr = X[perm[:n_te]], X[perm[n_te:]]
        for m in methods:
            row = {"dataset": key, "n": len(Xtr), "method": m.name}
            try:
                fit = m.fit(Xtr, None, g=g, seed=0)
                row.update(ok=True, runtime=fit.runtime, g_est=fit.params.g,
                           train_loglik=fit.loglik / len(Xtr),
                           test_loglik=float(np.mean(family.mixture_logpdf_np(Xte, None, fit.params))))
            except Exception as exc:  # noqa: BLE001
                row.update(ok=False, error=f"{type(exc).__name__}: {exc}")
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "runs.csv", index=False)
    df[df.ok == True].groupby("method").agg(  # noqa: E712
        test_loglik=("test_loglik", "mean"), runtime=("runtime", "sum"), n_datasets=("dataset", "count")
    ).to_csv(out / "summary.csv")
    # whole-collection throughput (the "why amortise" numbers, H7). Batched GPU EM is the fair
    # comparator; the batched hybrid starts it from the amortised estimates.
    Xs = list(datasets.values())
    g_tp = g or cfg.get("throughput_g", 2)
    throughput = {"n_datasets": len(Xs), "g": g_tp}
    bem = None
    if family.exponential_family:
        from aemnet.baselines import BatchedEM

        bem = BatchedEM(family, **cfg.get("batched_em", {}))
        _, info = bem.fit_many(Xs, g_tp)
        throughput["batched_em_seconds"] = info["seconds"]
        throughput["batched_em_iterations_mean"] = float(np.mean(info["iterations"]))
    for spec in cfg["methods"]:
        if spec.get("type") == "neural" and spec.get("mode", "amortized") == "amortized":
            name = spec.get("name", "neural")
            est = AmortizedEstimator.from_checkpoint(spec["checkpoint"], spec.get("device", "cpu"))
            t0 = time.perf_counter()
            inits = est.estimate_many(Xs, g=g_tp)
            throughput[f"{name}_seconds"] = time.perf_counter() - t0
            if bem is not None:
                _, info = bem.refine(Xs, inits)
                throughput[f"{name}+batched_em_seconds"] = throughput[f"{name}_seconds"] + info["seconds"]
                throughput[f"{name}+batched_em_iterations_mean"] = float(np.mean(info["iterations"]))
    with open(out / "throughput.json", "w") as f:
        json.dump(throughput, f, indent=2)
    print(throughput)


def exp_ppc(cfg: dict, out: Path) -> None:
    family = get_family(cfg["family"])
    datasets = list(load_real(cfg["data"]).values())
    res = prior_predictive_check(family, datasets, MixtureSimulator(family, sim_config(cfg)),
                                 n_sim=cfg.get("n_sim", 300), n_perm=cfg.get("n_perm", 500), seed=cfg.get("seed", 0))
    with open(out / "ppc.json", "w") as f:
        json.dump(res, f, indent=2)
    print(res)


def exp_replication(cfg: dict, out: Path) -> None:
    """E9: Vaidyanathan & Vani Lakshmi (2017), Cases 1-4, with the paper's metrics
    (AVG / Bias / SD / MSE per parameter after alignment, misclassification rate)."""
    from aemnet.simulation.paper_cases import PAPER_CASES, sample_case

    family = get_family({"name": "gamma", "k": 2})
    methods = methods_from(cfg, family)
    rows, err_rows = [], []
    for name in cfg.get("cases", list(PAPER_CASES)):
        case = PAPER_CASES[name]
        true = case.params
        for rep in range(cfg.get("reps", 500)):
            rng = np.random.default_rng([cfg.get("seed", 0), rep, int(name[-1])])
            X, ev, labels = sample_case(case, rng)
            for m in methods:
                row = {"case": name, "rep": rep, "method": m.name}
                try:
                    fit = m.fit(X, ev, g=true.g, seed=rep)
                    row.update(ok=True, runtime=fit.runtime,
                               **evaluate_fit(family, X, ev, labels, true, fit.params, fit.resp, n_test=0))
                    _, perm = quotient_alignment(family, X, true, fit.params)
                    al = fit.params.permuted(perm)
                    for key in ("alpha", "beta"):
                        for i in range(true.g):
                            for p in range(2):
                                err_rows.append({"case": name, "method": m.name, "param": f"{key}{i + 1}{p + 1}",
                                                 "true": true.comps[key][i, p], "est": al.comps[key][i, p]})
                    for i in range(true.g):
                        err_rows.append({"case": name, "method": m.name, "param": f"pi{i + 1}",
                                         "true": true.pi[i], "est": al.pi[i]})
                except Exception as exc:  # noqa: BLE001
                    row.update(ok=False, error=f"{type(exc).__name__}: {exc}")
                rows.append(row)
        print(f"{name} done")
    df = pd.DataFrame(rows)
    df.to_csv(out / "runs.csv", index=False)
    E = pd.DataFrame(err_rows)
    E["err"] = E["est"] - E["true"]
    table = E.groupby(["case", "method", "param"]).agg(
        true=("true", "first"), AVG=("est", "mean"), Bias=("err", "mean"), SD=("est", "std"),
        MSE=("err", lambda e: float(np.mean(e ** 2)))).reset_index()
    table.to_csv(out / "paper_table.csv", index=False)
    mis = df[df.ok == True].groupby(["case", "method"])["misclassification"].mean().reset_index()  # noqa: E712
    mis["published_wh_mclust"] = mis["case"].map({k: v.published_misclassification for k, v in PAPER_CASES.items()})
    mis.to_csv(out / "misclassification.csv", index=False)


HANDLERS = {"benchmark": exp_benchmark, "robustness": exp_robustness, "symmetry": exp_symmetry,
            "calibration": exp_calibration, "contraction": exp_contraction, "real_data": exp_real_data,
            "ppc": exp_ppc, "replication": exp_replication}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = apply_overrides(load_yaml_with_base(args.config), args.set)
    out = Path(cfg.get("out", f"results/{Path(args.config).stem}"))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "experiment.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    HANDLERS[cfg["experiment"]](cfg, out)
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
