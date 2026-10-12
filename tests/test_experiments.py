"""Execute every experiment handler, the posterior/adaptation code, and short
training runs on the remaining families. Tiny settings: this checks the code
runs end to end, not that the methods work (that is what the experiments are for)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from aemnet.inference import AmortizedEstimator, self_consistency_loss, unlabeled_adaptation
from aemnet.losses.quotient import QuotientLoss
from aemnet.mixture_params import MixtureParams
from aemnet.models import build_model
from aemnet.simulation.families import get_family
from aemnet.simulation.simulator import MixtureSimulator, SimulatorConfig
from aemnet.training.trainer import AEMTrainer
from aemnet.utils.config import load_experiment

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_experiment", ROOT / "scripts" / "run_experiment.py")
RX = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RX)

TINY_TRAIN = ["training.steps=4", "training.val_every=2", "training.log_every=2", "training.u_stats_samples=30",
              "training.val_batches=1", "training.batch_size=4", "training.val_batch_size=4",
              "simulator.overlap_stratification=false", "simulator.param_bank=null"]


def _train(tmp, family_spec, k_slots=3, extra=()):
    exp = load_experiment(ROOT / "configs/train/smoke_gamma_k2.yaml",
                          TINY_TRAIN + [f"training.out_dir={Path(tmp).as_posix()}",
                                        f"simulator.g_max={k_slots}", *extra],)
    exp.family = family_spec
    torch.manual_seed(0)
    fam = get_family(family_spec)
    sim = MixtureSimulator(fam, exp.simulator)
    model = build_model(fam, exp.model, exp.simulator.g_max)
    raw = exp.to_dict()
    raw["family"] = family_spec
    AEMTrainer(model, QuotientLoss(fam, exp.loss), sim, exp.training, experiment=raw).train()
    return Path(tmp) / "best.pt"


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    return _train(tmp_path_factory.mktemp("ck"), {"name": "gamma", "k": 1})


def _neural(ckpt, **kw):
    return {"type": "neural", "name": "aem", "checkpoint": str(ckpt), **kw}


# ------------------------------------------------------------------ handlers
def test_handler_benchmark_known_and_unknown_g(ckpt, tmp_path):
    for known in (True, False):
        cfg = {"family": {"name": "gamma", "k": 1}, "grid": {"g": [2], "n": [80], "overlap": ["medium"]},
               "reps": 2, "n_test": 200, "known_g": known, "reference": "aem",
               "test_metrics": ["g_correct"] if not known else ["quotient_risk"],
               "methods": [_neural(ckpt), _neural(ckpt, mode="hybrid"), {"type": "em", "n_init": 2, "g_max": 3},
                           {"type": "batched_em", "n_restarts": 2, "device": "cpu"} if known else
                           {"type": "em", "n_init": 1, "g_max": 2}]}
        out = tmp_path / f"bench_{known}"
        out.mkdir()
        RX.exp_benchmark(cfg, out)
        import pandas as pd

        df = pd.read_csv(out / "runs.csv")
        assert df["ok"].all(), df.get("error")
        if known:
            assert "x_em_steps_total" in df and "x_em_steps" in df


def test_handler_robustness(ckpt, tmp_path):
    cfg = {"family": {"name": "gamma", "k": 1}, "reps": 2, "n_test": 200,
           "scenarios": [{"name": "in", "g": 2, "n": 80},
                         {"name": "contam", "g": 2, "n": 80, "contamination": 0.05},
                         {"name": "weib", "g": 2, "n": 80, "data_family": {"name": "weibull", "k": 1}}],
           "methods": [_neural(ckpt), {"type": "em", "n_init": 1}]}
    RX.exp_robustness(cfg, tmp_path)
    import pandas as pd

    df = pd.read_csv(tmp_path / "runs.csv")
    assert df["ok"].all() and df[df.scenario == "weib"]["quotient_risk"].isna().all()


def test_handler_symmetry(ckpt, tmp_path):
    cfg = {"family": {"name": "gamma", "k": 1}, "grid": {"g": [2], "n": [80], "overlap": ["low"]}, "reps": 1,
           "checkpoints": {"quotient": str(ckpt), "naive": str(ckpt)},
           "mcmc": {"n_iter": 400, "burn_in": 100, "thin": 5}}
    RX.exp_symmetry(cfg, tmp_path)
    import pandas as pd

    df = pd.read_csv(tmp_path / "runs.csv")
    assert {"quotient", "naive", "mcmc_quotient", "mcmc_naive", "mcmc_sorted"} <= set(df.variant)


def test_handler_calibration(ckpt, tmp_path):
    cfg = {"family": {"name": "gamma", "k": 1}, "checkpoint": str(ckpt), "n_samples": 30,
           "grid": {"g": [2], "n": [80], "overlap": ["low"]}, "reps": 2, "c2st_reps": 1,
           "mcmc": {"n_iter": 600, "burn_in": 100, "thin": 10}}
    RX.exp_calibration(cfg, tmp_path)
    assert (tmp_path / "coverage.csv").exists() and np.load(tmp_path / "sbc_ranks.npy").shape[0] == 2


def test_handler_contraction(tmp_path):
    cfg = {"family": {"name": "gamma", "k": 1}, "grid": {"g": [2], "n": [200], "overlap": ["low"]}, "reps": 1,
           "radii": [0.05, 0.1], "n_dirs": 2}
    RX.exp_contraction(cfg, tmp_path)
    assert (tmp_path / "contraction.csv").exists()


def _fake_allen_npz(path):
    rng = np.random.default_rng(0)
    units = {f"u{i}": np.cumsum(rng.gamma(2.0, 0.01, size=600) * rng.choice([1, 8], 600)) for i in range(4)}
    np.savez(path, **units)


def test_handler_real_data_and_ppc(ckpt, tmp_path):
    npz = tmp_path / "units.npz"
    _fake_allen_npz(npz)
    data = {"kind": "allen_npz", "path": str(npz), "min_spikes": 100}
    cfg = {"family": {"name": "gamma", "k": 1}, "data": data, "g": 2, "test_frac": 0.2, "throughput_g": 2,
           "batched_em": {"n_restarts": 2, "device": "cpu"},
           "methods": [_neural(ckpt), _neural(ckpt, mode="hybrid"), {"type": "em", "n_init": 1}]}
    RX.exp_real_data(cfg, tmp_path)
    tp = json.loads((tmp_path / "throughput.json").read_text())
    assert {"batched_em_seconds", "aem_seconds", "aem+batched_em_seconds"} <= set(tp)
    RX.exp_ppc({"family": {"name": "gamma", "k": 1}, "data": data, "n_sim": 20, "n_perm": 20}, tmp_path)
    assert 0 < json.loads((tmp_path / "ppc.json").read_text())["p_value"] <= 1


def test_ghcn_parser(tmp_path):
    line = "USW00000001" + "2001" + "01" + "PRCP"
    vals = [(25, " "), (-9999, " "), (3, " "), (40, "X")] + [(0, " ")] * 27
    line += "".join(f"{v:5d} {q} " for v, q in vals) + "\n"
    (tmp_path / "USW00000001.dly").write_text(line)
    from aemnet.data import load_stations, parse_dly, wet_day_amounts

    df = parse_dly(tmp_path / "USW00000001.dly")
    assert len(df) == 29                                  # missing and QC-flagged values dropped
    np.testing.assert_allclose(wet_day_amounts(df, threshold_mm=1.0), [2.5])
    assert load_stations(tmp_path, min_wet_days=1)["USW00000001"].shape == (1, 1)


# ------------------------------------------------------------------ posterior / adaptation
def test_flow_log_prob_and_self_consistency(ckpt):
    est = AmortizedEstimator.from_checkpoint(ckpt)
    model = est.model
    fam = model.family
    sim = MixtureSimulator(fam, SimulatorConfig(g_max=3, n_max=80, overlap_stratification=False))
    batch = sim.sample_batch(np.random.default_rng(0), 2)
    out = model(batch["X"], batch["events"], batch["obs_mask"])
    y = model.flow.sample(model.flow_cond(out), 1, steps=4)[0]
    exact = model.flow.log_prob(y, model.flow_cond(out), steps=4)
    hutch = torch.stack([model.flow.log_prob(y, model.flow_cond(out), steps=4, hutchinson=True) for _ in range(64)]).mean(0)
    assert torch.isfinite(exact).all()
    torch.testing.assert_close(hutch, exact, rtol=0.05, atol=0.5)       # unbiased divergence estimate
    model.train()
    loss = self_consistency_loss(model, out, batch, n_samples=3, flow_steps=3)
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None for p in model.flow.parameters())


def test_unlabeled_adaptation_runs(ckpt):
    est = AmortizedEstimator.from_checkpoint(ckpt)
    fam = est.family
    sim = MixtureSimulator(fam, SimulatorConfig(g_max=3, n_max=80, overlap_stratification=False))
    rng = np.random.default_rng(0)
    real = [fam.sample_dataset(rng, MixtureParams([0.5, 0.5], fam.sample_prior(rng, 2)), 100)[0] for _ in range(4)]
    hist = unlabeled_adaptation(est.model, QuotientLoss(fam, load_experiment(raw={}).loss), sim, real, steps=2,
                                sim_batch_size=2, real_batch_size=2)
    assert len(hist) == 2 and np.isfinite(hist[-1]["real_nll"])


# ------------------------------------------------------------------ other families train end to end
@pytest.mark.parametrize("spec", [{"name": "gandk", "k": 1}, {"name": "kibble_gamma", "k": 2},
                                  {"name": "noisy_gamma", "k": 1}, {"name": "copula_gamma", "k": 2},
                                  {"name": "censored", "base": {"name": "weibull", "k": 1}},
                                  {"name": "negbin", "k": 1}, {"name": "student_t", "k": 1}])
def test_family_trains_and_estimates(spec, tmp_path):
    path = _train(tmp_path, spec, k_slots=2, extra=["model.use_loglik=false"] if spec["name"] == "kibble_gamma" else [])
    est = AmortizedEstimator.from_checkpoint(path)
    fam = get_family(spec)
    rng = np.random.default_rng(0)
    X, ev, _ = fam.sample_dataset(rng, MixtureParams([0.5, 0.5], fam.sample_prior(rng, 2)), 120)
    mp, _ = est.estimate(X, ev, g=2)
    assert mp.g == 2 and np.isfinite(fam.mixture_logpdf_np(X, ev, mp)).all()


def test_handler_replication_paper_cases(ckpt, tmp_path):
    cfg = {"reps": 2, "cases": ["case2", "case4"],
           "methods": [{"type": "em", "n_init": 1}, {"type": "wh_mclust"}]}
    RX.exp_replication(cfg, tmp_path)
    import pandas as pd

    table = pd.read_csv(tmp_path / "paper_table.csv")
    assert {"AVG", "Bias", "SD", "MSE"} <= set(table.columns)
    assert set(table["param"]) >= {"alpha11", "beta32", "pi3"}
    mis = pd.read_csv(tmp_path / "misclassification.csv")
    assert mis["published_wh_mclust"].notna().all()


def test_parallel_benchmark_matches_serial(tmp_path):
    from aemnet.evaluation.benchmark import run_benchmark, run_benchmark_parallel
    from aemnet.baselines import build_baseline

    fam_spec = {"name": "gamma", "k": 1}
    cells = [{"g": 2, "n": 60, "overlap": "low"}, {"g": 2, "n": 90, "overlap": "medium"}]
    specs = [{"type": "em", "n_init": 1}]
    fam = get_family(fam_spec)
    serial = run_benchmark(fam, SimulatorConfig(), cells, [build_baseline(s, fam) for s in specs], reps=4,
                           n_test=0, verbose=False)
    par = run_benchmark_parallel(fam_spec, SimulatorConfig(), cells, specs, reps=4, workers=2, reps_per_shard=2,
                                 shard_dir=tmp_path / "shards", n_test=0)
    key = ["n", "rep", "method"]
    a = serial.sort_values(key).reset_index(drop=True)
    b = par.sort_values(key).reset_index(drop=True)
    np.testing.assert_allclose(a["quotient_risk"], b["quotient_risk"], rtol=1e-10)
