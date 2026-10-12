"""End-to-end wiring: a few training steps, checkpoint, reload, all inference modes.
(Verifies the pipeline runs -- it is NOT an accuracy experiment.)"""
import numpy as np
import torch

from aemnet.baselines import NeuralBaseline
from aemnet.inference import AmortizedEstimator
from aemnet.losses.quotient import QuotientLoss
from aemnet.mixture_params import MixtureParams
from aemnet.models import build_model
from aemnet.simulation.families import get_family
from aemnet.simulation.simulator import MixtureSimulator
from aemnet.training.trainer import AEMTrainer
from aemnet.utils.config import load_experiment


def test_train_checkpoint_reload_and_infer(tmp_path):
    exp = load_experiment("configs/train/smoke_gamma_k2.yaml",
                          [f"training.out_dir={tmp_path.as_posix()}", "training.steps=6", "training.val_every=3",
                           "training.log_every=3", "training.u_stats_samples=50", "training.val_batches=1",
                           "training.batch_size=4", "training.val_batch_size=4", "simulator.overlap_stratification=false"])
    torch.manual_seed(0)
    fam = get_family(exp.family)
    sim = MixtureSimulator(fam, exp.simulator)
    model = build_model(fam, exp.model, exp.simulator.g_max)
    hist = AEMTrainer(model, QuotientLoss(fam, exp.loss), sim, exp.training, seed=0,
                      experiment=exp.to_dict()).train()
    assert len(hist) == 2 and np.isfinite(hist[-1]["val_quotient"])
    assert (tmp_path / "best.pt").exists() and (tmp_path / "train_log.jsonl").exists()

    est = AmortizedEstimator.from_checkpoint(tmp_path / "best.pt")
    rng = np.random.default_rng(0)
    true = MixtureParams([0.5, 0.5], fam.sample_prior(rng, 2))
    X, ev, _ = fam.sample_dataset(rng, true, 300)
    mp, resp = est.estimate(X, g=2)
    assert mp.g == 2 and resp.shape == (300, 2)
    assert est.estimate(X)[0].g >= 1                                  # existence head chooses g
    assert len(est.estimate_many([X, X[:100]], g=2)) == 2
    post = est.posterior(X, g=2, n_samples=8, steps=4)
    assert len(post) == 8 and post[0].g == 2
    mp_h, info = est.hybrid(X, g=2)
    assert info["loglik_trace"][-1] >= info["loglik_trace"][0]
    for mode in ("amortized", "hybrid", "learned"):
        fit = NeuralBaseline(est, mode=mode).fit(X, ev, g=2)
        assert fit.params.g == 2 and np.isfinite(fit.loglik)
