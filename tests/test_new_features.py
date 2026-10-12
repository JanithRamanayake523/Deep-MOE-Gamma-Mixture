"""Parameter bank, aux-loss fade, batched GPU EM, noisy-gamma family."""
import time
import warnings
from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy import integrate
from scipy.stats import gamma as gamma_dist, norm

from aemnet.baselines import BatchedEM, EMBaseline
from aemnet.inference.refine import em_step
from aemnet.losses.quotient import QuotientLoss
from aemnet.mixture_params import MixtureParams
from aemnet.models import build_model
from aemnet.simulation.bank import ParameterBank
from aemnet.simulation.families import get_family
from aemnet.simulation.simulator import MixtureSimulator, SimulatorConfig
from aemnet.training.trainer import AEMTrainer
from aemnet.utils.config import AEMConfig, LossConfig, TrainConfig


# ------------------------------------------------------------------ parameter bank
@pytest.fixture(scope="module")
def bank_path(tmp_path_factory):
    fam = get_family("gamma", k=1)
    sim = MixtureSimulator(fam, SimulatorConfig(g_min=2, g_max=3, overlap_n_mc=300))
    bank = ParameterBank.build(sim, per_cell=40, workers=1, chunk=100, verbose=False)
    path = tmp_path_factory.mktemp("bank") / "gamma_k1.npz"
    bank.save(path)
    return path


def test_bank_roundtrip_bins_and_splits(bank_path):
    bank = ParameterBank.load(bank_path)
    fam = get_family("gamma", k=1)
    cfg = SimulatorConfig(g_min=2, g_max=3, overlap_n_mc=300, param_bank=str(bank_path))
    assert bank.compatible(fam, cfg)
    for (g, b), cell in bank.cells.items():
        lo, hi = bank.bins[b]
        assert cell["pi"].shape[1] == g and np.all((cell["err"] >= lo) & (cell["err"] < hi))
    rng = np.random.default_rng(0)
    N = bank.size(2, 0)
    n_val = max(1, int(bank.val_frac * N))
    val_pis = {tuple(bank.cells[(2, 0)]["pi"][i]) for i in range(N - n_val, N)}
    for _ in range(50):
        mp, _ = bank.draw(rng, 2, 0, "train")
        assert tuple(mp.pi) not in val_pis                     # validation tail never used for training
        mp, _ = bank.draw(rng, 2, 0, "val")
        assert tuple(mp.pi) in val_pis


def test_simulator_uses_bank_and_is_fast(bank_path):
    fam = get_family("gamma", k=1)
    cfg = SimulatorConfig(g_min=2, g_max=3, overlap_n_mc=300, param_bank=str(bank_path), n_max=300)
    sim = MixtureSimulator(fam, cfg)
    assert sim.bank is not None
    rng = np.random.default_rng(1)
    t = time.time()
    batch = sim.sample_batch(rng, 32)
    assert time.time() - t < 2.0 and batch["X"].shape[0] == 32
    # incompatible bank (different overlap bins) is ignored with a warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        other = MixtureSimulator(fam, replace(cfg, overlap_bins=[(0.0, 0.5)]))
        assert other.bank is None and any("different" in str(x.message) for x in w)


# ------------------------------------------------------------------ aux fade
def test_aux_fade_schedule_and_pure_phase(tmp_path):
    fam = get_family("gamma", k=1)
    sim = MixtureSimulator(fam, SimulatorConfig(g_max=2, n_max=100, overlap_stratification=False))
    model = build_model(fam, AEMConfig(d_model=16, expert_hidden=16, T=1, flow_head=False), 2)
    loss = QuotientLoss(fam, LossConfig())
    cfg = TrainConfig(steps=20, batch_size=4, val_every=5, val_batches=1, val_batch_size=4, log_every=5,
                        u_stats_samples=20, warmup_steps=2, aux_fade_frac=0.4, out_dir=str(tmp_path))
    tr = AEMTrainer(model, loss, sim, cfg)
    tr._plan_fade(int(0.6 * cfg.steps))
    assert tr.aux_scale(5) == 1.0 and tr.aux_scale(16) == 0.0 and 0.0 < tr.aux_scale(13) < 1.0
    hist = tr.train()
    pure = [h for h in hist if h["pure_phase"]]
    assert pure and (tmp_path / "best.pt").exists() and (tmp_path / "best_prefade.pt").exists()
    assert loss.aux_scale == 1.0 and model.exist_detach is False       # restored after training


def test_pure_phase_removes_aux_gradients():
    fam = get_family("gamma", k=1)
    sim = MixtureSimulator(fam, SimulatorConfig(g_max=2, n_max=100, overlap_stratification=False))
    batch = sim.sample_batch(np.random.default_rng(0), 4)
    torch.manual_seed(0)
    model = build_model(fam, AEMConfig(d_model=16, expert_hidden=16, T=2, flow_head=False), 2)
    loss = QuotientLoss(fam, LossConfig(exist=1.0, resp=1.0, deep_supervision=1.0))
    loss.aux_scale, model.exist_detach = 0.0, True
    out = model(batch["X"], batch["events"], batch["obs_mask"])
    total, stats, _ = loss(out, batch, model)
    pure = QuotientLoss(fam, LossConfig(exist=0.0, resp=0.0, deep_supervision=0.0))
    l_pure, _, _ = pure(out, batch, model)
    # backbone gradient of the pure-phase loss == gradient of L_Q (+ unmatched-pi) alone
    g1 = torch.autograd.grad(total, model.experts.nets[0][0].weight, retain_graph=True)[0]
    g2 = torch.autograd.grad(l_pure, model.experts.nets[0][0].weight)[0]
    torch.testing.assert_close(g1, g2)


# ------------------------------------------------------------------ batched EM
@pytest.mark.parametrize("name,k", [("gamma", 1), ("gamma", 2), ("gaussian", 2)])
def test_batched_em_step_matches_serial(name, k):
    fam = get_family(name, k=k)
    rng = np.random.default_rng(0)
    true = MixtureParams([0.4, 0.6], fam.sample_prior(rng, 2))
    Xs = [fam.sample_dataset(rng, true, n)[0] for n in (150, 400)]
    starts = []
    for X in Xs:
        u = fam.encode_np(true.comps) + 0.2 * rng.standard_normal((2, fam.d))
        starts.append(MixtureParams([0.5, 0.5], fam.decode_np(u)))
    bem = BatchedEM(fam, max_iter=1, device="cpu")
    refined, info = bem.refine(Xs, starts)
    for X, s, r in zip(Xs, starts, refined):
        ref, _ = em_step(fam, X, np.ones_like(X), s)
        np.testing.assert_allclose(r.pi, ref.pi, rtol=1e-6)
        for key in ref.comps:
            np.testing.assert_allclose(r.comps[key], ref.comps[key], rtol=1e-5, atol=1e-8)


def test_batched_em_fit_many_matches_serial_em_quality():
    fam = get_family("gamma", k=1)
    rng = np.random.default_rng(3)
    true = MixtureParams([0.4, 0.6], {"alpha": np.array([[2.0], [30.0]]), "beta": np.array([[1.0], [0.5]])})
    Xs = [fam.sample_dataset(rng, true, 500)[0] for _ in range(6)]
    fits, info = BatchedEM(fam, n_restarts=5, device="cpu").fit_many(Xs, 2)
    for X, mp in zip(Xs, fits):
        ll_b = fam.mixture_logpdf_np(X, None, mp).sum()
        ll_s = EMBaseline(fam, n_init=3).fit(X, g=2).loglik
        assert ll_b >= ll_s - 1e-3 * abs(ll_s)
    assert info["iterations"].shape == (6,)


# ------------------------------------------------------------------ noisy gamma
@pytest.mark.parametrize("a,b,s", [(0.5, 1.0, 0.035), (0.8, 1.0, 0.2), (3.0, 2.0, 0.5), (20.0, 0.3, 0.1)])
def test_noisy_gamma_density(a, b, s):
    fam = get_family("noisy_gamma", k=1)
    mp = MixtureParams([1.0], {"alpha": np.array([[a]]), "beta": np.array([[b]]), "sigma": np.array([[s]])})
    pdf = lambda y: float(np.exp(fam.component_logpdf_np(np.array([[y]]), None, mp))[0, 0])  # noqa: E731
    sd = np.sqrt(a) * b
    total = sum(integrate.quad(pdf, lo, hi, limit=800)[0] for lo, hi in [(-40 * s, 0), (0, a * b), (a * b, a * b + 80 * sd)])
    assert abs(total - 1.0) < 1e-5
    if a >= 1:  # direct convolution reference (smooth integrand)
        for y in (a * b - 2 * sd, a * b, a * b + 3 * sd):
            ref = integrate.quad(lambda x: gamma_dist.pdf(x, a, scale=b) * norm.pdf(y - x, scale=s), 0,
                                 a * b + 80 * sd, points=[max(y, 1e-9)], limit=2000, epsrel=1e-11)[0]
            assert abs(pdf(y) - ref) / ref < 1e-5


def test_noisy_gamma_is_likelihood_free_for_the_network():
    fam = get_family("noisy_gamma", k=2)
    assert not fam.has_torch_logprob
    model = build_model(fam, AEMConfig(d_model=16, expert_hidden=16, T=2, flow_head=False), 2)
    assert not model.eblocks[0].use_ll
