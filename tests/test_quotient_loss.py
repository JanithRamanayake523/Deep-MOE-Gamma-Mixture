"""The quotient loss is permutation-invariant, zero at the truth, and equals the
brute-force minimum over label permutations."""
import itertools

import numpy as np
import torch

from aemnet.losses.quotient import QuotientLoss, match_components
from aemnet.models import build_model
from aemnet.simulation.families import get_family
from aemnet.simulation.simulator import MixtureSimulator, SimulatorConfig
from aemnet.utils.config import AEMConfig, LossConfig


def _setup(g_max=3):
    fam = get_family("gamma", k=2)
    sim = MixtureSimulator(fam, SimulatorConfig(g_min=g_max, g_max=g_max, n_min=100, n_max=200,
                                                overlap_stratification=False))
    batch = sim.sample_batch(np.random.default_rng(0), 4)
    torch.manual_seed(0)
    model = build_model(fam, AEMConfig(d_model=16, flow_head=False, T=1), g_max)
    return fam, batch, model


def _fake_out(model, batch, z, pi):
    X, obs = batch["X"], batch["obs_mask"]
    Xc, loc, scale = model.canonical(X, obs)
    out = {"z": z, "pi": pi, "steps": [{"z": z, "pi": pi}], "loc": loc, "scale": scale, "Xc": Xc,
           "resp": None, "exist_logits": None, "slots": torch.zeros(z.shape[0], z.shape[1], 16)}
    return model.finalize(out)


def test_zero_at_truth_and_permutation_invariance():
    fam, batch, model = _setup()
    loss_fn = QuotientLoss(fam, LossConfig(deep_supervision=0.0, resp=0.0, exist=0.0, flow=0.0))
    Xc, loc, scale = model.canonical(batch["X"], batch["obs_mask"])
    tp = {k: batch[f"tp_{k}"] for k in fam.param_shapes()}
    z_true = model.u_to_z(fam.encode(fam.canonicalize(tp, loc, scale)))
    out = _fake_out(model, batch, z_true, batch["tp_pi"])
    l0, _, _ = loss_fn(out, batch, model)
    assert float(l0) < 1e-10
    perm = torch.tensor([2, 0, 1])
    out_p = _fake_out(model, batch, z_true[:, perm] + 0.3, batch["tp_pi"][:, perm])
    out_q = _fake_out(model, batch, z_true + 0.3, batch["tp_pi"])
    lp, _, _ = loss_fn(out_p, batch, model)
    lq, _, _ = loss_fn(out_q, batch, model)
    torch.testing.assert_close(lp, lq)


def test_hungarian_equals_bruteforce():
    rng = np.random.default_rng(1)
    C = torch.from_numpy(rng.uniform(size=(5, 4, 4)))
    assign = match_components(C, torch.full((5,), 4))
    for b in range(5):
        best = min(sum(C[b, i, p[i]] for i in range(4)) for p in itertools.permutations(range(4)))
        got = sum(C[b, i, assign[b, i]] for i in range(4))
        assert abs(float(best - got)) < 1e-12


def test_rectangular_matching_and_padding():
    C = torch.rand(2, 3, 5)
    assign = match_components(C, torch.tensor([2, 3]))
    assert assign[0, 2] == -1 and (assign[0, :2] >= 0).all()
    assert len(set(assign[1].tolist())) == 3


def test_naive_mode_is_not_invariant():
    fam, batch, model = _setup()
    loss_fn = QuotientLoss(fam, LossConfig(match="naive", deep_supervision=0.0, resp=0.0, exist=0.0, flow=0.0))
    Xc, loc, scale = model.canonical(batch["X"], batch["obs_mask"])
    tp = {k: batch[f"tp_{k}"] for k in fam.param_shapes()}
    z_true = model.u_to_z(fam.encode(fam.canonicalize(tp, loc, scale)))
    perm = torch.tensor([2, 0, 1])
    l_id, _, _ = loss_fn(_fake_out(model, batch, z_true, batch["tp_pi"]), batch, model)
    l_perm, _, _ = loss_fn(_fake_out(model, batch, z_true[:, perm], batch["tp_pi"][:, perm]), batch, model)
    assert float(l_id) < 1e-10 < float(l_perm)
