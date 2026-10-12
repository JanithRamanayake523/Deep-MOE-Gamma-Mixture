"""Prop 1: data-permutation invariance, slot equivariance of the E/M blocks, exact
scale / location-scale equivariance via canonicalisation."""
import numpy as np
import torch

from aemnet.mixture_params import MixtureParams
from aemnet.models import build_model
from aemnet.models.aem import EBlock, MBlock
from aemnet.simulation.families import get_family
from aemnet.utils.config import AEMConfig

D = torch.float64


def _model(fam, **kw):
    torch.manual_seed(0)
    cfg = AEMConfig(d_model=32, expert_hidden=32, T=3, flow_head=False, **kw)
    m = build_model(fam, cfg, n_slots=3).double().eval()
    return m


def _data(fam, n=120, seed=0):
    rng = np.random.default_rng(seed)
    mp = MixtureParams(np.ones(2) / 2, fam.sample_prior(rng, 2))
    X, ev, _ = fam.sample_dataset(rng, mp, n)
    return torch.from_numpy(X)[None], torch.from_numpy(ev)[None], torch.ones(1, n, dtype=torch.bool)


def test_data_permutation_invariance():
    fam = get_family("gamma", k=2)
    m = _model(fam)
    X, ev, mask = _data(fam)
    perm = torch.randperm(X.shape[1])
    with torch.no_grad():
        o1, o2 = m(X, ev, mask), m(X[:, perm], ev[:, perm], mask)
    torch.testing.assert_close(o1["z"], o2["z"], rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(o1["pi"], o2["pi"], rtol=1e-8, atol=1e-8)


def test_exact_scale_equivariance_gamma():
    fam = get_family("gamma", k=2)
    m = _model(fam)
    X, ev, mask = _data(fam)
    c = torch.tensor([7.3, 0.02], dtype=D)
    with torch.no_grad():
        o1, o2 = m(X, ev, mask), m(X * c, ev, mask)
    torch.testing.assert_close(o2["params"]["alpha"], o1["params"]["alpha"], rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(o2["params"]["beta"], o1["params"]["beta"] * c, rtol=1e-8, atol=1e-10)


def test_exact_location_scale_equivariance_gaussian():
    fam = get_family("gaussian", k=2)
    m = _model(fam)
    X, ev, mask = _data(fam)
    a, b = torch.tensor([3.0, 0.5], dtype=D), torch.tensor([-4.0, 10.0], dtype=D)
    with torch.no_grad():
        o1, o2 = m(X, ev, mask), m(b + a * X, ev, mask)
    torch.testing.assert_close(o2["params"]["mu"], b + a * o1["params"]["mu"], rtol=1e-7, atol=1e-8)
    torch.testing.assert_close(o2["params"]["L"], a[:, None] * o1["params"]["L"], rtol=1e-7, atol=1e-8)


def test_eblock_mblock_slot_equivariance():
    fam = get_family("gamma", k=1)
    cfg = AEMConfig(d_model=16)
    torch.manual_seed(1)
    eb, mb = EBlock(cfg, fam).double(), MBlock(cfg, fam).double()
    B, n, S, H = 2, 50, 4, 16
    rho, slots, ctx = torch.randn(B, n, H, dtype=D), torch.randn(B, S, H, dtype=D), torch.randn(B, H, dtype=D)
    Xc = torch.rand(B, n, 1, dtype=D) * 3 + 0.1
    ev, mask = torch.ones_like(Xc), torch.ones(B, n, dtype=torch.bool)
    params = {"alpha": torch.rand(B, S, 1, dtype=D) * 3 + 0.5, "beta": torch.rand(B, S, 1, dtype=D) + 0.2}
    pi = torch.softmax(torch.randn(B, S, dtype=D), -1)
    ss = fam.suff_stats(Xc, ev)
    perm = torch.randperm(S)
    with torch.no_grad():
        r1 = eb(rho, slots, params, pi, Xc, ev, mask)
        r2 = eb(rho, slots[:, perm], {k: v[:, perm] for k, v in params.items()}, pi[:, perm], Xc, ev, mask)
        torch.testing.assert_close(r2, r1[..., perm])
        s1, p1 = mb(rho, ss, r1, slots, ctx, mask)
        s2, p2 = mb(rho, ss, r1[..., perm], slots[:, perm], ctx, mask)
        torch.testing.assert_close(s2, s1[:, perm])
        torch.testing.assert_close(p2, p1[:, perm])
