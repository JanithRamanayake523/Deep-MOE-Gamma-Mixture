"""Model forward shapes, constraints, and permutation invariance."""
import numpy as np
import torch

from dmoegamma.models.moe import DeepMoEGammaEstimator
from dmoegamma.utils.config import ModelConfig


def _model(**overrides):
    cfg = ModelConfig(enc_hidden=32, enc_out=32, head_hidden=32, **overrides)
    return DeepMoEGammaEstimator(cfg, k=2, g_max=4)


def test_forward_shapes_and_constraints():
    model = _model()
    X = torch.rand(5, 30, 2) * 3 + 0.1
    mask = torch.zeros(5, 30, dtype=torch.bool)
    for b, n in enumerate([10, 15, 20, 25, 30]):
        mask[b, :n] = True
    out = model(X, mask)
    assert out["pi"].shape == (5, 4)
    assert out["alpha"].shape == (5, 4, 2)
    assert out["beta"].shape == (5, 4, 2)
    assert out["resp"].shape == (5, 30, 4)
    assert out["g_logits"].shape == (5, 4)
    assert torch.allclose(out["pi"].sum(-1), torch.ones(5), atol=1e-5)
    assert (out["alpha"] > 0).all() and (out["beta"] > 0).all()


def test_permutation_invariance_in_data():
    torch.manual_seed(0)
    model = _model()
    model.eval()
    X = torch.rand(1, 50, 2) * 3 + 0.1
    mask = torch.ones(1, 50, dtype=torch.bool)
    perm = torch.randperm(50)
    with torch.no_grad():
        out1 = model(X, mask)
        out2 = model(X[:, perm], mask)
    assert torch.allclose(out1["pi"], out2["pi"], atol=1e-5)
    assert torch.allclose(out1["alpha"], out2["alpha"], atol=1e-5)


def test_estimate_returns_theta_and_resp():
    torch.manual_seed(0)
    model = _model()
    X = np.random.default_rng(0).gamma(3.0, 2.0, size=(80, 2))
    theta, resp = model.estimate(X, g=2)
    assert theta.g == 2 and theta.k == 2
    assert resp.shape == (80, 2)
    assert np.isclose(theta.pi.sum(), 1.0)
