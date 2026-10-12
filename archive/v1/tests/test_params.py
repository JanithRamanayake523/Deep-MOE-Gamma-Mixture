"""Packing roundtrips and parameter bookkeeping."""
import numpy as np

from dmoegamma.params import (
    Theta,
    pack,
    pack_unconstrained,
    padded_batch,
    unpack,
    unpack_unconstrained,
)


def _theta(g=3, k=2, seed=0):
    rng = np.random.default_rng(seed)
    return Theta(
        pi=rng.dirichlet(np.ones(g)),
        alpha=rng.uniform(0.5, 10.0, (g, k)),
        beta=rng.uniform(0.5, 10.0, (g, k)),
    )


def test_free_dimension():
    th = _theta(g=3, k=2)
    assert th.dim == 2 * 3 * 2 + 3 - 1 == 14


def test_pack_roundtrip():
    th = _theta()
    back = unpack(pack(th), th.g, th.k)
    assert np.allclose(back.pi, th.pi)
    assert np.allclose(back.alpha, th.alpha)
    assert np.allclose(back.beta, th.beta)


def test_unconstrained_roundtrip():
    th = _theta(g=4, k=3)
    back = unpack_unconstrained(pack_unconstrained(th), th.g, th.k)
    assert np.allclose(back.pi, th.pi, atol=1e-8)
    assert np.allclose(back.alpha, th.alpha, atol=1e-8)
    assert np.allclose(back.beta, th.beta, atol=1e-8)


def test_padded_batch_shapes():
    thetas = [_theta(g=2), _theta(g=4, seed=1)]
    batch = padded_batch(thetas, g_max=4)
    assert batch["pi"].shape == (2, 4)
    assert batch["alpha"].shape == (2, 4, 2)
    assert batch["comp_mask"].sum().item() == 6
    assert batch["g"].tolist() == [2, 4]


def test_sorted_and_permuted_are_inverses():
    th = _theta()
    perm = np.array([2, 0, 1])
    assert np.allclose(th.permuted(perm).alpha, th.alpha[perm])
