"""Classical baselines recover parameters on an easy, well-separated case."""
import numpy as np

from dmoegamma.baselines import (
    DirectMLEBaseline,
    ECMBaseline,
    MethodOfMomentsBaseline,
    WHMclustBaseline,
)
from dmoegamma.losses.matching import hungarian_match
from dmoegamma.params import Theta
from dmoegamma.simulation.mixture import sample

TRUE = Theta(pi=[0.5, 0.5], alpha=[[12.0, 8.0], [6.0, 4.0]], beta=[[0.2, 0.3], [2.5, 3.0]])


def _data(n=2000, seed=7):
    return sample(TRUE, n=n, rng=seed)[0]


def _check_close(res_theta, tol=0.35):
    perm = hungarian_match(TRUE, res_theta)
    al = res_theta.permuted(perm)
    rel_a = np.abs(al.alpha - TRUE.alpha) / TRUE.alpha
    rel_b = np.abs(al.beta - TRUE.beta) / TRUE.beta
    assert (rel_a < tol).all() and (rel_b < tol).all()
    assert (np.abs(al.pi - TRUE.pi) < 0.1).all()


def test_ecm_recovers_parameters():
    res = ECMBaseline().fit(_data(), g=2)
    assert res.converged
    _check_close(res.theta, tol=0.20)


def test_direct_mle_recovers_parameters():
    res = DirectMLEBaseline(n_restarts=3).fit(_data(), g=2)
    _check_close(res.theta, tol=0.20)


def test_wh_mclust_runs_and_is_reasonable():
    res = WHMclustBaseline().fit(_data(), g=2)
    assert res.theta.g == 2
    _check_close(res.theta, tol=0.50)  # approximation error: looser tolerance


def test_mom_runs():
    res = MethodOfMomentsBaseline().fit(_data(), g=2)
    assert res.theta.g == 2
    assert (res.theta.alpha > 0).all()
