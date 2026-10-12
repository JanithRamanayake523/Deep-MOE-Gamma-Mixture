"""Sampler correctness and density sanity checks."""
import numpy as np

from dmoegamma.params import Theta
from dmoegamma.simulation.cases import PAPER_CASES
from dmoegamma.simulation.mixture import (
    bayes_error,
    component_logpdf,
    logpdf,
    responsibilities,
    sample,
)


def test_sample_counts_match_labels():
    case = PAPER_CASES["case1"]
    X, labels = sample(case.theta, counts=case.counts, rng=0)
    assert X.shape == (case.n, 2)
    assert (labels == 0).sum() == 50 and (labels == 1).sum() == 50
    assert (X > 0).all()


def test_moments_match_theory():
    th = Theta(pi=[1.0], alpha=[[3.0, 5.0]], beta=[[2.0, 0.5]])
    X, _ = sample(th, n=200_000, rng=1)
    # mean = a*b, var = a*b^2
    assert abs(X[:, 0].mean() - 6.0) < 0.15
    assert abs(X[:, 1].mean() - 2.5) < 0.05
    assert abs(X[:, 0].var() - 12.0) < 0.5


def test_single_component_density_integrates_to_one():
    from scipy.integrate import trapezoid

    th = Theta(pi=[1.0], alpha=[[2.5]], beta=[[1.3]])
    grid = np.linspace(1e-6, 60.0, 200_001).reshape(-1, 1)
    dens = np.exp(logpdf(grid, th))
    assert abs(trapezoid(dens, grid[:, 0]) - 1.0) < 1e-3


def test_responsibilities_are_probabilities():
    case = PAPER_CASES["case4"]
    X, _ = sample(case.theta, n=500, rng=2)
    r = responsibilities(X, case.theta)
    assert r.shape == (500, 3)
    assert np.allclose(r.sum(1), 1.0, atol=1e-8)


def test_component_logpdf_matches_known_value():
    # Gamma(a=2, b=3) pdf at x=1: (1/b^2 / Gamma(2)) * x * exp(-x/3)
    th = Theta(pi=[1.0], alpha=[[2.0]], beta=[[3.0]])
    lp = component_logpdf(np.array([[1.0]]), th)[0, 0]
    expected = np.log(1.0 / 9.0 * np.exp(-1.0 / 3.0))
    assert abs(lp - expected) < 1e-10


def test_bayes_error_orders_overlap():
    separated = Theta(pi=[0.5, 0.5], alpha=[[10.0], [10.0]], beta=[[0.1], [10.0]])
    overlapping = Theta(pi=[0.5, 0.5], alpha=[[10.0], [10.0]], beta=[[1.0], [1.1]])
    e_sep = bayes_error(separated, n_mc=8000, rng=3)
    e_ovl = bayes_error(overlapping, n_mc=8000, rng=4)
    assert e_sep < 0.05
    assert e_ovl > 0.3
    # paper's Case 1 is near chance level (published method rate: 0.4988)
    e1 = bayes_error(PAPER_CASES["case1"].theta, n_mc=20000, rng=5)
    assert e1 > 0.4
