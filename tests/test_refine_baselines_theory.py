"""Refinement, classical baselines, relabelling and theory utilities."""
import numpy as np

from aemnet.baselines import ABCSMCBaseline, EMBaseline, DirectMLEBaseline
from aemnet.evaluation.metrics import adjusted_rand_index, quotient_alignment
from aemnet.evaluation.relabel import quotient_frechet_mean, stephens_relabel
from aemnet.inference.refine import em_refine
from aemnet.mixture_params import MixtureParams
from aemnet.simulation.families import get_family
from aemnet.theory import collapse_index, estimate_em_contraction, ordering_bias_two_normals
from aemnet.theory.symmetry import naive_posterior_mean, symmetrize

FAM = get_family("gamma", k=1)
TRUE = MixtureParams([0.4, 0.6], {"alpha": np.array([[2.0], [30.0]]), "beta": np.array([[1.0], [0.5]])})


def _data(n=800, seed=0):
    return FAM.sample_dataset(np.random.default_rng(seed), TRUE, n)


def test_em_refine_monotone_and_close_to_truth():
    X, ev, _ = _data()
    start = MixtureParams([0.5, 0.5], {"alpha": np.array([[1.0], [10.0]]), "beta": np.array([[2.0], [1.0]])})
    mp, info = em_refine(FAM, X, ev, start, max_steps=200)
    tr = np.asarray(info["loglik_trace"])
    assert np.all(np.diff(tr) >= -1e-8)
    assert quotient_alignment(FAM, X, TRUE, mp)[0] < 0.02


def test_em_and_mle_baselines_agree():
    X, ev, labels = _data()
    em = EMBaseline(FAM, n_init=3).fit(X, ev, g=2)
    mle = DirectMLEBaseline(FAM, n_restarts=2).fit(X, ev, g=2)
    assert abs(em.loglik - mle.loglik) < 1e-2 * abs(em.loglik)
    assert adjusted_rand_index(labels, em.resp.argmax(1)) > 0.8


def test_em_bic_selects_order():
    X, ev, _ = _data(1500)
    fit = EMBaseline(FAM, n_init=2, g_max=3).fit(X, ev, g=None)
    assert fit.params.g == 2


def test_abc_smc_runs():
    X, ev, _ = _data(300)
    fit = ABCSMCBaseline(FAM, n_particles=60, n_rounds=2).fit(X, ev, g=2)
    assert fit.params.g == 2 and len(fit.extra["samples"]) > 0


def test_quotient_frechet_mean_undoes_label_switching():
    rng = np.random.default_rng(0)
    base = []
    for _ in range(300):
        a = TRUE.comps["alpha"] * np.exp(0.05 * rng.standard_normal((2, 1)))
        base.append(MixtureParams(TRUE.pi, {"alpha": a, "beta": TRUE.comps["beta"]}))
    switched = symmetrize(base, rng)
    est, _ = quotient_frechet_mean(FAM, switched)
    naive = naive_posterior_mean(FAM, switched)
    X, _, _ = _data(200)
    assert quotient_alignment(FAM, X, TRUE, est)[0] < 1e-2
    assert collapse_index(FAM, naive) < 0.2 * collapse_index(FAM, est)      # Thm 2a: collapse


def test_stephens_relabel_recovers_permutations():
    rng = np.random.default_rng(1)
    P0 = rng.dirichlet(np.ones(3) * 0.3, size=100)
    perms_true = [rng.permutation(3) for _ in range(40)]
    P = np.stack([P0[:, np.argsort(p)] for p in perms_true])     # draw m has columns permuted
    perms = stephens_relabel(P)
    aligned = np.stack([P[m][:, perms[m]] for m in range(40)])
    assert np.allclose(aligned, aligned[0], atol=1e-12)


def test_ordering_bias_positive_and_largest_at_zero():
    b = [ordering_bias_two_normals(d, 1.0) for d in (0.0, 0.5, 1.0, 3.0)]
    assert all(x >= 0 for x in b) and b[0] == max(b) and b[-1] < 0.05


def test_em_contracts_near_mle():
    X, ev, _ = _data(2000)
    mle = EMBaseline(FAM, n_init=3).fit(X, ev, g=2).params
    df = estimate_em_contraction(FAM, X, ev, mle, radii=(0.05,), n_dirs=5)
    assert float(df["median"].iloc[0]) < 1.0
