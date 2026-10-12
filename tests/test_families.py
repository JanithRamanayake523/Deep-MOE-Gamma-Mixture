"""Family contracts: coordinates, canonicalisation, densities."""
import numpy as np
import pytest
from scipy import integrate
from scipy.special import ive, gammaln

from aemnet.mixture_params import MixtureParams
from aemnet.simulation.families import get_family

SPECS = [
    {"name": "gamma", "k": 2}, {"name": "gaussian", "k": 3}, {"name": "weibull", "k": 2},
    {"name": "student_t", "k": 2}, {"name": "negbin", "k": 2}, {"name": "copula_gamma", "k": 3},
    {"name": "kibble_gamma", "k": 2}, {"name": "gandk", "k": 2},
    {"name": "censored", "base": {"name": "gamma", "k": 1}},
    {"name": "censored", "base": {"name": "weibull", "k": 2}},
]


@pytest.fixture(params=SPECS, ids=lambda s: s["name"] + str(s.get("k", s.get("base", {}).get("k"))))
def family(request):
    return get_family(dict(request.param))


def _mp(fam, g=3, seed=0):
    rng = np.random.default_rng(seed)
    return MixtureParams(np.ones(g) / g, fam.sample_prior(rng, g))


def test_encode_decode_roundtrip(family):
    mp = _mp(family)
    back = family.decode_np(family.encode_np(mp.comps))
    for k in mp.comps:
        np.testing.assert_allclose(back[k], mp.comps[k], rtol=1e-8, atol=1e-10)


def test_canonicalize_roundtrip(family):
    mp = _mp(family)
    loc, scale = np.linspace(-1, 1, family.k), np.linspace(0.5, 3, family.k)
    back = family.uncanonicalize_np(family.canonicalize_np(mp, loc, scale), loc, scale)
    for k in mp.comps:
        np.testing.assert_allclose(back.comps[k], mp.comps[k], rtol=1e-10, atol=1e-12)


def test_sampling_and_density_finite(family):
    mp = _mp(family)
    X, ev, lab = family.sample_dataset(np.random.default_rng(1), mp, 300)
    assert X.shape == (300, family.k) and ev.shape == X.shape and lab.shape == (300,)
    lp = family.component_logpdf_np(X, ev, mp)
    assert lp.shape == (300, 3) and np.all(np.isfinite(lp))


def test_density_change_of_variables(family):
    """log p(a x + b | canon^{-1} theta) = log p(x | theta) - sum log a  (the parameter action is right)."""
    if family.equivariance == "none" or family.name.startswith("censored"):
        pytest.skip("no continuous group / censoring is dataset-level")
    mp = _mp(family)
    X, ev, _ = family.sample_dataset(np.random.default_rng(2), mp, 200)
    a = np.linspace(0.3, 4.0, family.k)
    b = np.linspace(-2, 2, family.k) if family.equivariance == "location_scale" else np.zeros(family.k)
    mp2 = family.uncanonicalize_np(mp, b, a)
    lhs = family.mixture_logpdf_np(b + a * X, ev, mp2)
    rhs = family.mixture_logpdf_np(X, ev, mp) - np.log(a).sum()
    np.testing.assert_allclose(lhs, rhs, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("spec", [{"name": "gamma", "k": 1}, {"name": "weibull", "k": 1},
                                  {"name": "student_t", "k": 1}, {"name": "gandk", "k": 1}])
def test_univariate_density_integrates_to_one(spec):
    fam = get_family(dict(spec))
    mp = _mp(fam, g=1, seed=3)
    f = lambda x: float(np.exp(fam.component_logpdf_np(np.array([[x]]), None, mp))[0, 0])  # noqa: E731
    lo = 0.0 if fam.positive_support else -np.inf
    val, _ = integrate.quad(f, lo, np.inf, limit=400)
    assert abs(val - 1.0) < 2e-3


def test_negbin_pmf_sums_to_one():
    fam = get_family("negbin", k=1)
    mp = MixtureParams([1.0], {"m": np.array([[7.0]]), "r": np.array([[2.5]])})
    x = np.arange(0, 2000, dtype=float)[:, None]
    assert abs(np.exp(fam.component_logpdf_np(x, None, mp)).sum() - 1.0) < 1e-8


def test_kibble_series_matches_bessel_form():
    fam = get_family("kibble_gamma", k=2)
    a, b1, b2, rho = 2.3, 1.5, 0.7, 0.55
    mp = MixtureParams([1.0], {"alpha": np.array([[a]]), "beta": np.array([[b1, b2]]), "rho": np.array([[rho]])})
    X = np.random.default_rng(4).gamma(2.0, 1.0, size=(50, 2))
    x, y = X[:, 0] / b1, X[:, 1] / b2
    arg = 2 * np.sqrt(rho * x * y) / (1 - rho)
    exact = ((a - 1) / 2 * np.log(x * y / rho) - (x + y) / (1 - rho) - gammaln(a) - np.log(1 - rho)
             + np.log(ive(a - 1, arg)) + arg - np.log(b1 * b2))
    np.testing.assert_allclose(fam.component_logpdf_np(X, None, mp)[:, 0], exact, rtol=1e-6)


def test_copula_identity_equals_independent_gamma():
    cop, gam = get_family("copula_gamma", k=2), get_family("gamma", k=2)
    comps = {"alpha": np.array([[2.0, 5.0]]), "beta": np.array([[1.0, 0.3]])}
    X = np.random.default_rng(5).gamma(3.0, 1.0, size=(40, 2))
    lp_c = cop.component_logpdf_np(X, None, MixtureParams([1.0], {**comps, "Lc": np.eye(2)[None]}))
    lp_g = gam.component_logpdf_np(X, None, MixtureParams([1.0], comps))
    np.testing.assert_allclose(lp_c, lp_g, rtol=1e-6, atol=1e-6)


def test_censored_loglik_uses_survival():
    fam = get_family({"name": "censored", "base": {"name": "weibull", "k": 1}})
    mp = MixtureParams([1.0], {"kappa": np.array([[1.5]]), "lam": np.array([[2.0]])})
    X = np.array([[1.0], [3.0]])
    ev = np.array([[1.0], [0.0]])
    lp = fam.component_logpdf_np(X, ev, mp)[:, 0]
    np.testing.assert_allclose(lp[1], -(3.0 / 2.0) ** 1.5, rtol=1e-10)


def test_closed_form_msteps_match_numerical():
    for name in ("gamma", "weibull", "gaussian"):
        fam = get_family(name, k=1)
        mp = _mp(fam, g=1, seed=6)
        X, ev, _ = fam.sample_dataset(np.random.default_rng(7), mp, 2000)
        w = np.random.default_rng(8).uniform(0.2, 1.0, size=2000)
        closed = fam.mstep(X, ev, w)
        numer = type(fam).__mro__[1].mstep(fam, X, ev, w) if name != "gaussian" else closed
        ll = lambda c: float((w * fam.component_logpdf_np(X, ev, MixtureParams([1.0], {k: v[None] for k, v in c.items()}))[:, 0]).sum())  # noqa: E731
        assert ll(closed) >= ll(numer) - 1e-3 * abs(ll(numer))
