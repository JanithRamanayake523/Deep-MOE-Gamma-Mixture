"""Sample-based Bayesian baselines.

AdaptiveMCMCBaseline (any family)
    Adaptive random-walk Metropolis (Haario et al., 2001) on canonicalised data in
    v = [alr(pi), u_1..u_g]; weakly-informative Gaussian prior on u from the family
    prior statistics, Dirichlet(1) on pi (alr Jacobian included). Point estimate =
    quotient Fréchet mean (Thm 3 target); samples in `extra["samples"]`.

NumPyroNUTSBaseline (gamma family; optional dependency `numpyro`)
    NUTS on the marginalised gamma mixture -- the "gold standard" of E1/E3/E5.
"""
from __future__ import annotations

import numpy as np

from ..evaluation.relabel import quotient_frechet_mean
from ..mixture_params import MixtureParams
from ..simulation.families.base import Family
from ..simulation.families.stats import prior_u_stats
from .em import EMBaseline
from .generic import GenericBaseline, GenericFit
from .mle import pack_v, unpack_v


class AdaptiveMCMCBaseline(GenericBaseline):
    name = "mcmc"

    def __init__(self, family: Family, n_iter: int = 20000, burn_in: int = 5000, thin: int = 10,
                 prior_inflation: float = 2.0, adapt_start: int = 1000):
        super().__init__(family)
        self.n_iter, self.burn_in, self.thin = n_iter, burn_in, thin
        self.prior_inflation, self.adapt_start = prior_inflation, adapt_start

    def log_prior(self, v, g, u_mean, u_sd):
        u = v[g - 1:].reshape(g, -1)
        a = np.concatenate([v[: g - 1], [0.0]])
        log_pi = a - a.max() - np.log(np.exp(a - a.max()).sum())
        return float(-0.5 * (((u - u_mean) / u_sd) ** 2).sum() + log_pi.sum())   # Dirichlet(1) + alr Jacobian

    def _fit(self, X, events, g, rng):
        fam = self.family
        loc, scale = fam.canonical_loc_scale_np(X)
        Xc = (X - loc) / scale
        u_mean, u_std = prior_u_stats(fam)
        u_sd = self.prior_inflation * u_std
        init = EMBaseline(fam, n_init=2, max_iter=100).fit(Xc, events, g, seed=int(rng.integers(2 ** 31))).params
        v = pack_v(fam, init)
        D = v.size

        def log_post(v):
            mp = unpack_v(fam, v, g)
            ll = float(fam.mixture_logpdf_np(Xc, events, mp).sum())
            return ll + self.log_prior(v, g, u_mean, u_sd) if np.isfinite(ll) else -np.inf

        lp = log_post(v)
        cov = np.diag(np.full(D, 0.01))
        chain, accepted = [], 0
        mean_run, M2, cnt = np.zeros(D), np.zeros((D, D)), 0
        for it in range(self.n_iter):
            prop = v + rng.multivariate_normal(np.zeros(D), cov)
            lp_prop = log_post(prop)
            if np.log(rng.uniform()) < lp_prop - lp:
                v, lp = prop, lp_prop
                accepted += 1
            cnt += 1
            delta = v - mean_run
            mean_run += delta / cnt
            M2 += np.outer(delta, v - mean_run)
            if it >= self.adapt_start and it % 100 == 0:
                cov = (2.38 ** 2 / D) * (M2 / max(cnt - 1, 1)) + 1e-8 * np.eye(D)
            if it >= self.burn_in and (it - self.burn_in) % self.thin == 0:
                chain.append(v.copy())
        samples_c = [unpack_v(fam, s, g) for s in chain]
        est_c, aligned_c = quotient_frechet_mean(fam, samples_c, u_std)
        est = fam.uncanonicalize_np(est_c, loc, scale)
        samples = [fam.uncanonicalize_np(s, loc, scale) for s in aligned_c]
        return GenericFit(est, fam.responsibilities_np(X, events, est),
                          extra={"samples": samples, "acceptance": accepted / self.n_iter})


class NumPyroNUTSBaseline(GenericBaseline):
    name = "nuts"

    def __init__(self, family: Family, num_warmup: int = 1000, num_samples: int = 1000, chains: int = 1):
        if family.name != "gamma":
            raise ValueError("NumPyroNUTSBaseline implements the gamma family only")
        super().__init__(family)
        self.num_warmup, self.num_samples, self.chains = num_warmup, num_samples, chains

    def _fit(self, X, events, g, rng):
        try:
            import jax
            import jax.numpy as jnp
            import numpyro
            import numpyro.distributions as dist
            from numpyro.infer import MCMC, NUTS
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("NumPyroNUTSBaseline needs `pip install numpyro jax`") from exc
        fam = self.family
        loc, scale = fam.canonical_loc_scale_np(X)
        Xc = X / scale
        u_mean, u_std = prior_u_stats(fam)
        k = fam.k

        def model(x):
            pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(g)))
            la = numpyro.sample("log_alpha", dist.Normal(u_mean[:k], 2 * u_std[:k]).expand([g, k]).to_event(2))
            lb = numpyro.sample("log_beta", dist.Normal(u_mean[k:], 2 * u_std[k:]).expand([g, k]).to_event(2))
            a, b = jnp.exp(la), jnp.exp(lb)
            comp = dist.Gamma(a[None], 1.0 / b[None]).log_prob(x[:, None, :]).sum(-1)       # (n, g)
            numpyro.factor("ll", jax.scipy.special.logsumexp(comp + jnp.log(pi)[None], axis=-1).sum())

        mcmc = MCMC(NUTS(model), num_warmup=self.num_warmup, num_samples=self.num_samples,
                    num_chains=self.chains, progress_bar=False)
        mcmc.run(jax.random.PRNGKey(int(rng.integers(2 ** 31))), jnp.asarray(Xc))
        s = mcmc.get_samples()
        samples_c = [MixtureParams(np.asarray(s["pi"][i]), {"alpha": np.exp(np.asarray(s["log_alpha"][i])),
                                                            "beta": np.exp(np.asarray(s["log_beta"][i]))})
                     for i in range(len(s["pi"]))]
        est_c, aligned_c = quotient_frechet_mean(fam, samples_c, u_std)
        est = fam.uncanonicalize_np(est_c, loc, scale)
        samples = [fam.uncanonicalize_np(m, loc, scale) for m in aligned_c]
        return GenericFit(est, fam.responsibilities_np(X, events, est), extra={"samples": samples})
