"""Thm 1 (constructive): a hand-set AEM block equals one EM iteration."""
import numpy as np
import pytest
import torch

from aemnet.inference.refine import em_step
from aemnet.mixture_params import MixtureParams
from aemnet.models.exact import ExactEMBlock
from aemnet.simulation.families import get_family


@pytest.mark.parametrize("name,k", [("gamma", 1), ("gamma", 3), ("gaussian", 2)])
def test_exact_block_equals_em_step(name, k):
    fam = get_family(name, k=k)
    rng = np.random.default_rng(0)
    true = MixtureParams(np.array([0.3, 0.7]), fam.sample_prior(rng, 2))
    X, ev, _ = fam.sample_dataset(rng, true, 800)
    # a non-degenerate start: the truth perturbed in unconstrained coordinates
    u = fam.encode_np(true.comps) + 0.3 * rng.standard_normal((2, fam.d))
    start = MixtureParams(np.array([0.45, 0.55]), fam.decode_np(u))

    ref, _ = em_step(fam, X, ev, start)                             # classical EM (numpy M-step)

    block = ExactEMBlock(fam).double()
    Xt, Et = torch.from_numpy(X)[None], torch.from_numpy(ev)[None]
    mask = torch.ones(1, X.shape[0], dtype=torch.bool)
    with torch.no_grad():
        params, pi, _ = block(Xt, Et, mask, fam.to_torch(start.comps), torch.from_numpy(start.pi)[None])
    np.testing.assert_allclose(pi[0].numpy(), ref.pi, rtol=1e-6)
    for key in ref.comps:
        np.testing.assert_allclose(params[key][0].numpy(), ref.comps[key], rtol=1e-5, atol=1e-8)


def test_exact_block_rejects_non_exponential_family():
    with pytest.raises(ValueError):
        ExactEMBlock(get_family("weibull", k=1))
