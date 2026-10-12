"""Hungarian alignment and composite loss sanity."""
import numpy as np
import torch

from dmoegamma.losses.composite import CompositeLoss
from dmoegamma.losses.matching import batch_permutations, hungarian_match
from dmoegamma.params import Theta, padded_batch
from dmoegamma.utils.config import LossWeights


def test_hungarian_recovers_label_permutation():
    true = Theta(pi=[0.3, 0.7], alpha=[[1.0], [9.0]], beta=[[2.0], [0.5]])
    pred = true.permuted([1, 0])  # same parameters, swapped labels
    perm = hungarian_match(true, pred)
    assert list(perm) == [1, 0]
    assert np.allclose(pred.permuted(perm).alpha, true.alpha)


def test_batch_permutations_padded():
    th = Theta(pi=[0.4, 0.6], alpha=[[2.0], [8.0]], beta=[[1.0], [2.0]])
    batch = padded_batch([th], g_max=3)
    pred = {
        "pi": torch.tensor([[0.6, 0.4, 0.0]]),
        "alpha": torch.tensor([[[8.0], [2.0], [1.0]]]),  # (B, g_max, k)
        "beta": torch.tensor([[[2.0], [1.0], [1.0]]]),
    }
    perms = batch_permutations(batch, pred)
    assert perms.shape == (1, 3)
    assert perms[0, :2].tolist() == [1, 0]  # swapped
    assert perms[0, 2].item() == 2          # identity on padding


def test_composite_loss_is_finite_and_label_robust():
    g_max, k, n = 3, 2, 40
    th = Theta(pi=[0.4, 0.6], alpha=[[2.0, 3.0], [8.0, 5.0]], beta=[[1.0, 2.0], [0.5, 1.0]])
    batch = padded_batch([th], g_max=g_max)
    rng = np.random.default_rng(0)
    from dmoegamma.simulation.mixture import sample

    X, labels = sample(th, n=n, rng=rng)
    batch["X"] = torch.from_numpy(X).float().unsqueeze(0)
    batch["obs_mask"] = torch.ones(1, n, dtype=torch.bool)
    batch["labels"] = torch.from_numpy(labels).unsqueeze(0)

    def pred_from(theta_perm):
        full = padded_batch([theta_perm], g_max=g_max)
        return {"pi": full["pi"], "alpha": full["alpha"], "beta": full["beta"],
                "resp": torch.full((1, n, g_max), 1.0 / g_max),
                "g_logits": torch.zeros(1, g_max)}

    loss_fn = CompositeLoss(LossWeights(load_balance=0.0, g=0.0), g_max=g_max)
    l1, stats1 = loss_fn(pred_from(th), batch)
    l2, stats2 = loss_fn(pred_from(th.permuted([1, 0])), batch)
    for s in (stats1, stats2):
        assert np.isfinite(s["loss"]) and np.isfinite(s["nll"])
    # parameter term must be (near-)invariant to label permutation
    assert abs(stats1["param"] - stats2["param"]) < 1e-4
    assert stats1["param"] < 0.05  # exact parameters -> tiny aligned error
