"""Adapting an amortised estimator to real data (RESEARCH_METHODOLOGY.md 5, item 4; E6/E7).

unlabeled_adaptation
    Semi-supervised fine-tuning: L_Q on fresh simulations (keeps the Bayes target)
    + lambda * NLL of the network's own estimates on *unlabelled* real datasets
    (pulls it toward the real-data likelihood). Tractable families only.

self_consistency_loss
    Bayesian self-consistency (Schmitt et al., 2023; Mishra et al., 2025): for the
    true posterior, log p(theta) + log p(D | theta) - log q(theta | D) is constant
    in theta, so its variance over posterior draws is a label-free training signal.
    With priors that are flat in the flow coordinates (log-uniform priors, Dirichlet(1))
    the prior term is constant and drops out of the variance.
"""
from __future__ import annotations

import numpy as np
import torch

from ..simulation.simulator import MixtureSimulator, batch_to_device, collate


def _real_batch(family, datasets, n_slots, device):
    data = []
    for d in datasets:
        X, ev = (d if isinstance(d, tuple) else (d, None))
        X = np.asarray(X, dtype=np.float64)
        X = X[:, None] if X.ndim == 1 else X
        data.append({"X": X, "events": np.ones_like(X) if ev is None else np.asarray(ev), "labels": None,
                     "params": None})
    return batch_to_device(collate(data, family, n_slots), device)


def mixture_nll(model, out, batch) -> torch.Tensor:
    fam = model.family
    lp = fam.log_prob(out["Xc"], batch["events"], out["params_canon"]) + out["pi"].clamp_min(1e-8).log().unsqueeze(1)
    ll = torch.logsumexp(lp, -1)
    m = batch["obs_mask"].to(ll.dtype)
    return -(ll * m).sum() / m.sum()


def unlabeled_adaptation(model, loss_fn, simulator: MixtureSimulator, real_datasets: list, steps: int = 500,
                         lr: float = 3e-5, lam_nll: float = 0.1, sim_batch_size: int = 32, real_batch_size: int = 16,
                         device: str = "cpu", seed: int = 0) -> list[dict]:
    if not model.family.has_torch_logprob:
        raise ValueError("unlabeled_adaptation needs a tractable torch likelihood")
    rng = np.random.default_rng(seed)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    hist = []
    for step in range(steps):
        sim_b = batch_to_device(simulator.sample_batch(rng, sim_batch_size), device)
        out = model(sim_b["X"], sim_b["events"], sim_b["obs_mask"])
        l_sim, stats, _ = loss_fn(out, sim_b, model)
        ids = rng.choice(len(real_datasets), size=min(real_batch_size, len(real_datasets)), replace=False)
        real_b = _real_batch(model.family, [real_datasets[i] for i in ids], model.S, device)
        out_r = model(real_b["X"], real_b["events"], real_b["obs_mask"])
        l_real = mixture_nll(model, out_r, real_b)
        loss = l_sim + lam_nll * l_real
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist.append({"step": step, "sim_quotient": stats["quotient"], "real_nll": float(l_real.detach())})
    model.eval()
    return hist


def self_consistency_loss(model, out, batch, n_samples: int = 16, flow_steps: int = 16,
                          hutchinson: bool = True) -> torch.Tensor:
    """Variance over posterior draws of log p(D | theta) - log q(theta | D), averaged over datasets."""
    if model.flow is None or not model.family.has_torch_logprob:
        raise ValueError("self-consistency needs a flow head and a tractable likelihood")
    cond = model.flow_cond(out)
    B, S, _ = cond.shape
    with torch.no_grad():
        y = model.flow.sample(cond, n_samples, flow_steps)                 # (N,B,S,dy)
    yf = y.reshape(n_samples * B, S, -1)
    cond_rep = cond.unsqueeze(0).expand(n_samples, -1, -1, -1).reshape(n_samples * B, S, -1)
    log_q = model.flow.log_prob(yf, cond_rep, flow_steps, hutchinson=hutchinson).view(n_samples, B)
    z, logpi = yf[..., :-1], yf[..., -1]
    params = model.decode_z(z)
    pi = torch.softmax(logpi, -1)
    Xc = out["Xc"].unsqueeze(0).expand(n_samples, -1, -1, -1).reshape(n_samples * B, *out["Xc"].shape[1:])
    ev = batch["events"].unsqueeze(0).expand(n_samples, -1, -1, -1).reshape(n_samples * B, *batch["events"].shape[1:])
    mask = batch["obs_mask"].unsqueeze(0).expand(n_samples, -1, -1).reshape(n_samples * B, -1)
    lp = model.family.log_prob(Xc, ev, params) + pi.clamp_min(1e-8).log().unsqueeze(1)
    loglik = (torch.logsumexp(lp, -1) * mask.to(lp.dtype)).sum(-1).view(n_samples, B)
    s = loglik.detach() - log_q
    return s.var(dim=0).mean()
