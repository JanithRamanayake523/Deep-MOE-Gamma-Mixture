"""Baselines behind one interface: GenericBaseline.fit(X, events, g) -> GenericFit (MixtureParams)."""
from .generic import GenericBaseline, GenericFit, init_from_labels, kmeanspp
from .em import EMBaseline
from .batched_em import BatchedEM
from .mle import DirectMLEBaseline
from .mcmc import AdaptiveMCMCBaseline, NumPyroNUTSBaseline
from .abc_smc import ABCSMCBaseline
from .wh_mclust import WHMclustBaseline
from .neural import NeuralBaseline
from .external import ClusterPFNBaseline, TGMMBaseline, VBGaussianBaseline

BASELINES = {
    "em": EMBaseline,
    "batched_em": BatchedEM,
    "mle": DirectMLEBaseline,
    "mcmc": AdaptiveMCMCBaseline,
    "nuts": NumPyroNUTSBaseline,
    "abc_smc": ABCSMCBaseline,
    "wh_mclust": WHMclustBaseline,
    "tgmm": TGMMBaseline,
    "cluster_pfn": ClusterPFNBaseline,
    "vb": VBGaussianBaseline,
}


def build_baseline(spec: dict, family):
    """spec: {"type": <key of BASELINES>, ...kwargs}
    or {"type": "neural", "checkpoint": path, "name": ..., "mode": amortized|hybrid|learned, ...}."""
    spec = dict(spec)
    kind = spec.pop("type")
    if kind == "neural":
        ckpt = spec.pop("checkpoint")
        device = spec.pop("device", "cpu")
        return NeuralBaseline.from_checkpoint(ckpt, device=device, **spec)
    if kind not in BASELINES:
        raise KeyError(f"unknown baseline type '{kind}'; available: {sorted(BASELINES)} + 'neural'")
    return BASELINES[kind](family, **spec)


__all__ = [
    "GenericBaseline", "GenericFit", "init_from_labels", "kmeanspp", "EMBaseline", "BatchedEM",
    "DirectMLEBaseline", "AdaptiveMCMCBaseline", "NumPyroNUTSBaseline", "ABCSMCBaseline", "WHMclustBaseline",
    "NeuralBaseline", "TGMMBaseline", "ClusterPFNBaseline", "VBGaussianBaseline", "BASELINES", "build_baseline",
]
