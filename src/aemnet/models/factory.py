"""Model construction from an AEMConfig (`model.arch`)."""
from __future__ import annotations

from ..simulation.families.base import Family
from .aem import AEMNet
from .baseline_nets import NBEDeepSetsNet, SetTransformerNet

ARCHITECTURES = {"aem": AEMNet, "nbe_deepsets": NBEDeepSetsNet, "set_transformer": SetTransformerNet}


def build_model(family: Family, cfg, n_slots: int):
    if cfg.arch not in ARCHITECTURES:
        raise KeyError(f"unknown arch '{cfg.arch}'; available: {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[cfg.arch](family, cfg, n_slots)
