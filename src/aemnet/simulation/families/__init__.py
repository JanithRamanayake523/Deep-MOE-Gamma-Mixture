"""Component-distribution families for AEM-Net (RESEARCH_METHODOLOGY.md section 1).

    get_family("gamma", k=2)                         -> GammaFamily
    get_family({"name": "censored", "base": {"name": "weibull", "k": 1}})
    family_from_spec(family.spec())                  -> round trip (checkpoints)
"""
from __future__ import annotations

from .base import Family
from .censored import CensoredFamily
from .copula_gamma import CopulaGammaFamily
from .gamma import GammaFamily
from .gandk import GAndKFamily
from .gaussian import GaussianFamily
from .kibble_gamma import KibbleGammaFamily
from .negbin import NegBinFamily
from .noisy_gamma import NoisyGammaFamily
from .student_t import StudentTFamily
from .weibull import WeibullFamily

FAMILY_REGISTRY: dict[str, type[Family]] = {
    "gamma": GammaFamily,
    "gaussian": GaussianFamily,
    "weibull": WeibullFamily,
    "student_t": StudentTFamily,
    "negbin": NegBinFamily,
    "copula_gamma": CopulaGammaFamily,
    "kibble_gamma": KibbleGammaFamily,
    "gandk": GAndKFamily,
    "noisy_gamma": NoisyGammaFamily,
}


def family_from_spec(spec: dict) -> Family:
    spec = dict(spec)
    name = spec.pop("name")
    if name == "censored":
        base = family_from_spec(spec.pop("base"))
        return CensoredFamily(base, **spec)
    if name not in FAMILY_REGISTRY:
        raise KeyError(f"unknown family '{name}'; available: {sorted(FAMILY_REGISTRY)} + 'censored'")
    return FAMILY_REGISTRY[name](**spec)


def get_family(name_or_spec, k: int | None = None, **kwargs) -> Family:
    if isinstance(name_or_spec, dict):
        return family_from_spec(name_or_spec)
    spec = {"name": name_or_spec, **kwargs}
    if k is not None:
        spec["k"] = k
    return family_from_spec(spec)


__all__ = [
    "Family", "GammaFamily", "GaussianFamily", "WeibullFamily", "StudentTFamily", "NegBinFamily",
    "CopulaGammaFamily", "KibbleGammaFamily", "GAndKFamily", "NoisyGammaFamily", "CensoredFamily",
    "FAMILY_REGISTRY", "family_from_spec", "get_family",
]
