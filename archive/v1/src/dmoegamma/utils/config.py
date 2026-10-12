"""Configuration dataclasses and YAML loading."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SimConfig:
    k: int = 2
    g_min: int = 1
    g_max: int = 4
    alpha_range: tuple[float, float] = (0.1, 30.0)
    beta_range: tuple[float, float] = (0.1, 30.0)
    dirichlet_concentration: float = 1.0
    n_min: int = 25
    n_max: int = 5000
    overlap_stratification: bool = True
    overlap_bins: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.05), (0.05, 0.20), (0.20, 0.50)]
    )


@dataclass
class ModelConfig:
    enc_hidden: int = 128
    enc_layers: int = 3
    enc_out: int = 128
    pooling: str = "mean"  # mean | sum
    use_wh_features: bool = True
    use_summary_stats: bool = True
    head_hidden: int = 128
    head_layers: int = 2
    use_g_head: bool = True


@dataclass
class LossWeights:
    param: float = 1.0         # lambda_1: Hungarian-aligned parameter MSE
    nll: float = 1.0           # lambda_2: label-invariant negative log-likelihood
    resp: float = 0.5          # responsibility cross-entropy supervision
    load_balance: float = 0.01 # Switch-style expert load balancing
    g: float = 1.0             # model-order cross-entropy


@dataclass
class TrainConfig:
    batch_size: int = 64
    steps: int = 20000
    lr: float = 1e-3
    min_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 1.0
    val_every: int = 500
    val_batches: int = 8
    patience: int = 10
    out_dir: str = "runs/default"


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _coerce_sim(d: dict) -> SimConfig:
    d = dict(d)
    if "alpha_range" in d:
        d["alpha_range"] = tuple(d["alpha_range"])
    if "beta_range" in d:
        d["beta_range"] = tuple(d["beta_range"])
    if "overlap_bins" in d:
        d["overlap_bins"] = [tuple(b) for b in d["overlap_bins"]]
    return SimConfig(**d)


def experiment_from_yaml(
    path: str | Path,
) -> tuple[int, SimConfig, ModelConfig, LossWeights, TrainConfig]:
    """Load a training experiment config (e.g. configs/default.yaml)."""
    raw = load_yaml(path)
    seed = int(raw.get("seed", 123))
    sim = _coerce_sim(raw.get("simulation", {}))
    model = ModelConfig(**raw.get("model", {}))
    loss = LossWeights(**raw.get("loss", {}))
    train = TrainConfig(**raw.get("training", {}))
    return seed, sim, model, loss, train
