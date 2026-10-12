"""v2 configuration: family spec, simulator, model, loss, training (YAML-backed).

Experiment YAML layout:
    seed: 0
    family:    {name: gamma, k: 2, ...prior kwargs}
    simulator: {...SimulatorConfig}
    model:     {...AEMConfig}
    loss:      {...LossConfig}
    training:  {...TrainConfig}
Command-line overrides use dotted keys, e.g. `model.T=6 loss.match=naive`.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

from ..simulation.simulator import SimulatorConfig


@dataclass
class AEMConfig:
    arch: str = "aem"                  # aem | nbe_deepsets | set_transformer
    d_model: int = 128
    n_heads: int = 4
    encoder: str = "isab"              # isab | deepsets
    n_enc_layers: int = 2
    n_inducing: int = 16
    ff_mult: int = 2
    # unrolled EM blocks
    T: int = 4
    tie_iterations: bool = True
    slot_init: str = "learned"         # learned | random
    use_loglik: bool = True            # exact log f(z; theta) term in the E-block (if the family allows)
    loglik_scale_init: float = 1.0
    learn_loglik_scale: bool = True
    detach_loglik: bool = False
    loglik_floor: float = -1000.0
    temperature_init: float = 1.0
    pooling: str = "responsibility"    # responsibility | global (v1-style ablation)
    use_suffstats: bool = True         # pool raw sufficient statistics alongside learned psi
    shared_expert: bool = True         # False = one expert per slot (v1-style ablation)
    slot_attention: bool = False       # cross-slot self-attention after each M-block
    expert_hidden: int = 128
    expert_layers: int = 2
    existence_head: bool = True
    canonicalize: bool = True
    # posterior head (C4)
    flow_head: bool = False
    flow_hidden: int = 128
    flow_layers: int = 3
    flow_detach: bool = True
    flow_steps: int = 32


@dataclass
class LossConfig:
    match: str = "quotient"            # quotient | naive | sorted | none_nll
    w_pi: float = 1.0                  # weight of pi inside the quotient cost
    deep_supervision: float = 0.5      # lambda_T on intermediate unrolled steps
    resp: float = 0.5                  # responsibility cross-entropy (aligned labels)
    exist: float = 0.5                 # existence BCE
    unmatched_pi: float = 1.0          # pi of unmatched slots pushed to 0
    nll: float = 0.0                   # ablation only: amortised-MLE target
    flow: float = 1.0                  # flow-matching posterior loss (if model.flow_head)


@dataclass
class TrainConfig:
    batch_size: int = 64
    steps: int = 100_000
    lr: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    val_every: int = 1000
    val_batches: int = 16
    val_batch_size: int = 64
    patience: int = 20
    num_workers: int = 0
    amp: bool = False                  # bf16 autocast on CUDA
    log_every: int = 100
    u_stats_samples: int = 2000
    aux_fade_frac: float = 0.2         # final fraction of training: fade aux losses, then pure L_Q (0 = off)
    out_dir: str = "runs/default"


@dataclass
class ExperimentConfig:
    seed: int
    family: dict
    simulator: SimulatorConfig
    model: AEMConfig
    loss: LossConfig
    training: TrainConfig
    raw: dict

    def to_dict(self) -> dict:
        return {"seed": self.seed, "family": self.family, "simulator": asdict(self.simulator),
                "model": asdict(self.model), "loss": asdict(self.loss), "training": asdict(self.training)}


def _build(cls, d: dict | None):
    d = dict(d or {})
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise KeyError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    if cls is SimulatorConfig:
        if "dirichlet_conc_range" in d:
            d["dirichlet_conc_range"] = tuple(d["dirichlet_conc_range"])
        if "overlap_bins" in d and d["overlap_bins"] is not None:
            d["overlap_bins"] = [tuple(b) for b in d["overlap_bins"]]
    return cls(**d)


def _parse_scalar(s: str):
    return yaml.safe_load(s)


def apply_overrides(raw: dict, overrides: list[str] | None) -> dict:
    raw = copy.deepcopy(raw)
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        cur = raw
        parts = key.strip().split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = _parse_scalar(val)
    return raw


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "family":
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml_with_base(path: str | Path) -> dict:
    """Load YAML; a top-level `base: <relative path>` is loaded first and deep-merged
    (the `family` block is replaced wholesale, never merged)."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if "base" in raw:
        base_path = (path.parent / raw.pop("base")).resolve()
        raw = deep_merge(load_yaml_with_base(base_path), raw)
    return raw


def load_experiment(path: str | Path | None = None, overrides: list[str] | None = None,
                    raw: dict | None = None) -> ExperimentConfig:
    if raw is None:
        raw = load_yaml_with_base(path)
    raw = apply_overrides(raw, overrides)
    return ExperimentConfig(
        seed=int(raw.get("seed", 0)),
        family=dict(raw.get("family", {"name": "gamma", "k": 2})),
        simulator=_build(SimulatorConfig, raw.get("simulator")),
        model=_build(AEMConfig, raw.get("model")),
        loss=_build(LossConfig, raw.get("loss")),
        training=_build(TrainConfig, raw.get("training")),
        raw=raw,
    )
