#!/usr/bin/env python
"""Study 2: factorial simulation study.

    python scripts/evaluate.py --config configs/study2_factorial.yaml \
        --checkpoint runs/default/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dmoegamma.baselines import BASELINE_REGISTRY, NeuralEstimatorBaseline
from dmoegamma.evaluation.benchmark import run_factorial_study
from dmoegamma.evaluation.stats import wilcoxon_table
from dmoegamma.models.moe import load_estimator
from dmoegamma.utils.config import load_yaml
from dmoegamma.utils.misc import get_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    methods = [BASELINE_REGISTRY[name]() for name in cfg.get("methods", [])]
    if args.checkpoint:
        device = args.device or get_device()
        methods = [NeuralEstimatorBaseline(load_estimator(args.checkpoint, device), device)] + methods

    sim_overrides = {k: v for k, v in cfg.get("simulation", {}).items()
                     if k in ("alpha_range", "beta_range", "dirichlet_concentration")}
    df = run_factorial_study(cfg["grid"], methods, reps_per_cell=cfg["reps_per_cell"],
                             seed=cfg["seed"], sim_overrides=sim_overrides,
                             out_dir=cfg.get("out_dir"))
    if args.checkpoint:
        tests = wilcoxon_table(df, metric="rmse_logalpha", reference="deep_moe")
        print(tests.to_string(index=False))
        out = cfg.get("out_dir")
        if out:
            tests.to_csv(Path(out) / "wilcoxon_rmse_logalpha.csv", index=False)


if __name__ == "__main__":
    main()
