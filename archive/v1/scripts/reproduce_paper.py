#!/usr/bin/env python
"""Study 1: replicate the base paper's four cases with all methods, paired.

    python scripts/reproduce_paper.py --config configs/study1_case1.yaml \
        --checkpoint runs/default/best.pt --out results/case1

Without --checkpoint only the classical baselines are run. Per-method
AVG/Bias/SD/MSE tables (the paper's Tables 3-4 format) and the
misclassification table (its Table 5) are written to --out as CSVs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dmoegamma.baselines import BASELINE_REGISTRY, NeuralEstimatorBaseline
from dmoegamma.evaluation.benchmark import run_replication_study
from dmoegamma.models.moe import load_estimator
from dmoegamma.simulation.cases import PAPER_CASES
from dmoegamma.utils.config import load_yaml
from dmoegamma.utils.misc import get_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    case = PAPER_CASES[cfg["case"]]
    methods = [BASELINE_REGISTRY[name]() for name in cfg.get("methods", ["wh_mclust", "ecm", "mle", "mom"])]
    if args.checkpoint:
        device = args.device or get_device()
        methods = [NeuralEstimatorBaseline(load_estimator(args.checkpoint, device), device)] + methods

    print(f"case: {case.name} -- {case.description} (n={case.n})")
    tables = run_replication_study(case, methods, n_runs=cfg["n_runs"], seed=cfg["seed"],
                                   out_dir=args.out)
    for name, df in tables.items():
        if name.startswith("summary_"):
            print(f"\n== {name} ==")
            print(df.round(4).to_string(index=False))
    print("\n== misclassification (published: "
          f"{case.published_misclassification}) ==")
    print(tables["misclassification"].round(4).to_string(index=False))
    print("\n== efficiency ==")
    print(tables["efficiency"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
