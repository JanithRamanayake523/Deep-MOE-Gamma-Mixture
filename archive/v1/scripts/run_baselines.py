#!/usr/bin/env python
"""Study 1 helper: run the classical baselines on one base-paper case.

    python scripts/run_baselines.py --config configs/study1_case1.yaml [--out results/case1]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dmoegamma.baselines import BASELINE_REGISTRY
from dmoegamma.evaluation.benchmark import run_replication_study
from dmoegamma.simulation.cases import PAPER_CASES
from dmoegamma.utils.config import load_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    case = PAPER_CASES[cfg["case"]]
    methods = [BASELINE_REGISTRY[name]() for name in cfg["methods"]]
    tables = run_replication_study(case, methods, n_runs=cfg["n_runs"], seed=cfg["seed"],
                                   out_dir=args.out)
    print(tables["efficiency"].to_string(index=False))
    print(tables["misclassification"].to_string(index=False))


if __name__ == "__main__":
    main()
