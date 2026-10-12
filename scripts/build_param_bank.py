#!/usr/bin/env python
"""Pre-compute the overlap-labelled parameter bank for a training config (run once).

    python scripts/build_param_bank.py --config configs/train/gamma_k2.yaml --workers 8
writes the file named by `simulator.param_bank` in the config (or --out).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aemnet.simulation.bank import ParameterBank  # noqa: E402
from aemnet.simulation.families import get_family  # noqa: E402
from aemnet.simulation.simulator import MixtureSimulator  # noqa: E402
from aemnet.utils.config import load_experiment  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--per-cell", type=int, default=20000, help="parameter sets per (g, overlap bin)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    exp = load_experiment(args.config, args.set)
    out = args.out or exp.simulator.param_bank
    if not out:
        raise SystemExit("no output path: set simulator.param_bank in the config or pass --out")
    sim = MixtureSimulator(get_family(exp.family), exp.simulator)
    t0 = time.time()
    bank = ParameterBank.build(sim, per_cell=args.per_cell, workers=args.workers, seed=args.seed)
    bank.save(out)
    sizes = {f"g{g}_b{b}": bank.size(g, b) for (g, b) in sorted(bank.cells)}
    print(f"saved {out} in {time.time() - t0:.0f}s: {sizes}")


if __name__ == "__main__":
    main()
