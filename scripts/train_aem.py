#!/usr/bin/env python
"""Train AEM-Net (or a neural baseline) on simulated mixtures.

    python scripts/train_aem.py --config configs/train/gamma_k2.yaml
    python scripts/train_aem.py --config configs/train/gamma_k2.yaml --device cuda \
        --set model.T=6 training.out_dir=runs/gamma_k2_T6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aemnet.losses.quotient import QuotientLoss  # noqa: E402
from aemnet.models.factory import build_model  # noqa: E402
from aemnet.simulation.families import get_family  # noqa: E402
from aemnet.simulation.simulator import MixtureSimulator  # noqa: E402
from aemnet.training.trainer import AEMTrainer  # noqa: E402
from aemnet.utils.config import load_experiment  # noqa: E402
from aemnet.utils.misc import get_device, seed_all  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    args = ap.parse_args()

    exp = load_experiment(args.config, args.set)
    device = args.device or get_device()
    seed_all(exp.seed)
    family = get_family(exp.family)
    simulator = MixtureSimulator(family, exp.simulator)
    model = build_model(family, exp.model, n_slots=exp.simulator.g_max)
    loss_fn = QuotientLoss(family, exp.loss)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"family={family.name} k={family.k} d={family.d} arch={exp.model.arch} params={n_params:,} "
          f"device={device} out={exp.training.out_dir}")
    trainer = AEMTrainer(model, loss_fn, simulator, exp.training, device=device, seed=exp.seed,
                         experiment=exp.to_dict())
    trainer.train()


if __name__ == "__main__":
    main()
