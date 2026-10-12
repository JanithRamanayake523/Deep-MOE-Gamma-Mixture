#!/usr/bin/env python
"""Train the Deep-MoE estimator.

    python scripts/train.py --config configs/default.yaml [--device cuda] [--out runs/exp1]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dmoegamma.losses.composite import CompositeLoss
from dmoegamma.models.moe import DeepMoEGammaEstimator
from dmoegamma.training.trainer import Trainer
from dmoegamma.utils.config import experiment_from_yaml
from dmoegamma.utils.misc import get_device, seed_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="override training.out_dir")
    args = ap.parse_args()

    seed, sim_cfg, model_cfg, loss_w, train_cfg = experiment_from_yaml(args.config)
    if args.out:
        train_cfg.out_dir = args.out
    device = args.device or get_device()
    seed_all(seed)

    model = DeepMoEGammaEstimator(model_cfg, k=sim_cfg.k, g_max=sim_cfg.g_max)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,} | device: {device} | out: {train_cfg.out_dir}")

    loss_fn = CompositeLoss(loss_w, g_max=sim_cfg.g_max)
    trainer = Trainer(model, loss_fn, sim_cfg, train_cfg, device=device, seed=seed)
    trainer.train()


if __name__ == "__main__":
    main()
