"""Training loop: AdamW, warmup + cosine schedule, gradient clipping, periodic
validation on fresh simulations, early stopping on validation NLL
(RESEARCH_METHODOLOGY.md 5.5)."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ..losses.composite import CompositeLoss
from ..simulation.priors import PriorConfig, ThetaPrior
from ..utils.config import SimConfig, TrainConfig
from .datagen import OnlineGenerator


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: CompositeLoss,
        sim_cfg: SimConfig,
        train_cfg: TrainConfig,
        device: str = "cpu",
        seed: int = 123,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.sim_cfg = sim_cfg
        self.cfg = train_cfg
        self.device = device
        prior = ThetaPrior(
            PriorConfig(
                k=sim_cfg.k,
                g_min=sim_cfg.g_min,
                g_max=sim_cfg.g_max,
                alpha_range=sim_cfg.alpha_range,
                beta_range=sim_cfg.beta_range,
                dirichlet_concentration=sim_cfg.dirichlet_concentration,
                n_min=sim_cfg.n_min,
                n_max=sim_cfg.n_max,
                overlap_bins=sim_cfg.overlap_bins,
            )
        )
        self.gen = OnlineGenerator(prior, sim_cfg, train_cfg.batch_size)
        self.rng = np.random.default_rng(seed)
        self.val_rng = np.random.default_rng(seed + 1)

        self.opt = torch.optim.AdamW(
            model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
        )

        def lr_lambda(step: int) -> float:
            if step < train_cfg.warmup_steps:
                return (step + 1) / train_cfg.warmup_steps
            t = (step - train_cfg.warmup_steps) / max(1, train_cfg.steps - train_cfg.warmup_steps)
            floor = train_cfg.min_lr / train_cfg.lr
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)

    def _to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device) for k, v in batch.items()}

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        nlls = []
        for _ in range(self.cfg.val_batches):
            batch = self._to_device(self.gen.sample_batch(self.val_rng))
            _, stats = self.loss_fn(self.model(batch["X"], batch["obs_mask"]), batch)
            nlls.append(stats["nll"])
        self.model.train()
        return float(np.mean(nlls))

    def train(self) -> list[dict]:
        cfg = self.cfg
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        history: list[dict] = []
        best_val = float("inf")
        bad_epochs = 0
        self.model.train()
        t0 = time.time()
        for step in range(1, cfg.steps + 1):
            batch = self._to_device(self.gen.sample_batch(self.rng))
            loss, stats = self.loss_fn(self.model(batch["X"], batch["obs_mask"]), batch)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.opt.step()
            self.sched.step()

            if step % cfg.val_every == 0 or step == cfg.steps:
                val_nll = self.validate()
                rec = {"step": step, "val_nll": val_nll, "elapsed_s": round(time.time() - t0, 1),
                       **{k: round(v, 5) for k, v in stats.items()}}
                history.append(rec)
                print(json.dumps(rec))
                if val_nll < best_val - 1e-4:
                    best_val = val_nll
                    bad_epochs = 0
                    torch.save(
                        {"model_state": self.model.state_dict(),
                         "model_config": getattr(self.model, "cfg", None).__dict__
                             if hasattr(self.model, "cfg") else None,
                         "sim_config": self.sim_cfg.__dict__,
                         "val_nll": best_val},
                        out_dir / "best.pt",
                    )
                else:
                    bad_epochs += 1
                    if bad_epochs >= cfg.patience:
                        print(f"early stopping at step {step} (best val NLL {best_val:.4f})")
                        break
        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        return history
