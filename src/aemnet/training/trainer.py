"""Training loop for AEM-Net and the neural baselines (RESEARCH_METHODOLOGY.md 6.3).

AdamW + warmup/cosine schedule, gradient clipping, optional bf16 autocast,
online simulation (optionally multi-worker), validation on a *fixed* simulated
suite, early stopping on validation quotient risk, JSONL logging, and
self-describing checkpoints (family spec + full experiment config + u-stats).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ..simulation.simulator import MixtureSimulator, batch_to_device, make_fixed_set, make_loader


def estimate_u_stats(simulator: MixtureSimulator, n_datasets: int = 2000, seed: int = 0):
    """Mean/std of canonical unconstrained coordinates under the simulation prior.
    Used to standardise network outputs and the loss (z-coordinates)."""
    fam = simulator.family
    sim = MixtureSimulator(fam, replace(simulator.cfg, overlap_stratification=False))
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_datasets):
        n = min(sim.sample_n(rng), 2000)
        d = sim.sample_dataset(rng, n=n)
        loc, scale = fam.canonical_loc_scale_np(d["X"])
        rows.append(fam.encode_np(fam.canonicalize_np(d["params"], loc, scale).comps))
    U = np.concatenate(rows, 0)
    U = U[np.all(np.isfinite(U), 1)]
    return U.mean(0), np.maximum(U.std(0), 1e-3)


class AEMTrainer:
    def __init__(self, model, loss_fn, simulator: MixtureSimulator, cfg, device: str = "cpu",
                 seed: int = 0, experiment: dict | None = None):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.sim = simulator
        self.cfg = cfg
        self.device = device
        self.seed = seed
        self.experiment = experiment or {}
        self.out_dir = Path(cfg.out_dir)
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        def lr_lambda(step: int) -> float:
            if step < cfg.warmup_steps:
                return (step + 1) / max(1, cfg.warmup_steps)
            t = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
            floor = cfg.min_lr / cfg.lr
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)
        self.use_amp = bool(cfg.amp and str(device).startswith("cuda"))
        self.val_set = None
        self.monitor = "nll" if getattr(loss_fn.cfg, "match", "") == "none_nll" else "quotient"

    # ------------------------------------------------------------------ helpers
    def _forward_loss(self, batch):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp):
            out = self.model(batch["X"], batch["events"], batch["obs_mask"])
        return self.loss_fn(out, batch, self.model)

    @torch.no_grad()
    def validate(self) -> dict:
        if self.val_set is None:
            # validation parameters come from the bank's held-out tail (never seen in training)
            self.val_set = make_fixed_set(self.sim.with_split("val"), self.cfg.val_batches,
                                          self.cfg.val_batch_size, self.seed + 10_007)
        self.model.eval()
        agg: dict[str, list] = {}
        for batch in self.val_set:
            _, stats, _ = self._forward_loss(batch_to_device(batch, self.device))
            stats = {k: v for k, v in stats.items() if k != "aux_scale"}
            for k, v in stats.items():
                agg.setdefault(k, []).append(v)
        self.model.train()
        return {f"val_{k}": float(np.mean(v)) for k, v in agg.items()}

    def save(self, path: Path, step: int, val: dict) -> None:
        torch.save({
            "model_state": self.model.state_dict(),
            "family_spec": self.model.family.spec(),
            "n_slots": self.model.S,
            "experiment": self.experiment,
            "step": step,
            "val": val,
        }, path)

    # ------------------------------------------------------------------ aux-loss fade
    def aux_scale(self, step: int) -> float:
        """1 before the fade, linear to 0 over the fade, 0 in the final pure-L_Q phase."""
        if self.fade_start is None or step < self.fade_start:
            return 1.0
        if step >= self.fade_end:
            return 0.0
        return 1.0 - (step - self.fade_start) / max(1, self.fade_end - self.fade_start)

    def _set_aux(self, scale: float) -> None:
        self.loss_fn.aux_scale = scale
        self.model.exist_detach = scale < 1.0          # existence head keeps learning, backbone does not

    def _plan_fade(self, start: int) -> None:
        f = self.cfg.aux_fade_frac
        self.fade_start = start
        self.fade_end = start + max(1, int(0.5 * f * self.cfg.steps))
        self.end_step = max(self.fade_end + 1, start + int(f * self.cfg.steps))

    # ------------------------------------------------------------------ loop
    def train(self, u_stats: tuple | None = None) -> list[dict]:
        """Training ends with a pure-L_Q phase when aux_fade_frac > 0: auxiliary losses
        (deep supervision, responsibility CE, existence gradients into the backbone)
        fade to zero, so the final network minimises exactly the quotient loss whose
        Bayes estimator Thm 3 characterises. If validation plateaus before the fade,
        the fade starts then (instead of early stopping). best.pt is selected among
        pure-phase checkpoints only; best_prefade.pt keeps the best earlier one."""
        cfg = self.cfg
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if u_stats is None:
            u_stats = estimate_u_stats(self.sim, cfg.u_stats_samples, self.seed + 1)
        self.model.set_u_stats(torch.as_tensor(u_stats[0]), torch.as_tensor(u_stats[1]))
        with open(self.out_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.experiment, f, indent=2)

        fading = cfg.aux_fade_frac > 0
        self.fade_start = self.fade_end = None
        self.end_step = cfg.steps
        if fading:
            self._plan_fade(int((1 - cfg.aux_fade_frac) * cfg.steps))
            self.end_step = cfg.steps

        loader = iter(make_loader(self.sim, cfg.batch_size, self.seed, cfg.num_workers))
        history, best, best_pre, bad = [], float("inf"), float("inf"), 0
        log_f = open(self.out_dir / "train_log.jsonl", "a", encoding="utf-8")
        t0 = time.time()
        running: dict[str, list] = {}
        self.model.train()
        step = 0
        try:
            while step < self.end_step:
                step += 1
                self._set_aux(self.aux_scale(step))
                batch = batch_to_device(next(loader), self.device)
                loss, stats, _ = self._forward_loss(batch)
                if not torch.isfinite(loss):
                    print(f"[step {step}] non-finite loss, batch skipped: {stats}")
                    self.opt.zero_grad(set_to_none=True)
                    continue
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.opt.step()
                self.sched.step()
                for k, v in stats.items():
                    running.setdefault(k, []).append(v)
                running.setdefault("grad_norm", []).append(float(gnorm))

                if step % cfg.log_every == 0:
                    rec = {"step": step, "lr": self.sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1),
                           **{k: round(float(np.mean(v)), 5) for k, v in running.items()}}
                    running = {}
                    log_f.write(json.dumps(rec) + "\n")
                    log_f.flush()
                    print(json.dumps(rec))

                if step % cfg.val_every == 0 or step == self.end_step:
                    val = self.validate()
                    pure = (not fading) or step >= self.fade_end
                    rec = {"step": step, "pure_phase": pure, **val}
                    history.append(rec)
                    log_f.write(json.dumps(rec) + "\n")
                    log_f.flush()
                    print(json.dumps(rec))
                    self.save(self.out_dir / "last.pt", step, val)
                    score = val.get(f"val_{self.monitor}", float("inf"))
                    if pure:
                        if score < best - 1e-5:
                            best, bad = score, 0
                            self.save(self.out_dir / "best.pt", step, val)
                        elif not fading:
                            bad += 1
                            if bad >= cfg.patience:
                                print(f"early stopping at step {step} (best val {self.monitor} {best:.5f})")
                                break
                    else:
                        if score < best_pre - 1e-5:
                            best_pre, bad = score, 0
                            self.save(self.out_dir / "best_prefade.pt", step, val)
                        elif step < self.fade_start:
                            bad += 1
                            if bad >= cfg.patience:
                                print(f"plateau at step {step}: starting the pure-L_Q phase now")
                                self._plan_fade(step)
                                bad = 0
        finally:
            log_f.close()
            self._set_aux(1.0)
        if not (self.out_dir / "best.pt").exists():
            self.save(self.out_dir / "best.pt", step, history[-1] if history else {})
        with open(self.out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        return history
