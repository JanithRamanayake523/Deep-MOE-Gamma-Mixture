# AEM-Net: Amortized EM for Mixtures Beyond Closed-Form M-Steps

Code for the research project *Amortized EM: Symmetry-Aware Neural Estimation for
Mixtures Beyond Closed-Form M-Steps* (working title).

| Document | Contents |
|---|---|
| [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) | **Start here.** Idea, claims and their status, experiments, compute, risks, questions for the reviewer |
| [`RESEARCH_METHODOLOGY.md`](RESEARCH_METHODOLOGY.md) | Full design: setting, hypotheses, architecture, theory (with proof sketches), training, experiments, timeline |
| [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md) | Related work, the gap, threats to novelty |
| [`references/`](references/) | The original base paper (Vaidyanathan & Vani Lakshmi, 2017, CC BY 4.0) |
| [`archive/v1/`](archive/v1/) | The superseded first version (design and code), kept for the record |

**Status:** implementation complete and tested; **no experiments run yet.**

## What AEM-Net does

AEM-Net maps a whole dataset to the parameters of a finite mixture in one forward
pass. It works like unrolled EM:

- **E-block:** computes per-observation responsibilities, using the exact
  log-likelihood when the family has one and a learned score otherwise.
- **M-block:** pools each component's responsibility-weighted sufficient
  statistics. One *shared* expert network then acts as a learned inverse link.

The network is invariant to the order of the data and equivariant to the order of
the components. Canonicalizing the data makes it exactly equivariant to scale (or
location and scale). It is trained on simulations with a Hungarian **quotient
loss**, and the last 20% of training uses that loss alone. Optional extras:

- a flow-matching posterior on the quotient space;
- an existence head, to choose the number of components;
- hybrid EM refinement.

## Repository layout

```
src/aemnet/
  mixture_params.py       MixtureParams: family-agnostic parameter container
  simulation/
    families/             Family interface + gamma, gaussian, weibull, student_t, negbin, copula_gamma,
                          kibble_gamma, noisy_gamma (gamma + measurement error; intractable),
                          gandk (likelihood-free), censored(<base>); stats.py (prior coordinate stats)
    simulator.py          MixtureSimulator: priors, overlap stratification, scale augmentation, loaders
    bank.py               pre-computed overlap-labelled parameter bank (fast stratified simulation)
    paper_cases.py        the base paper's Cases 1-4 (E9)
  models/
    aem.py                AEM-Net: canonicaliser, ISAB encoder, slot init, EBlock, MBlock, shared expert
    exact.py              ExactEMBlock: the Thm 1 construction (hand-set block == one EM step)
    flow.py               slot-level conditional flow-matching posterior head
    baseline_nets.py      NBE-DeepSets and Set Transformer (+ flow & sorted loss = ordering-constrained NPE)
    base_net.py, layers.py, factory.py
  losses/quotient.py      quotient loss (Hungarian); naive / sorted / NLL variants for E3
  training/trainer.py     AEMTrainer: online simulation, pure-L_Q final phase, checkpoints
  inference/              AmortizedEstimator; EM / gradient refinement (Thm 4); real-data adaptation
  baselines/              EM/ECM, batched GPU EM, direct MLE, adaptive MCMC, NumPyro NUTS, ABC-SMC,
                          WH-MCLUST (base paper), neural wrapper, adapters for TGMM / Cluster-PFN, sklearn VB
  evaluation/             metrics, relabelling (quotient Frechet mean, Stephens), calibration (SBC,
                          coverage, C2ST), diagnostics (prior-predictive MMD), paired benchmark (sharded), stats
  theory/                 numerical checks: EM contraction (Thm 4), symmetry illustrations (Thm 2)
  data/                   loaders: GHCN-Daily, CAS insurance, Allen spike ISIs, Sentinel-1 SAR, Madawaska
configs/
  make_configs.py         generates every config below (edit there, then re-run)
  train/                  _base.yaml + one per family; baselines/, symmetry/ (E3), ablations/ (E8)
  experiments/            e1 ... e9, contraction
scripts/
  build_param_bank.py     one-off bank build per training config (run before training)
  train_aem.py            train AEM-Net or a neural baseline
  run_experiment.py       benchmark | robustness | symmetry | calibration | contraction | real_data | ppc | replication
  make_ablations.py       materialise the E8 ablation configs
tests/                    families, symmetry, Thm 1, quotient loss, baselines, theory, every experiment handler
```

## Install and test

```bash
pip install -e .[dev]
```

```bash
python -m pytest tests -q
```

The optional extras are `.[bayes]` (NumPyro NUTS), `.[sklearn]` (VB) and
`.[realdata]` (AllenSDK, rasterio).

## Workflow

**1. Build the parameter bank** for each training config. This is a one-off.
Without it, overlap stratification falls back to rejection sampling, which is
about 40 times slower and leaves the GPU idle.

```bash
python scripts/build_param_bank.py --config configs/train/gamma_k2.yaml --workers 8
```

**2. Train.** Any key can be overridden with `--set`, for example `--set model.T=6`.

```bash
python scripts/train_aem.py --config configs/train/gamma_k2.yaml --device cuda
```

The neural baselines (`configs/train/baselines/`) and the E3 loss variants
(`configs/train/symmetry/`) train the same way. Generate the E8 ablation configs with:

```bash
python scripts/make_ablations.py --spec configs/experiments/e8_ablations.yaml
```

**3. Evaluate.** Every experiment is one YAML file in `configs/experiments/`. For example, E2:

```bash
python scripts/run_experiment.py --config configs/experiments/e2_n_regime_gamma_k1.yaml
```

| Config | Experiment |
|---|---|
| `e1_*.yaml` | accuracy benchmark; `e1_order_*` chooses the number of components |
| `e2_*.yaml` | headline: risk against n, plus EM iterations to convergence |
| `e3_*.yaml` | symmetry study (Thms 2–3) |
| `e4_*.yaml` | intractable or awkward likelihoods (`noisy_gamma`, g-and-k, copula, Kibble, censored) |
| `e5_*.yaml` | posterior calibration |
| `e6_*.yaml` | robustness |
| `e7_*.yaml` | real data and throughput vs. batched GPU EM |
| `e9_*.yaml` | base-paper replication |
| `contraction_*.yaml` | Thm 4 assumption check |

Benchmark configs run sharded over processes (`workers:`). Results match a serial
run, and an interrupted run resumes from finished shards.

**Using a trained model from Python:**

```python
from aemnet.inference import AmortizedEstimator
est = AmortizedEstimator.from_checkpoint("runs/gamma_k2/best.pt")
mp, resp = est.estimate(X)                  # number of components chosen by the existence head
mp, info = est.hybrid(X, g=2)               # + ceil(log2 n) EM steps (Thm 4)
samples = est.posterior(X, g=2, n_samples=1000)
many = est.estimate_many(list_of_datasets)  # batched amortized inference
```

## Design notes

- **Fair comparison.** Every method sees identical simulated datasets, and
  failures are recorded rather than dropped.
- **Common scoring coordinates.** Metrics use Hungarian alignment in the same
  standardized canonical coordinates the network is trained in, so risks are
  comparable across data scales.
- **Fair speed comparison.** The real-data experiment times batched GPU EM and
  AEM followed by batched EM, not only per-dataset EM. EM iteration counts are
  logged as `x_em_steps` / `x_em_steps_total`.
- **No re-implementations of external baselines.** TGMM and Cluster-PFN load
  their official code through environment variables (`baselines/external.py`).
  The ordering-constraint approach of Kucharský & Bürkner is trained in this code
  base with the same encoder (`configs/train/baselines/npe_sorted_gamma_k2.yaml`).
- **Known approximations.**
  - The Kibble likelihood uses a truncated series, which is ample for the default prior.
  - The noisy-gamma density is computed by panelled quadrature, verified to about
    1e-6 relative error. It is used by the baselines and metrics only; the network
    never sees it.
  - torch has no gradient of the incomplete gamma function with respect to the
    shape, so the shape gets no gradient through the copula and censoring terms.
    It still gets gradients through the marginal density.
- **No redistributed data.** Each loader's docstring explains how to obtain its data.
