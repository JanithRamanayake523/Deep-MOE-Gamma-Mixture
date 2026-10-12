# Project Overview (for review)

**Working title:** *Amortized EM: Symmetry-Aware Neural Estimation for Mixtures Beyond Closed-Form M-Steps*
**Method:** AEM-Net (Amortized Expectation–Maximization Network)
**Status (October 2026):** research design, literature review and a complete, tested code base. **No experiments have been run; no results exist yet.** The theorems are stated with proof sketches; full proofs are still to be written.
**Target:** NeurIPS 2027 (main track); fallbacks AISTATS 2028 / TMLR; thesis version in a statistics journal.

> **Disclosure.** The research redesign, the literature review and the code were developed with substantial assistance from an AI system (Claude, Anthropic), working with the student. The student must verify every citation, every claim and every line of the proofs, and should be able to explain the code, before any submission. Entries marked **[verify]** in the literature review still need a bibliographic check.

---

## 1. Origin

The project started from a master's idea built on Vaidyanathan & Vani Lakshmi (2017, *Sri Lankan J. Appl. Stat.*). That paper estimates finite mixtures of multivariate gamma distributions through a Wilson–Hilferty transform, MCLUST and a grid search. The first plan (v1, in `archive/v1/`) attached a "deep mixture of experts" neural estimator to this model. A critical review found three problems:

- **Unmotivated amortisation.** The likelihood is tractable and EM is fast, so there was little reason to amortise.
- **Weak comparator.** The main baseline was the base paper's own method.
- **Hollow mixture of experts.** The experts never saw the responsibilities, so the "MoE" was a multi-head output in name only.

The project was therefore redesigned for a machine-learning venue. The gamma family is kept as the flagship application.

## 2. The research question

Neural networks can now be trained on simulations to estimate the parameters of a mixture model in one forward pass ("amortised inference"). Existing work has three gaps:

1. **Gaussian components only.** TGMM (2025), transformer-vs-EM theory (2025), Cluster-PFN (2025) and Set Transformer all handle Gaussian components. Amortized Bayesian Mixture Models (*Stat. Comput.* 2026) uses Gaussian and related models and forces component labels into a fixed order. None handles mixtures whose **M-step has no closed form** (gamma, Weibull, Student-t) or whose **likelihood is intractable** (for example, gamma observed with measurement error).
2. **Label switching is handled heuristically.** Methods sort components, use unanalysed matching losses, or fall back to likelihood training. Nobody says **which estimator** the trained network actually converges to.
3. **Purely amortised estimators lose to maximum likelihood on large samples.** There is no method with guarantees that is Bayes-like at small n and efficient at large n.

## 3. The method in five points

1. **Unrolled, learned EM.** The E-block computes per-observation responsibilities, using the exact log-likelihood when one exists and a learned score otherwise. The M-block pools each component's responsibility-weighted sufficient statistics, and **one shared expert network** turns them into parameters, acting as a learned inverse link. This makes the mixture of experts genuine.
2. **Built-in symmetry.** The output is invariant to the order of the data and equivariant to the order of the components. Canonicalising the data makes it exactly equivariant to scale (or location and scale).
3. **The quotient loss.** Training uses a Hungarian-matched squared loss. The last 20% of training uses this loss alone, so the delivered network minimises exactly the objective the theory characterises.
4. **Hybrid inference.** The amortised estimate is followed by a few exact EM steps (`ceil(log2 n)` by default), or by extra learned iterations when there is no likelihood.
5. **Posterior on the quotient space.** A flow-matching head gives uncertainty estimates, with posterior samples aligned to the point estimate.

## 4. Claims and their status

| # | Claim | Status |
|---|---|---|
| Thm 1 | For exponential families, one AEM block can represent one EM step exactly. Hence the trained network is never worse than T-step EM in Bayes risk, at infinite capacity. | Construction verified numerically for gamma and Gaussian (`tests/test_em_embedding.py`); proof sketch only |
| Thm 2 | Training on plain squared error makes all components collapse to the same values. Sorting components to fix their labels biases the estimates when components overlap. | Short argument (section 4.3 of the methodology); E3 will show it empirically |
| Thm 3 | Training on the quotient loss learns the quotient-space Bayes estimator, which is a fixed point of Stephens' (2000) relabelling algorithm. | Proof sketch; part (iii), consistency of the trained network, needs real work |
| Thm 4 | Starting from a consistent amortised estimate, O(log n) EM steps give an asymptotically efficient estimator. | Rests on a local contraction assumption, which is checked numerically (`theory/contraction.py`) and not proven for gamma mixtures |
| H1–H8 | Accuracy, behaviour across sample sizes, symmetry, architecture choices, intractable likelihoods, calibration, speed, choosing the number of components | Hypotheses; every experiment is configured, **none run** |

## 5. Experiments (all configured in `configs/experiments/`)

| ID | Question | Main comparison |
|---|---|---|
| E1 | Accuracy across families, numbers of components, sample sizes and overlap | EM / ECM, MLE, NBE-DeepSets, Set Transformer, ordering-constrained flow posterior (ABMM-style), TGMM (Gaussian), WH-MCLUST |
| E1-order | Choosing the number of components (H8) | Existence head vs. EM with BIC (Cluster-PFN optional) |
| **E2** | **Headline: estimation risk against n** | Amortised, hybrid, EM (also iterations to convergence), batched EM, MLE |
| E3 | Collapse and ordering bias (Thms 2–3) | Four loss variants vs. reference estimators built from MCMC samples |
| E4 | Intractable likelihoods | Noisy gamma, g-and-k, copula, Kibble, censored: AEM vs. ABC-SMC and EM |
| E5 | Posterior calibration | Simulation-based calibration, coverage, classifier two-sample test vs. MCMC |
| E6 | Robustness | Prior shift, contamination, misspecified data family |
| E7 | Real data at scale | GHCN rainfall (thousands of stations); throughput vs. batched GPU EM |
| E8 | Ablations | 18 variants of the architecture and loss |
| E9 | Base-paper replication (appendix) | Paper Cases 1–4 with the paper's own metrics |

## 6. Title options (choose after the E2 and E3 pilots)

1. *Amortized EM: Learning to Fit Mixture Models Beyond Closed-Form M-Steps*: leads with the method.
2. *What Does a Neural Estimator Learn Under Label Switching? Quotient Bayes Estimators for Amortized Mixture Inference*: leads with the theory. Choose this if E3 is clean.
3. *Bayes at Small n, Efficient at Large n: Hybrid Amortized Estimation for Mixture Models*: choose this if E2 shows a clean crossover.
4. *Fit Once, Estimate Millions: Amortized Mixture Inference for Collections of Small Datasets*: leads with scale (E7).
5. *Learned M-Steps for Mixtures with Intractable Likelihoods*: leads with simulation-based inference (E4).
6. *Amortized Neural Estimation of Multivariate Gamma Mixtures with Measurement Error*: the thesis / statistics-journal version.

## 7. Compute (rough estimate, extrapolated, not measured on a GPU)

- **Model size.** AEM-Net has about 1.45M parameters. On an 8-thread CPU one training step (batch 64) takes 1.5 s at n = 200 and 5 s at n = 2000. On a single modern GPU that is plausibly 0.05–0.25 s per step, so **about 3–14 GPU-hours per model** at 200k steps.
- **Total training.** The full plan trains about 40 models (12 families, 5 neural baselines, 4 symmetry variants, 18 ablations), roughly **150–550 GPU-hours** in total. Cheaper options: 100k steps for the ablations, and k = 1 / k = 2 gamma plus Gaussian for the first submission.
- **Classical baselines.** These are CPU-heavy: E1 at full size is about 72,000 datasets × 9 methods. The benchmark runner is sharded across processes and resumable (`workers:` in the configs).
- **Parameter banks.** Each family needs a one-off bank build before training (`scripts/build_param_bank.py`), taking minutes to an hour with 8 workers.

## 8. Main risks

| Risk | Mitigation |
|---|---|
| Overlap with TGMM and slot attention ("this is slot attention plus EM") | Non-Gaussian and intractable families, exact EM embedding, symmetry theory, hybrid efficiency; say the relationship openly in the paper |
| AEM only matches EM on tractable families | The headline becomes the hybrid (E2), intractable families (E4) and scale (E7) |
| Proofs of Thm 3(iii) and Thm 4 harder than expected | Prove special cases (g = 2, k = 1) and present the general case as a conjecture with strong empirical support |
| Gap between real data and the simulator | Prior-predictive check, hybrid refinement, self-consistency fine-tuning |
| Scooped | Post an arXiv preprint after E1–E3 |

## 9. Questions for the reviewer

1. Is the **quotient Bayes estimator** angle (Thms 2–3) strong enough to lead the paper, or should the method lead?
2. Are Theorems 3(iii) and 4 provable at a reasonable level of generality, or should they be narrowed now?
3. Is **gamma with measurement error** a convincing "intractable" case for statisticians, or is a different intractable family better?
4. Is NeurIPS realistic for a student-led paper on this timeline, or is AISTATS / TMLR the better first target?
5. Is the planned compute (section 7) available, or should the scope be cut before starting?

## 10. Reproducing (once compute is available)

```bash
pip install -e .[dev]
```

```bash
python -m pytest tests -q
```

The suite currently gives 98 passed, 3 skipped, in about 1–2 minutes on CPU. Then build the parameter bank and train a model:

```bash
python scripts/build_param_bank.py --config configs/train/gamma_k2.yaml --workers 8
```

```bash
python scripts/train_aem.py --config configs/train/gamma_k2.yaml --device cuda
```

Finally, run an experiment:

```bash
python scripts/run_experiment.py --config configs/experiments/e2_n_regime_gamma_k1.yaml
```

The full workflow is in [`README.md`](README.md), the design in [`RESEARCH_METHODOLOGY.md`](RESEARCH_METHODOLOGY.md), and related work in [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md).
