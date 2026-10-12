# Research Methodology

## Deep Mixture-of-Experts for Amortized Estimation of Multivariate Gamma Mixture Parameters

**Working title (refined):** *Learning to Estimate: A Deep Mixture-of-Experts Framework for Amortized Parameter Estimation of Multivariate Gamma Mixture Models*

**Base paper:** Vaidyanathan, V. S. & Vani Lakshmi, R. (2017). *Estimation of Parameters in a Finite Mixture of Multivariate Gamma Distributions using Gaussian Approximation.* Sri Lankan Journal of Applied Statistics, 17(3), 187–200. DOI: 10.4038/sljastats.v17i3.7902

---

## 0. Topic Decision (Expert Assessment)

You gave full freedom to modify the topic. My assessment as a statistician + deep learning researcher:

- **The core niche is genuinely open.** A literature scan (see §2) confirms: neural/amortized inference is a mature and active field (BayesFlow, neural Bayes estimators, amortized mixture models), and classical gamma-mixture estimation is an active statistical field (ECM algorithms, WH-approximation methods, variational Bayes) — but **no published work applies deep learning to parameter estimation of (multivariate) gamma mixture models**. The gap is real and defensible.
- **The phrase "Deep Mixture of Experts" must be made architecturally meaningful**, or reviewers will read it as buzzword-stacking. The sharpened framing: an MoE architecture whose **gating network mirrors the E-step** (soft component assignments / responsibilities) and whose **expert networks mirror the M-step** (each expert estimates one component's parameters). This is not decoration — classical MoE was itself trained by EM (Jordan & Jacobs, 1994), so the architecture is a natural neural analogue of mixture-model estimation. This gives the "MoE" claim substance.
- **What changed from your original wording:** the topic is reframed as *amortized, simulation-based point estimation* (a neural Bayes estimator) with an MoE output head. This connects it to a rigorous theoretical foundation (the trained network approximates a Bayes estimator) instead of being "a neural net applied to a problem."

Two alternative framings were considered and rejected: (a) a plain monolithic neural Bayes estimator without MoE structure — sound but architecturally uninteresting, and it wastes the interpretability angle; (b) a "Deep-EM" hybrid that unrolls EM iterations inside a network — viable as future work, but it inherits EM's per-dataset cost and weakens the amortization selling point.

---

## 1. Background and Problem Statement

### 1.1 The base paper

Vaidyanathan & Vani Lakshmi (2017) consider a finite mixture of g multivariate gamma distributions with **independent marginals**. For component i, the joint density of Z = (Z₁,…,Z_k) factorizes as a product of univariate Gamma(α_ij, β_ij) densities (shape α_ij > 0, scale β_ij > 0):

  f(z; π, α, β) = Σᵢ₌₁..g πᵢ fᵢ(z; αᵢ, βᵢ),  Σᵢπᵢ = 1,  0 < πᵢ < 1

with **(2gk + g − 1) free parameters**. Direct ML estimation is awkward: the log-likelihood score equations involve digamma functions and have no closed-form solution, and gradient/iterative methods are sensitive to starting values. The paper's pipeline:

1. **Wilson–Hilferty (WH) cube-root transform** (Wilson & Hilferty, 1931; extended to the two-parameter gamma by Krishnamoorthy et al., 2008): if X ~ Gamma(α, β), then X^(1/3) is approximately N(μ, σ²) with μ, σ² explicit functions of (α, β). This converts the gamma mixture into an approximate **Gaussian mixture**.
2. **MCLUST** (Fraley & Raftery, 1998) on the transformed data → component memberships, Gaussian component parameters (μ_ij, σ_ij²), mixing proportions, and BIC-based choice of g.
3. **Back-mapping:** for each component/marginal, build a wide 100(1−γ)% CI for μ_ij (γ = 0.001), grid-search (α_ij, β_ij) pairs consistent with the CI via the WH relations, and pick the pair maximizing the normal log-likelihood of that marginal.

Validated on 4 simulation cases (2-component bivariate with high/low/medium overlap; 3-component bivariate) over 500 Monte Carlo runs, and on the Madawaska flood data (Yue, 2001).

### 1.2 Limitations that motivate this research

- **L1 — Approximation error compounding.** WH is an approximation (poor for small shape α ≲ 1, where the gamma has a singularity at 0); the estimate quality is bounded by transform error + clustering error + grid-search error, with no end-to-end optimality notion.
- **L2 — Fragility under overlap.** In the paper's own Case 1 (equal mean vectors, high overlap), the misclassification rate is ≈ 0.499 — essentially chance level — and bias/MSE are largest exactly there.
- **L3 — Ad hoc CI-search.** The back-mapping depends on an arbitrary coverage parameter γ, searches each component separately (cost grows with g·k), and has no theoretical justification as an estimator.
- **L4 — Restrictive model class.** Independent marginals, no location parameters; the authors themselves name dependent marginals and location parameters as future work. The WH+MCLUST trick does not survive these extensions (a dependent-marginal multivariate gamma does not transform into anything MCLUST can handle).
- **L5 — No uncertainty quantification.** Point estimates only; SD/MSE known only via expensive Monte Carlo.
- **L6 — Per-dataset cost.** Every new dataset re-runs the whole transform–cluster–search pipeline.

---

## 2. Literature Review (verified)

### 2.1 Classical estimation for gamma mixtures

- **Teicher (1963)**, *Identifiability of finite mixtures*, Ann. Math. Statist. — gamma mixtures are identifiable, so point estimation is well-posed (modulo label permutation).
- **Redner & Walker (1984)**, *Mixture densities, maximum likelihood and the EM algorithm*, SIAM Review 26 — canonical EM treatment.
- **McLachlan & Peel (2000)**, *Finite Mixture Models*, Wiley; **Frühwirth-Schnatter (2006)**, *Finite Mixture and Markov Switching Models*, Springer — standard references.
- **Jones, Lai & Rayner (2000)**, *A bivariate gamma mixture distribution*, Commun. Stat. Theory Methods 29(12) — rare prior work on multivariate gamma mixtures (generalized-Bernoulli scales).
- **Wiper, Insua & Ruggeri (2001)**, *Mixtures of gamma distributions with applications*, J. Comput. Graph. Statist. 10(3) — Bayesian MCMC estimation.
- **Young, Chen, Hewage & Nilo-Poyanco (2019)**, *Finite mixture-of-gamma distributions: estimation, inference, and model-based clustering*, Advances in Data Analysis and Classification 13 — ECM algorithm for univariate gamma mixtures; implemented in R `mixtools::gammamixEM`. **Key baseline.**
- **Liu et al. (2019)**, *Bayesian estimation of generalized Gamma mixture model based on variational EM algorithm*, Pattern Recognition 88 — variational EM for generalized-gamma mixtures (SAR imagery).
- **Vani Lakshmi & Vaidyanathan (2016)**, J. Stat. Theory Appl. 15(1) — univariate precursor of the base paper (normal approximation).
- **Wilson & Hilferty (1931)**, PNAS 17; **Krishnamoorthy, Mathew & Mukherjee (2008)**, Technometrics 50(1), DOI: 10.1198/004017007000000353 — the normal approximation the base paper relies on.
- **Fraley & Raftery (1998)**, The Computer Journal 41(8); Fraley et al. (2012) `mclust` v4 — MCLUST, the base paper's engine.

### 2.2 Simulation-based / amortized neural inference

- **Sainsbury-Dale, Zammit-Mangion & Huser (2024)**, *Likelihood-Free Parameter Estimation with Neural Bayes Estimators*, JASA (arXiv:2208.12942) — trains a network mapping a full sample to parameter point estimates; proves it approximates the Bayes estimator under the training loss; uses **permutation-invariant DeepSets architectures for i.i.d. samples**; includes bootstrap-based uncertainty quantification. **The methodological backbone of this proposal.**
- **Radev et al. (2020)**, *BayesFlow*, arXiv:2003.06281; **Radev et al. (2023)**, *BayesFlow: Amortized Bayesian Workflows with Neural Networks*, JOSS (arXiv:2306.16015) — amortized *posterior* inference with invertible networks + summary networks.
- **Kucharský et al. (2025)**, *Amortized Bayesian Mixture Models*, arXiv:2501.10229 — closest existing work: amortized inference for general Bayesian mixture models, factorizing the posterior into a parameter distribution and a categorical mixture-indicator distribution, with membership classification networks; explicitly handles label switching. **Differentiation from our work:** (i) full posterior via generative networks, not point estimation; (ii) not tailored to gamma/multivariate gamma structure; (iii) no MoE architecture; (iv) never benchmarked against classical gamma-mixture pipelines (WH-MCLUST, ECM) or on hydrology applications.
- **Zaheer et al. (2017)**, *Deep Sets*, NeurIPS — permutation-invariant set functions; **Lee et al. (2019)**, *Set Transformer*, ICML — attention-based set encoding. Architectural basis for handling variable-size samples.
- **Bishop (1994)**, *Mixture Density Networks*, NCRG Tech. Rep. — conditional mixture density estimation; related but solves a different problem (regression, not sample-to-parameter estimation).

### 2.3 Mixture of Experts

- **Jacobs, Jordan, Nowlan & Hinton (1991)**, *Adaptive Mixtures of Local Experts*, Neural Computation 3 — original MoE.
- **Jordan & Jacobs (1994)**, *Hierarchical Mixtures of Experts and the EM Algorithm*, Neural Computation 6 — **classical MoE is trained by EM; the gate is literally a responsibility estimator.** This is the conceptual bridge our architecture exploits.
- **Shazeer et al. (2017)**, *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*, ICLR; **Fedus, Zoph & Shazeer (2022)**, *Switch Transformers*, JMLR 23 — modern sparse MoE scaling.

### 2.4 Label switching / permutation symmetry

- **Jasra, Holmes & Stephens (2005)**, *MCMC methods and the label switching problem in Bayesian mixture models*, Statistical Science 20 — survey of the problem.
- **Monteiller et al. (2019)**, *Alleviating Label Switching with Optimal Transport*, NeurIPS — optimal-transport alignment of mixture components.
- **Carion et al. (2020)**, DETR, ECCV — Hungarian (bipartite matching) losses for set-structured predictions; the template for our permutation-invariant parameter loss.

### 2.5 Gap statement

> No existing work (i) estimates the parameters of multivariate gamma mixture models with deep networks, (ii) uses an MoE architecture that mirrors the E/M structure of mixture estimation, (iii) handles the label-switching symmetry of mixture parameters inside a *point-estimation* loss, or (iv) benchmarks amortized neural estimators against the WH–MCLUST and ECM statistical pipelines. All four are combined here for the first time.

---

## 3. Research Questions, Objectives, Hypotheses

**RQ1 (Accuracy).** Does the proposed Deep-MoE estimator achieve lower RMSE/bias than WH–MCLUST (base paper), ECM gamma-mixture EM, and direct numerical MLE, especially under medium/high component overlap and small shape parameters (α < 1)?

**RQ2 (Architecture).** Does the MoE head (gate + per-component experts) outperform a monolithic estimator head of matched capacity? (Ablation.)

**RQ3 (Model order).** Can the framework estimate the number of components g as accurately as BIC-based selection (MCLUST) — via a g-classification head — when g is unknown?

**RQ4 (Robustness & scope).** How does the estimator behave under model misspecification (Weibull/lognormal components, dependent marginals via copula, outliers), and does it extend to settings where WH–MCLUST fails entirely (dependent marginals, location parameters, censoring)?

**RQ5 (Efficiency).** What is the amortized inference cost vs. per-dataset classical pipelines (orders-of-magnitude speedup after one offline training cost)?

**RQ6 (Uncertainty).** Can bootstrap ensembles of the neural estimator produce intervals with empirical coverage near nominal levels?

**Hypotheses.**
- H1: Deep-MoE RMSE ≤ WH–MCLUST RMSE in all overlap regimes, with the largest relative gain in the high-overlap regime (base paper's Case 1) and in small-α regimes.
- H2: MoE head ≥ monolithic head in accuracy at equal parameter count, with interpretable gate behavior (gates ≈ empirical responsibilities).
- H3: The permutation-invariant composite loss (§5.3) dominates naive MSE training (which is corrupted by label switching).
- H4: g-classification accuracy ≥ MCLUST/BIC selection accuracy across n.
- H5: Inference time per dataset is ≥ 100× lower than WH–MCLUST/EM at n ≥ 10³ after amortization.

**Objectives.** O1: build the simulator + estimator; O2: reproduce the base paper's four cases and benchmark; O3: factorial simulation study; O4: real-data validation (Madawaska flood data + ≥1 more dataset); O5: robustness/extension studies; O6: theoretical analysis; O7: open reproducible code.

---

## 4. Mathematical Formulation

**Model.** Data D = {z₁,…,z_n}, z_j ∈ ℝ₊ᵏ, drawn i.i.d. from

  f(z; θ) = Σᵢ₌₁..g πᵢ Π_{p=1..k} Gam(z_p; α_ip, β_ip)

with θ = (π₁..g, {α_ip, β_ip}) ∈ Θ, dim(Θ) = 2gk + g − 1. Shape α_ip > 0, scale β_ip > 0, π on the simplex. Extensions (Study E): location vector δ_ip, and dependence between marginals induced by a Gaussian copula with correlation matrix R_i.

**Estimator as a learned statistic.** We seek a function φ_θ̂: (ℝ₊ᵏ)ⁿ → Θ — a *set function*, invariant to the n! orderings of D — trained on simulated pairs (θ^(m), D^(m)), θ^(m) ~ p(θ), minimizing E[L(θ, φ(D))]. Under squared-error loss the population minimizer is the posterior mean E[θ|D] — a Bayes estimator; under L1 loss, the posterior median (Sainsbury-Dale et al., 2024). Hence the trained network is a **neural Bayes point estimator (NBE)** for the mixture parameters.

**Design requirements on φ:**
1. Permutation invariance in the data (DeepSets encoder).
2. Equivariance-free output handling of component-label symmetry (permutation-invariant loss, §5.3).
3. Positivity/simplex constraints on outputs (softplus for α, β; softmax for π).
4. Variable n at test time (set encoder) and variable g (g-head or padded experts + gate sparsity).

---

## 5. Proposed Framework: Deep-MoE Estimator

### 5.1 Architecture (three blocks)

**Block A — Permutation-invariant encoder.**
Each observation z_j is embedded by an MLP ρ(·) (input optionally augmented with WH features z_j^(1/3) and log z_j — *domain-knowledge feature engineering, not an approximation crutch*), pooled by sum/mean (DeepSets) or ISAB attention (Set Transformer) into a sample embedding h. Optionally append a vector of classical summary statistics (empirical moments, quantiles, MCLUST BIC profile) concatenated to h — a "data + classical-hint" fusion variant to be ablated.

**Block B — Gating network (the "E-step").**
A gated head over h producing:
- π̂ ∈ Δ^{g_max−1} (softmax mixing weights), and
- optionally, per-observation soft assignments r_ji = softmax(gate([ρ(z_j), h])) — interpretable as learned responsibilities, directly comparable to EM responsibilities and usable for clustering/misclassification metrics.

**Block C — Expert networks (the "M-step").**
G_max parallel expert MLPs E_i(h) → (α̂_i, β̂_i) ∈ ℝ₊^{2k} via softplus outputs. Component i's parameter vector is read from expert i; the final estimate is the set {(π̂_i, α̂_i, β̂_i)} with trailing experts masked when ĝ < g_max. Load-balancing regularization (auxiliary gate-entropy or Switch-style load-balancing loss) prevents expert collapse during training.

**Optional Block D — Model-order head.**
A classifier over h predicting ĝ ∈ {1,…,g_max}, trained on datasets simulated with random g; at test time, mask experts beyond ĝ. Alternative: keep g_max experts and threshold π̂_i < ε (sparsity selection), mirroring BIC selection in MCLUST. Both variants evaluated under RQ3.

### 5.2 Why this is a principled "Mixture of Experts" and not a buzzword

Jordan & Jacobs (1994) showed classical MoE training *is* EM: gates estimate responsibilities, experts specialize by region. Here the correspondence is explicit and statistical: the gate's r_ji are responsibilities of observation j for mixture component i; expert i is specialized to estimate component i's (α_i, β_i); π̂_i is the gate's marginal responsibility mass. The architecture *is* the mixture model's structure, embedded in the estimator. This yields three concrete benefits to test: (H2) expert specialization under heterogeneous component regimes; interpretability (gate ≈ responsibilities → misclassification rate measurable without extra machinery); modular extensibility (add an expert per extra component; conditional computation at inference if desired).

### 5.3 Training losses — solving label switching

Mixture parameters are set-valued: permuting component labels leaves the model unchanged, so naive MSE between θ̂ and θ is ill-defined and trains the network toward the *average over labelings* (destructive interference). Composite loss per training pair (θ, D):

  L = λ₁ · L_Hungarian + λ₂ · L_NLL + λ₃ · L_aux

- **L_Hungarian (permutation-aligned parameter error).** Compute the g! (Hungarian-solved in O(g³)) assignment σ* = argmin_σ Σᵢ ||θᵢ − θ̂_{σ(i)}||₁ (on standardized log-parameters), then L = Σᵢ ||θᵢ − θ̂_{σ*(i)}||²₂. g ≤ 5 keeps the assignment solve negligible.
- **L_NLL (label-invariant likelihood anchor).** L_NLL = −(1/n) Σ_j log f(z_j; θ̂) evaluated on the *training-sample* D (or a fresh same-θ sample). Invariant to label permutation by construction; ties training to the ML principle; scale-aware (errors on well-separated components weigh differently than on overlapping ones). Also supplies a natural early-stopping metric.
- **L_aux:** gate load-balancing + (Block D) cross-entropy on g + optional responsibility supervision (since the simulator knows the true component labels z_j came from, we can supervise r_ji with cross-entropy — a luxury classical EM lacks; ablate its effect).

### 5.4 Training-data simulator and prior design

- **Prior p(θ):** independent ranges informed by the base paper's cases but widened: α_ip ~ LogUniform(0.1, 30) (covers the singular α<1 regime where WH is poor and the near-Gaussian large-α regime); β_ip ~ LogUniform(0.1, 30); π ~ Dirichlet(1,…,1); n ~ Uniform{25,…,5000}; g ~ Uniform{1,…,g_max=4} (for the unknown-g variant); k ∈ {1,2,3,5} (train separate networks per k, or a k-conditioned network with masked pooling).
- **Overlap-stratified sampling:** stratify simulation draws by pairwise component overlap (approx. misclassification probability computed from the sampled θ, e.g., via Monte Carlo estimate of the Bayes error between components) into low/medium/high bins, so training coverage matches the evaluation design (§6.2).
- **Online (on-the-fly) generation:** fresh (θ, D) pairs each step — effectively infinite training set, no overfitting to a fixed grid. ~10⁵–10⁶ pairs/epoch equivalent; train until validation NLL plateaus.

### 5.5 Optimization protocol

AdamW, cosine LR schedule (peak 1e-3 → 1e-5), batch = 64 datasets, gradient clipping 1.0; hyperparameters (encoder width/depth, number of experts, λ's, pooling type) selected by random search (≤ 60 trials) on validation NLL of a held-out simulation suite; early stopping; 3 random seeds per final config to report training variance. Implementation: PyTorch; classical baselines in R (`mclust`, `mixtools`, `optim`).

---

## 6. Evaluation Methodology

### 6.1 Study 1 — Direct replication benchmark (comparability with the base paper)

Reproduce the base paper's exact four cases (Table 1 & 2 therein): Cases 1–3 = two-component bivariate mixtures with high/low/medium overlap (equal means / equal scales / equal covariance respectively); Case 4 = three-component bivariate. 500 Monte Carlo runs per case (matching the paper). Report AVG, Bias, SD, MSE per parameter and misclassification rate — the paper's exact metrics — side by side with published WH–MCLUST tables. All baselines re-run on identical simulated datasets (paired design).

### 6.2 Study 2 — Factorial simulation study (generalization)

Full factorial: g ∈ {2,3,4} × k ∈ {1,2,3,5} × n ∈ {50, 100, 250, 500, 1000, 5000} × overlap ∈ {low, medium, high} × shape regime {mixed α, all α ≥ 1, includes α < 1}. Cells sampled with randomized θ per replication (not one fixed θ per cell — avoids anecdotal parameter choices); ≥ 500 replications per cell; overlap quantified per draw by estimated Bayes error. Stratified analysis by cell.

### 6.3 Baselines (all paired on identical datasets)

- **B1 — WH + MCLUST + CI-search** (base paper method; R, `mclust`; γ = 0.001 as published). *The primary comparator.*
- **B2 — ECM for gamma mixtures** (Young et al., 2019; `mixtools::gammamixEM` for k=1; for k>1 with independent marginals, EM with per-marginal numerical M-step via Newton on the digamma score equations).
- **B3 — Direct numerical MLE**: BFGS/`optim` on the observed-data log-likelihood with 10 random restarts + best-of selection (strong honest baseline the base paper did not compare against).
- **B4 — Method of moments** (first 2gk+g−1 sample moments) — fast classical floor.
- **B5 — Bayesian MCMC** (Wiper et al., 2001 style; JAGS/Stan) on a small-n subset only, as gold-standard accuracy reference where affordable.
- **B6 — Ablation nets:** monolithic DeepSets→MLP head (no MoE); MoE without L_NLL; MoE without L_Hungarian (NLL only); MoE without responsibility supervision; encoder with/without WH/log feature hints and with/without classical-statistics fusion.

### 6.4 Metrics

- Per-parameter Bias, SD, RMSE — **after optimal component alignment** (Hungarian match of estimates to truth per replication) to avoid label-switch artifacts in reporting.
- Achieved log-likelihood of estimated parameters on an independent same-θ test sample (label-free accuracy measure).
- Misclassification rate (MAP responsibility vs. true label, Hungarian-aligned).
- Model-order accuracy: P(ĝ = g) and mean |ĝ − g|.
- Calibration/coverage of bootstrap ensemble intervals (Study 4).
- Runtime: per-dataset inference seconds; total cost including training amortization curves (accuracy vs. #training simulations).
- Convergence/failure rate of iterative baselines (non-convergence counts as a failure mode; NBE has none).

### 6.5 Statistical analysis of results

Paired per-replication comparisons: Wilcoxon signed-rank tests of RMSE differences (Deep-MoE vs each baseline) within each cell; Holm correction across cells; effect sizes (median ratio of RMSEs) with bootstrap CIs. Pre-registered analysis plan frozen before final runs. Power: 500 reps/cell detects ≈ 0.13 SD paired RMSE differences at 90% power, α = 0.05.

### 6.6 Study 3 — Real-data application

- **Madawaska basin flood data** (Yue, 2001; n = 77 years; flood duration × flood peak, independent gamma marginals) — the base paper's dataset: validate a one-component fit, KS tests + probability plots as in the paper, plus bootstrap intervals and posterior-predictive-style checks (simulate from fitted model, compare ECDFs).
- **≥1 multivariate dataset with genuine mixture structure** (candidates: streamflow/peak-volume flood data with multiple flood-generating mechanisms; insurance claim severity; reliability lifetime data). Success criteria: goodness-of-fit (KS/AD per marginal + energy-distance multivariate test), likelihood vs. baselines, and plausibility of recovered components.
- **Prior-mismatch safeguard:** before applying the amortized net, run prior predictive checks (does the real sample look like training simulations?); if mismatch, fine-tune briefly on a re-centered prior (transfer step, minutes not days).

### 6.7 Study 4 — Uncertainty quantification

Nonparametric bootstrap of the test sample → distribution of θ̂ over bootstrap resamples (Sainsbury-Dale et al., 2024 §5 recipe); ensemble-of-seeds variant; report empirical coverage of nominal 90/95% intervals across Study-1 cases. Optional extension head: NPE (BayesFlow-style normalizing-flow posterior) for full joint posterior — future work, kept out of the core claims.

### 6.8 Study 5 — Robustness and scope extensions (RQ4)

- Misspecification: fit gamma mixture to data simulated from Weibull mixtures, lognormal mixtures, and gamma + 2–5% outlier contamination; measure degradation vs. likelihood-free retraining (the amortized framework can simply *retrain on misspecified simulators* — impossible for WH–MCLUST).
- Dependent marginals: simulate multivariate gamma with Gaussian-copula dependence (extension the base paper explicitly lists as future work and cannot handle); train the same architecture; compare against B3-style numeric MLE where feasible.
- Location parameters: 3-parameter gamma components.
- Censoring: right-censored lifetimes — handled by adding the censoring mechanism to the simulator only.

---

## 7. Theoretical Analysis Plan

1. **Bayes-optimality link.** State and use the NBE result: as training pairs → ∞ and capacity → ∞, the estimator converges to the Bayes estimator under p(θ) (Sainsbury-Dale et al., 2024). Derive the implication for our composite loss (Hungarian-MSE selects the permutation-aligned posterior mean; NLL term approximates MAP-like behavior).
2. **Identifiability.** Gamma mixtures are identifiable (Teicher, 1963); the only non-identifiability is label permutation — formally justifying the permutation-invariant loss as *complete* symmetry handling.
3. **Amortization gap.** Empirically bound the gap between the amortized NBE and per-dataset exact Bayes/MLE across n; relate gap to training-set size (learning curves).
4. **Prior sensitivity.** Systematic study: evaluate under shifted priors; quantify robustness; document the effective implicit prior.

## 8. Threats to Validity and Mitigations

- **Prior-data mismatch (internal):** amortized estimator is Bayes only under the training prior → sensitivity analysis + fine-tuning protocol (§6.6).
- **Label-switch artifacts in metrics:** all per-parameter metrics computed after Hungarian alignment.
- **Simulation gap (external):** real data may differ from simulator → prior predictive checks, misspecification study (§6.8), honest reporting of where degradation appears.
- **Baseline fairness:** equal compute budgets considered; B3 given 10 restarts; report convergence failures of baselines separately rather than silently excluding them.
- **Selection bias in fixed-θ cells:** randomized θ per replication in Study 2.
- **Overclaiming g-estimation:** unknown-g claims restricted to the trained g_max range and n ≥ a data-driven floor.

## 9. Expected Contributions

1. **First deep-learning estimator for multivariate gamma mixture parameters** — a new, open niche (verified against current literature).
2. **A new architecture pattern:** an MoE estimator whose gate/experts instantiate the E/M structure of mixture estimation — interpretable (gate = responsibilities) and extensible (experts = components); connects 1991/1994 classical MoE theory to modern amortized inference.
3. **A permutation-invariant composite training loss** for mixture *parameter* point estimation (Hungarian alignment + likelihood anchor + responsibility supervision).
4. **The first rigorous benchmark** of WH–MCLUST vs. ECM vs. direct MLE vs. amortized neural estimation on gamma mixtures, including the base paper's exact cases.
5. **Scope extensions the base paper could not address:** dependent marginals (copula), location parameters, censoring, unknown g — delivered by simulator-only changes.
6. Reproducible open-source pipeline (simulator + PyTorch estimator + R baselines + experiment configs + seeds).

## 10. Publication and Dissemination Targets

Primary: *Statistics and Computing* or *Computational Statistics & Data Analysis* (statistics audience, direct successor of the base paper's venue class). Secondary: *Journal of Statistical Computation and Simulation*; an abridged architecture-focused version for an ML workshop (e.g., NeurIPS/ICML simulation-based-inference workshops). Code + data repository with CC-BY license; arXiv preprint first.

## 11. Timeline and Resources (9 months)

| Phase | Months | Deliverable |
|---|---|---|
| P1: simulator + baselines (R/Python) | 1–2 | Verified data generator; B1–B4 reproduced base-paper Tables 3–5 |
| P2: architecture + losses + training loop | 2–3 | Trained Deep-MoE v1; ablation grid defined |
| P3: Studies 1–2 (replication + factorial) | 3–5 | Benchmark tables; statistical tests |
| P4: Studies 3–4 (real data + UQ) | 5–7 | Application section; coverage results |
| P5: Study 5 + theory + writing | 7–9 | Robustness results; manuscript + repo |

Compute: single GPU (≥ 16 GB) sufficient — datasets are small (n ≤ 5×10³, k ≤ 5); training dominated by simulation throughput (CPU-parallel gamma generation); no large-scale cluster needed.

## 12. Reproducibility and Ethics

Fixed seeds per study; containerized environment; all configs versioned; synthetic data only for Studies 1–2 (no privacy issues); real hydrological data are published/public; license: MIT (code) / CC-BY (paper artifacts). An analysis plan (§6.5) is frozen before final benchmark runs to avoid metric shopping.

---

## References (key verified items; full bib in manuscript)

1. Vaidyanathan & Vani Lakshmi (2017), *Sri Lankan J. Appl. Stat.* 17(3). DOI: 10.4038/sljastats.v17i3.7902 — base paper.
2. Krishnamoorthy, Mathew & Mukherjee (2008), *Technometrics* 50(1):69–78. DOI: 10.1198/004017007000000353.
3. Wilson & Hilferty (1931), *PNAS* 17:684–688.
4. Fraley & Raftery (1998), *The Computer Journal* 41(8):578–588.
5. Young, Chen, Hewage & Nilo-Poyanco (2019), *Advances in Data Analysis and Classification* 13 — ECM gamma mixtures (`mixtools::gammamixEM`).
6. Liu et al. (2019), *Pattern Recognition* 88 — variational EM generalized-gamma mixtures.
7. Wiper, Insua & Ruggeri (2001), *JCGS* 10(3) — Bayesian gamma mixtures.
8. Jones, Lai & Rayner (2000), *Commun. Stat. Theory Methods* 29(12):2775–2790.
9. Sainsbury-Dale, Zammit-Mangion & Huser (2024), *JASA* — Neural Bayes Estimators. arXiv:2208.12942.
10. Radev et al. (2020), BayesFlow. arXiv:2003.06281; Radev et al. (2023), *JOSS*. arXiv:2306.16015.
11. Kucharský et al. (2025), *Amortized Bayesian Mixture Models*. arXiv:2501.10229.
12. Jacobs, Jordan, Nowlan & Hinton (1991), *Neural Computation* 3:79–87.
13. Jordan & Jacobs (1994), *Neural Computation* 6:181–214.
14. Shazeer et al. (2017), ICLR — Sparsely-Gated MoE. arXiv:1701.06538.
15. Fedus, Zoph & Shazeer (2022), *JMLR* 23 — Switch Transformers. arXiv:2101.03961.
16. Zaheer et al. (2017), NeurIPS — Deep Sets. arXiv:1703.06114.
17. Lee et al. (2019), ICML — Set Transformer. arXiv:1810.00825.
18. Jasra, Holmes & Stephens (2005), *Statistical Science* 20(1):50–67.
19. Monteiller et al. (2019), NeurIPS — label switching via optimal transport.
20. Carion et al. (2020), ECCV — DETR / Hungarian matching loss. arXiv:2005.12872.
21. Teicher (1963), *Ann. Math. Statist.* 34:1265–1269.
22. Redner & Walker (1984), *SIAM Review* 26:195–239.
23. McLachlan & Peel (2000), *Finite Mixture Models*, Wiley.
24. Yue (2001), *Hydrological Processes* 15(6):1033–1045. DOI: 10.1002/hyp.259.
25. Bishop (1994), *Mixture Density Networks*, Aston Univ. NCRG/94/004.
