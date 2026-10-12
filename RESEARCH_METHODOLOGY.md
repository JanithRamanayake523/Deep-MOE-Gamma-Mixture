# Research Methodology

# Amortized EM: Symmetry-Aware Neural Estimation for Mixtures Beyond Closed-Form M-Steps

*(Working title. Alternatives and how to choose between them after the first experiments: [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) section 6.)*

**Method:** AEM-Net (Amortized Expectation–Maximization Network)
**Target venue:** NeurIPS 2027 (main track). Fallbacks: ICML 2027 (reduced version), AISTATS 2028, TMLR. arXiv preprint and an SBI-workshop version first.
**Companion documents:** [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md) (verified related work and the gap). [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) (one-page summary and status). The superseded v1 design and code are in [`archive/v1/`](archive/v1/).

> **Integrity note for the student.** Rewrite every section in your own words before submission. Check every citation against the original paper. Report every negative result. Nothing in this document is a result. Theorems are *targets with proof sketches*, and each must be proven rigorously (or weakened) before it is claimed.

---

## 0. Why v1 was redesigned (one paragraph)

v1 applied a DeepSets neural Bayes estimator to independent-marginal gamma mixtures and benchmarked it mainly against WH+MCLUST. That has three problems:

- The likelihood is tractable and EM is fast, so the amortization is unmotivated.
- The comparator is weak.
- The "MoE" was hollow: the experts read only a global embedding, the responsibilities never reached them, and the heads were not equivariant to component order.

v2 keeps the student's domain (gamma and multivariate-gamma mixtures) but reframes the contribution for an ML audience. It becomes a **general, symmetry-aware, EM-structured amortized estimator for mixtures whose M-step has no closed form or whose likelihood is intractable**, with theory that says *what* the network learns and *when* it matches MLE.

---

## 1. Problem setting and notation

- **Data:** D = {z_1, …, z_n}, with z_j ∈ 𝒵 ⊆ ℝᵏ i.i.d. from a finite mixture
  p(z | ϑ) = Σ_{s=1}^{g} π_s f(z; θ_s), with ϑ = (π, θ_1, …, θ_g).
- **Component family** ℱ = {f(·; θ)}. We study:

| Family | Density tractable? | Closed-form M-step? | Role |
|---|---|---|---|
| Gaussian (full covariance) | yes | yes | Comparison with TGMM, Set Transformer, Cluster-PFN |
| Independent-marginal gamma (the base paper's model) | yes | **no** (inverse digamma) | Flagship; Thm 1 approximate case |
| Weibull, Student-t (ν unknown) | yes | no | M-step with no closed form |
| Negative binomial (counts) | yes | no (for dispersion) | Discrete family |
| Gaussian-copula multivariate gamma | awkward (copula + gamma CDFs) | no | Dependent marginals |
| Kibble–Moran bivariate gamma | Bessel-series density | no | Intractable-leaning likelihood |
| Right-censored gamma/Weibull | censored likelihood | no | Survival / reliability |
| Gamma + Gaussian measurement error | **no closed form** (convolution integral) | no | Intractable gamma case (E4) |

- **Symmetry groups acting on ϑ:**
  - S_g permutes the component labels. Label switching: p(D | σϑ) = p(D | ϑ).
  - S_n permutes the data.
  - For scale families, the scaling group (ℝ₊)ᵏ acts: z ↦ c ⊙ z maps (α, β) ↦ (α, c ⊙ β).
- **Prior** p(ϑ, g, n) over simulation settings. It is exchangeable in the components: p(σϑ) = p(ϑ).
- **Goal:** an estimator φ: D ↦ (ĝ, ϑ̂, q(ϑ | D)) that is fast (amortized), accurate at small n, efficient at large n, and calibrated.

---

## 2. Research questions and falsifiable hypotheses

| RQ | Hypothesis | Falsified if |
|---|---|---|
| RQ1 Accuracy | **H1:** AEM-Net has lower quotient risk (§4.2) than NBE-DeepSets, Set Transformer, TGMM and ABMM on every non-Gaussian family, and matches TGMM on Gaussian | No significant gain (Holm-corrected paired Wilcoxon) on ≥ 2 of 3 non-Gaussian families |
| RQ2 n-regime | **H2:** amortized AEM beats EM/MLE for n ≲ 200; the hybrid (AEM + K EM steps) matches the MLE's risk at n ≥ 2000 and is never worse than the better of the two by > 5% | The hybrid is significantly worse than ECM with 10 restarts at large n |
| RQ3 Symmetry | **H3:** MSE training collapses the components (Thm 2); ordering constraints bias estimates under overlap; the quotient loss matches Stephens-relabelled NUTS posterior means | The quotient estimator deviates from the relabelled NUTS mean by more than the NUTS Monte-Carlo error on the E3 grid |
| RQ4 Architecture | **H4:** each component (likelihood-aware E-block, responsibility-pooled M-block, shared expert, canonicalization, deep supervision) contributes | Removing an ingredient changes risk by < 2% (then drop it from the claims) |
| RQ5 Intractable | **H5:** on copula, Kibble and censored mixtures, AEM-Net beats NPE-flow, ABC-SMC and MCEM at equal wall-clock | No gain over NPE-flow with the same encoder |
| RQ6 UQ | **H6:** the flow head's quotient-space posteriors are calibrated (SBC uniform; 90/95% coverage within ±2%) and close to NUTS (C2ST ≤ 0.6) | Systematic under-coverage that conservative calibration cannot fix |
| RQ7 Scale | **H7:** on "many-dataset" real problems (≥ 10⁴ datasets), AEM-Net gives EM-level fit at ≥ 100× lower total compute, including training | EM parallelized on the same hardware is within 10× |
| RQ8 Order | **H8:** the existence head estimates g at least as well as BIC-EM and Cluster-PFN | Worse P(ĝ = g) at every n |

---

## 3. AEM-Net architecture

The network has five stages (§3.1–3.5). Stage 3 is repeated T times with shared weights.

### 3.1 Canonicalization (exact scale equivariance)

For each marginal p, compute a scale-equivariant statistic c_p(D), for example the median, with c_p(aD) = a_p c_p(D). Work with the standardized data z̃_j = z_j ⊘ c(D). The network estimates θ̃. For scale parameters, map back with β_{s,p} = c_p β̃_{s,p}. Shape parameters and weights are unchanged. For families without a scale group, this stage is the identity. For location-scale families, also subtract a location statistic.

- **Why:** it removes an entire nuisance dimension from what must be learned, so the prior over β becomes much narrower.
- **Guarantee:** exactness is Prop 1(c).

### 3.2 Point encoder

x_j = [φ_ℱ(z̃_j), log z̃_j, z̃_j^{1/3}, 𝟙_censored,j]. Here φ_ℱ holds the family's known sufficient statistics; for gamma these are z and log z. The cube root is the Wilson–Hilferty feature, which keeps the base paper's idea as an input feature.

ρ_j = ISAB_L(MLP(x_j)): L induced-set-attention blocks (Set Transformer) with m inducing points. The cost is O(nm), so n = 10⁴ is cheap. The encoder is permutation-equivariant over j.

### 3.3 Slot initialization (symmetry breaking)

S⁰ = PMA_{g_max}(ρ) uses g_max *distinct learned seed vectors*.

- **Important subtlety (paper-worthy):** a strictly slot-equivariant network whose slots all start identical *cannot* output distinct components. It would collapse, just as in Thm 2. What must be symmetric is the *set* output D ↦ {(π_s, θ_s)}, not the individual slots. Distinct seeds break the symmetry. The quotient loss (§4.2) makes the choice of which seed takes which component irrelevant.
- **Ablation:** random Gaussian seeds, as in Slot Attention, versus learned seeds.

### 3.4 T unrolled EM blocks (weights shared across slots; tied or untied across iterations)

For t = 1..T:

**E-block (the gate: learned responsibilities)**

  ℓ_js = a · log f(z̃_j; θ_s^{t−1}) · 𝟙[ℱ tractable] + log π_s^{t−1} + MLP_E([ρ_j, S_s^{t−1}])
  r_js = softmax_s(ℓ_js / τ)

- a and τ are learned scalars.
- With a = 1, MLP_E = 0 and τ = 1, this is exactly the EM E-step.
- For intractable families, a = 0 and the learned compatibility MLP_E does the job.
- For censored observations, log f is replaced by log survival: log S(z̃_j; θ_s).

**M-block (the experts: learned M-step)**

  N_s = Σ_j r_js
  m_s = (1/N_s) Σ_j r_js ψ(ρ_j, x_j)   (responsibility-weighted pooling of learned sufficient statistics; ψ includes the raw φ_ℱ(z̃_j))
  v_s = (1/N_s) Σ_j r_js (ψ_j − m_s)²   (a second-moment channel)
  π_s^t = N_s / n
  U_s = MLP_U([m_s, v_s, log N_s, S_s^{t−1}, h_global])
  S_s^t = (1 − u_s) ⊙ S_s^{t−1} + u_s ⊙ U_s, where u_s = σ(MLP_u(·)) is a gated update (u = 1 means overwrite)
  θ_s^t = Expert(S_s^t)   (**one shared expert network** for all slots)

- h_global = mean_j ρ_j gives every slot cross-slot context. An optional slot self-attention layer lets slots "negotiate" with each other; this is ablated.
- The Expert has constrained outputs: softplus for α and β, Cholesky factors for covariances, (−1, 1) via tanh-parameterized correlation for copulas, and so on.
- The Expert is a learned **inverse of the mean-to-natural map** (∇A)⁻¹. For gamma, that inverse has no closed form (log α − ψ(α) = log E z − E log z).

**Why this is a genuine mixture of experts.** The gate's outputs are per-observation responsibilities. Each expert's input is *its own* responsibility-weighted data. π is the responsibility mass. This is exactly the structure of Jordan & Jacobs (1994), unrolled.

### 3.5 Output heads

- **Existence head:** e_s = σ(MLP_e(S_s^T)) gives ĝ = Σ_s 𝟙[e_s > ½]. A slot counts as active when it exists and π_s > ε.
- **Posterior head (C4):** a conditional flow-matching vector field v_ω(θ, τ | S_s^T, h_global) per slot, plus a Dirichlet-concentration head for π. Samples live in the *pivot-aligned* coordinates of §4.4.
- **Refinement (C3, applied at test time):** see §5.

### 3.6 Complexity

Encoder: O(n m L). E/M blocks: O(n g_max T). Inference for n = 10⁴, g_max = 6, T = 5 is a few milliseconds on a GPU, and batched inference runs 10⁴–10⁶ datasets per GPU-minute.

---

## 4. Theory

**Proof targets.** Each item below is a *target statement with a sketch*. Full proofs go in the appendix. If a proof does not go through, weaken the statement; do not overclaim.

### 4.1 Prop 1 (Symmetry)

- (a) AEM-Net is invariant to permutations of D, because every aggregation over j is a sum or attention pooling.
- (b) Its output multiset {(π_s, θ_s, e_s)} is invariant to relabelling the slot seeds together with the parameters. The E and M blocks are S_{g_max}-equivariant given the slot states.
- (c) For scale families, with a scale-equivariant c(·): φ(aD) = a ⋆ φ(D) *exactly*, where ⋆ multiplies the scale parameters by a.

*Sketch:* direct composition. For (c), z̃ is invariant under D ↦ aD, so θ̃ is invariant, and the back-transform is equivariant.

### 4.2 The quotient loss

Let Θ_g be the parameter space of a g-component mixture, in standardized log-coordinates for positive parameters. Define

  L_Q(ϑ, ϑ̂) = min_{σ∈S_g} Σ_s w ⊙ ‖u(ϑ_s) − u(ϑ̂_{σ(s)})‖²,

where u is a fixed reparameterization (log α, log β, logit-π, …) and w holds per-coordinate weights. L_Q is a squared metric on the quotient Θ_g / S_g. It is computed with the Hungarian algorithm in O(g³) (`losses/quotient.py`).

### 4.3 Thm 2 (Label-switching collapse, and ordering bias)

- **(a) Collapse.** If the prior is exchangeable in components, then for every D the Bayes estimator under the *naive* squared loss Σ_s ‖ϑ_s − ϑ̂_s‖² satisfies ϑ̂_1 = … = ϑ̂_g. That is, all components equal the symmetrized posterior mean. This holds **for any function class rich enough to represent the posterior mean**, equivariant or not.
  - *Sketch:* the likelihood is S_g-invariant, so the posterior is exchangeable. Then E[ϑ | D] = E[σϑ | D] = σE[ϑ | D] for all σ, so every block of the mean is identical.
- **(b) Ordering bias.** If the training targets are relabelled by an identifiability constraint (e.g. sort by component mean, as ABMM does), the Bayes estimator is the posterior mean of the *order statistics*. Under strong overlap this is biased away from the truth: it is pushed toward separation. For a two-component, one-dimensional example, give a closed form for the bias as a function of overlap.
  - *Sketch:* E[max(X₁, X₂)] − E[min(X₁, X₂)] ≥ |E X₁ − E X₂|, with the gap largest under overlap.
- **Empirical check (E3):** train with naive MSE, sort-then-MSE, NLL and L_Q, and plot the estimated components against the truth across overlap levels.

### 4.4 Thm 3 (Quotient Bayes estimator = Stephens relabelling fixed point)

Let R(ϑ̂ | D) = E[L_Q(ϑ, ϑ̂) | D].

- (i) A minimizer ϑ̂* exists. R is continuous and coercive in ϑ̂ when the posterior has finite second moments in u-coordinates.
- (ii) At any minimizer, for posterior-almost-every ϑ the optimal permutation σ_ϑ is unique, and ϑ̂* = E[σ_ϑ⁻¹ ϑ | D]. So ϑ̂* is a fixed point of Stephens' (2000) relabelling algorithm, "align each draw to the current estimate, then average", and it is a Fréchet mean on Θ_g / S_g.
- (iii) Consistency of amortized training: let 𝒜 be the AEM-Net class and assume universality on compact sets. Assume a compact parameter set, a bounded loss, and M → ∞ simulated pairs. Then the empirical-risk minimizer's Bayes risk converges to inf_𝒜 r(φ), which equals the Bayes risk of ϑ̂*. This extends the neural Bayes estimator argument of Sainsbury-Dale et al. (2024) to quotient losses.

*Sketch:* for (ii), the argmin of a finite minimum of quadratics is locally a single quadratic away from ties; ties form a null set under a density; then apply the first-order condition. For (iii), a uniform law of large numbers over a Lipschitz class, then universal approximation of a measurable selection of ϑ̂*.

*Remark:* uniqueness is **not** claimed, since quotient Fréchet means can be non-unique. Report the multiplicity empirically.

### 4.5 Thm 1 (Exact EM embedding)

- **(a) Exact case.** Let ℱ be a regular exponential family, f(z; η) = h(z) exp(⟨η, T(z)⟩ − A(η)). One EM iteration for the g-component mixture equals one AEM block with:
  - ψ = T;
  - a = 1, MLP_E = 0, τ = 1;
  - gated update u = 1 and U_s = m_s;
  - Expert = (∇A)⁻¹.
  
  So T AEM blocks contain T-step EM from any initialization the PMA can represent.
  - *Sketch:* the E-step is softmax_s(⟨η_s, T(z_j)⟩ − A(η_s) + log π_s). The M-step is μ_s = Σ_j r_js T(z_j) / Σ_j r_js, then η_s = (∇A)⁻¹(μ_s) and π_s = N_s / n. This is the Bregman soft-clustering form of Banerjee et al. (2005).
- **(b) Approximate case (gamma, Weibull, …).** If (∇A)⁻¹ is not closed-form but is Lipschitz on the relevant compact set, a width-W expert approximates it to within ε(W). Since the EM map is locally a contraction near the MLE, the T-step error accumulates to at most ε(W)/(1 − κ).
  - For gamma, the inverse is log α − ψ(α) = s. Minka's (2002) closed-form approximation plus a Newton step gives a reference for ε. [cite: Minka 2002, *Estimating a gamma distribution*, tech. report, verify]
- **(c) Corollary.** inf_{φ∈𝒜_T} r(φ) ≤ r(EM_T ∘ init) for every representable init. So, at infinite capacity, AEM-Net never has higher Bayes risk than T-step EM.
- **Empirical check (implemented):** `models/exact.py` builds the hand-set block from the same EBlock / MBlock code as AEM-Net; `tests/test_em_embedding.py` checks that it equals one classical EM step for gamma (k = 1, 3) and Gaussian (k = 2).

### 4.6 Thm 4 (Hybrid efficiency)

- **Assumption A (basin).** There exist r > 0 and κ < 1 such that, with probability → 1, the sample EM operator M_n is a κ-contraction on B(ϑ̂_n^{MLE}, r) up to a label permutation. Balakrishnan, Wainwright & Yu (2017) prove this for certain GMMs. For gamma mixtures, A is an *assumption*; we verify it numerically by estimating κ on the E2 grid.
- **Assumption B (amortized consistency).** P(L_Q(ϑ̃_n, ϑ*) < r²/4) → 1. The Bayes estimator is consistent under a prior with positive density near ϑ*; the neural approximation error must be below r/2.
- **Claim.** With K_n ≥ log n / (2 log(1/κ)) EM steps from ϑ̃_n, the refined estimator satisfies √n (ϑ^{K_n} − ϑ*) → N(0, I(ϑ*)⁻¹) up to a permutation. So it is asymptotically efficient.
  - *Sketch:* ‖ϑ^{K} − ϑ̂_n‖ ≤ κ^K r = o(n^{−1/2}); then Slutsky's lemma together with MLE asymptotics (identifiable, interior, regular).
- **For intractable families:** refinement uses learned iterations (weight-tied AEM blocks run for extra steps) or stochastic-gradient steps on the likelihood where it can be evaluated. There is **no guarantee** in that case, and we say so.

### 4.7 Prop 2 (Posterior on the quotient space)

For a posterior sample ϑ^{(b)}, define the pivot-aligned sample σ_b⁻¹ ϑ^{(b)}, where σ_b = argmin_σ L(ϑ^{(b)}, σϑ̂*).

- This alignment is measurable, and continuous off a null set of ties.
- By the impossibility of continuous canonicalization (Dym et al., ICML 2024), it cannot be continuous everywhere, so discontinuities are unavoidable.
- Credible sets built in aligned coordinates project to valid credible sets on Θ_g / S_g.
- The flow head is trained on pivot-aligned targets: each simulated ϑ is aligned to the network's own point estimate.

---

## 5. Inference modes

1. **Amortized:** one forward pass. Outputs ϑ̂, responsibilities r, ĝ, and posterior samples.
2. **Hybrid-exact (tractable ℱ):** ϑ̂ is the initialization, then K exact EM or ECM steps using `Family.exact_mstep`; for gamma that is `weighted_gamma_mle`. Default K = ⌈log₂ n⌉. Stop early when the log-likelihood gain is < 10⁻⁸·n.
3. **Hybrid-learned (intractable ℱ):** extra weight-tied AEM iterations, then optional likelihood-gradient steps where the likelihood can be evaluated.
4. **Self-consistency adaptation (real data):** fine-tune on unlabeled real datasets with the Mishra et al. (2025) self-consistency loss. Use this only when a prior-predictive MMD check flags a mismatch.

---

## 6. Training

### 6.1 Loss

  𝓛 = L_Q(ϑ, ϑ̂^{T}) + λ_T Σ_{t<T} L_Q(ϑ, ϑ̂^{t})   (deep supervision)
    + λ_r CE(r, true labels ∘ σ*)   (responsibility supervision, labels aligned by the Hungarian permutation σ*)
    + λ_e BCE(e, 𝟙[slot matched to a true component])
    + λ_f 𝓛_FM(v_ω; pivot-aligned ϑ)   (flow-matching posterior head)

- **Pure-L_Q final phase:** over the last 20% of training the auxiliary terms (deep supervision, responsibility CE, existence gradients into the backbone) fade to zero. The delivered network therefore minimises exactly L_Q, which is the condition under which Thm 3 applies. Checkpoints are selected in this phase only.
- **Deliberately excluded from the main objective:** NLL. It would change the target from the Bayes estimator to an amortized MLE (Thm 3 would no longer apply). NLL is used only in refinement and in the "NLL-target" ablation.
- **Unmatched slots** (g < g_max): only the existence loss applies to them.

### 6.2 Priors (in canonical, standardized coordinates)

- g ~ Uniform{1..6}; n ~ LogUniform(20, 10⁴); k ∈ {1, 2, 3, 5}. Use one network per k in the main results; a k-agnostic tokenized variant is an optional extension.
- π ~ Dirichlet(γ·1) with γ ~ LogUniform(0.3, 3), so the data include unbalanced components.
- **Gamma:** log α ~ U(log 0.2, log 50), which includes the α < 1 regime where Wilson–Hilferty fails. Scale parameters are relative (after canonicalization, log β̃ ~ U(−3, 3)).
- **Overlap stratification:** a pre-computed parameter bank (`simulation/bank.py`) files each prior draw into its Bayes-error bin once. Training resamples parameters from the bank and simulates fresh data each time. This is about 40x faster than rejection sampling, and validation uses a held-out part of the bank.
- **Families:** a separate network per family, with the same architecture and hyperparameters. The paper will stress that nothing is tuned per family.

### 6.3 Optimization

- AdamW, lr 3e-4 with cosine decay, batch of 128 datasets, bucketed so that n is similar within a batch, gradient clipping at 1.0, bf16.
- Online simulation: about 10⁷ datasets per family, using CPU workers for sampling.
- Hyperparameter search (≤ 40 trials) on a held-out simulation suite, using quotient risk.
- 3 seeds for each final model.
- Compute: 1–2 GPUs with 24 GB or more; about 1–2 GPU-days per family.

---

## 7. Experiments

**Common protocol for all experiments:**
- All methods run **paired on identical datasets**.
- Metrics are computed after Hungarian alignment (reuse `evaluation/metrics.py`).
- Paired Wilcoxon tests with Holm correction (reuse `evaluation/stats.py`), bootstrap 95% confidence intervals, 3 training seeds.
- Iterative baselines report failures and non-convergence as failures, never silently excluded.
- The analysis plan is frozen before the final runs.

### E1. Synthetic benchmark across families (RQ1)

- **Grid:** family × g ∈ {2, 3, 4, 6} × k ∈ {1, 2, 5} × n ∈ {30, 100, 300, 1000, 3000, 10⁴} × overlap {low, med, high} × (gamma only) shape regime {α < 1 present, all α ≥ 1}.
- **Replications:** 500 random-ϑ replications per cell.

**Baselines:**

| Group | Methods |
|---|---|
| Classical | EM/ECM with k-means++ and 10 restarts; direct MLE (L-BFGS, 10 restarts); method of moments; variational Bayes; NUTS (Stan, n ≤ 1000, subset of cells) as gold standard |
| Amortized | NBE-DeepSets (Sainsbury-Dale); Set Transformer MoG; **TGMM** (official code, Gaussian, then adapted with the same family heads); **ABMM-style** ordering-constrained flow posterior (trained in this code base with the same encoder; the official BayesFlow ABMM is optional); v1-style hollow MoE (`configs/train/baselines/v1style_gamma_k2.yaml`) |
| Gamma-specific | WH-MCLUST (base paper; `baselines/wh_mclust.py`); ECM M-step of Young et al. (2019) inside the EM baseline |
| Batched | Batched GPU EM (`baselines/batched_em.py`) for the speed comparison (H7) |

- **Metrics:**
  - Quotient risk (L_Q).
  - Aligned RMSE for each parameter.
  - Held-out log-likelihood on 10⁴ fresh points.
  - ARI and misclassification rate.
  - Wall-clock time, per dataset and total including training (amortization curve).
  - Failure rate.

### E2. The n-regime (headline figure, RQ2)

- Plot quotient risk against n, on log-log axes, for: AEM (amortized), AEM + K EM steps (hybrid), EM with 10 restarts, MLE, NUTS posterior mean (relabelled), and the Cramér–Rao bound (the trace of the inverse Fisher information, computed numerically).
- Also report the estimated contraction factor κ, as the empirical check of Assumption A.

### E3. Symmetry study (RQ3, Thm 2 and Thm 3)

- Train with each of four losses: naive MSE, sort-then-MSE, NLL, L_Q.
- Show: (i) collapse under MSE; (ii) ordering bias rising with overlap; (iii) the L_Q estimator matching Stephens-relabelled NUTS means (via the `label.switching` R package) within Monte-Carlo error.
- Setting: g = 2 or 3, k = 1 or 2, so NUTS is cheap.

### E4. Intractable and awkward likelihoods (RQ5)

- **Families:** gamma with Gaussian measurement error (`noisy_gamma`; no closed-form density, so this is the headline intractable case); Gaussian-copula gamma mixtures; Kibble–Moran bivariate gamma mixtures; right-censored gamma and Weibull mixtures; g-and-k.
- **Baselines:** NPE-flow with the same ISAB encoder (sbi / BayesFlow); ABC-SMC; Monte-Carlo EM or a stochastic-approximation EM where feasible; ABMM.
- **Report:** risk against simulation budget and against wall-clock time.

### E5. Uncertainty quantification (RQ6)

- SBC rank histograms on the quotient space.
- Coverage at 50/90/95% for each parameter.
- C2ST against NUTS on a 200-dataset subset.
- Expected coverage curves, following Hermans et al. (2022).
- Compare with bootstrap-NBE intervals and ABMM.

### E6. Robustness (RQ1 and RQ6 under shift)

- (i) Prior shift: test with shape and scale ranges 2× wider than the training prior, and with more components than seen in training.
- (ii) Contamination: 2–10% outliers.
- (iii) Misspecification: Weibull or lognormal data fitted as gamma.
- (iv) Mitigation: hybrid refinement and self-consistency fine-tuning.
- Report the degradation curves honestly.

### E7. Real data at scale (RQ7, the "why amortize" section)

| Dataset | Unit | #datasets | Model | Check |
|---|---|---|---|---|
| GHCN-Daily precipitation (NOAA, public) | wet-day amounts, per station and season | 10⁴+ | univariate gamma mixture, g ≤ 3 | Held-out log-likelihood vs. EM; KS/AD; spatial smoothness of the fitted parameters; total compute |
| Insurance severity (R `CASdatasets`, e.g. freMTPL2sev; bivariate claim datasets if available) | portfolio segments | 10²–10³ | (bivariate) gamma or copula-gamma mixture | Tail fit (QQ, VaR backtest) |
| Neural inter-spike intervals (Allen Brain Observatory, public) | per neuron | 10⁴+ | gamma or Weibull mixture (bursting vs. tonic firing) | Agreement with EM; cell-type separation |
| Madawaska flood data (Yue 2001; the base paper's data) | single dataset | 1 | bivariate gamma | Continuity with the base paper; appendix only |
| *(optional)* Sentinel-1 SAR intensity | per patch | 10⁶ | gamma mixture (speckle) | Segmentation quality; throughput |

Protocol: a prior-predictive MMD check first; self-consistency adaptation if it flags a mismatch; always report the hybrid alongside the amortized fit.

### E8. Ablations (RQ4)

The following are ablated one at a time:

- T ∈ {0, 1, 3, 5, 8};
- tied vs. untied iterations;
- E-block with the exact ℓ_ℱ term vs. MLP_E only;
- shared vs. per-slot experts (the v1 design);
- responsibility pooling vs. global-embedding experts (the v1 design);
- canonicalization on vs. off;
- deep supervision on vs. off;
- responsibility supervision on vs. off;
- ISAB vs. DeepSets encoder;
- learned vs. random seeds;
- slot self-attention on vs. off;
- sufficient-statistic features on vs. off.

### E9. Base-paper replication (appendix)

Run the four cases of Vaidyanathan & Vani Lakshmi (2017), with 500 Monte Carlo runs each and the paper's own metrics (AVG, Bias, SD, MSE, misclassification rate). Implemented as `simulation/paper_cases.py` + `configs/experiments/e9_paper_cases.yaml` (handler `replication`), which writes the paper-style AVG / Bias / SD / MSE table and the misclassification rates next to the published ones.

---

## 8. Expected paper structure (9 pages + appendix)

1. **Introduction:** the "many small datasets" motivation, Figure 1 (the n-regime plot), and the contributions.
2. **Related work:** condensed from `LITERATURE_REVIEW.md`.
3. **Setup and symmetry:** Thm 2 and the quotient loss; Thm 3.
4. **AEM-Net:** the architecture and Thm 1.
5. **Hybrid inference:** Thm 4. **Posterior on the quotient space:** Prop 2.
6. **Experiments:** E1–E7, with E8 condensed.
7. **Limitations:** prior dependence, no guarantee for learned refinement, fixed k per network, g_max.

**Appendix:** proofs, the full E1 tables, E9, compute accounting, and a reproducibility checklist.

---

## 9. Risk register

| Risk | Likelihood | Impact | Mitigation / pivot |
|---|---|---|---|
| AEM-Net fails to beat TGMM on Gaussian | Med | Med | Acceptable if it matches; the claims centre on non-Gaussian families. Report honestly. |
| Thm 3(iii) or Thm 4 proofs are harder than expected | Med | Med | State the proved special cases (e.g. g = 2, k = 1); move the general case to a conjecture backed by strong empirics |
| The gain over NPE-flow on intractable families is small | Med | High | Emphasize the point estimator, the hybrid, speed and g-estimation; NPE-flow cannot estimate g |
| Real-data prior mismatch | High | Med | MMD check, self-consistency fine-tuning, and always reporting the hybrid |
| Compute shortfall | Low–Med | Med | Reduce to gamma + Gaussian + copula-gamma for the main paper; move the rest to the appendix or future work |
| Scoop (a TGMM-like paper for non-Gaussian mixtures) | Med | High | Post the arXiv preprint early (after E1–E3, month 3) |

---

## 10. Timeline (7 months, to NeurIPS 2027)

| Month | Work | Exit criterion |
|---|---|---|
| 1 | Build parameter banks; train gamma k = 1 / k = 2 models and the E3 loss variants; E3 pilot | E3 shows collapse vs. the quotient estimator |
| 2 | Baseline harness (NBE, Set Transformer, TGMM, ABMM, ECM, NUTS); E1 on gamma and Gaussian | H1 decision on gamma |
| 3 | Hybrid refinement; E2; drafts of the Thm 2 and Thm 3 proofs; **arXiv v1** | E2 headline figure done |
| 4 | Flow head, E5; Thm 4 proof; κ estimation | Coverage within ±2% |
| 5 | E4 (copula, Kibble, censored); E6 | H5 decision |
| 6 | E7 real data; E8 ablations | Real-data tables |
| 7 | Writing, appendix proofs, code release | Submission |

---

## 11. Implementation status

The full code base is implemented and tested (`python -m pytest tests`: 98 passed, 3 skipped). **No experiment has been run yet;** every number in this document is a hypothesis.

| Methodology item | Where | Status |
|---|---|---|
| Families (section 1), incl. noisy gamma (intractable) | `simulation/families/` | implemented, densities verified numerically |
| Simulator, overlap stratification, scale augmentation | `simulation/simulator.py`, `simulation/bank.py` | implemented; the parameter bank makes stratified simulation as cheap as unstratified |
| AEM-Net (section 3) | `models/aem.py` | implemented; symmetry/equivariance unit-tested |
| Thm 1 construction | `models/exact.py`, `tests/test_em_embedding.py` | verified numerically |
| Quotient loss + E3 variants (section 4.2, Thm 2) | `losses/quotient.py` | implemented; equals brute-force minimum |
| Pure-L_Q final phase (section 6.1, Thm 3 condition) | `training/trainer.py` | implemented; gradient equality tested |
| Flow posterior (C4, Prop 2) | `models/flow.py`, `inference/estimator.py` | implemented |
| Hybrid refinement (Thm 4) | `inference/refine.py`, `baselines/batched_em.py` | implemented |
| Thm 4 Assumption A check | `theory/contraction.py` | implemented |
| Baselines (section 7 table) | `baselines/` | implemented; TGMM and Cluster-PFN load their official code (adapters only) |
| Experiments E1-E9 | `configs/experiments/`, `scripts/run_experiment.py` | configured; every handler exercised end to end in tests |
| Proofs of Thms 1-4 | -- | **sketches only (section 4); to be written** |

The superseded v1 implementation is archived in `archive/v1/` (not maintained).

---

## 12. Reproducibility and ethics

- Fixed seeds, versioned configs and a container.
- All simulation code is released.
- Real data are public (NOAA, CASdatasets, Allen Institute), with licences respected.
- No personal data.
- Negative results are reported.
- The analysis plan for E1, E2 and E5 is frozen before the final runs.
