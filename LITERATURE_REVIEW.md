# Literature Review

**Project:** *Amortized EM: Symmetry-Aware Neural Estimation for Mixtures Beyond Closed-Form M-Steps* (working title)
**Status:** search done September 2026. Every entry has an arXiv ID, DOI or venue. Entries marked **[verify]** still need a bibliographic detail (year, pages, exact ID) checked against the publisher before citing. **The student must personally read and check every reference before it goes into a manuscript.**

---

## 0. How to read this review

The project sits where five lines of work meet:

1. **Amortized inference for mixture models** (§1). This is the closest competition.
2. **Neural Bayes estimators and simulation-based inference** (§2). This is the statistical backbone.
3. **EM-structured and slot-based neural networks** (§3). This is the architectural lineage.
4. **Label switching and symmetry in estimation** (§4). This is the theoretical lineage.
5. **Classical estimation for non-Gaussian, especially gamma, mixtures** (§5). This is the application domain and the source of the strongest baselines.

§6 states the gap. §7 lists threats to novelty and how to handle each.

---

## 1. Amortized inference for mixture models (closest competitors)

| # | Work | Method | Target | Families | Label switching | Limitation we exploit |
|---|---|---|---|---|---|---|
| 1.1 | **Set Transformer**: Lee, Lee, Kim, Kosiorek, Choi, Teh. ICML 2019. arXiv:1810.00825 | ISAB/PMA attention; the "amortized clustering" experiment outputs MoG parameters from k seed vectors | Point estimate (amortized maximum likelihood) | Gaussian | NLL loss (label-invariant) | Gaussian only; no Bayes interpretation; no uncertainty quantification; no theory |
| 1.2 | **Deep Amortized Clustering (DAC)**: Lee, Lee, Teh. arXiv:1909.13433 (ICLR 2020 submission) | Iterative "filtering" of one cluster at a time | Assignments and count | Gaussian and learned | Sequential, so the issue does not arise | Assignments, not calibrated parameters |
| 1.3 | **Neural Clustering Processes**: Pakman, Wang, Mitelut, Lee, Paninski. ICML 2020 (PMLR 119). arXiv:1901.00409 | Sequential amortized posterior over partitions | Posterior over assignments | Gaussian, MFM | Exchangeable sequential construction | No component parameters; cost grows with n |
| 1.4 | **Transformers versus the EM Algorithm in Multi-class Clustering**: He, Chen, Cao, Fan, Liu. arXiv:2502.06007 (2025) | Theory: softmax attention approximates E and M steps; transformers reach minimax rates given enough pre-training | Clustering | Gaussian (isotropic) | Not a focus | GMM only; theory-first; no non-Gaussian or intractable M-step |
| 1.5 | **TGMM, "Transformers as Unsupervised Learning Algorithms: A Study on Gaussian Mixtures"**: Chen, Wu, Fang. arXiv:2505.11918 (2025). Code: github.com/Rorschach1989/transformer-for-gmm | One shared transformer solves many GMM tasks; proves approximation of EM and cubic tensor power iteration | Parameters | Gaussian | [verify in paper] | **Most direct competitor.** Gaussian only; no label-symmetry theory; no hybrid refinement; no UQ |
| 1.6 | **Cluster-PFN, "Transformers can do Bayesian Clustering"**: Bhaskaran, Viering. arXiv:2510.24318 (2025) | Prior-data fitted network trained on a finite-GMM prior | Posterior over K and assignments | Gaussian (handles missing data) | Via assignment posterior | No component parameters; GMM only |
| 1.7 | **Amortized Bayesian Mixture Models (ABMM)**: Kucharský, Bürkner. *Statistics and Computing* (2026), doi:10.1007/s11222-026-10911-y; arXiv:2501.10229 | Summary net + spline-coupling flow for parameters + classifier for memberships | Full posterior | GMM, Gaussian HMM, evidence-accumulation mixture | **Order constraints enforced by rejection sampling** (e.g. μ₁<μ₂<μ₃) | Fixed K; ordering fails in multivariate or overlapping settings and for families with no natural order; no EM structure |
| 1.8 | **Can Transformers Learn Full Bayesian Inference in Context?**: Reuter, Rudner, Fortuin, Rügamer. ICML 2025. arXiv:2501.16825 | PFN + continuous normalizing flow / flow matching | Full posterior | GLMs, factor models, GMM | Not mixture-specific | Generic architecture; mixtures are a side case |
| 1.9 | **Neural-g**: arXiv:2406.05986 (2024) | MLP estimates a *mixing density* (NPMLE-style) | Mixing distribution | Empirical-Bayes mixtures | n/a | Different problem (nonparametric mixing density), but must be cited to pre-empt reviewer confusion |

**Takeaway.** Every amortized mixture method with parameter outputs is Gaussian (1.1, 1.4, 1.5, 1.7, 1.8), or it outputs assignments only (1.2, 1.3, 1.6). None handles mixtures whose **M-step has no closed form** (gamma, Weibull, Student-t) or whose **likelihood is intractable** (dependent multivariate gamma, censored data). None says **which estimator the network converges to** under label symmetry.

---

## 2. Neural Bayes estimators (NBE) and simulation-based inference (SBI)

- **Sainsbury-Dale, Zammit-Mangion, Huser (2024).** *Likelihood-free parameter estimation with neural Bayes estimators.* *The American Statistician* 78(1) **[verify venue: often cited as TAS 2024]**. arXiv:2208.12942.
  - Shows that a network trained to minimize empirical Bayes risk over simulated (θ, D) pairs approximates the Bayes estimator.
  - Uses DeepSets for i.i.d. replicates. Software: `NeuralEstimators.jl` and R.
  - **This is our starting point:** Thm 3 extends it to the quotient space of mixture parameters.
- **Zammit-Mangion, Sainsbury-Dale, Huser (2025).** *Neural methods for amortized inference.* *Annual Review of Statistics and Its Application* 12:311–335. doi:10.1146/annurev-statistics-112723-034123. The standard review; use it for framing.
- **Radev, Mertens, Voss, Ardizzone, Köthe (2020).** *BayesFlow.* IEEE TNNLS. arXiv:2003.06281. **BayesFlow 2** (2026), arXiv:2602.07098. Software for the ABMM and NPE baselines.
- **Papamakarios & Murray (2016)** NPE (arXiv:1605.06376); **Greenberg, Nonnenmacher, Macke (2019)** APT (arXiv:1905.07488). Neural posterior estimation.
- **Lipman, Chen, Ben-Hamu, Nickel, Le (2023).** *Flow Matching for Generative Modeling.* ICLR. arXiv:2210.02747. **Wildberger et al. (2023)**, *Flow matching for scalable SBI*, NeurIPS, arXiv:2305.17161. Basis for our posterior head.
- **Lueckmann et al. (2021).** *Benchmarking SBI (sbibm).* AISTATS. arXiv:2101.04653. **Tejero-Cantero et al. (2020)**, `sbi`, JOSS. Benchmarking conventions to follow.
- **Talts et al. (2018).** *Simulation-Based Calibration.* arXiv:1804.06788. **Hermans et al. (2022)**, *A trust crisis in SBI? Towards conservative inference*, TMLR, arXiv:2110.06581. Calibration protocol for E5.
- **Müller et al. (2022).** *Transformers can do Bayesian inference (PFNs).* ICLR. arXiv:2112.10510. **Bai et al. (2023)**, *Transformers as statisticians*, NeurIPS, arXiv:2306.04637. In-context estimation lineage.
- **Robustness and misspecification:**
  - Ward, Cannon, Beaumont, Fasiolo, Schmon (2022). *Robust NPE and statistical model criticism.* NeurIPS. [verify arXiv ID]
  - Huang et al. (2023). *Learning robust statistics for SBI under misspecification.* NeurIPS.
  - Mishra, Habermann, Schmitt, Radev, Bürkner (2025). *Robust amortized Bayesian inference with self-consistency losses on unlabeled data.* arXiv:2501.13483. **We use this for real-data adaptation (E6).**
  - Schmitt et al. (2023). *Leveraging self-consistency for data-efficient ABI.* arXiv:2310.04395.
  - *Amortized SBI in generalized Bayes via NPE* (2026). arXiv:2601.22367.
- **Amortization gap:**
  - Cremer, Li, Duvenaud (2018). *Inference suboptimality in VAEs.* ICML. arXiv:1801.03558.
  - Marino, Yue, Mandt (2018). *Iterative amortized inference.* ICML. arXiv:1807.09356.
  - Motivates our hybrid "amortize, then refine" stage (C3, Thm 4).

---

## 3. EM-structured, slot-based and unrolled networks (architectural lineage)

- **Jacobs, Jordan, Nowlan, Hinton (1991).** *Adaptive mixtures of local experts.* Neural Computation 3:79–87. **Jordan & Jacobs (1994).** *Hierarchical mixtures of experts and the EM algorithm.* Neural Computation 6:181–214.
  - The gate is a responsibility estimator. Our architecture makes this literal: responsibilities route data to experts.
- **Greff, van Steenkiste, Schmidhuber (2017).** *Neural Expectation Maximization.* NeurIPS. arXiv:1708.03498. EM-style perceptual grouping. It learns the generative model, not estimators of mixture parameters.
- **Locatello et al. (2020).** *Object-centric learning with Slot Attention.* NeurIPS. arXiv:2006.15055.
  - Iterative soft assignment plus a GRU slot update. It reduces to soft k-means.
  - **Our E/M blocks are slot attention with (i) likelihood-based rather than dot-product assignment, (ii) pooling of sufficient statistics, and (iii) statistically supervised outputs.** This must be said openly in the paper.
- **Kori et al. (2024).** *Identifiable object-centric representation learning via Probabilistic Slot Attention.* NeurIPS. arXiv:2406.07141. Slot attention viewed as EM for a GMM, with identifiability up to permutation and affine maps. Closest *architectural* relative; its goal is representation learning, not parameter estimation.
- **Zhang, Hare, Prügel-Bennett (2019).** *Deep Set Prediction Networks.* NeurIPS. arXiv:1906.06565. Iterative set prediction. Also: *Attention normalization impacts cardinality generalization in slot attention* (2024), arXiv:2407.04170, which is relevant to generalizing to unseen g.
- **Deep unfolding:** Gregor & LeCun (2010), *LISTA*, ICML; Hershey, Le Roux, Weninger (2014), *Deep unfolding*, arXiv:1409.2574. Unrolling iterative algorithms into trainable layers.
  - **Novelty caveat:** "unrolled EM" by itself is **not** new. Our novelty is the combination of (a) exact EM containment for exponential families, (b) learned M-steps where no closed form exists, (c) training by amortized Bayes risk, and (d) the symmetry theory.
- **Zaheer et al. (2017)**, *Deep Sets*, NeurIPS, arXiv:1703.06114. **Wagstaff et al. (2019)**, *On the limitations of representing functions on sets*, ICML. Set-function expressivity.

---

## 4. Label switching and symmetric estimation (theoretical lineage)

- **Teicher (1963).** *Identifiability of finite mixtures.* Ann. Math. Statist. 34:1265–1269. Finite gamma mixtures are identifiable up to permutation.
- **Stephens (2000).** *Dealing with label switching in mixture models.* JRSS-B 62(4):795–809. doi:10.1111/1467-9868.00265.
  - Relabelling as minimizing posterior expected loss under label-invariant losses.
  - **Thm 3 connects the quotient-loss neural Bayes estimator to Stephens' relabelling fixed point.**
- **Celeux, Hurn, Robert (2000).** *Computational and inferential difficulties with mixture posterior distributions.* JASA 95:957–970. Loss-based point estimation for mixtures.
- **Jasra, Holmes, Stephens (2005).** *MCMC methods and the label switching problem.* Statistical Science 20(1):50–67. doi:10.1214/088342305000000016.
- **Papastamoulis (2016).** `label.switching` R package, J. Stat. Softw. 69. **Egidi et al. (2018 / 2021)**, `pivmet` pivotal relabelling, arXiv:2103.16948. Pivot-based relabelling, which our posterior canonicalization (Prop 2) formalizes.
- **Monteiller et al. (2019).** *Alleviating label switching with optimal transport.* NeurIPS. arXiv:1911.02053. Relabelling via Wasserstein barycenters. Closely related to our quotient Fréchet mean, which is a barycenter on Θ/S_g.
- **Carion et al. (2020).** DETR. ECCV. arXiv:2005.12872. Hungarian set loss, used in deep learning without statistical analysis.
- **Kaba et al. (2023).** *Equivariance with learned canonicalization functions.* ICML. arXiv:2211.06489. **Dym, Lawrence, Siegel (2024).** *Equivariant frames and the impossibility of continuous canonicalization.* ICML. [verify arXiv 2402.16077]. Any canonicalization, including sorting and relabelling, must be discontinuous somewhere. Prop 2 accounts for this.

---

## 5. Non-Gaussian (gamma) mixture estimation: classical methods and strongest baselines

- **Dempster, Laird, Rubin (1977)** EM, JRSS-B. **Wu (1983)**, convergence of EM, Ann. Statist. **Redner & Walker (1984)**, SIAM Review 26.
- **Balakrishnan, Wainwright, Yu (2017).** *Statistical guarantees for the EM algorithm: from population to sample-based analysis.* Ann. Statist. 45(1). arXiv:1408.2156. Local contraction basins, used in Thm 4.
- **Banerjee, Merugu, Dhillon, Ghosh (2005).** *Clustering with Bregman divergences.* JMLR 6:1705–1749. Regular exponential-family mixtures correspond one-to-one with Bregman soft clustering, so the M-step is a responsibility-weighted mean of sufficient statistics. **This is the basis of Thm 1.**
- **Young, Chen, Hewage, Nilo-Poyanco (2019).** *Finite mixture-of-gamma distributions: estimation, inference, and model-based clustering.* ADAC 13. doi:10.1007/s11634-019-00361-y. ECM with a Wilson–Hilferty starting-value strategy; `mixtools::gammamixEM`. **Strong classical baseline.**
- **Wiper, Ríos Insua, Ruggeri (2001).** *Mixtures of gamma distributions with applications.* JCGS 10(3). Bayesian MCMC for gamma mixtures.
- **Vaidyanathan & Vani Lakshmi (2017).** Sri Lankan J. Appl. Stat. 17(3), doi:10.4038/sljastats.v17i3.7902. WH + MCLUST. This was the *original base paper*; it is now a minor baseline for the gamma family only.
- **Wilson & Hilferty (1931)**, PNAS 17. **Krishnamoorthy, Mathew, Mukherjee (2008)**, Technometrics 50(1). Cube-root normal approximation, used as input features.
- **Multivariate and bivariate gamma families:**
  - McKay (1934); Kibble (1941) and Moran (1969) (density involves a modified Bessel function); Cheriyan (1941).
  - Kotz, Balakrishnan, Johnson (2000), *Continuous Multivariate Distributions*, ch. 48.
  - Jones, Lai, Rayner (2000), *A bivariate gamma mixture distribution*, Commun. Stat. Theory Methods 29(12).
  - Mathai & Moschopoulos (1991), *On a multivariate gamma*, J. Multivariate Anal. 39.
- **Applications:**
  - *Bivariate gamma mixture-of-experts models for joint insurance claims*, arXiv:1904.04699 (EM for bivariate gamma).
  - Liu et al. (2019), variational EM for generalized-gamma mixtures (SAR imagery), Pattern Recognition 88 [verify].
  - Rainfall: daily precipitation amounts are classically modelled with gamma and mixed-gamma distributions (e.g. Wilks 1998, J. Hydrology) [verify].
  - SAR speckle: multilook intensity is gamma-distributed (Goodman / Oliver & Quegan) [verify textbook reference].

---

## 6. Gap statement

Existing amortized mixture estimators:

- **(G1)** are limited to **Gaussian** components (§1). None addresses families where the **M-step has no closed form** (gamma: inverse digamma; Weibull; Student-t) or the **likelihood is intractable** (copula- or Kibble-dependent multivariate gamma, censored lifetimes).
- **(G2)** handle label symmetry with **ordering constraints** (ABMM), **unanalysed Hungarian losses** (DETR-style), or **NLL**. None characterizes **which estimator** the trained network converges to.
- **(G3)** are purely amortized, so they **lose to MLE at large n**. No method combines Bayes-like small-sample accuracy with MLE efficiency at large n, with guarantees.
- **(G4)** are rarely deployed where amortization is essential: **very many small datasets** (per station, per patch, per neuron).

**Our contribution closes G1–G4 with one architecture (AEM-Net), four theoretical results, and a benchmark that spans families.**

---

## 7. Threats to novelty and mitigations

| Threat | Reviewer's words | Mitigation |
|---|---|---|
| TGMM / He et al. already show transformers ≈ EM | "Incremental over TGMM" | Our setting is non-Gaussian and intractable-likelihood; we contain EM *exactly* for exponential families (not only approximately); we add symmetry theory and a hybrid with an efficiency guarantee. Run TGMM on Gaussian data and **match or beat it** there. |
| Slot attention / PSA already do EM-style slots | "Architecture is slot attention" | Say so explicitly. Novelty lies in the statistical supervision, the likelihood-aware E-block, the learned inverse link in the M-block, and the theory. The ablation (E8) shows each ingredient matters. |
| ABMM already does amortized Bayesian mixtures | "Just use BayesFlow" | Show ABMM's ordering constraint failing (multivariate, overlapping, g ≥ 3), and ABMM's lack of g-estimation and non-Gaussian M-steps. |
| "Why not EM?" | "Likelihood is tractable" | E2 (the n-regime), E4 (intractable families), E7 (millions of datasets), plus the hybrid, which *includes* EM. |
| Unrolled EM is known | "Deep unfolding" | Cite Hershey 2014 and LISTA; novelty is the combination and the theory, not unrolling itself. |
