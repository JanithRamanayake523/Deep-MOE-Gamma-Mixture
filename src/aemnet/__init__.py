"""AEM-Net: Amortized Expectation-Maximization Networks -- symmetry-aware amortized
inference for mixture models beyond closed-form M-steps.

Subpackages
    simulation   component families, simulator, parameter bank, base-paper cases
    models       AEM-Net, neural baselines, exact-EM block (Thm 1), flow posterior head
    losses       quotient (Hungarian) loss and the E3 variants
    training     trainer (pure-L_Q final phase)
    inference    amortized estimator, EM refinement (Thm 4), real-data adaptation
    baselines    EM / batched EM / MLE / MCMC / NUTS / ABC-SMC / WH-MCLUST / neural / external
    evaluation   metrics, relabelling, calibration, diagnostics, paired benchmark, statistics
    theory       numerical checks of the theory (contraction, symmetry)
    data         real-data loaders (GHCN, CAS insurance, Allen ISIs, Sentinel-1, Madawaska)
"""

__version__ = "0.3.0"
