"""Evaluation. benchmark and diagnostics are imported explicitly (they pull in baselines)."""
from .metrics import adjusted_rand_index, evaluate_fit, heldout_loglik, quotient_alignment, rmse_by_param
from .relabel import align_to_pivot, quotient_frechet_mean, stephens_relabel
from .calibration import c2st, coverage, expected_coverage_curve, sbc_ranks, sbc_uniformity
from .stats import holm_correction, wilcoxon_table

__all__ = [
    "adjusted_rand_index", "evaluate_fit", "heldout_loglik", "quotient_alignment", "rmse_by_param",
    "align_to_pivot", "quotient_frechet_mean", "stephens_relabel",
    "c2st", "coverage", "expected_coverage_curve", "sbc_ranks", "sbc_uniformity",
    "holm_correction", "wilcoxon_table",
]
