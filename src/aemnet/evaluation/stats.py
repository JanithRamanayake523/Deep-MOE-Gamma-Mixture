"""Paired statistical analysis of benchmark results (RESEARCH_METHODOLOGY.md 6.5):
Wilcoxon signed-rank tests per cell/metric with Holm correction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def wilcoxon_table(runs: pd.DataFrame, metric: str, reference: str = "aem") -> pd.DataFrame:
    """Paired Wilcoxon signed-rank of `metric` between `reference` and each
    other method, matched on run/cell identity.

    `runs` is the long table from a benchmark runner, with a method column and
    one row per (replication, method). Returns one row per comparison with the
    raw and Holm-adjusted p-value and the median ratio (reference / other).
    """
    key_cols = [c for c in ("g", "k", "n", "overlap", "shape_regime", "rep", "run")
                if c in runs.columns]
    ref = runs[(runs["method"] == reference) & runs["ok"]][key_cols + [metric]]
    rows = []
    for other in sorted(set(runs["method"]) - {reference}):
        oth = runs[(runs["method"] == other) & runs["ok"]][key_cols + [metric]]
        paired = ref.merge(oth, on=key_cols, suffixes=("_ref", "_oth")).dropna()
        if len(paired) < 10:
            continue
        d = paired[f"{metric}_ref"] - paired[f"{metric}_oth"]
        if np.allclose(d, 0):
            p, stat = 1.0, 0.0
        else:
            stat, p = wilcoxon(d)
        ratio = float((paired[f"{metric}_ref"] / paired[f"{metric}_oth"].replace(0, np.nan)).median())
        rows.append(dict(metric=metric, reference=reference, other=other, n_paired=len(paired),
                         median_ratio=round(ratio, 4), statistic=float(stat), p_value=float(p)))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm_correction(out["p_value"].tolist())
    return out


def holm_correction(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running_max = max(running_max, val)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()
