"""Insurance claim severities (E7(b)) from the R package CASdatasets.

The data are not redistributed. Export once from R, for example:

    library(CASdatasets); data(freMTPL2sev); data(freMTPL2freq)
    sev <- merge(freMTPL2sev, freMTPL2freq[, c("IDpol", "Region")], by = "IDpol")
    write.csv(sev, "data/freMTPL2sev.csv", row.names = FALSE)

then load portfolio segments as separate datasets (one mixture fit per segment).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_severity_csv(path: str | Path, amount_col: str = "ClaimAmount", group_col: str | None = "Region",
                      min_size: int = 50, max_quantile: float | None = None) -> dict[str, np.ndarray]:
    df = pd.read_csv(path)
    df = df[df[amount_col] > 0]
    if max_quantile is not None:
        df = df[df[amount_col] <= df[amount_col].quantile(max_quantile)]
    if group_col is None:
        return {"all": df[amount_col].to_numpy(float)[:, None]}
    out = {}
    for key, sub in df.groupby(group_col):
        if len(sub) >= min_size:
            out[str(key)] = sub[amount_col].to_numpy(float)[:, None]
    return out


def load_bivariate_csv(path: str | Path, cols: tuple[str, str], group_col: str | None = None,
                       min_size: int = 50) -> dict[str, np.ndarray]:
    """Bivariate positive claims (e.g. loss and ALAE) for copula / Kibble gamma mixtures."""
    df = pd.read_csv(path)
    df = df[(df[cols[0]] > 0) & (df[cols[1]] > 0)]
    groups = [("all", df)] if group_col is None else list(df.groupby(group_col))
    return {str(k): s[list(cols)].to_numpy(float) for k, s in groups if len(s) >= min_size}
