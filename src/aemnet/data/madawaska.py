"""Madawaska basin flood data (the base paper's real dataset; E9 appendix / continuity check).

Source: Yue (2001), *Hydrological Processes* 15(6):1033-1045,
DOI: 10.1002/hyp.259 -- 77 years (1919-1995) of streamflow measurements from
the Madawaska basin, Quebec, Canada. Variables: flood duration (days) and
flood peak (m^3/s), reported independent (rho = -0.0118) with gamma marginals;
a single bivariate gamma component is expected (base paper, Table 6:
alpha ~ (9.6, 13), beta ~ (8.4, 19.7)).

The data are NOT redistributed in this repository. Place a CSV at
`data/madawaska.csv` with a header row and columns `duration,peak`, one row
per year, e.g.:

    duration,peak
    10.0,140.2
    ...
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_COLS = ("duration", "peak")


def load_madawaska(path: str | Path = "data/madawaska.csv") -> np.ndarray:
    """Returns X : (77, 2) with columns [duration, peak]."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Madawaska data not found at {path}. It is not redistributed with "
            "this repo; transcribe it from Yue (2001), Table therein, into a "
            "CSV with columns duration,peak. See this module's docstring."
        )
    df = pd.read_csv(path)
    missing = set(EXPECTED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns {missing}; expected {EXPECTED_COLS}")
    X = df[list(EXPECTED_COLS)].to_numpy(dtype=np.float64)
    if (X <= 0).any():
        raise ValueError("gamma data must be strictly positive")
    return X
