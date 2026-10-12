"""GHCN-Daily precipitation (NOAA; public) -- E7(a): per-station gamma mixtures of
wet-day rainfall amounts, thousands of small datasets (the "why amortise" case).

Station files use the fixed-width `.dly` format:
    ID(11) YEAR(4) MONTH(2) ELEMENT(4) then 31 x [VALUE(5) MFLAG(1) QFLAG(1) SFLAG(1)]
PRCP values are tenths of mm; -9999 = missing; a non-blank QFLAG = failed QC (dropped).

Download (explicit, never automatic):
    download_station("USW00094728", "data/ghcn")      # one station
or bulk: https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd_all.tar.gz
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station}.dly"


def parse_dly(path: str | Path, element: str = "PRCP") -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            if line[17:21] != element:
                continue
            year, month = int(line[11:15]), int(line[15:17])
            for day in range(31):
                off = 21 + 8 * day
                field = line[off:off + 5]
                if not field.strip():
                    continue
                val = int(field)
                qflag = line[off + 6:off + 7]
                if val == -9999 or qflag.strip():
                    continue
                rows.append((year, month, day + 1, val))
    df = pd.DataFrame(rows, columns=["year", "month", "day", "value"])
    df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
    return df.dropna(subset=["date"])


def wet_day_amounts(df: pd.DataFrame, threshold_mm: float = 1.0, months: list[int] | None = None,
                    years: tuple[int, int] | None = None) -> np.ndarray:
    """Wet-day precipitation amounts in mm (PRCP is stored in tenths of mm)."""
    d = df
    if months is not None:
        d = d[d["month"].isin(months)]
    if years is not None:
        d = d[(d["year"] >= years[0]) & (d["year"] <= years[1])]
    mm = d["value"].to_numpy(dtype=float) / 10.0
    return mm[mm >= threshold_mm]


def load_stations(directory: str | Path, min_wet_days: int = 200, threshold_mm: float = 1.0,
                  months: list[int] | None = None, limit: int | None = None) -> dict[str, np.ndarray]:
    """{station_id: wet-day amounts (n, 1)} for every .dly file with enough wet days."""
    out = {}
    for p in sorted(Path(directory).glob("*.dly")):
        x = wet_day_amounts(parse_dly(p), threshold_mm, months)
        if len(x) >= min_wet_days:
            out[p.stem] = x[:, None]
        if limit and len(out) >= limit:
            break
    return out


def download_station(station: str, dest: str | Path) -> Path:
    import urllib.request

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / f"{station}.dly"
    urllib.request.urlretrieve(BASE_URL.format(station=station), target)
    return target
