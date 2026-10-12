"""Sentinel-1 SAR intensity patches (optional E7(d)). Multilook SAR intensity is
gamma-distributed (speckle), so per-patch gamma mixtures segment a scene -- up to
millions of small datasets, the extreme amortisation regime."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def patches_from_array(img: np.ndarray, patch: int = 32, stride: int = 32, min_valid: float = 0.9):
    """Split a 2-D intensity image into patches.
    Returns (list of (m, 1) arrays, list of (row, col) positions)."""
    H, W = img.shape
    out, pos = [], []
    for r in range(0, H - patch + 1, stride):
        for c in range(0, W - patch + 1, stride):
            p = img[r:r + patch, c:c + patch].ravel()
            p = p[np.isfinite(p) & (p > 0)]
            if p.size >= min_valid * patch * patch:
                out.append(p[:, None].astype(float))
                pos.append((r, c))
    return out, pos


def load_geotiff(path: str | Path, band: int = 1) -> np.ndarray:  # pragma: no cover - optional dependency
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError("load_geotiff needs `pip install rasterio`") from exc
    with rasterio.open(path) as src:
        return src.read(band).astype(float)
