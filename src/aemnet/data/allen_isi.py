"""Neural inter-spike intervals (E7(c)): per-neuron gamma / Weibull mixtures
(e.g. bursting vs tonic firing), from the public Allen Brain Observatory.

Two entry points:
  * load_units_npz(path): an .npz holding {unit_id: spike_times in seconds}, produced
    once by any exporter (recommended: dump with the AllenSDK and keep it offline);
  * load_from_allensdk(session_id, cache_dir): requires `pip install allensdk`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def isis_from_spike_times(times: np.ndarray, min_isi: float = 1e-4, max_isi: float | None = None) -> np.ndarray:
    isi = np.diff(np.sort(np.asarray(times, dtype=float)))
    isi = isi[isi >= min_isi]
    if max_isi is not None:
        isi = isi[isi <= max_isi]
    return isi


def load_units_npz(path: str | Path, min_spikes: int = 500, **isi_kw) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    out = {}
    for key in data.files:
        t = data[key]
        if t.size >= min_spikes:
            out[key] = isis_from_spike_times(t, **isi_kw)[:, None]
    return out


def load_from_allensdk(session_id: int, cache_dir: str | Path, min_spikes: int = 500,
                       **isi_kw) -> dict[str, np.ndarray]:  # pragma: no cover - optional dependency
    try:
        from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
    except ImportError as exc:
        raise ImportError("load_from_allensdk needs `pip install allensdk`") from exc
    cache = EcephysProjectCache.from_warehouse(manifest=str(Path(cache_dir) / "manifest.json"))
    session = cache.get_session_data(session_id)
    out = {}
    for unit_id, times in session.spike_times.items():
        if len(times) >= min_spikes:
            out[str(unit_id)] = isis_from_spike_times(times, **isi_kw)[:, None]
    return out
