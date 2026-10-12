"""Loaders for real datasets (E7). No data are redistributed."""
from .madawaska import load_madawaska
from .ghcn import download_station, load_stations, parse_dly, wet_day_amounts
from .cas_insurance import load_bivariate_csv, load_severity_csv
from .allen_isi import isis_from_spike_times, load_from_allensdk, load_units_npz
from .sentinel_sar import load_geotiff, patches_from_array

__all__ = [
    "load_madawaska", "download_station", "load_stations", "parse_dly", "wet_day_amounts",
    "load_bivariate_csv", "load_severity_csv", "isis_from_spike_times", "load_from_allensdk",
    "load_units_npz", "load_geotiff", "patches_from_array",
]
