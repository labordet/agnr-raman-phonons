"""Read the two recorded configuration-III acquisition selection stages."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REJECTED_MEAN_TEMPERATURES_K = (125.0, 145.0, 185.0, 195.0)


def acquisition_selection(data_root: Path) -> pd.DataFrame:
    """Return released before-averaging column decisions for configuration III.

    Decisions are recorded human selections; no SNR threshold is inferred.
    """
    path = Path(data_root) / "metadata/spectrum_selection.csv"
    rows = pd.read_csv(path)
    return rows.loc[rows["family"] == "MIRA_Au_unaligned_8A"].copy()


def accepted_mean_temperatures(data_root: Path) -> list[float]:
    """Read accepted III RBLM/D/G temperatures from the final peak table."""
    table = pd.read_csv(Path(data_root) / "data/derived/peak_parameters.csv")
    rows = table.loc[table["family"] == "MIRA_Au_unaligned_8A"]
    temperatures = sorted(float(x) for x in rows["temperature_K"].unique())
    if set(REJECTED_MEAN_TEMPERATURES_K) & set(temperatures):
        raise ValueError("A rejected configuration-III mean entered accepted peaks")
    return temperatures
