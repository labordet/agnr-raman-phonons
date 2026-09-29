#!/usr/bin/env python3
"""Lorentzian-only Raman deconvolution for GOOD_SPECTRA TXT folders.

This script is designed for the Euler cluster workflow:

* Read the peak bounds from ``raman_peak_bounds_settings.json``.
* Treat every input TXT file as read-only.
* Use column 1 as Raman shift X and fit every remaining Y column separately.
* Fit only sums of Lorentzians:

      h / (1 + ((x - pos) / (0.5 * width)) ** 2)

* Write incremental CSV and HTML results every time a full TXT file is done.
* Build one XLSX workbook per family at the end.

The HTML is intentionally self-contained with embedded PNG previews so the
cluster run does not create thousands of separate image files.
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import hashlib
import html
import io
import json
import math
import os
import re
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares


TARGET_SAMPLE_FOLDERS = [
    "Aligned_Au_3A",
    "Aligned_Au_8A",
    "Aligned_RO_8A",
    "MIRA_Au_unaligned_8A",
    "MIRA_RO_unaligned_8A",
]

MODEL_NAME = "sum_of_lorentzians_no_baseline"
SCRIPT_VERSION = "2026-07-05-aligned-ro-passive-ch-mid-v1"
TEMP_RE = re.compile(r"(?<![A-Za-z0-9])([0-9]+(?:[.,p][0-9]+)?)\s*K", re.IGNORECASE)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def single_lorentzian(x: np.ndarray, h: float, pos: float, width: float) -> np.ndarray:
    return h / (1 + ((x - pos) / (0.5 * width)) ** 2)


def multiple_lorentzian(x: np.ndarray, *params: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 3):
        h = params[i]
        pos = params[i + 1]
        width = params[i + 2]
        y += single_lorentzian(x, h, pos, width)
    return y


@dataclass(frozen=True)
class PeakSpec:
    peak_id: str
    label: str
    required: bool
    center: float
    x_min: float
    x_max: float
    width_min: float
    width_max: float
    width_guess: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit every TXT spectrum column with Lorentzian peaks from JSON bounds."
    )
    parser.add_argument("--root", default=".", help="Folder containing family folders.")
    parser.add_argument(
        "--settings",
        default="raman_peak_bounds_settings.json",
        help="Bounds JSON produced by the Raman peak bounds tuner.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output folder. Default uses $SCRATCH or /cluster/scratch/$USER when available.",
    )
    parser.add_argument(
        "--families",
        nargs="*",
        default=None,
        help="Optional subset of family folder names to process.",
    )
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("RAMAN_N_JOBS", "1")))
    parser.add_argument("--bootstrap-runs", type=int, default=100)
    parser.add_argument("--bootstrap-jitter", type=float, default=0.03)
    parser.add_argument("--de-maxiter", type=int, default=100000)
    parser.add_argument("--de-popsize", type=int, default=200)
    parser.add_argument("--de-tol", type=float, default=1e-7)
    parser.add_argument("--ls-max-nfev", type=int, default=20000)
    parser.add_argument("--fit-padding", type=float, default=8.0)
    parser.add_argument("--fit-x-min", type=float, default=200.0)
    parser.add_argument("--fit-x-max", type=float, default=2000.0)
    peak_weight_group = parser.add_mutually_exclusive_group()
    peak_weight_group.add_argument(
        "--peak-window-weighting",
        dest="peak_window_weighting",
        action="store_true",
        default=env_bool("RAMAN_PEAK_WINDOW_WEIGHTING", True),
        help="Upweight residuals for X points inside enabled peak windows from the JSON bounds.",
    )
    peak_weight_group.add_argument(
        "--no-peak-window-weighting",
        dest="peak_window_weighting",
        action="store_false",
        help="Disable peak-window residual weighting.",
    )
    parser.add_argument(
        "--peak-window-weight",
        type=float,
        default=float(os.environ.get("RAMAN_PEAK_WINDOW_WEIGHT", "6.0")),
        help="Squared-error weight for points inside any enabled peak X window.",
    )
    parser.add_argument(
        "--peak-window-margin",
        type=float,
        default=float(os.environ.get("RAMAN_PEAK_WINDOW_MARGIN", "0.0")),
        help="Optional cm-1 margin added to each side of peak X windows for weighting only.",
    )
    peak_height_group = parser.add_mutually_exclusive_group()
    peak_height_group.add_argument(
        "--peak-window-height-weighting",
        dest="peak_window_height_weighting",
        action="store_true",
        default=env_bool("RAMAN_PEAK_WINDOW_HEIGHT_WEIGHTING", True),
        help="Within each peak window, gently upweight taller points relative to local window noise.",
    )
    peak_height_group.add_argument(
        "--no-peak-window-height-weighting",
        dest="peak_window_height_weighting",
        action="store_false",
        help="Disable local height weighting inside peak windows.",
    )
    parser.add_argument(
        "--peak-window-height-weight",
        type=float,
        default=float(os.environ.get("RAMAN_PEAK_WINDOW_HEIGHT_WEIGHT", "2.0")),
        help="Maximum multiplier applied to the tallest points inside a peak window.",
    )
    parser.add_argument(
        "--peak-window-height-floor-percentile",
        type=float,
        default=float(os.environ.get("RAMAN_PEAK_WINDOW_HEIGHT_FLOOR_PERCENTILE", "10.0")),
        help="Local percentile used as the zero-height floor for peak-window height weighting.",
    )
    parser.add_argument(
        "--height-percentile-weighting",
        action="store_true",
        default=env_bool("RAMAN_HEIGHT_PERCENTILE_WEIGHTING", False),
        help="Optional diagnostic mode: also upweight high-Y points above a percentile. Disabled by default.",
    )
    parser.add_argument(
        "--height-weight-percentile",
        type=float,
        default=float(os.environ.get("RAMAN_HEIGHT_WEIGHT_PERCENTILE", "75.0")),
        help="Percentile threshold used only with --height-percentile-weighting.",
    )
    parser.add_argument(
        "--height-weight",
        type=float,
        default=float(os.environ.get("RAMAN_HEIGHT_WEIGHT", "2.5")),
        help="Squared-error weight for high-Y points used only with --height-percentile-weighting.",
    )
    parser.add_argument(
        "--adaptive-bounds",
        dest="adaptive_bounds",
        action="store_true",
        default=True,
        help="Refit with expanded bounds when fitted position/width/height parameters hit a bound.",
    )
    parser.add_argument(
        "--no-adaptive-bounds",
        dest="adaptive_bounds",
        action="store_false",
        help="Disable adaptive bound expansion and use JSON bounds exactly.",
    )
    parser.add_argument("--adaptive-bound-max-iterations", type=int, default=4)
    parser.add_argument("--adaptive-bound-tol", type=float, default=0.05)
    parser.add_argument("--adaptive-width-expand-factor", type=float, default=1.35)
    parser.add_argument("--adaptive-position-expand-cm", type=float, default=3.0)
    parser.add_argument("--adaptive-height-expand-factor", type=float, default=1.5)
    parser.add_argument("--adaptive-height-upper-rel-tol", type=float, default=0.01)
    parser.add_argument("--plot-dpi", type=int, default=135)
    parser.add_argument("--plot-width", type=float, default=11.0)
    parser.add_argument("--plot-height", type=float, default=5.0)
    parser.add_argument(
        "--anti-burial-penalty",
        type=float,
        default=120.0,
        help="Penalty strength that keeps required peaks visible instead of buried under neighbors.",
    )
    parser.add_argument(
        "--required-min-visibility",
        type=float,
        default=0.18,
        help="Minimum component fraction at its own center for required peaks.",
    )
    parser.add_argument(
        "--ch-left-min-visibility",
        type=float,
        default=0.40,
        help="Stronger center-visibility fraction for CH_L1 and CH_L2.",
    )
    parser.add_argument(
        "--required-min-area-share",
        type=float,
        default=0.10,
        help="Minimum component area share inside its own bounds for required peaks.",
    )
    parser.add_argument(
        "--ch-left-min-area-share",
        type=float,
        default=0.30,
        help="Stronger area-share fraction for CH_L1 and CH_L2.",
    )
    parser.add_argument(
        "--anti-burial-optional",
        action="store_true",
        help="Also apply the anti-burial penalty to optional enabled peaks.",
    )
    parser.add_argument(
        "--passive-peaks",
        nargs="*",
        default=os.environ.get("RAMAN_PASSIVE_PEAKS", "CH_MID").split(),
        help="Peak IDs treated as passive filler components instead of active Raman modes.",
    )
    parser.add_argument(
        "--passive-peak-penalty",
        type=float,
        default=float(os.environ.get("RAMAN_PASSIVE_PEAK_PENALTY", "260.0")),
        help="Penalty strength keeping passive peaks broad and non-dominant.",
    )
    parser.add_argument(
        "--passive-max-height-fraction",
        type=float,
        default=float(os.environ.get("RAMAN_PASSIVE_MAX_HEIGHT_FRACTION", "0.55")),
        help="Maximum passive peak height relative to the tallest active peak.",
    )
    parser.add_argument(
        "--passive-max-area-share",
        type=float,
        default=float(os.environ.get("RAMAN_PASSIVE_MAX_AREA_SHARE", "0.38")),
        help="Maximum passive component area share inside its own window.",
    )
    parser.add_argument(
        "--passive-min-width",
        type=float,
        default=float(os.environ.get("RAMAN_PASSIVE_MIN_WIDTH", "24.0")),
        help="Soft minimum width for passive filler peaks.",
    )
    parser.add_argument(
        "--ch-left-master",
        default="CH_L2",
        choices=["CH_L1", "CH_L2", "none"],
        help="Soft identity rule for the crowded CH_L1/CH_L2 pair.",
    )
    parser.add_argument(
        "--ch-left-identity-penalty",
        type=float,
        default=60.0,
        help="Penalty strength discouraging CH_L1/CH_L2 area-ratio identity swaps.",
    )
    parser.add_argument(
        "--ch-left-slave-min-area-ratio",
        type=float,
        default=0.45,
        help="Minimum slave/master area ratio for CH_L1/CH_L2.",
    )
    parser.add_argument(
        "--ch-left-slave-max-area-ratio",
        type=float,
        default=0.90,
        help="Maximum slave/master area ratio for CH_L1/CH_L2; below 1 discourages swaps.",
    )
    parser.add_argument(
        "--ch-left-center-dominance-penalty",
        type=float,
        default=90.0,
        help="Penalty strength requiring each CH-left component to dominate at its own center.",
    )
    parser.add_argument(
        "--ch-left-own-center-share",
        type=float,
        default=0.55,
        help="Minimum share of CH_L1/CH_L2 pair intensity contributed by each component at its own center.",
    )
    parser.add_argument(
        "--ch-left-width-ratio-penalty",
        type=float,
        default=80.0,
        help="Penalty strength keeping CH_L1/CH_L2 fitted widths comparable.",
    )
    parser.add_argument(
        "--ch-left-width-ratio-min",
        type=float,
        default=0.75,
        help="Minimum CH_L1/CH_L2 width ratio after ordering by master/slave.",
    )
    parser.add_argument(
        "--ch-left-width-ratio-max",
        type=float,
        default=1.30,
        help="Maximum CH_L1/CH_L2 width ratio after ordering by master/slave.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Dry-run limit.")
    parser.add_argument("--max-spectra-per-file", type=int, default=None, help="Dry-run limit.")
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore completed-file checkpoints and refit every discovered TXT file.",
    )
    parser.add_argument(
        "--min-r2-warning",
        type=float,
        default=0.95,
        help="Flag fit quality below this R2 in HTML/CSV.",
    )
    parser.add_argument(
        "--html-max-points",
        type=int,
        default=2500,
        help="Maximum data points drawn per preview trace.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260622,
        help="Base seed for reproducible differential evolution and bootstrap jitter.",
    )
    return parser.parse_args()


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return cleaned or "item"


def default_output_root(root: Path) -> Path:
    stamp = now_stamp()
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"
    scratch_env = os.environ.get("SCRATCH")
    candidates = []
    if scratch_env:
        candidates.append(Path(scratch_env))
    if user != "unknown":
        candidates.append(Path("/cluster/scratch") / user)
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.W_OK):
            return candidate / f"raman_lorentzian_fit_{stamp}"
    return root / f"LORENTZIAN_FIT_RESULTS_{stamp}"


def parse_temperature_from_text(text: str) -> float | None:
    matches = TEMP_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ".").replace("p", "."))
    except ValueError:
        return None


def format_temp(value: float | int | str | None) -> str:
    if value is None:
        return "NA"
    try:
        f = float(str(value).replace(",", "."))
    except ValueError:
        return str(value)
    if abs(f - round(f)) < 1e-8:
        return f"{int(round(f))}K"
    return f"{f:g}K"


def numeric_temp_key(key: str) -> float:
    return float(str(key).replace(",", "."))


def load_settings(settings_path: Path) -> dict[str, Any]:
    with settings_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    families = data.get("families")
    if not isinstance(families, dict) or not families:
        raise ValueError(f"No family settings found in {settings_path}")
    return data


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_temperature_bounds(
    settings: dict[str, Any], family: str, temperature: float
) -> tuple[str, float, dict[str, Any]]:
    family_settings = settings["families"].get(family)
    if not family_settings:
        raise KeyError(f"No settings for family {family!r}")
    resolved = family_settings.get("resolved_temperature_bounds") or {}
    if not resolved:
        raise KeyError(f"No resolved temperature bounds for family {family!r}")
    key = min(resolved, key=lambda k: abs(numeric_temp_key(k) - temperature))
    bounds_temp = numeric_temp_key(key)
    return key, bounds_temp, resolved[key]


def _float_from_peak(peak: dict[str, Any], names: tuple[str, ...], default: float | None = None) -> float:
    for name in names:
        value = peak.get(name)
        if value is not None and value != "":
            return float(value)
    if default is None:
        raise KeyError(f"Missing peak value; expected one of {names}")
    return float(default)


def peak_specs_for_temperature(
    settings: dict[str, Any], family: str, temperature: float
) -> tuple[str, float, list[PeakSpec]]:
    key, bounds_temp, entry = pick_temperature_bounds(settings, family, temperature)
    raw_peaks = entry.get("peaks") or entry.get("peak_bounds") or entry
    peaks: list[PeakSpec] = []
    for peak_id, peak in raw_peaks.items():
        if not isinstance(peak, dict):
            continue
        enabled = peak.get("enabled", peak.get("on", True))
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() not in {"0", "false", "no", "n", ""}
        if not enabled:
            continue
        x_min = _float_from_peak(peak, ("x_min", "xmin", "X min", "x lower"))
        x_max = _float_from_peak(peak, ("x_max", "xmax", "X max", "x upper"))
        if x_max < x_min:
            x_min, x_max = x_max, x_min
        center = _float_from_peak(peak, ("center", "ctr", "position"), (x_min + x_max) / 2.0)
        width_min = _float_from_peak(peak, ("width_min", "w_min", "wmin", "W min"), 1.0)
        width_max = _float_from_peak(peak, ("width_max", "w_max", "wmax", "W max"), max(width_min, 50.0))
        if width_max < width_min:
            width_min, width_max = width_max, width_min
        width_guess = _float_from_peak(
            peak,
            ("width_guess", "width0", "w_guess", "w0", "W0"),
            (width_min + width_max) / 2.0,
        )
        width_guess = min(max(width_guess, width_min), width_max)
        required = peak.get("required", peak.get("req", False))
        if isinstance(required, str):
            required = required.strip().lower() in {"1", "true", "yes", "y"}
        label = str(peak.get("label") or peak.get("name") or peak_id)
        peaks.append(
            PeakSpec(
                peak_id=str(peak_id),
                label=label,
                required=bool(required),
                center=center,
                x_min=x_min,
                x_max=x_max,
                width_min=max(width_min, 1e-6),
                width_max=max(width_max, width_min + 1e-6),
                width_guess=width_guess,
            )
        )
    peaks.sort(key=lambda p: p.center)
    if not peaks:
        raise ValueError(f"No enabled peaks for {family} at {temperature:g} K")
    return key, bounds_temp, peaks


def split_line(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped:
        return []
    if "\t" in stripped:
        return [token.strip() for token in stripped.split("\t") if token.strip()]
    if "," in stripped and stripped.count(",") >= stripped.count(" "):
        return [token.strip() for token in stripped.replace(";", ",").split(",") if token.strip()]
    return [token for token in re.split(r"[\s,;]+", stripped) if token]


def load_txt_table(path: Path) -> tuple[np.ndarray, list[str]]:
    rows: list[list[float]] = []
    header: list[str] | None = None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            tokens = split_line(line)
            if not tokens:
                continue
            try:
                row = [float(token.replace(",", ".")) for token in tokens]
            except ValueError:
                if header is None:
                    header = tokens
                continue
            if len(row) >= 2 and all(math.isfinite(v) for v in row):
                rows.append(row)
    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    counts = Counter(len(row) for row in rows)
    ncols = counts.most_common(1)[0][0]
    filtered = [row[:ncols] for row in rows if len(row) >= ncols]
    data = np.asarray(filtered, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected at least X plus one Y column in {path}")
    if header is None or len(header) != ncols:
        header = [f"Column {i}" for i in range(1, ncols + 1)]
    order = np.argsort(data[:, 0])
    data = data[order]
    return data, header


def discover_txt_files(root: Path, settings: dict[str, Any], families: list[str] | None) -> list[dict[str, Any]]:
    selected = families or TARGET_SAMPLE_FOLDERS
    tasks: list[dict[str, Any]] = []
    for family in selected:
        if family not in settings.get("families", {}):
            print(f"WARNING: skipping {family}: not present in settings JSON", flush=True)
            continue
        family_root = root / family
        if not family_root.exists():
            print(f"WARNING: skipping {family}: folder not found at {family_root}", flush=True)
            continue
        for folder in sorted(family_root.rglob("*")):
            if not folder.is_dir():
                continue
            lower = folder.name.lower()
            if "spikes" not in lower or "removed" not in lower:
                continue
            if folder.name.lower() == "raw":
                continue
            for txt_path in sorted(folder.glob("*.txt")):
                rel = txt_path.relative_to(root)
                tasks.append(
                    {
                        "path": str(txt_path),
                        "rel_path": rel.as_posix(),
                        "family": family,
                        "sequence": folder.name,
                    }
                )
    tasks.sort(key=lambda t: (t["family"], t["sequence"], t["rel_path"]))
    return tasks


def stable_seed(base_seed: int, *items: object) -> int:
    h = hashlib.sha256()
    h.update(str(base_seed).encode("utf-8"))
    for item in items:
        h.update(repr(item).encode("utf-8", errors="replace"))
    return int.from_bytes(h.digest()[:4], "little", signed=False)


def choose_fit_region(x: np.ndarray, roi_min: float, roi_max: float) -> np.ndarray:
    x_min = min(roi_min, roi_max)
    x_max = max(roi_min, roi_max)
    mask = np.isfinite(x) & (x >= x_min) & (x <= x_max)
    return mask


def make_weight_vector(
    x: np.ndarray,
    y_scaled: np.ndarray,
    peaks: list[PeakSpec],
    options: dict[str, Any],
) -> np.ndarray:
    weights = np.ones_like(y_scaled, dtype=float)
    if bool(options.get("peak_window_weighting", True)):
        peak_weight = max(1.0, float(options.get("peak_window_weight", 6.0)))
        margin = max(0.0, float(options.get("peak_window_margin", 0.0)))
        use_local_height_weight = bool(options.get("peak_window_height_weighting", True))
        max_height_multiplier = max(1.0, float(options.get("peak_window_height_weight", 2.0)))
        floor_percentile = min(
            max(float(options.get("peak_window_height_floor_percentile", 10.0)), 0.0),
            95.0,
        )
        for peak in peaks:
            x_min = min(peak.x_min, peak.x_max) - margin
            x_max = max(peak.x_min, peak.x_max) + margin
            in_peak = (x >= x_min) & (x <= x_max)
            local_weight = np.full(np.count_nonzero(in_peak), peak_weight, dtype=float)
            if use_local_height_weight and local_weight.size:
                local_y = y_scaled[in_peak]
                finite_local = local_y[np.isfinite(local_y)]
                if finite_local.size >= 3:
                    y_floor = float(np.nanpercentile(finite_local, floor_percentile))
                    y_top = float(np.nanmax(finite_local))
                    if math.isfinite(y_top) and math.isfinite(y_floor) and y_top > y_floor:
                        normalized_height = np.clip((local_y - y_floor) / (y_top - y_floor), 0.0, 1.0)
                        height_multiplier = 1.0 + (max_height_multiplier - 1.0) * normalized_height
                        local_weight *= height_multiplier
            weights[in_peak] = np.maximum(weights[in_peak], local_weight)
    if bool(options.get("height_percentile_weighting", False)):
        finite_y = y_scaled[np.isfinite(y_scaled)]
        if finite_y.size:
            percentile = min(max(float(options.get("height_weight_percentile", 75.0)), 0.0), 100.0)
            high = np.percentile(finite_y, percentile)
            height_weight = max(1.0, float(options.get("height_weight", 2.5)))
            weights[y_scaled >= high] = np.maximum(weights[y_scaled >= high], height_weight)
    return np.sqrt(weights)


def initial_and_bounds(
    x: np.ndarray,
    y_scaled: np.ndarray,
    peaks: list[PeakSpec],
    height_upper_factors: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    global_max = float(np.nanmax(np.maximum(y_scaled, 0.0))) if y_scaled.size else 1.0
    global_max = max(global_max, 0.05)
    height_upper_factors = dict(height_upper_factors or {})
    p0: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    per_peak: list[dict[str, float]] = []
    for peak in peaks:
        in_window = (x >= peak.x_min) & (x <= peak.x_max)
        if np.any(in_window):
            yw = y_scaled[in_window]
            xw = x[in_window]
            local_idx = int(np.nanargmax(yw))
            local_height = max(float(yw[local_idx]), global_max * 0.05)
            local_pos = float(xw[local_idx])
        else:
            local_height = global_max * 0.15
            local_pos = peak.center
        height_factor = float(height_upper_factors.get(peak.peak_id, 1.0))
        height_upper = max(global_max * 1.25, local_height * 3.0, 0.1) * max(height_factor, 1.0)
        height_guess = min(max(local_height, 1e-8), height_upper)
        pos_guess = min(max(local_pos, peak.x_min), peak.x_max)
        width_guess = min(max(peak.width_guess, peak.width_min), peak.width_max)
        p0.extend([height_guess, pos_guess, width_guess])
        lower.extend([0.0, peak.x_min, peak.width_min])
        upper.extend([height_upper, peak.x_max, peak.width_max])
        per_peak.append(
            {
                "height_guess_scaled": height_guess,
                "height_upper_scaled": height_upper,
                "position_guess": pos_guess,
                "width_guess": width_guess,
            }
        )
    return np.asarray(p0), np.asarray(lower), np.asarray(upper), per_peak


def peak_with_bounds(
    peak: PeakSpec,
    x_min: float,
    x_max: float,
    width_min: float,
    width_max: float,
) -> PeakSpec:
    x_min, x_max = min(x_min, x_max), max(x_min, x_max)
    width_min = max(min(width_min, width_max), 1e-6)
    width_max = max(width_min + 1e-6, max(width_min, width_max))
    width_guess = min(max(peak.width_guess, width_min), width_max)
    return PeakSpec(
        peak_id=peak.peak_id,
        label=peak.label,
        required=peak.required,
        center=peak.center,
        x_min=x_min,
        x_max=x_max,
        width_min=width_min,
        width_max=width_max,
        width_guess=width_guess,
    )


def detect_bound_hits(
    params_scaled: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    peaks: list[PeakSpec],
    position_width_tol: float,
    height_upper_rel_tol: float,
) -> dict[str, set[str]]:
    hits: dict[str, set[str]] = {}
    for i, peak in enumerate(peaks):
        j = 3 * i
        peak_hits = hits.setdefault(peak.peak_id, set())
        height = float(params_scaled[j])
        position = float(params_scaled[j + 1])
        width = float(params_scaled[j + 2])
        if height <= float(lower[j]) + 1e-10:
            peak_hits.add("height_min")
        if height >= float(upper[j]) * (1.0 - max(height_upper_rel_tol, 0.0)):
            peak_hits.add("height_max")
        if position <= peak.x_min + position_width_tol:
            peak_hits.add("pos_min")
        if position >= peak.x_max - position_width_tol:
            peak_hits.add("pos_max")
        if width <= peak.width_min + position_width_tol:
            peak_hits.add("width_min")
        if width >= peak.width_max - position_width_tol:
            peak_hits.add("width_max")
        if not peak_hits:
            hits.pop(peak.peak_id, None)
    return hits


def expand_peaks_for_bound_hits(
    peaks: list[PeakSpec],
    hits: dict[str, set[str]],
    height_factors: dict[str, float],
    options: dict[str, Any],
) -> tuple[list[PeakSpec], dict[str, float], bool]:
    width_factor = max(float(options.get("adaptive_width_expand_factor", 1.35)), 1.01)
    pos_expand = max(float(options.get("adaptive_position_expand_cm", 3.0)), 0.0)
    height_factor = max(float(options.get("adaptive_height_expand_factor", 1.5)), 1.01)
    expanded: list[PeakSpec] = []
    changed = False
    new_height_factors = dict(height_factors)
    for peak in peaks:
        peak_hits = hits.get(peak.peak_id, set())
        x_min = peak.x_min
        x_max = peak.x_max
        width_min = peak.width_min
        width_max = peak.width_max
        if _peak_is_passive(peak, options):
            expanded.append(peak_with_bounds(peak, x_min, x_max, width_min, width_max))
            continue
        if "pos_min" in peak_hits:
            x_min -= pos_expand
            changed = True
        if "pos_max" in peak_hits:
            x_max += pos_expand
            changed = True
        if "width_min" in peak_hits:
            width_min = max(0.1, width_min / width_factor)
            changed = True
        if "width_max" in peak_hits:
            width_max *= width_factor
            changed = True
        if "height_max" in peak_hits:
            new_height_factors[peak.peak_id] = max(1.0, new_height_factors.get(peak.peak_id, 1.0)) * height_factor
            changed = True
        expanded.append(peak_with_bounds(peak, x_min, x_max, width_min, width_max))
    return expanded, new_height_factors, changed


def encode_bound_hits(hits: dict[str, set[str]]) -> str:
    if not hits:
        return ""
    return json.dumps({peak: sorted(values) for peak, values in sorted(hits.items())}, sort_keys=True)


def data_residual_vector(
    params: np.ndarray, x: np.ndarray, y_scaled: np.ndarray, sqrt_weights: np.ndarray
) -> np.ndarray:
    return (multiple_lorentzian(x, *params) - y_scaled) * sqrt_weights


def _peak_is_ch_left(peak: PeakSpec) -> bool:
    peak_id = peak.peak_id.upper()
    label = peak.label.upper()
    return peak_id in {"CH_L1", "CH_L2"} or label.startswith("CH LEFT")


def _peak_id_upper(peak: PeakSpec) -> str:
    return peak.peak_id.upper().replace(" ", "_")


def _passive_peak_ids(options: dict[str, Any]) -> set[str]:
    raw = options.get("passive_peaks", ["CH_MID"])
    if isinstance(raw, str):
        values = raw.replace(",", " ").split()
    else:
        values = list(raw or [])
    return {str(value).upper().replace(" ", "_") for value in values if str(value).strip()}


def _peak_is_passive(peak: PeakSpec, options: dict[str, Any]) -> bool:
    return _peak_id_upper(peak) in _passive_peak_ids(options)


def anti_burial_residuals(
    params: np.ndarray,
    x: np.ndarray,
    peaks: list[PeakSpec],
    options: dict[str, Any],
) -> np.ndarray:
    penalty = float(options.get("anti_burial_penalty", 0.0))
    if penalty <= 0:
        return np.empty(0, dtype=float)
    include_optional = bool(options.get("anti_burial_optional", False))
    sqrt_penalty = math.sqrt(penalty)
    eps = 1e-12
    residuals: list[float] = []

    components = []
    for i in range(len(peaks)):
        j = 3 * i
        components.append(single_lorentzian(x, params[j], params[j + 1], params[j + 2]))
    if not components:
        return np.empty(0, dtype=float)
    total = np.sum(np.vstack(components), axis=0)

    passive_penalty = float(options.get("passive_peak_penalty", 0.0))
    passive_indices = [i for i, peak in enumerate(peaks) if _peak_is_passive(peak, options)]
    if passive_penalty > 0.0 and passive_indices:
        active_heights = [
            float(params[3 * i])
            for i, peak in enumerate(peaks)
            if i not in passive_indices and float(params[3 * i]) > eps
        ]
        active_ref_height = max(active_heights) if active_heights else eps
        max_height_fraction = float(options.get("passive_max_height_fraction", 0.55))
        max_area_share = float(options.get("passive_max_area_share", 0.38))
        min_width = float(options.get("passive_min_width", 24.0))
        sqrt_passive = math.sqrt(passive_penalty)
        for passive_idx in passive_indices:
            peak = peaks[passive_idx]
            j = 3 * passive_idx
            passive_height_fraction = float(params[j]) / max(active_ref_height, eps)
            residuals.append(sqrt_passive * max(0.0, passive_height_fraction - max_height_fraction))
            passive_width = float(params[j + 2])
            if min_width > 0:
                residuals.append(sqrt_passive * max(0.0, (min_width - passive_width) / min_width))
            in_window = (x >= peak.x_min) & (x <= peak.x_max)
            if np.count_nonzero(in_window) >= 3:
                own_area = float(np.trapezoid(components[passive_idx][in_window], x[in_window]))
                total_area = float(np.trapezoid(total[in_window], x[in_window]))
                area_share = own_area / max(total_area, eps)
                residuals.append(sqrt_passive * max(0.0, area_share - max_area_share))

    for i, peak in enumerate(peaks):
        if _peak_is_passive(peak, options):
            continue
        if not peak.required and not include_optional:
            continue
        j = 3 * i
        pos = float(params[j + 1])
        own_at_center = float(params[j])
        total_at_center = 0.0
        for k in range(len(peaks)):
            kk = 3 * k
            total_at_center += float(single_lorentzian(np.asarray([pos]), params[kk], params[kk + 1], params[kk + 2])[0])
        center_fraction = own_at_center / max(total_at_center, eps)
        if _peak_is_ch_left(peak):
            min_center_fraction = float(options.get("ch_left_min_visibility", 0.22))
            min_area_fraction = float(options.get("ch_left_min_area_share", 0.14))
        else:
            min_center_fraction = float(options.get("required_min_visibility", 0.12))
            min_area_fraction = float(options.get("required_min_area_share", 0.08))

        residuals.append(sqrt_penalty * max(0.0, min_center_fraction - center_fraction))

        in_window = (x >= peak.x_min) & (x <= peak.x_max)
        if np.count_nonzero(in_window) >= 3:
            own_area = float(np.trapezoid(components[i][in_window], x[in_window]))
            total_area = float(np.trapezoid(total[in_window], x[in_window]))
            area_fraction = own_area / max(total_area, eps)
            residuals.append(sqrt_penalty * max(0.0, min_area_fraction - area_fraction))

    master_id = str(options.get("ch_left_master", "CH_L2")).upper()
    identity_penalty = float(options.get("ch_left_identity_penalty", 0.0))
    if master_id in {"CH_L1", "CH_L2"} and identity_penalty > 0:
        ch_lookup = {_peak_id_upper(peak): idx for idx, peak in enumerate(peaks)}
        if "CH_L1" in ch_lookup and "CH_L2" in ch_lookup:
            master_idx = ch_lookup[master_id]
            slave_idx = ch_lookup["CH_L1" if master_id == "CH_L2" else "CH_L2"]
            pair_min = min(peaks[master_idx].x_min, peaks[slave_idx].x_min)
            pair_max = max(peaks[master_idx].x_max, peaks[slave_idx].x_max)
            in_pair = (x >= pair_min) & (x <= pair_max)
            if np.count_nonzero(in_pair) >= 3:
                master_area = float(np.trapezoid(components[master_idx][in_pair], x[in_pair]))
                slave_area = float(np.trapezoid(components[slave_idx][in_pair], x[in_pair]))
                ratio = slave_area / max(master_area, eps)
                min_ratio = float(options.get("ch_left_slave_min_area_ratio", 0.20))
                max_ratio = float(options.get("ch_left_slave_max_area_ratio", 0.95))
                sqrt_identity = math.sqrt(identity_penalty)
                residuals.append(sqrt_identity * max(0.0, min_ratio - ratio))
                residuals.append(sqrt_identity * max(0.0, ratio - max_ratio))

            center_penalty = float(options.get("ch_left_center_dominance_penalty", 0.0))
            own_center_share = float(options.get("ch_left_own_center_share", 0.55))
            if center_penalty > 0:
                sqrt_center = math.sqrt(center_penalty)
                for own_idx, other_idx in (
                    (ch_lookup["CH_L1"], ch_lookup["CH_L2"]),
                    (ch_lookup["CH_L2"], ch_lookup["CH_L1"]),
                ):
                    own_j = 3 * own_idx
                    other_j = 3 * other_idx
                    own_pos = float(params[own_j + 1])
                    own_value = float(params[own_j])
                    other_value = float(
                        single_lorentzian(
                            np.asarray([own_pos]),
                            params[other_j],
                            params[other_j + 1],
                            params[other_j + 2],
                        )[0]
                    )
                    share = own_value / max(own_value + other_value, eps)
                    residuals.append(sqrt_center * max(0.0, own_center_share - share))

            width_penalty = float(options.get("ch_left_width_ratio_penalty", 0.0))
            if width_penalty > 0:
                sqrt_width = math.sqrt(width_penalty)
                width_master = float(params[3 * master_idx + 2])
                width_slave = float(params[3 * slave_idx + 2])
                width_ratio = width_slave / max(width_master, eps)
                width_min = float(options.get("ch_left_width_ratio_min", 0.70))
                width_max = float(options.get("ch_left_width_ratio_max", 1.35))
                residuals.append(sqrt_width * max(0.0, width_min - width_ratio))
                residuals.append(sqrt_width * max(0.0, width_ratio - width_max))

    return np.asarray(residuals, dtype=float)


def residual_vector(
    params: np.ndarray,
    x: np.ndarray,
    y_scaled: np.ndarray,
    sqrt_weights: np.ndarray,
    peaks: list[PeakSpec],
    options: dict[str, Any],
) -> np.ndarray:
    data_residuals = data_residual_vector(params, x, y_scaled, sqrt_weights)
    penalties = anti_burial_residuals(params, x, peaks, options)
    if penalties.size:
        return np.concatenate([data_residuals, penalties])
    return data_residuals


def objective_value(
    params: np.ndarray,
    x: np.ndarray,
    y_scaled: np.ndarray,
    sqrt_weights: np.ndarray,
    peaks: list[PeakSpec],
    options: dict[str, Any],
) -> float:
    r = residual_vector(params, x, y_scaled, sqrt_weights, peaks, options)
    return float(np.dot(r, r))


def data_objective_value(
    params: np.ndarray, x: np.ndarray, y_scaled: np.ndarray, sqrt_weights: np.ndarray
) -> float:
    r = data_residual_vector(params, x, y_scaled, sqrt_weights)
    return float(np.dot(r, r))


def clip_params(params: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(params, lower), upper)


def jitter_params(
    best: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    peaks: list[PeakSpec],
    rng: np.random.Generator,
    jitter_fraction: float,
) -> np.ndarray:
    out = best.copy()
    for i, peak in enumerate(peaks):
        j = 3 * i
        out[j] *= 1.0 + rng.normal(0.0, jitter_fraction)
        pos_span = max(peak.x_max - peak.x_min, 1.0)
        out[j + 1] += rng.normal(0.0, pos_span * jitter_fraction)
        out[j + 2] *= 1.0 + rng.normal(0.0, jitter_fraction)
    return clip_params(out, lower, upper)


def bootstrap_parameter_std(
    x: np.ndarray,
    y_scaled: np.ndarray,
    sqrt_weights: np.ndarray,
    best_scaled: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    peaks: list[PeakSpec],
    scale: float,
    runs: int,
    jitter_fraction: float,
    max_nfev: int,
    rng: np.random.Generator,
    options: dict[str, Any],
) -> tuple[np.ndarray, int, int]:
    if runs <= 0:
        return np.full_like(best_scaled, np.nan, dtype=float), 0, 0
    fitted = multiple_lorentzian(x, *best_scaled)
    residuals = y_scaled - fitted
    if residuals.size < 4:
        return np.full_like(best_scaled, np.nan, dtype=float), 0, runs

    samples: list[np.ndarray] = []
    failures = 0
    for _ in range(runs):
        boot_y = fitted + rng.choice(residuals, size=residuals.size, replace=True)
        x0 = jitter_params(best_scaled, lower, upper, peaks, rng, jitter_fraction)
        try:
            ls = least_squares(
                residual_vector,
                x0,
                bounds=(lower, upper),
                args=(x, boot_y, sqrt_weights, peaks, options),
                max_nfev=max_nfev,
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9,
                x_scale="jac",
            )
        except Exception:
            failures += 1
            continue
        if ls.success and np.all(np.isfinite(ls.x)):
            sample = ls.x.copy()
            sample[0::3] *= scale
            samples.append(sample)
        else:
            failures += 1
    if len(samples) < 2:
        return np.full_like(best_scaled, np.nan, dtype=float), len(samples), failures
    return np.nanstd(np.vstack(samples), axis=0, ddof=1), len(samples), failures


def fit_spectrum_column_once(
    x_full: np.ndarray,
    y_full: np.ndarray,
    peaks: list[PeakSpec],
    options: dict[str, Any],
    seed: int,
    height_upper_factors: dict[str, float] | None = None,
) -> dict[str, Any]:
    finite = np.isfinite(x_full) & np.isfinite(y_full)
    x_all = x_full[finite].astype(float)
    y_all = y_full[finite].astype(float)
    region = choose_fit_region(
        x_all,
        float(options["fit_x_min"]),
        float(options["fit_x_max"]),
    )
    x = x_all[region]
    y = y_all[region]
    if x.size < max(12, len(peaks) * 5):
        raise ValueError(f"Not enough data points in fit region ({x.size})")

    scale = float(np.nanmax(np.abs(y)))
    if not math.isfinite(scale) or scale <= 0:
        scale = 1.0
    y_scaled = y / scale
    sqrt_weights = make_weight_vector(x, y_scaled, peaks, options)
    p0, lower, upper, per_peak_guesses = initial_and_bounds(
        x, y_scaled, peaks, height_upper_factors
    )

    rng = np.random.default_rng(seed)
    de_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    de_result = differential_evolution(
        objective_value,
        bounds=list(zip(lower, upper)),
        args=(x, y_scaled, sqrt_weights, peaks, options),
        maxiter=int(options["de_maxiter"]),
        popsize=int(options["de_popsize"]),
        tol=float(options["de_tol"]),
        polish=False,
        seed=de_seed,
        updating="immediate",
        workers=1,
        init="latinhypercube",
        x0=p0,
    )

    ls_result = least_squares(
        residual_vector,
        clip_params(de_result.x, lower, upper),
        bounds=(lower, upper),
        args=(x, y_scaled, sqrt_weights, peaks, options),
        max_nfev=int(options["ls_max_nfev"]),
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        x_scale="jac",
    )
    best_scaled = ls_result.x if ls_result.success else de_result.x
    best_scaled = clip_params(best_scaled, lower, upper)
    best_original = best_scaled.copy()
    best_original[0::3] *= scale

    fitted = multiple_lorentzian(x, *best_original)
    residuals = y - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(ss_res / max(1, y.size)))
    mae = float(np.mean(np.abs(residuals)))
    weighted_sse_data = float(data_objective_value(best_scaled, x, y_scaled, sqrt_weights))
    penalty_residuals = anti_burial_residuals(best_scaled, x, peaks, options)
    anti_burial_sse = float(np.dot(penalty_residuals, penalty_residuals))
    weighted_sse = weighted_sse_data + anti_burial_sse

    bootstrap_std, bootstrap_success, bootstrap_failures = bootstrap_parameter_std(
        x=x,
        y_scaled=y_scaled,
        sqrt_weights=sqrt_weights,
        best_scaled=best_scaled,
        lower=lower,
        upper=upper,
        peaks=peaks,
        scale=scale,
        runs=int(options["bootstrap_runs"]),
        jitter_fraction=float(options["bootstrap_jitter"]),
        max_nfev=max(500, int(options["ls_max_nfev"]) // 2),
        rng=rng,
        options=options,
    )

    return {
        "x_fit": x,
        "y_fit_region": y,
        "params_scaled": best_scaled,
        "params_original": best_original,
        "bootstrap_std": bootstrap_std,
        "bootstrap_success": bootstrap_success,
        "bootstrap_failures": bootstrap_failures,
        "r_squared": r_squared,
        "rmse": rmse,
        "mae": mae,
        "weighted_sse": weighted_sse,
        "weighted_sse_data": weighted_sse_data,
        "anti_burial_sse": anti_burial_sse,
        "peak_window_weighting": bool(options.get("peak_window_weighting", True)),
        "peak_window_weight": float(options.get("peak_window_weight", 6.0)),
        "peak_window_margin": float(options.get("peak_window_margin", 0.0)),
        "peak_window_height_weighting": bool(options.get("peak_window_height_weighting", True)),
        "peak_window_height_weight": float(options.get("peak_window_height_weight", 2.0)),
        "peak_window_height_floor_percentile": float(
            options.get("peak_window_height_floor_percentile", 10.0)
        ),
        "height_percentile_weighting": bool(options.get("height_percentile_weighting", False)),
        "height_weight_percentile": float(options.get("height_weight_percentile", 75.0)),
        "height_weight": float(options.get("height_weight", 2.5)),
        "n_weighted_points": int(np.count_nonzero(sqrt_weights > 1.0)),
        "scale": scale,
        "n_points": int(y.size),
        "fit_x_min": float(np.min(x)),
        "fit_x_max": float(np.max(x)),
        "de_fun": float(de_result.fun),
        "de_nit": int(getattr(de_result, "nit", -1)),
        "de_success": bool(de_result.success),
        "de_message": str(de_result.message),
        "ls_success": bool(ls_result.success),
        "ls_cost": float(getattr(ls_result, "cost", float("nan"))),
        "ls_nfev": int(getattr(ls_result, "nfev", -1)),
        "ls_message": str(ls_result.message),
        "per_peak_guesses": per_peak_guesses,
        "_lower": lower,
        "_upper": upper,
        "_y_scaled": y_scaled,
        "_sqrt_weights": sqrt_weights,
    }


def add_final_bootstrap(
    fit: dict[str, Any],
    peaks: list[PeakSpec],
    options: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    runs = int(options["bootstrap_runs"])
    if runs <= 0:
        return fit
    updated = dict(fit)
    rng = np.random.default_rng(seed)
    bootstrap_std, bootstrap_success, bootstrap_failures = bootstrap_parameter_std(
        x=updated["x_fit"],
        y_scaled=updated["_y_scaled"],
        sqrt_weights=updated["_sqrt_weights"],
        best_scaled=updated["params_scaled"],
        lower=updated["_lower"],
        upper=updated["_upper"],
        peaks=peaks,
        scale=updated["scale"],
        runs=runs,
        jitter_fraction=float(options["bootstrap_jitter"]),
        max_nfev=max(500, int(options["ls_max_nfev"]) // 2),
        rng=rng,
        options=options,
    )
    updated["bootstrap_std"] = bootstrap_std
    updated["bootstrap_success"] = bootstrap_success
    updated["bootstrap_failures"] = bootstrap_failures
    return updated


def fit_spectrum_column(
    x_full: np.ndarray,
    y_full: np.ndarray,
    peaks: list[PeakSpec],
    options: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    if not bool(options.get("adaptive_bounds", True)):
        fit = fit_spectrum_column_once(x_full, y_full, peaks, options, seed)
        fit["peaks"] = peaks
        fit["adaptive_bounds_enabled"] = False
        fit["adaptive_bound_attempts"] = 1
        fit["adaptive_bound_expansions"] = 0
        fit["adaptive_bound_history"] = ""
        fit["adaptive_bound_unresolved_hits"] = ""
        return fit

    search_options = dict(options)
    search_options["bootstrap_runs"] = 0
    current_peaks = list(peaks)
    height_factors: dict[str, float] = {}
    max_expansions = max(0, int(options.get("adaptive_bound_max_iterations", 4)))
    bound_tol = float(options.get("adaptive_bound_tol", 0.05))
    height_tol = float(options.get("adaptive_height_upper_rel_tol", 0.01))
    history: list[dict[str, Any]] = []
    expansions = 0
    fit: dict[str, Any] | None = None
    hits: dict[str, set[str]] = {}

    for attempt in range(max_expansions + 1):
        attempt_seed = stable_seed(seed, "adaptive_bounds", attempt)
        fit = fit_spectrum_column_once(
            x_full,
            y_full,
            current_peaks,
            search_options,
            attempt_seed,
            height_factors,
        )
        hits = detect_bound_hits(
            fit["params_scaled"],
            fit["_lower"],
            fit["_upper"],
            current_peaks,
            bound_tol,
            height_tol,
        )
        history.append(
            {
                "attempt": attempt + 1,
                "hits": {peak: sorted(values) for peak, values in sorted(hits.items())},
                "r_squared": fit["r_squared"],
                "rmse": fit["rmse"],
            }
        )
        if not hits or attempt >= max_expansions:
            break
        next_peaks, next_height_factors, changed = expand_peaks_for_bound_hits(
            current_peaks, hits, height_factors, options
        )
        if not changed:
            break
        current_peaks = next_peaks
        height_factors = next_height_factors
        expansions += 1

    if fit is None:
        raise RuntimeError("Adaptive bound fitting did not produce a fit")

    fit = add_final_bootstrap(
        fit,
        current_peaks,
        options,
        stable_seed(seed, "adaptive_bounds_final_bootstrap"),
    )
    final_hits = detect_bound_hits(
        fit["params_scaled"],
        fit["_lower"],
        fit["_upper"],
        current_peaks,
        bound_tol,
        height_tol,
    )
    fit["peaks"] = current_peaks
    fit["adaptive_bounds_enabled"] = True
    fit["adaptive_bound_attempts"] = len(history)
    fit["adaptive_bound_expansions"] = expansions
    fit["adaptive_bound_history"] = json.dumps(history, sort_keys=True)
    fit["adaptive_bound_unresolved_hits"] = encode_bound_hits(final_hits)
    return fit


def downsample_for_plot(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or x.size <= max_points:
        return x, y
    step = int(math.ceil(x.size / max_points))
    return x[::step], y[::step]


def make_fit_preview_png(
    fit: dict[str, Any],
    peaks: list[PeakSpec],
    title: str,
    options: dict[str, Any],
) -> str:
    x = fit["x_fit"]
    y = fit["y_fit_region"]
    params = fit["params_original"]
    x_plot, y_plot = downsample_for_plot(x, y, int(options["html_max_points"]))

    fig, ax = plt.subplots(
        figsize=(float(options["plot_width"]), float(options["plot_height"])),
        dpi=int(options["plot_dpi"]),
    )
    ax.scatter(x_plot, y_plot, s=8, c="black", alpha=0.65, linewidths=0, label="data")
    total = multiple_lorentzian(x, *params)
    ax.plot(x, total, color="#d62728", lw=1.8, label="total Lorentzian fit")
    colors = plt.cm.tab10(np.linspace(0, 1, max(3, len(peaks))))
    for i, peak in enumerate(peaks):
        j = 3 * i
        component = single_lorentzian(x, params[j], params[j + 1], params[j + 2])
        ax.plot(x, component, ls="--", lw=1.0, color=colors[i], label=peak.peak_id)
        ax.axvspan(peak.x_min, peak.x_max, color=colors[i], alpha=0.08)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.text(
        0.01,
        0.98,
        f"R2={fit['r_squared']:.5f} | RMSE={fit['rmse']:.4g} | n={fit['n_points']}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
    )
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=int(options["plot_dpi"]), bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_rows_for_fit(
    fit: dict[str, Any],
    peaks: list[PeakSpec],
    metadata: dict[str, Any],
    min_r2_warning: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    params = fit["params_original"]
    std = fit["bootstrap_std"]
    quality_flag = "OK"
    if math.isfinite(fit["r_squared"]) and fit["r_squared"] < min_r2_warning:
        quality_flag = "LOW_R2"
    if not fit["ls_success"]:
        quality_flag = "CHECK_LS"

    base = {
        **metadata,
        "model": MODEL_NAME,
        "quality_flag": quality_flag,
        "r_squared": fit["r_squared"],
        "rmse": fit["rmse"],
        "mae": fit["mae"],
        "weighted_sse": fit["weighted_sse"],
        "weighted_sse_data": fit["weighted_sse_data"],
        "anti_burial_sse": fit["anti_burial_sse"],
        "peak_window_weighting": fit.get("peak_window_weighting", False),
        "peak_window_weight": fit.get("peak_window_weight", float("nan")),
        "peak_window_margin": fit.get("peak_window_margin", float("nan")),
        "peak_window_height_weighting": fit.get("peak_window_height_weighting", False),
        "peak_window_height_weight": fit.get("peak_window_height_weight", float("nan")),
        "peak_window_height_floor_percentile": fit.get(
            "peak_window_height_floor_percentile", float("nan")
        ),
        "height_percentile_weighting": fit.get("height_percentile_weighting", False),
        "height_weight_percentile": fit.get("height_weight_percentile", float("nan")),
        "height_weight": fit.get("height_weight", float("nan")),
        "n_weighted_points": fit.get("n_weighted_points", 0),
        "n_points_fit": fit["n_points"],
        "fit_x_min": fit["fit_x_min"],
        "fit_x_max": fit["fit_x_max"],
        "requested_fit_x_min": metadata["requested_fit_x_min"],
        "requested_fit_x_max": metadata["requested_fit_x_max"],
        "scale_used": fit["scale"],
        "de_success": fit["de_success"],
        "de_nit": fit["de_nit"],
        "de_fun": fit["de_fun"],
        "de_message": fit["de_message"],
        "ls_success": fit["ls_success"],
        "ls_nfev": fit["ls_nfev"],
        "ls_cost": fit["ls_cost"],
        "ls_message": fit["ls_message"],
        "bootstrap_requested": metadata["bootstrap_requested"],
        "bootstrap_success": fit["bootstrap_success"],
        "bootstrap_failures": fit["bootstrap_failures"],
        "adaptive_bounds_enabled": fit.get("adaptive_bounds_enabled", False),
        "adaptive_bound_attempts": fit.get("adaptive_bound_attempts", 1),
        "adaptive_bound_expansions": fit.get("adaptive_bound_expansions", 0),
        "adaptive_bound_unresolved_hits": fit.get("adaptive_bound_unresolved_hits", ""),
        "adaptive_bound_history": fit.get("adaptive_bound_history", ""),
    }
    long_rows: list[dict[str, Any]] = []
    wide_row = dict(base)
    guesses = fit["per_peak_guesses"]
    for i, peak in enumerate(peaks):
        j = 3 * i
        height = float(params[j])
        position = float(params[j + 1])
        width = float(params[j + 2])
        height_std = float(std[j]) if j < std.size else float("nan")
        position_std = float(std[j + 1]) if j + 1 < std.size else float("nan")
        width_std = float(std[j + 2]) if j + 2 < std.size else float("nan")
        peak_row = {
            **base,
            "peak_id": peak.peak_id,
            "peak_label": peak.label,
            "peak_required": peak.required,
            "height": height,
            "position_cm-1": position,
            "width_fwhm_cm-1": width,
            "height_std": height_std,
            "position_std_cm-1": position_std,
            "width_fwhm_std_cm-1": width_std,
            "x_bound_min": peak.x_min,
            "x_bound_max": peak.x_max,
            "width_bound_min": peak.width_min,
            "width_bound_max": peak.width_max,
            "center_guess_from_json": peak.center,
            "height_guess_scaled": guesses[i]["height_guess_scaled"],
            "height_upper_scaled": guesses[i]["height_upper_scaled"],
            "position_guess": guesses[i]["position_guess"],
            "width_guess": guesses[i]["width_guess"],
        }
        long_rows.append(peak_row)
        prefix = safe_name(peak.peak_id)
        wide_row[f"{prefix}_height"] = height
        wide_row[f"{prefix}_position_cm-1"] = position
        wide_row[f"{prefix}_width_fwhm_cm-1"] = width
        wide_row[f"{prefix}_height_std"] = height_std
        wide_row[f"{prefix}_position_std_cm-1"] = position_std
        wide_row[f"{prefix}_width_fwhm_std_cm-1"] = width_std
    return long_rows, wide_row


def process_file_task(task: dict[str, Any]) -> dict[str, Any]:
    path = Path(task["path"])
    family = task["family"]
    sequence = task["sequence"]
    settings = task["settings"]
    options = task["options"]
    started = time.time()

    result: dict[str, Any] = {
        "family": family,
        "sequence": sequence,
        "rel_path": task["rel_path"],
        "long_rows": [],
        "wide_rows": [],
        "failure_rows": [],
        "html_fragment": "",
        "n_spectra": 0,
        "n_success": 0,
        "seconds": 0.0,
    }
    try:
        data, headers = load_txt_table(path)
        x = data[:, 0]
        temp = parse_temperature_from_text(path.name) or parse_temperature_from_text(str(path))
        if temp is None:
            raise ValueError("Could not parse temperature from file name/path")
        temp_key, bounds_temp, peaks = peak_specs_for_temperature(settings, family, temp)
        y_indices = list(range(1, data.shape[1]))
        max_spectra = options.get("max_spectra_per_file")
        if max_spectra is not None:
            y_indices = y_indices[: int(max_spectra)]
        result["n_spectra"] = len(y_indices)

        details: list[str] = []
        details.append(
            "<details open>"
            f"<summary><strong>{html.escape(task['rel_path'])}</strong> | "
            f"{html.escape(sequence)} | {temp:g} K | {len(y_indices)} Y column(s)</summary>"
        )
        details.append("<div class='file-meta'>")
        details.append(
            f"Bounds temperature: {bounds_temp:g} K (key {html.escape(temp_key)}), "
            f"peaks: {', '.join(html.escape(p.peak_id) for p in peaks)}"
        )
        details.append("</div>")

        for y_idx in y_indices:
            y = data[:, y_idx]
            y_name = headers[y_idx] if y_idx < len(headers) else f"Column {y_idx + 1}"
            spectrum_in_file = y_idx
            metadata = {
                "script_version": SCRIPT_VERSION,
                "family": family,
                "sequence": sequence,
                "temperature_K": temp,
                "bounds_temperature_K": bounds_temp,
                "bounds_temperature_key": temp_key,
                "bounds_temperature_delta_K": temp - bounds_temp,
                "file_path": task["rel_path"],
                "file_name": path.name,
                "txt_x_column_name": headers[0] if headers else "Column 1",
                "spectrum_in_file": spectrum_in_file,
                "y_column_number": y_idx + 1,
                "y_column_name": y_name,
                "bootstrap_requested": int(options["bootstrap_runs"]),
                "requested_fit_x_min": float(options["fit_x_min"]),
                "requested_fit_x_max": float(options["fit_x_max"]),
            }
            seed = stable_seed(int(options["seed"]), task["rel_path"], y_idx, family, temp)
            try:
                fit = fit_spectrum_column(x, y, peaks, options, seed)
                fit_peaks = fit.get("peaks", peaks)
                long_rows, wide_row = build_rows_for_fit(
                    fit, fit_peaks, metadata, float(options["min_r2_warning"])
                )
                result["long_rows"].extend(long_rows)
                result["wide_rows"].append(wide_row)
                result["n_success"] += 1
                title = (
                    f"{family} | {sequence} | {format_temp(temp)} | "
                    f"{path.name} | {y_name}"
                )
                png64 = make_fit_preview_png(fit, fit_peaks, title, options)
                details.append("<figure>")
                details.append(
                    f"<img src='data:image/png;base64,{png64}' alt='{html.escape(title)}'>"
                )
                details.append(
                    "<figcaption>"
                    f"{html.escape(title)} | R2={fit['r_squared']:.5f}, "
                    f"RMSE={fit['rmse']:.4g}, bootstrap={fit['bootstrap_success']}/"
                    f"{options['bootstrap_runs']}"
                    "</figcaption>"
                )
                details.append("</figure>")
            except Exception as exc:
                failure = {
                    **metadata,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
                result["failure_rows"].append(failure)
                details.append(
                    "<div class='failure'>"
                    f"FAILED {html.escape(path.name)} {html.escape(y_name)}: "
                    f"{html.escape(type(exc).__name__)}: {html.escape(str(exc))}"
                    "</div>"
                )
        details.append("</details>")
        result["html_fragment"] = "\n".join(details)
    except Exception as exc:
        result["failure_rows"].append(
            {
                "script_version": SCRIPT_VERSION,
                "family": family,
                "sequence": sequence,
                "temperature_K": "",
                "file_path": task["rel_path"],
                "file_name": path.name,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }
        )
        result["html_fragment"] = (
            "<details open><summary><strong>"
            f"{html.escape(task['rel_path'])}</strong> | FILE FAILED</summary>"
            f"<div class='failure'>{html.escape(type(exc).__name__)}: "
            f"{html.escape(str(exc))}</div></details>"
        )
    result["seconds"] = time.time() - started
    return result


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fieldnames: list[str] = []
    seen: set[str] = set()
    if exists:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                fieldnames = next(reader)
            except StopIteration:
                fieldnames = []
        seen.update(fieldnames)
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    if exists:
        # If new columns appear later, rewrite with the expanded header.
        with path.open("r", newline="", encoding="utf-8") as f:
            old_rows = list(csv.DictReader(f))
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(old_rows)
            writer.writerows(rows)
    else:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def html_header(title: str, config: dict[str, Any]) -> str:
    escaped_title = html.escape(title)
    escaped_config = html.escape(json.dumps(config, indent=2, sort_keys=True))
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{escaped_title}</title>
<style>
body {{
  font-family: Arial, Helvetica, sans-serif;
  margin: 20px;
  color: #161616;
}}
h1, h2 {{ margin: 0.3rem 0 0.8rem; }}
details {{
  border: 1px solid #cfcfcf;
  border-radius: 6px;
  padding: 10px;
  margin: 12px 0;
  background: #fff;
}}
summary {{ cursor: pointer; }}
img {{ max-width: 100%; height: auto; border: 1px solid #dedede; }}
figure {{ margin: 12px 0; }}
figcaption {{ font-size: 12px; color: #333; }}
.file-meta {{ margin: 6px 0 10px; color: #444; font-size: 13px; }}
.failure {{
  color: #9b1c1c;
  background: #fff2f2;
  border: 1px solid #e4b4b4;
  padding: 8px;
  margin: 8px 0;
}}
pre {{
  background: #f7f7f7;
  border: 1px solid #ddd;
  padding: 10px;
  overflow-x: auto;
}}
</style>
</head>
<body>
<h1>{escaped_title}</h1>
<p>Generated incrementally by <code>{SCRIPT_VERSION}</code>.</p>
<details><summary>Run configuration</summary><pre>{escaped_config}</pre></details>
"""


def html_footer() -> str:
    return f"\n<p>Finished at {html.escape(dt.datetime.now().isoformat(timespec='seconds'))}</p>\n</body>\n</html>\n"


def reopen_html_for_resume(path: Path, family: str) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    footer_idx = text.rfind("</body>")
    if footer_idx != -1:
        text = text[:footer_idx]
    text += (
        "\n<hr>\n"
        f"<h2>Resume started {html.escape(dt.datetime.now().isoformat(timespec='seconds'))}</h2>\n"
        f"<p>Continuing report for <code>{html.escape(family)}</code>; completed TXT files are skipped by checkpoint.</p>\n"
    )
    path.write_text(text, encoding="utf-8")


def initialize_html_reports(
    output_root: Path,
    families: list[str],
    config: dict[str, Any],
    resume: bool,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for family in families:
        family_dir = output_root / safe_name(family)
        family_dir.mkdir(parents=True, exist_ok=True)
        html_path = family_dir / f"{safe_name(family)}_fit_report.html"
        if resume and html_path.exists() and html_path.stat().st_size > 0:
            reopen_html_for_resume(html_path, family)
        else:
            html_path.write_text(
                html_header(f"{family} Lorentzian fit report", config),
                encoding="utf-8",
            )
        paths[family] = html_path
    write_master_index(output_root, families, paths, config, finished=False)
    return paths


def append_html(path: Path, fragment: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write("\n")
        f.write(fragment)
        f.write("\n")
        f.flush()


def finalize_html_reports(paths: dict[str, Path]) -> None:
    for path in paths.values():
        with path.open("a", encoding="utf-8") as f:
            f.write(html_footer())


def write_master_index(
    output_root: Path,
    families: list[str],
    report_paths: dict[str, Path],
    config: dict[str, Any],
    finished: bool,
) -> None:
    lines = [html_header("Raman Lorentzian fitting index", config)]
    lines.append(f"<h2>Status: {'finished' if finished else 'running'}</h2>")
    lines.append("<ul>")
    for family in families:
        report = report_paths.get(family)
        if report:
            rel = report.relative_to(output_root).as_posix()
            lines.append(f"<li><a href='{html.escape(rel)}'>{html.escape(family)}</a></li>")
        else:
            lines.append(f"<li>{html.escape(family)}: no report</li>")
    lines.append("</ul>")
    lines.append(html_footer())
    (output_root / "index.html").write_text("\n".join(lines), encoding="utf-8")


def csv_paths_for_family(output_root: Path, family: str) -> dict[str, Path]:
    family_dir = output_root / safe_name(family)
    return {
        "long": family_dir / f"{safe_name(family)}_long_results.csv",
        "wide": family_dir / f"{safe_name(family)}_wide_results.csv",
        "failures": family_dir / f"{safe_name(family)}_failures.csv",
        "workbook": family_dir / f"{safe_name(family)}_Lorentzian_fit_results.xlsx",
    }


def run_signature(config: dict[str, Any], settings_sha256: str) -> dict[str, Any]:
    keys = [
        "script_version",
        "bootstrap_runs",
        "de_maxiter",
        "de_popsize",
        "de_tol",
        "ls_max_nfev",
        "fit_padding",
        "fit_x_min",
        "fit_x_max",
        "adaptive_bounds",
        "adaptive_bound_max_iterations",
        "adaptive_bound_tol",
        "adaptive_width_expand_factor",
        "adaptive_position_expand_cm",
        "adaptive_height_expand_factor",
        "adaptive_height_upper_rel_tol",
        "anti_burial_penalty",
        "required_min_visibility",
        "ch_left_min_visibility",
        "required_min_area_share",
        "ch_left_min_area_share",
        "anti_burial_optional",
        "passive_peaks",
        "passive_peak_penalty",
        "passive_max_height_fraction",
        "passive_max_area_share",
        "passive_min_width",
        "ch_left_master",
        "ch_left_identity_penalty",
        "ch_left_slave_min_area_ratio",
        "ch_left_slave_max_area_ratio",
        "ch_left_center_dominance_penalty",
        "ch_left_own_center_share",
        "ch_left_width_ratio_penalty",
        "ch_left_width_ratio_min",
        "ch_left_width_ratio_max",
        "max_spectra_per_file",
        "model",
    ]
    return {key: config.get(key) for key in keys} | {"settings_sha256": settings_sha256}


def checkpoint_path(output_root: Path, task: dict[str, Any]) -> Path:
    rel_path = task["rel_path"]
    digest = hashlib.sha1(rel_path.encode("utf-8", errors="replace")).hexdigest()[:16]
    stem = safe_name(Path(rel_path).stem)
    family = safe_name(task["family"])
    sequence = safe_name(task["sequence"])
    return output_root / "_checkpoints" / family / f"{sequence}__{stem}__{digest}.done.json"


def input_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def checkpoint_matches(
    output_root: Path,
    task: dict[str, Any],
    signature: dict[str, Any],
) -> bool:
    marker = checkpoint_path(output_root, task)
    if not marker.exists():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return False
    if data.get("rel_path") != task["rel_path"]:
        return False
    if data.get("family") != task["family"]:
        return False
    if data.get("run_signature") != signature:
        return False
    try:
        current_input = input_signature(Path(task["path"]))
    except OSError:
        return False
    return data.get("input_signature") == current_input and data.get("status") == "complete"


def write_checkpoint(
    output_root: Path,
    task: dict[str, Any],
    result: dict[str, Any],
    signature: dict[str, Any],
) -> None:
    marker = checkpoint_path(output_root, task)
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "complete",
        "completed_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_version": SCRIPT_VERSION,
        "family": task["family"],
        "sequence": task["sequence"],
        "rel_path": task["rel_path"],
        "n_spectra": result.get("n_spectra"),
        "n_success": result.get("n_success"),
        "run_signature": signature,
        "input_signature": input_signature(Path(task["path"])),
    }
    tmp = marker.with_suffix(marker.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(marker)


def sanitize_sheet_name(name: str, used: set[str]) -> str:
    cleaned = re.sub(r"[\[\]:*?/\\]", "_", name)[:31] or "Sheet"
    base = cleaned
    idx = 1
    while cleaned in used:
        suffix = f"_{idx}"
        cleaned = f"{base[:31 - len(suffix)]}{suffix}"
        idx += 1
    used.add(cleaned)
    return cleaned


def write_family_workbook(output_root: Path, family: str) -> Path | None:
    paths = csv_paths_for_family(output_root, family)
    long_path = paths["long"]
    wide_path = paths["wide"]
    failures_path = paths["failures"]
    if not long_path.exists() and not wide_path.exists() and not failures_path.exists():
        return None

    long_df = pd.read_csv(long_path) if long_path.exists() else pd.DataFrame()
    wide_df = pd.read_csv(wide_path) if wide_path.exists() else pd.DataFrame()
    failures_df = pd.read_csv(failures_path) if failures_path.exists() else pd.DataFrame()

    stats_by_sequence = pd.DataFrame()
    stats_all_sequences = pd.DataFrame()
    if not long_df.empty:
        value_cols = ["height", "position_cm-1", "width_fwhm_cm-1", "r_squared", "rmse"]
        for col in value_cols:
            long_df[col] = pd.to_numeric(long_df[col], errors="coerce")
        stats_by_sequence = (
            long_df.groupby(["sequence", "temperature_K", "peak_id", "peak_label"], dropna=False)
            .agg(
                n_spectra=("height", "count"),
                height_mean=("height", "mean"),
                height_std=("height", "std"),
                position_mean_cm_1=("position_cm-1", "mean"),
                position_std_cm_1=("position_cm-1", "std"),
                width_fwhm_mean_cm_1=("width_fwhm_cm-1", "mean"),
                width_fwhm_std_cm_1=("width_fwhm_cm-1", "std"),
                r_squared_mean=("r_squared", "mean"),
                rmse_mean=("rmse", "mean"),
            )
            .reset_index()
        )
        stats_all_sequences = (
            long_df.groupby(["temperature_K", "peak_id", "peak_label"], dropna=False)
            .agg(
                n_spectra=("height", "count"),
                height_mean=("height", "mean"),
                height_std=("height", "std"),
                position_mean_cm_1=("position_cm-1", "mean"),
                position_std_cm_1=("position_cm-1", "std"),
                width_fwhm_mean_cm_1=("width_fwhm_cm-1", "mean"),
                width_fwhm_std_cm_1=("width_fwhm_cm-1", "std"),
                r_squared_mean=("r_squared", "mean"),
                rmse_mean=("rmse", "mean"),
            )
            .reset_index()
        )

    workbook_path = paths["workbook"]
    engine = "xlsxwriter"
    try:
        import xlsxwriter  # noqa: F401
    except Exception:
        engine = "openpyxl"
    with pd.ExcelWriter(workbook_path, engine=engine) as writer:
        used: set[str] = set()
        if not long_df.empty:
            long_df.to_excel(writer, sheet_name=sanitize_sheet_name("all_long", used), index=False)
        if not wide_df.empty:
            wide_df.to_excel(writer, sheet_name=sanitize_sheet_name("all_wide", used), index=False)
        if not stats_by_sequence.empty:
            stats_by_sequence.to_excel(
                writer, sheet_name=sanitize_sheet_name("stats_by_sequence", used), index=False
            )
        if not stats_all_sequences.empty:
            stats_all_sequences.to_excel(
                writer, sheet_name=sanitize_sheet_name("stats_all_sequences", used), index=False
            )
        if not failures_df.empty:
            failures_df.to_excel(writer, sheet_name=sanitize_sheet_name("failures", used), index=False)
        if not long_df.empty:
            for temp, temp_df in long_df.groupby("temperature_K", dropna=False):
                sheet = sanitize_sheet_name(f"T_{format_temp(temp)}", used)
                temp_df.to_excel(writer, sheet_name=sheet, index=False)
    return workbook_path


def write_run_manifest(output_root: Path, tasks: list[dict[str, Any]], config: dict[str, Any]) -> None:
    manifest_path = output_root / "run_manifest.json"
    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "task_count": len(tasks),
        "tasks": [
            {k: task[k] for k in ("family", "sequence", "rel_path")}
            for task in tasks
        ],
        "config": config,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def options_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "bootstrap_runs": args.bootstrap_runs,
        "bootstrap_jitter": args.bootstrap_jitter,
        "de_maxiter": args.de_maxiter,
        "de_popsize": args.de_popsize,
        "de_tol": args.de_tol,
        "ls_max_nfev": args.ls_max_nfev,
        "fit_padding": args.fit_padding,
        "fit_x_min": args.fit_x_min,
        "fit_x_max": args.fit_x_max,
        "peak_window_weighting": args.peak_window_weighting,
        "peak_window_weight": args.peak_window_weight,
        "peak_window_margin": args.peak_window_margin,
        "peak_window_height_weighting": args.peak_window_height_weighting,
        "peak_window_height_weight": args.peak_window_height_weight,
        "peak_window_height_floor_percentile": args.peak_window_height_floor_percentile,
        "height_percentile_weighting": args.height_percentile_weighting,
        "height_weight_percentile": args.height_weight_percentile,
        "height_weight": args.height_weight,
        "adaptive_bounds": args.adaptive_bounds,
        "adaptive_bound_max_iterations": args.adaptive_bound_max_iterations,
        "adaptive_bound_tol": args.adaptive_bound_tol,
        "adaptive_width_expand_factor": args.adaptive_width_expand_factor,
        "adaptive_position_expand_cm": args.adaptive_position_expand_cm,
        "adaptive_height_expand_factor": args.adaptive_height_expand_factor,
        "adaptive_height_upper_rel_tol": args.adaptive_height_upper_rel_tol,
        "plot_dpi": args.plot_dpi,
        "plot_width": args.plot_width,
        "plot_height": args.plot_height,
        "anti_burial_penalty": args.anti_burial_penalty,
        "required_min_visibility": args.required_min_visibility,
        "ch_left_min_visibility": args.ch_left_min_visibility,
        "required_min_area_share": args.required_min_area_share,
        "ch_left_min_area_share": args.ch_left_min_area_share,
        "anti_burial_optional": args.anti_burial_optional,
        "passive_peaks": args.passive_peaks,
        "passive_peak_penalty": args.passive_peak_penalty,
        "passive_max_height_fraction": args.passive_max_height_fraction,
        "passive_max_area_share": args.passive_max_area_share,
        "passive_min_width": args.passive_min_width,
        "ch_left_master": args.ch_left_master,
        "ch_left_identity_penalty": args.ch_left_identity_penalty,
        "ch_left_slave_min_area_ratio": args.ch_left_slave_min_area_ratio,
        "ch_left_slave_max_area_ratio": args.ch_left_slave_max_area_ratio,
        "ch_left_center_dominance_penalty": args.ch_left_center_dominance_penalty,
        "ch_left_own_center_share": args.ch_left_own_center_share,
        "ch_left_width_ratio_penalty": args.ch_left_width_ratio_penalty,
        "ch_left_width_ratio_min": args.ch_left_width_ratio_min,
        "ch_left_width_ratio_max": args.ch_left_width_ratio_max,
        "max_spectra_per_file": args.max_spectra_per_file,
        "min_r2_warning": args.min_r2_warning,
        "html_max_points": args.html_max_points,
        "seed": args.seed,
    }


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    settings_path = Path(args.settings)
    if not settings_path.is_absolute():
        settings_path = root / settings_path
    settings = load_settings(settings_path)
    settings_hash = sha256_file(settings_path)

    output_root = Path(args.output_root).expanduser() if args.output_root else default_output_root(root)
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    discovered_tasks = discover_txt_files(root, settings, args.families)
    if args.max_files is not None:
        discovered_tasks = discovered_tasks[: args.max_files]
    if not discovered_tasks:
        raise RuntimeError("No TXT files discovered. Check root/family folders.")

    families = sorted(
        {task["family"] for task in discovered_tasks},
        key=lambda f: TARGET_SAMPLE_FOLDERS.index(f) if f in TARGET_SAMPLE_FOLDERS else 999,
    )
    config = {
        "script_version": SCRIPT_VERSION,
        "root": str(root),
        "settings": str(settings_path),
        "settings_sha256": settings_hash,
        "output_root": str(output_root),
        "families": families,
        "n_jobs": args.n_jobs,
        "bootstrap_runs": args.bootstrap_runs,
        "de_maxiter": args.de_maxiter,
        "de_popsize": args.de_popsize,
        "de_tol": args.de_tol,
        "ls_max_nfev": args.ls_max_nfev,
        "fit_padding": args.fit_padding,
        "fit_x_min": args.fit_x_min,
        "fit_x_max": args.fit_x_max,
        "adaptive_bounds": args.adaptive_bounds,
        "adaptive_bound_max_iterations": args.adaptive_bound_max_iterations,
        "adaptive_bound_tol": args.adaptive_bound_tol,
        "adaptive_width_expand_factor": args.adaptive_width_expand_factor,
        "adaptive_position_expand_cm": args.adaptive_position_expand_cm,
        "adaptive_height_expand_factor": args.adaptive_height_expand_factor,
        "adaptive_height_upper_rel_tol": args.adaptive_height_upper_rel_tol,
        "anti_burial_penalty": args.anti_burial_penalty,
        "required_min_visibility": args.required_min_visibility,
        "ch_left_min_visibility": args.ch_left_min_visibility,
        "required_min_area_share": args.required_min_area_share,
        "ch_left_min_area_share": args.ch_left_min_area_share,
        "anti_burial_optional": args.anti_burial_optional,
        "passive_peaks": args.passive_peaks,
        "passive_peak_penalty": args.passive_peak_penalty,
        "passive_max_height_fraction": args.passive_max_height_fraction,
        "passive_max_area_share": args.passive_max_area_share,
        "passive_min_width": args.passive_min_width,
        "ch_left_master": args.ch_left_master,
        "ch_left_identity_penalty": args.ch_left_identity_penalty,
        "ch_left_slave_min_area_ratio": args.ch_left_slave_min_area_ratio,
        "ch_left_slave_max_area_ratio": args.ch_left_slave_max_area_ratio,
        "ch_left_center_dominance_penalty": args.ch_left_center_dominance_penalty,
        "ch_left_own_center_share": args.ch_left_own_center_share,
        "ch_left_width_ratio_penalty": args.ch_left_width_ratio_penalty,
        "ch_left_width_ratio_min": args.ch_left_width_ratio_min,
        "ch_left_width_ratio_max": args.ch_left_width_ratio_max,
        "max_files": args.max_files,
        "max_spectra_per_file": args.max_spectra_per_file,
        "resume_enabled": not args.force_rerun,
        "model": MODEL_NAME,
    }
    signature = run_signature(config, settings_hash)
    skipped_tasks: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    if args.force_rerun:
        tasks = discovered_tasks
    else:
        for task in discovered_tasks:
            if checkpoint_matches(output_root, task, signature):
                skipped_tasks.append(task)
            else:
                tasks.append(task)
    config["discovered_txt_files"] = len(discovered_tasks)
    config["skipped_by_checkpoint"] = len(skipped_tasks)
    config["pending_txt_files"] = len(tasks)

    write_run_manifest(output_root, discovered_tasks, config)
    report_paths = initialize_html_reports(output_root, families, config, resume=not args.force_rerun)

    opts = options_dict(args)
    task_payloads = []
    for task in tasks:
        payload = dict(task)
        payload["settings"] = settings
        payload["options"] = opts
        task_payloads.append(payload)

    print(
        f"Raman Lorentzian fitting started: {len(task_payloads)} pending TXT files "
        f"({len(skipped_tasks)} skipped by checkpoint)",
        flush=True,
    )
    print(f"Output root: {output_root}", flush=True)
    print(f"Families: {', '.join(families)}", flush=True)
    print(f"Workers: {args.n_jobs}", flush=True)

    if not task_payloads:
        finalize_html_reports(report_paths)
        write_master_index(output_root, families, report_paths, config, finished=True)
        summary = {
            "script_version": SCRIPT_VERSION,
            "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
            "output_root": str(output_root),
            "txt_files_discovered": len(discovered_tasks),
            "txt_files_skipped_by_checkpoint": len(skipped_tasks),
            "txt_files_processed": 0,
            "spectra_success": 0,
            "failures": 0,
            "workbooks": [
                str(path)
                for family in families
                if (path := write_family_workbook(output_root, family)) is not None
            ],
        }
        (output_root / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("All discovered TXT files were already complete for this run signature.", flush=True)
        print(json.dumps(summary, indent=2), flush=True)
        return 0

    completed = 0
    total_success = 0
    total_failures = 0
    started = time.time()
    n_workers = max(1, int(args.n_jobs))
    if n_workers == 1:
        iterator = ((None, process_file_task(payload)) for payload in task_payloads)
    else:
        executor = ProcessPoolExecutor(max_workers=n_workers)
        future_map = {executor.submit(process_file_task, payload): payload for payload in task_payloads}
        iterator = ((future, future.result()) for future in as_completed(future_map))

    try:
        for _future, result in iterator:
            completed += 1
            family = result["family"]
            paths = csv_paths_for_family(output_root, family)
            write_csv_rows(paths["long"], result["long_rows"])
            write_csv_rows(paths["wide"], result["wide_rows"])
            write_csv_rows(paths["failures"], result["failure_rows"])
            append_html(report_paths[family], result["html_fragment"])
            if result["n_spectra"] > 0 and result["n_success"] == result["n_spectra"] and not result["failure_rows"]:
                source_task = next(t for t in discovered_tasks if t["rel_path"] == result["rel_path"])
                write_checkpoint(output_root, source_task, result, signature)
            total_success += int(result["n_success"])
            total_failures += len(result["failure_rows"])
            elapsed = time.time() - started
            print(
                f"[{completed}/{len(task_payloads)}] {result['rel_path']} | "
                f"{result['n_success']}/{result['n_spectra']} spectra fit | "
                f"{len(result['failure_rows'])} failure(s) | "
                f"{result['seconds']:.1f}s file | {elapsed/60:.1f} min elapsed",
                flush=True,
            )
    finally:
        if n_workers != 1:
            executor.shutdown(wait=True, cancel_futures=False)

    workbooks: list[str] = []
    for family in families:
        workbook = write_family_workbook(output_root, family)
        if workbook is not None:
            workbooks.append(str(workbook))
    finalize_html_reports(report_paths)
    write_master_index(output_root, families, report_paths, config, finished=True)

    summary = {
        "script_version": SCRIPT_VERSION,
        "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "txt_files_discovered": len(discovered_tasks),
        "txt_files_skipped_by_checkpoint": len(skipped_tasks),
        "txt_files_processed": completed,
        "spectra_success": total_success,
        "failures": total_failures,
        "workbooks": workbooks,
    }
    (output_root / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Raman Lorentzian fitting finished.", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if total_success > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
