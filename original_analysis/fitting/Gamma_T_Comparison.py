"""
Temperature-dependent Lorentzian fit-parameter plots from ETH cluster fits.

Reads the cluster-generated *_long_results.csv files and plots fitted height,
FWHM (Gamma), and position versus temperature for Raman modes. By default,
the app opens the newest paper-clean or PC-refitted fitted-results folder if
one exists, then falls back to the second-pass averaged cluster result folder.

1. G-mode FWHM versus T with all sample families overlaid.
2. A compact RBLM/D/G grid inspired by the reference figure.
3. Sequence-aware height, FWHM, and position grids.

Run with --app for interactive family/UP-DOWN/peak inspection and manual
outlier rejection. Manual rejections are saved outside the fitted data folder
and are applied to all three parameter trends for the same fitted peak row.

The script treats all fitted spectra/data files as read-only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import matplotlib as mpl

if "--app" not in sys.argv:
    mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

try:
    from scipy.optimize import curve_fit, least_squares
except Exception:  # pragma: no cover - fallback only if scipy is unavailable
    curve_fit = None
    least_squares = None


APP_DIR = Path(__file__).resolve().parent
DEFAULT_BEFORE_FITTED_ROOT = APP_DIR / "Fitted_Spectra_ALL_Cluster_ETH"
DEFAULT_AFTER_FITTED_ROOT = APP_DIR / "RESULTS_AFTER_AVERAGING_CLEANED_SPECTRA_NO_OUTLIERS"
DEFAULT_OUTPUT_DIR = APP_DIR / "GAMMA_T_COMPARISON"
DEFAULT_PC_REFIT_OUTPUT_ROOT = APP_DIR / "Fitted_Spectra_ALL_Cluster_ETH_Updated"
DEFAULT_PC_REFIT_SOURCE_ROOT = DEFAULT_BEFORE_FITTED_ROOT / "Spectra_Fitted"
DEFAULT_SPECTRA_ROOT = APP_DIR / "ALS_BASELINE_CORRECTED"
DEFAULT_BOUNDS_JSON = APP_DIR / "raman_peak_bounds_settings.json"
DEFAULT_ACCEPTED_REFIT_ROOT = APP_DIR / "Fitted_Spectra_ALL_Cluster_ETH_Refitted_PC_Updated"
DEFAULT_PAPER_CLEAN_ROOT = APP_DIR / "Fitted_Spectra_ALL_PAPER_CLEAN"
DEFAULT_SECOND_PASS_AVERAGED_ROOT = APP_DIR / "Fitting_Cluster_ETH_Second_Pass_After_Deselecting_Bad_Spectra"
PC_REFIT_SCRIPT = APP_DIR / "refit_bound_hit_spectra_pc.py"


def latest_fitted_root(base: Path) -> Optional[Path]:
    candidates = []
    if base.parent.exists():
        for path in base.parent.glob(f"{base.name}*"):
            if not path.is_dir():
                continue
            if path.name != base.name and not path.name.startswith(f"{base.name}_"):
                continue
            if any(path.rglob("*_long_results.csv")):
                candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


DEFAULT_FITTED_ROOT = (
    latest_fitted_root(DEFAULT_PAPER_CLEAN_ROOT)
    or latest_fitted_root(DEFAULT_ACCEPTED_REFIT_ROOT)
    or (DEFAULT_AFTER_FITTED_ROOT if DEFAULT_AFTER_FITTED_ROOT.exists() else DEFAULT_BEFORE_FITTED_ROOT)
)
DEFAULT_ANALYSIS_STAGE = "BEFORE" if DEFAULT_FITTED_ROOT == DEFAULT_BEFORE_FITTED_ROOT else "AFTER"

TARGET_SAMPLE_FOLDERS = [
    "Aligned_Au_3A",
    "Aligned_Au_8A",
    "Aligned_RO_8A",
    "MIRA_Au_unaligned_8A",
    "MIRA_RO_unaligned_8A",
]

FAMILY_TO_KEY = {
    "Aligned_Au_3A": "aligned_au_3a",
    "Aligned_Au_8A": "aligned_au_8a",
    "Aligned_RO_8A": "aligned_ro_8a",
    "MIRA_Au_unaligned_8A": "unaligned_au_8a",
    "MIRA_RO_unaligned_8A": "unaligned_ro_8a",
}

SAMPLE_COLORS = {
    "aligned_au_3a": "#feb715",
    "aligned_au_8a": "#f08228",
    "aligned_ro_8a": "#00c8c8",
    "unaligned_au_8a": "#ae540b",
    "unaligned_ro_8a": "#008686",
}

SAMPLE_LABELS = {
    "aligned_au_3a": "Aligned Au 3A",
    "aligned_au_8a": "Aligned Au 8A",
    "aligned_ro_8a": "Aligned RO 8A",
    "unaligned_au_8a": "Unaligned Au 8A",
    "unaligned_ro_8a": "Unaligned RO 8A",
}

FAMILY_PLOT_ORDER = [
    "aligned_au_3a",
    "aligned_au_8a",
    "unaligned_au_8a",
    "unaligned_ro_8a",
    "aligned_ro_8a",
]

PEAK_LABELS = {
    "RBLM": "RBLM",
    "CH_L1": "CH L1",
    "CH_L2": "CH L2",
    "CH_MID": "CH MID",
    "D": "D",
    "G": "G",
}

PEAK_COLORS = {
    "RBLM": "#2a7bc0",
    "CH_L1": "#35a853",
    "CH_L2": "#8e63d6",
    "CH_MID": "#e377c2",
    "D": "#b8b800",
    "G": "#17becf",
}

PEAK_PREVIEW_ORDER = {
    "RBLM": 0,
    "CH_L1": 1,
    "CH_L2": 2,
    "CH_MID": 3,
    "D": 4,
    "G": 5,
}

PARAMETER_SPECS = {
    "height": {
        "value": "height",
        "std": "height_std",
        "label": "Height (a.u.)",
        "title": "Height",
        "filename": "Height",
    },
    "width": {
        "value": "width_fwhm_cm-1",
        "std": "width_fwhm_std_cm-1",
        "label": r"FWHM (cm$^{-1}$)",
        "title": "FWHM",
        "filename": "FWHM",
    },
    "position": {
        "value": "position_cm-1",
        "std": "position_std_cm-1",
        "label": r"Position (cm$^{-1}$)",
        "title": "Position",
        "filename": "Position",
    },
}

PARAMETER_ORDER = ["height", "width", "position"]

FIXED_NON_ALIGNED_RO_Y_LIMITS = {
    "RBLM": (10.0, 43.0),
    "D": (11.0, 18.0),
    "G": (11.0, 15.5),
}

EXCLUDED_TEMPERATURE_POINTS = {
    ("Aligned_Au_3A", "Spikes_Removed", 300.0),
}

INDEPENDENT_Y_FAMILIES = {"Aligned_RO_8A"}

HC_OVER_KB = 1.4387769  # cm K


def setup_paper_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Nimbus Sans L"],
            "font.size": 15,
            "axes.labelsize": 18,
            "axes.titlesize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "axes.linewidth": 1.4,
            "xtick.major.width": 1.3,
            "ytick.major.width": 1.3,
            "axes.grid": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "rm",
        }
    )


def family_key(family: str) -> str:
    return FAMILY_TO_KEY.get(str(family), str(family).lower())


def family_label(family: str) -> str:
    return SAMPLE_LABELS.get(family_key(family), str(family).replace("_", " "))


def family_sort_key(family: str) -> Tuple[int, str]:
    key = family_key(family)
    try:
        return FAMILY_PLOT_ORDER.index(key), family
    except ValueError:
        return len(FAMILY_PLOT_ORDER), family


def marker_style(family: str) -> Dict[str, object]:
    key = family_key(family)
    color = SAMPLE_COLORS.get(key, "#333333")
    marker = "o" if key == "aligned_au_3a" else "D"
    hollow = key in {"unaligned_au_8a", "unaligned_ro_8a"}
    return {
        "marker": marker,
        "facecolors": "none" if hollow else color,
        "edgecolors": color,
        "linewidths": 1.8 if hollow else 0.8,
        "color": color,
    }


def _hex_to_rgb(color: str) -> Tuple[float, float, float]:
    color = color.strip().lstrip("#")
    if len(color) != 6:
        return 0.2, 0.2, 0.2
    return tuple(int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    vals = [max(0, min(255, int(round(channel * 255)))) for channel in rgb]
    return "#{:02x}{:02x}{:02x}".format(*vals)


def blend_color(color: str, target: str, amount: float) -> str:
    rgb = _hex_to_rgb(color)
    tgt = _hex_to_rgb(target)
    amount = max(0.0, min(1.0, float(amount)))
    return _rgb_to_hex(tuple((1.0 - amount) * rgb[i] + amount * tgt[i] for i in range(3)))


def sequence_label(sequence: object) -> str:
    label = str(sequence) if sequence is not None else "Sequence"
    label = re.sub(r"(?i)^spikes[_ -]*removed[_ -]*", "", label)
    label = label.replace("_", " ").strip()
    return label or "Sequence"


def sequence_sort_key(sequence: object) -> Tuple[int, int, str]:
    text = str(sequence)
    upper = text.upper()
    match = re.search(r"(UP|DOWN)[_ -]*(\d+)?", upper)
    if not match:
        return 9, 999, text
    direction = 0 if match.group(1) == "UP" else 1
    number = int(match.group(2) or 1)
    return number, direction, text


def sequence_color(family: str, sequence: object, sequences: Optional[List[object]] = None) -> str:
    base = SAMPLE_COLORS.get(family_key(family), "#333333")
    if sequences:
        ordered = sorted([str(item) for item in sequences], key=sequence_sort_key)
        try:
            idx = ordered.index(str(sequence))
        except ValueError:
            idx = 0
    else:
        idx = sequence_sort_key(sequence)[0] - 1
    variants = [
        base,
        blend_color(base, "#000000", 0.24),
        blend_color(base, "#ffffff", 0.30),
        blend_color(base, "#000000", 0.42),
        blend_color(base, "#ffffff", 0.48),
    ]
    return variants[max(0, idx) % len(variants)]


def sequence_marker_style(family: str, sequence: object, sequences: Optional[List[object]] = None) -> Dict[str, object]:
    style = marker_style(family).copy()
    color = sequence_color(family, sequence, sequences)
    style["color"] = color
    style["edgecolors"] = color
    if style["facecolors"] != "none":
        style["facecolors"] = color
    return style


def safe_filename(text: object, max_len: int = 160) -> str:
    value = re.sub(r"[<>:\"/\\|?*\x00-\x1f]+", "_", str(text))
    value = re.sub(r"\s+", "_", value).strip("._ ")
    return (value[:max_len] or "item").strip("._ ")


def n_be(omega_cm1: float, temperature: np.ndarray) -> np.ndarray:
    temperature = np.asarray(temperature, dtype=float)
    clipped = np.clip(temperature, 1e-9, None)
    x = HC_OVER_KB * float(omega_cm1) / clipped
    return 1.0 / (np.exp(x) - 1.0)


def gamma_model(temperature: np.ndarray, gamma0: float, c3: float, omega_cm1: float) -> np.ndarray:
    return gamma0 + c3 * (1.0 + 2.0 * n_be(float(omega_cm1) / 2.0, temperature))


def single_lorentzian(x: np.ndarray, height: float, position: float, width: float) -> np.ndarray:
    width = max(float(width), 1e-12)
    return float(height) / (1.0 + ((x - float(position)) / (0.5 * width)) ** 2)


def multiple_lorentzian(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    params = np.asarray(params, dtype=float)
    y = np.zeros_like(np.asarray(x, dtype=float), dtype=float)
    for idx in range(0, len(params), 3):
        y += single_lorentzian(x, params[idx], params[idx + 1], params[idx + 2])
    return y


def read_txt_matrix(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=r"\s+|\t|,", engine="python")
    except Exception:
        data = np.loadtxt(path)
        columns = [f"Column {idx + 1}" for idx in range(data.shape[1])]
        return pd.DataFrame(data, columns=columns)


def load_bounds_settings(settings_path: Path) -> Dict[str, object]:
    if not settings_path.exists():
        raise FileNotFoundError(f"Bounds JSON does not exist: {settings_path}")
    with settings_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "families" not in payload:
        raise ValueError(f"Bounds JSON has no 'families' section: {settings_path}")
    return payload


def bounds_for_family_temperature(settings: Dict[str, object], family: str, temperature_k: float) -> Dict[str, dict]:
    families = settings.get("families", {})
    if family not in families:
        raise KeyError(f"Family {family!r} not present in bounds JSON.")
    family_data = families[family]
    resolved = family_data.get("resolved_temperature_bounds") or family_data.get("temperature_anchors") or {}
    if not resolved:
        raise KeyError(f"Family {family!r} has no temperature bounds in JSON.")
    temp_items = []
    for key, value in resolved.items():
        try:
            temp_items.append((abs(float(key) - float(temperature_k)), str(key), value))
        except Exception:
            continue
    if not temp_items:
        raise KeyError(f"No numeric temperature bounds for family {family!r}.")
    _delta, _key, selected = min(temp_items, key=lambda item: item[0])
    peaks = selected.get("peaks", selected)
    return {
        str(pid): dict(bounds)
        for pid, bounds in peaks.items()
        if bool(bounds.get("enabled", True))
    }


def unique_existing_roots(*roots: Path | str | None) -> List[Path]:
    unique: List[Path] = []
    seen: Set[str] = set()
    for root in roots:
        if root is None:
            continue
        path = Path(root)
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def locate_spectrum_path(row: pd.Series, spectra_root: Path, *extra_roots: Path | str | None) -> Path:
    roots = unique_existing_roots(spectra_root, *extra_roots, DEFAULT_SECOND_PASS_AVERAGED_ROOT)
    candidates: List[Path] = []
    file_path = str(row.get("file_path", "")).strip()
    family = str(row.get("family", "")).strip()
    sequence = str(row.get("sequence", "")).strip()
    file_name = str(row.get("file_name", "")).strip()
    for root in roots:
        if file_path:
            candidates.append(root / file_path)
        if family and sequence and file_name:
            candidates.append(root / family / sequence / file_name)
        if file_name and root.exists():
            candidates.extend(root.rglob(file_name))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n  - ".join(str(root) for root in roots)
    raise FileNotFoundError(
        f"Could not locate TXT spectrum for {file_path or file_name} under:\n  - {searched}"
    )


def spectrum_column_index(row: pd.Series, matrix: pd.DataFrame) -> int:
    y_column = pd.to_numeric(row.get("y_column_number", np.nan), errors="coerce")
    if math.isfinite(float(y_column)):
        idx = int(y_column) - 1
        if 1 <= idx < matrix.shape[1]:
            return idx
    spectrum_index = pd.to_numeric(row.get("spectrum_in_file", np.nan), errors="coerce")
    if math.isfinite(float(spectrum_index)):
        idx = int(spectrum_index)
        if 1 <= idx < matrix.shape[1]:
            return idx
    if matrix.shape[1] < 2:
        raise ValueError("TXT matrix has no Y columns.")
    return 1


def peak_windows_from_bounds(peak_bounds: Dict[str, dict], margin_cm: float = 0.0) -> List[Tuple[float, float]]:
    windows = []
    for bounds in peak_bounds.values():
        try:
            windows.append((float(bounds["x_min"]) - margin_cm, float(bounds["x_max"]) + margin_cm))
        except Exception:
            continue
    return windows


def noise_mask_from_windows(x: np.ndarray, windows: List[Tuple[float, float]], x_min: float, x_max: float) -> np.ndarray:
    mask = np.isfinite(x) & (x >= float(x_min)) & (x <= float(x_max))
    for left, right in windows:
        mask &= ~((x >= float(left)) & (x <= float(right)))
    return mask


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return float("nan")
    median = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - median)))
    sigma = 1.4826 * mad
    if not math.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(values, ddof=1)) if values.size > 1 else float("nan")
    return sigma


def load_cluster_results(fitted_root: Path) -> pd.DataFrame:
    if not fitted_root.exists():
        raise FileNotFoundError(f"Fitted root does not exist: {fitted_root}")

    csv_paths = sorted(
        path
        for path in fitted_root.rglob("*_long_results.csv")
        if "_checkpoints" not in {part.lower() for part in path.parts}
    )
    if not csv_paths:
        raise FileNotFoundError(f"No *_long_results.csv files found under {fitted_root}")

    required = {
        "family",
        "sequence",
        "temperature_K",
        "file_name",
        "spectrum_in_file",
        "y_column_number",
        "peak_id",
        "position_cm-1",
        "width_fwhm_cm-1",
    }
    frames = []
    skipped = []
    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            skipped.append(f"{csv_path}: {exc}")
            continue
        missing = required - set(df.columns)
        if missing:
            skipped.append(f"{csv_path}: missing {sorted(missing)}")
            continue
        df["_source_csv"] = str(csv_path)
        frames.append(df)

    if not frames:
        detail = "\n".join(skipped[:10])
        raise ValueError(f"No usable cluster fit CSVs found.\n{detail}")
    if skipped:
        print(f"Skipped {len(skipped)} unusable CSV(s).")

    results = pd.concat(frames, ignore_index=True)
    numeric_cols = [
        "temperature_K",
        "spectrum_in_file",
        "y_column_number",
        "height",
        "height_std",
        "position_cm-1",
        "position_std_cm-1",
        "width_fwhm_cm-1",
        "width_fwhm_std_cm-1",
        "r_squared",
        "rmse",
    ]
    for col in numeric_cols:
        if col in results.columns:
            results[col] = pd.to_numeric(results[col], errors="coerce")
    print(f"Loaded {len(results)} fitted peak rows from {len(csv_paths)} cluster CSV file(s).")
    return results


def linewidth_table(results: pd.DataFrame, peaks: Iterable[str], min_r2: Optional[float]) -> pd.DataFrame:
    peaks = [str(peak) for peak in peaks]
    df = results[
        results["family"].astype(str).isin(TARGET_SAMPLE_FOLDERS)
        & results["peak_id"].astype(str).isin(peaks)
    ].copy()
    if min_r2 is not None and "r_squared" in df.columns:
        df = df[df["r_squared"] >= float(min_r2)].copy()
    df = df[np.isfinite(df["temperature_K"]) & np.isfinite(df["width_fwhm_cm-1"])].copy()
    df = df[df["width_fwhm_cm-1"] > 0].copy()
    df["family_key"] = df["family"].map(family_key)
    df["family_label"] = df["family"].map(family_label)
    df["_family_order"] = df["family"].map(lambda name: family_sort_key(str(name))[0])
    df["sequence_label"] = df["sequence"].map(sequence_label)
    df["_sequence_order"] = df["sequence"].map(sequence_sort_key)
    return df.sort_values(["_family_order", "peak_id", "_sequence_order", "temperature_K", "y_column_number"])


def fit_row_key(row: pd.Series) -> str:
    parts = [
        str(row.get("family", "")),
        str(row.get("sequence", "")),
        f"{float(row.get('temperature_K', float('nan'))):.6g}",
        str(row.get("file_path", row.get("file_name", ""))),
        str(row.get("spectrum_in_file", "")),
        str(row.get("y_column_number", "")),
        str(row.get("peak_id", "")),
    ]
    return "|".join(parts)


def fit_parameter_table(results: pd.DataFrame, peaks: Iterable[str], min_r2: Optional[float]) -> pd.DataFrame:
    peaks = [str(peak) for peak in peaks]
    df = results[
        results["family"].astype(str).isin(TARGET_SAMPLE_FOLDERS)
        & results["peak_id"].astype(str).isin(peaks)
    ].copy()
    if min_r2 is not None and "r_squared" in df.columns:
        df = df[df["r_squared"] >= float(min_r2)].copy()

    for spec in PARAMETER_SPECS.values():
        for column in (spec["value"], spec["std"]):
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

    finite_required = np.isfinite(pd.to_numeric(df["temperature_K"], errors="coerce"))
    for spec in PARAMETER_SPECS.values():
        finite_required &= np.isfinite(pd.to_numeric(df[spec["value"]], errors="coerce"))
    df = df[finite_required].copy()
    df = df[(df["height"] >= 0) & (df["width_fwhm_cm-1"] > 0)].copy()
    df["family_key"] = df["family"].map(family_key)
    df["family_label"] = df["family"].map(family_label)
    df["_family_order"] = df["family"].map(lambda name: family_sort_key(str(name))[0])
    df["sequence_label"] = df["sequence"].map(sequence_label)
    df["_sequence_order"] = df["sequence"].map(sequence_sort_key)
    df["fit_row_key"] = df.apply(fit_row_key, axis=1)
    return df.sort_values(
        ["_family_order", "sequence", "peak_id", "temperature_K", "file_name", "y_column_number"]
    ).reset_index(drop=True)


def outlier_mask(values: np.ndarray, method: str, mad_threshold: float, iqr_factor: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    mask = finite.copy()
    finite_values = values[finite]
    if method == "none" or finite_values.size < 3:
        return mask

    if method == "mad":
        median = float(np.nanmedian(finite_values))
        mad = float(np.nanmedian(np.abs(finite_values - median)))
        if mad > 0:
            robust_z = np.abs(values - median) / (1.4826 * mad)
            mask &= robust_z <= float(mad_threshold)
            return mask

    q1, q3 = np.nanpercentile(finite_values, [25.0, 75.0])
    iqr = float(q3 - q1)
    if iqr <= 0:
        return mask
    lower = q1 - float(iqr_factor) * iqr
    upper = q3 + float(iqr_factor) * iqr
    mask &= (values >= lower) & (values <= upper)
    return mask


def average_linewidth_by_temperature(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    grouped_rows = []
    group_cols = [
        "family",
        "family_key",
        "family_label",
        "_family_order",
        "sequence",
        "sequence_label",
        "_sequence_order",
        "peak_id",
        "temperature_K",
    ]
    for keys, group in df.groupby(group_cols, dropna=False, sort=False):
        values = group["width_fwhm_cm-1"].to_numpy(dtype=float)
        keep = outlier_mask(values, args.outlier_method, args.mad_threshold, args.iqr_factor)
        kept = group.loc[keep].copy()
        if kept.empty:
            kept = group[np.isfinite(group["width_fwhm_cm-1"])].copy()
        if kept.empty:
            continue

        row = dict(zip(group_cols, keys))
        widths = kept["width_fwhm_cm-1"].to_numpy(dtype=float)
        row["width_fwhm_cm-1"] = float(np.nanmean(widths))
        row["width_group_std_cm-1"] = float(np.nanstd(widths, ddof=1)) if widths.size > 1 else 0.0
        row["width_group_sem_cm-1"] = (
            float(row["width_group_std_cm-1"] / math.sqrt(widths.size)) if widths.size > 1 else 0.0
        )
        if "width_fwhm_std_cm-1" in kept.columns:
            boot = pd.to_numeric(kept["width_fwhm_std_cm-1"], errors="coerce").to_numpy(dtype=float)
            boot = boot[np.isfinite(boot) & (boot >= 0)]
        else:
            boot = np.array([], dtype=float)
        if boot.size:
            row["width_bootstrap_mean_std_cm-1"] = float(np.nanmean(boot))
            row["width_bootstrap_sem_cm-1"] = float(np.sqrt(np.nansum(boot**2)) / widths.size)
        else:
            row["width_bootstrap_mean_std_cm-1"] = 0.0
            row["width_bootstrap_sem_cm-1"] = 0.0
        row["width_total_sem_cm-1"] = float(
            math.sqrt(row["width_group_sem_cm-1"] ** 2 + row["width_bootstrap_sem_cm-1"] ** 2)
        )
        row["width_total_std_visual_cm-1"] = float(
            math.sqrt(row["width_group_std_cm-1"] ** 2 + row["width_bootstrap_mean_std_cm-1"] ** 2)
        )
        if "position_cm-1" in kept.columns:
            row["position_cm-1"] = float(np.nanmedian(kept["position_cm-1"].to_numpy(dtype=float)))
        if "r_squared" in kept.columns:
            row["r_squared"] = float(np.nanmedian(kept["r_squared"].to_numpy(dtype=float)))
        if "rmse" in kept.columns:
            row["rmse"] = float(np.nanmedian(kept["rmse"].to_numpy(dtype=float)))
        row["n_raw"] = int(np.isfinite(values).sum())
        row["n_used"] = int(widths.size)
        row["n_outliers_removed"] = int(row["n_raw"] - row["n_used"])
        row["outlier_method"] = args.outlier_method
        grouped_rows.append(row)

    if not grouped_rows:
        return df.iloc[0:0].copy()
    averaged = pd.DataFrame(grouped_rows)
    return averaged.sort_values(["_family_order", "peak_id", "_sequence_order", "temperature_K"]).reset_index(drop=True)


def average_parameters_by_temperature(
    df: pd.DataFrame,
    args: argparse.Namespace,
    rejected_keys: Optional[Set[str]] = None,
) -> pd.DataFrame:
    rejected_keys = rejected_keys or set()
    source = df[~df["fit_row_key"].astype(str).isin(rejected_keys)].copy()
    grouped_rows = []
    group_cols = [
        "family",
        "family_key",
        "family_label",
        "_family_order",
        "sequence",
        "sequence_label",
        "_sequence_order",
        "peak_id",
        "temperature_K",
    ]
    for keys, group in source.groupby(group_cols, dropna=False, sort=False):
        row = dict(zip(group_cols, keys))
        row["n_raw"] = int(len(group))
        row["n_used"] = int(len(group))
        row["n_manual_rejected"] = int(
            df[
                (df["family"].astype(str) == str(row["family"]))
                & (df["sequence"].astype(str) == str(row["sequence"]))
                & (df["peak_id"].astype(str) == str(row["peak_id"]))
                & np.isclose(df["temperature_K"].astype(float), float(row["temperature_K"]))
                & df["fit_row_key"].astype(str).isin(rejected_keys)
            ].shape[0]
        )

        for param, spec in PARAMETER_SPECS.items():
            values = pd.to_numeric(group[spec["value"]], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(values)
            kept_values = values[valid]
            if kept_values.size and args.outlier_method != "none":
                keep_param = outlier_mask(kept_values, args.outlier_method, args.mad_threshold, args.iqr_factor)
                kept_values = kept_values[keep_param]
            if kept_values.size == 0:
                row[spec["value"]] = float("nan")
                row[f"{param}_group_std"] = 0.0
                row[f"{param}_group_sem"] = 0.0
                row[f"{param}_bootstrap_mean_std"] = 0.0
                row[f"{param}_bootstrap_sem"] = 0.0
                row[f"{param}_total_sem"] = 0.0
                row[f"{param}_total_std_visual"] = 0.0
                continue

            row[spec["value"]] = float(np.nanmean(kept_values))
            row[f"{param}_group_std"] = float(np.nanstd(kept_values, ddof=1)) if kept_values.size > 1 else 0.0
            row[f"{param}_group_sem"] = (
                float(row[f"{param}_group_std"] / math.sqrt(kept_values.size)) if kept_values.size > 1 else 0.0
            )
            std_col = spec["std"]
            if std_col in group.columns:
                boot = pd.to_numeric(group[std_col], errors="coerce").to_numpy(dtype=float)
                boot = boot[np.isfinite(boot) & (boot >= 0)]
            else:
                boot = np.array([], dtype=float)
            if boot.size:
                row[f"{param}_bootstrap_mean_std"] = float(np.nanmean(boot))
                row[f"{param}_bootstrap_sem"] = float(np.sqrt(np.nansum(boot**2)) / max(kept_values.size, 1))
            else:
                row[f"{param}_bootstrap_mean_std"] = 0.0
                row[f"{param}_bootstrap_sem"] = 0.0
            row[f"{param}_total_sem"] = float(
                math.sqrt(row[f"{param}_group_sem"] ** 2 + row[f"{param}_bootstrap_sem"] ** 2)
            )
            row[f"{param}_total_std_visual"] = float(
                math.sqrt(row[f"{param}_group_std"] ** 2 + row[f"{param}_bootstrap_mean_std"] ** 2)
            )

        if "r_squared" in group.columns:
            row["r_squared"] = float(np.nanmedian(group["r_squared"].to_numpy(dtype=float)))
        if "rmse" in group.columns:
            row["rmse"] = float(np.nanmedian(group["rmse"].to_numpy(dtype=float)))
        grouped_rows.append(row)

    if not grouped_rows:
        return source.iloc[0:0].copy()
    averaged = pd.DataFrame(grouped_rows)
    return averaged.sort_values(["_family_order", "peak_id", "_sequence_order", "temperature_K"]).reset_index(drop=True)


def spectrum_identity_mask(df: pd.DataFrame, row: pd.Series) -> pd.Series:
    def numeric_match(series: pd.Series, value: object) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        target = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(target):
            return pd.Series(np.isclose(numeric, float(target), equal_nan=False), index=series.index)
        return series.astype(str) == str(value)

    temps = pd.to_numeric(df["temperature_K"], errors="coerce").to_numpy(dtype=float)
    target_temp = pd.to_numeric(pd.Series([row.get("temperature_K", np.nan)]), errors="coerce").iloc[0]
    mask = (
        (df["family"].astype(str) == str(row.get("family", "")))
        & (df["sequence"].astype(str) == str(row.get("sequence", "")))
        & pd.Series(np.isclose(temps, float(target_temp), equal_nan=False), index=df.index)
        & (df["file_name"].astype(str) == str(row.get("file_name", "")))
    )
    if "y_column_number" in df.columns and "y_column_number" in row:
        mask &= numeric_match(df["y_column_number"], row.get("y_column_number", ""))
    if "spectrum_in_file" in df.columns and "spectrum_in_file" in row:
        mask &= numeric_match(df["spectrum_in_file"], row.get("spectrum_in_file", ""))
    return mask


def build_trust_region_initial_values(
    x: np.ndarray,
    y: np.ndarray,
    peak_bounds: Dict[str, dict],
    current_rows: pd.DataFrame,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    peak_ids = sorted(peak_bounds.keys(), key=lambda pid: (float(peak_bounds[pid].get("center", 0.0)), pid))
    lower: List[float] = []
    upper: List[float] = []
    p0: List[float] = []
    global_max = float(np.nanmax(y[np.isfinite(y)])) if np.isfinite(y).any() else 1.0
    global_min = float(np.nanmin(y[np.isfinite(y)])) if np.isfinite(y).any() else 0.0
    global_span = max(global_max - global_min, 1.0)

    for peak_id in peak_ids:
        bounds = peak_bounds[peak_id]
        x_min = float(bounds["x_min"])
        x_max = float(bounds["x_max"])
        width_min = max(float(bounds.get("width_min", 1.0)), 1e-6)
        width_max = max(float(bounds.get("width_max", width_min + 1.0)), width_min + 1e-6)
        center = float(bounds.get("center", 0.5 * (x_min + x_max)))
        width_guess = float(bounds.get("width_guess", 0.5 * (width_min + width_max)))
        in_window = np.isfinite(x) & np.isfinite(y) & (x >= x_min) & (x <= x_max)
        local_height = float(np.nanmax(y[in_window])) if in_window.any() else global_max
        local_floor = float(np.nanmedian(y[in_window])) if in_window.any() else global_min
        height_guess = max(local_height - min(local_floor, 0.0), local_height, 1e-6)

        current = current_rows[current_rows["peak_id"].astype(str) == peak_id]
        if not current.empty:
            first = current.iloc[0]
            height_guess = float(first.get("height", height_guess))
            center = float(first.get("position_cm-1", center))
            width_guess = float(first.get("width_fwhm_cm-1", width_guess))

        height_upper = max(global_span * 6.0, height_guess * 3.0, 1.0)
        lower.extend([0.0, x_min, width_min])
        upper.extend([height_upper, x_max, width_max])
        p0.extend(
            [
                min(max(height_guess, 1e-9), height_upper * 0.98),
                min(max(center, x_min + 1e-9), x_max - 1e-9),
                min(max(width_guess, width_min + 1e-9), width_max - 1e-9),
            ]
        )

    return peak_ids, np.asarray(p0, dtype=float), np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def _fit_peak_is_ch_left(peak_id: str, bounds: Dict[str, object]) -> bool:
    label = str(bounds.get("label", "")).upper()
    peak = str(peak_id).upper().replace(" ", "_")
    return peak in {"CH_L1", "CH_L2"} or label.startswith("CH LEFT")


def trust_anti_burial_residuals(
    params: np.ndarray,
    x: np.ndarray,
    peak_ids: List[str],
    peak_bounds: Dict[str, dict],
    args: argparse.Namespace,
) -> np.ndarray:
    penalty = float(getattr(args, "trust_anti_burial_penalty", 120.0))
    if penalty <= 0:
        return np.empty(0, dtype=float)
    include_optional = bool(getattr(args, "trust_anti_burial_optional", False))
    sqrt_penalty = math.sqrt(penalty)
    eps = 1e-12
    residuals: List[float] = []
    components = []
    for idx, _peak_id in enumerate(peak_ids):
        j = 3 * idx
        components.append(single_lorentzian(x, params[j], params[j + 1], params[j + 2]))
    if not components:
        return np.empty(0, dtype=float)
    total = np.sum(np.vstack(components), axis=0)

    for idx, peak_id in enumerate(peak_ids):
        bounds = peak_bounds[peak_id]
        required = bool(bounds.get("required", True))
        if not required and not include_optional:
            continue
        j = 3 * idx
        pos = float(params[j + 1])
        own_at_center = float(params[j])
        total_at_center = 0.0
        for other_idx in range(len(peak_ids)):
            k = 3 * other_idx
            total_at_center += float(single_lorentzian(np.asarray([pos]), params[k], params[k + 1], params[k + 2])[0])
        center_fraction = own_at_center / max(total_at_center, eps)
        if _fit_peak_is_ch_left(peak_id, bounds):
            min_center_fraction = float(getattr(args, "trust_ch_left_min_visibility", 0.40))
            min_area_fraction = float(getattr(args, "trust_ch_left_min_area_share", 0.30))
        else:
            min_center_fraction = float(getattr(args, "trust_required_min_visibility", 0.18))
            min_area_fraction = float(getattr(args, "trust_required_min_area_share", 0.10))
        residuals.append(sqrt_penalty * max(0.0, min_center_fraction - center_fraction))

        x_min = float(bounds.get("x_min", np.nan))
        x_max = float(bounds.get("x_max", np.nan))
        in_window = np.isfinite(x) & (x >= x_min) & (x <= x_max)
        if np.count_nonzero(in_window) >= 3:
            own_area = float(np.trapezoid(components[idx][in_window], x[in_window]))
            total_area = float(np.trapezoid(total[in_window], x[in_window]))
            area_fraction = own_area / max(total_area, eps)
            residuals.append(sqrt_penalty * max(0.0, min_area_fraction - area_fraction))

    master_id = str(getattr(args, "trust_ch_left_master", "CH_L2")).upper()
    identity_penalty = float(getattr(args, "trust_ch_left_identity_penalty", 60.0))
    ch_lookup = {str(pid).upper().replace(" ", "_"): idx for idx, pid in enumerate(peak_ids)}
    if master_id in {"CH_L1", "CH_L2"} and identity_penalty > 0 and {"CH_L1", "CH_L2"} <= set(ch_lookup):
        master_idx = ch_lookup[master_id]
        slave_idx = ch_lookup["CH_L1" if master_id == "CH_L2" else "CH_L2"]
        master_bounds = peak_bounds[peak_ids[master_idx]]
        slave_bounds = peak_bounds[peak_ids[slave_idx]]
        pair_min = min(float(master_bounds["x_min"]), float(slave_bounds["x_min"]))
        pair_max = max(float(master_bounds["x_max"]), float(slave_bounds["x_max"]))
        in_pair = np.isfinite(x) & (x >= pair_min) & (x <= pair_max)
        sqrt_identity = math.sqrt(identity_penalty)
        if np.count_nonzero(in_pair) >= 3:
            master_area = float(np.trapezoid(components[master_idx][in_pair], x[in_pair]))
            slave_area = float(np.trapezoid(components[slave_idx][in_pair], x[in_pair]))
            ratio = slave_area / max(master_area, eps)
            min_ratio = float(getattr(args, "trust_ch_left_slave_min_area_ratio", 0.45))
            max_ratio = float(getattr(args, "trust_ch_left_slave_max_area_ratio", 0.90))
            residuals.append(sqrt_identity * max(0.0, min_ratio - ratio))
            residuals.append(sqrt_identity * max(0.0, ratio - max_ratio))

        center_penalty = float(getattr(args, "trust_ch_left_center_dominance_penalty", 90.0))
        if center_penalty > 0:
            sqrt_center = math.sqrt(center_penalty)
            own_center_share = float(getattr(args, "trust_ch_left_own_center_share", 0.55))
            for own_idx, other_idx in ((ch_lookup["CH_L1"], ch_lookup["CH_L2"]), (ch_lookup["CH_L2"], ch_lookup["CH_L1"])):
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

        width_penalty = float(getattr(args, "trust_ch_left_width_ratio_penalty", 80.0))
        if width_penalty > 0:
            sqrt_width = math.sqrt(width_penalty)
            width_master = float(params[3 * master_idx + 2])
            width_slave = float(params[3 * slave_idx + 2])
            width_ratio = width_slave / max(width_master, eps)
            width_min = float(getattr(args, "trust_ch_left_width_ratio_min", 0.75))
            width_max = float(getattr(args, "trust_ch_left_width_ratio_max", 1.30))
            residuals.append(sqrt_width * max(0.0, width_min - width_ratio))
            residuals.append(sqrt_width * max(0.0, width_ratio - width_max))

    return np.asarray(residuals, dtype=float)


def trust_region_refit_spectrum(
    selected_row: pd.Series,
    all_rows: pd.DataFrame,
    spectra_root: Path,
    settings_path: Path,
    args: argparse.Namespace,
    fit_x_min: float = 200.0,
    fit_x_max: float = 2000.0,
    peak_window_weight: float = 6.0,
    bootstrap_runs: int = 100,
    random_seed: int = 12345,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if least_squares is None:
        raise RuntimeError("scipy.optimize.least_squares is unavailable in this Python environment.")

    settings = load_bounds_settings(settings_path)
    family = str(selected_row["family"])
    temperature = float(selected_row["temperature_K"])
    peak_bounds = bounds_for_family_temperature(settings, family, temperature)
    if not peak_bounds:
        raise ValueError(f"No enabled peaks found in bounds JSON for {family} at {temperature:g} K.")

    path = locate_spectrum_path(selected_row, spectra_root, getattr(args, "second_pass_output_root", None))
    matrix = read_txt_matrix(path)
    if matrix.shape[1] < 2:
        raise ValueError(f"TXT file has no Y columns: {path}")
    y_index = spectrum_column_index(selected_row, matrix)
    x = pd.to_numeric(matrix.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(matrix.iloc[:, y_index], errors="coerce").to_numpy(dtype=float)
    fit_mask = np.isfinite(x) & np.isfinite(y) & (x >= float(fit_x_min)) & (x <= float(fit_x_max))
    if int(fit_mask.sum()) < 20:
        raise ValueError(f"Not enough finite points in fit ROI for {path}")
    x_fit = x[fit_mask]
    y_fit = y[fit_mask]

    same_spectrum = all_rows.loc[spectrum_identity_mask(all_rows, selected_row)].copy()
    peak_ids, p0, lower, upper = build_trust_region_initial_values(x_fit, y_fit, peak_bounds, same_spectrum)
    windows = peak_windows_from_bounds(peak_bounds, margin_cm=0.0)
    weights = np.ones_like(x_fit, dtype=float)
    for left, right in windows:
        weights[(x_fit >= left) & (x_fit <= right)] = max(float(peak_window_weight), 1.0)
    sqrt_weights = np.sqrt(weights)

    def residual(params: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        data_residual = (multiple_lorentzian(x_fit, params) - y_target) * sqrt_weights
        penalty_residual = trust_anti_burial_residuals(params, x_fit, peak_ids, peak_bounds, args)
        if penalty_residual.size:
            return np.concatenate([data_residual, penalty_residual])
        return data_residual

    result = least_squares(
        lambda params: residual(params, y_fit),
        p0,
        bounds=(lower, upper),
        method="trf",
        loss="linear",
        max_nfev=20000,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    params = result.x
    model = multiple_lorentzian(x_fit, params)
    raw_residual = y_fit - model
    anti_burial = trust_anti_burial_residuals(params, x_fit, peak_ids, peak_bounds, args)

    bootstrap_params = []
    rng = np.random.default_rng(int(random_seed))
    if bootstrap_runs > 0 and raw_residual.size > 3:
        for _idx in range(int(bootstrap_runs)):
            resampled = rng.choice(raw_residual, size=raw_residual.size, replace=True)
            y_boot = model + resampled
            try:
                boot = least_squares(
                    lambda par: residual(par, y_boot),
                    params,
                    bounds=(lower, upper),
                    method="trf",
                    loss="linear",
                    max_nfev=12000,
                    xtol=1e-9,
                    ftol=1e-9,
                    gtol=1e-9,
                )
                if boot.success and np.all(np.isfinite(boot.x)):
                    bootstrap_params.append(boot.x)
            except Exception:
                continue
    boot_array = np.vstack(bootstrap_params) if bootstrap_params else np.empty((0, len(params)))
    boot_std = np.nanstd(boot_array, axis=0, ddof=1) if boot_array.shape[0] > 1 else np.zeros(len(params))

    ss_res = float(np.nansum((y_fit - model) ** 2))
    ss_tot = float(np.nansum((y_fit - np.nanmean(y_fit)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(math.sqrt(ss_res / max(len(y_fit), 1)))
    rows = []
    for idx, peak_id in enumerate(peak_ids):
        param_idx = idx * 3
        rows.append(
            {
                "family": family,
                "sequence": selected_row.get("sequence", ""),
                "temperature_K": temperature,
                "file_path": selected_row.get("file_path", str(path)),
                "source_txt_path": str(path),
                "file_name": path.name,
                "spectrum_in_file": selected_row.get("spectrum_in_file", ""),
                "y_column_number": selected_row.get("y_column_number", ""),
                "peak_id": peak_id,
                "peak_label": peak_bounds[peak_id].get("label", peak_id),
                "height": float(params[param_idx]),
                "position_cm-1": float(params[param_idx + 1]),
                "width_fwhm_cm-1": float(params[param_idx + 2]),
                "height_std": float(boot_std[param_idx]) if boot_std.size else 0.0,
                "position_std_cm-1": float(boot_std[param_idx + 1]) if boot_std.size else 0.0,
                "width_fwhm_std_cm-1": float(boot_std[param_idx + 2]) if boot_std.size else 0.0,
                "x_bound_min": float(peak_bounds[peak_id]["x_min"]),
                "x_bound_max": float(peak_bounds[peak_id]["x_max"]),
                "width_bound_min": float(peak_bounds[peak_id].get("width_min", np.nan)),
                "width_bound_max": float(peak_bounds[peak_id].get("width_max", np.nan)),
                "r_squared": r_squared,
                "rmse": rmse,
                "weighted_sse": float(np.dot(residual(params, y_fit), residual(params, y_fit))),
                "weighted_sse_data": float(np.dot((model - y_fit) * sqrt_weights, (model - y_fit) * sqrt_weights)),
                "anti_burial_sse": float(np.dot(anti_burial, anti_burial)),
                "n_points_fit": int(len(x_fit)),
                "fit_x_min": float(np.nanmin(x_fit)),
                "fit_x_max": float(np.nanmax(x_fit)),
                "bootstrap_requested": int(bootstrap_runs),
                "bootstrap_success": int(boot_array.shape[0]),
                "trust_region_success": bool(result.success),
                "trust_region_message": str(result.message),
                "refit_method": "trust_region_least_squares",
                "quality_flag": "TRUST_REGION_REFIT_ACCEPTED",
            }
        )

    meta = {
        "path": str(path),
        "y_column_index": int(y_index),
        "n_peaks": int(len(peak_ids)),
        "success": bool(result.success),
        "message": str(result.message),
        "r_squared": r_squared,
        "rmse": rmse,
        "bootstrap_success": int(boot_array.shape[0]),
    }
    return pd.DataFrame(rows), meta


def trust_region_refit_single_peak(
    selected_row: pd.Series,
    all_rows: pd.DataFrame,
    spectra_root: Path,
    settings_path: Path,
    args: argparse.Namespace,
    target_peak_id: str,
    x_min: float,
    x_max: float,
    width_min: float,
    width_max: float,
    fit_x_min: float = 200.0,
    fit_x_max: float = 2000.0,
    peak_window_weight: float = 6.0,
    bootstrap_runs: int = 100,
    random_seed: int = 12345,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Refit one Lorentzian while keeping the other peaks fixed."""
    if least_squares is None:
        raise RuntimeError("scipy.optimize.least_squares is unavailable in this Python environment.")

    target_peak_id = str(target_peak_id)
    if not target_peak_id:
        raise ValueError("No target peak was selected for single-peak refit.")

    settings = load_bounds_settings(settings_path)
    family = str(selected_row["family"])
    temperature = float(selected_row["temperature_K"])
    peak_bounds = bounds_for_family_temperature(settings, family, temperature)
    if target_peak_id not in peak_bounds:
        raise ValueError(f"Peak {target_peak_id!r} is not enabled in the bounds JSON for {family} at {temperature:g} K.")

    x_min, x_max = sorted([float(x_min), float(x_max)])
    width_min, width_max = sorted([float(width_min), float(width_max)])
    width_min = max(width_min, 1e-6)
    width_max = max(width_max, width_min + 1e-6)

    path = locate_spectrum_path(selected_row, spectra_root, getattr(args, "second_pass_output_root", None))
    matrix = read_txt_matrix(path)
    if matrix.shape[1] < 2:
        raise ValueError(f"TXT file has no Y columns: {path}")
    y_index = spectrum_column_index(selected_row, matrix)
    x_all = pd.to_numeric(matrix.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    y_all = pd.to_numeric(matrix.iloc[:, y_index], errors="coerce").to_numpy(dtype=float)
    fit_mask = np.isfinite(x_all) & np.isfinite(y_all) & (x_all >= float(fit_x_min)) & (x_all <= float(fit_x_max))
    if int(fit_mask.sum()) < 20:
        raise ValueError(f"Not enough finite points in fit ROI for {path}")
    x_fit = x_all[fit_mask]
    y_fit = y_all[fit_mask]

    same = all_rows.loc[spectrum_identity_mask(all_rows, selected_row)].copy()
    same = same.drop_duplicates("peak_id", keep="last")
    same = same[
        np.isfinite(pd.to_numeric(same["height"], errors="coerce"))
        & np.isfinite(pd.to_numeric(same["position_cm-1"], errors="coerce"))
        & np.isfinite(pd.to_numeric(same["width_fwhm_cm-1"], errors="coerce"))
    ].copy()
    if same.empty:
        raise ValueError("No current fitted peak rows were found for this spectrum-column.")
    target_current = same[same["peak_id"].astype(str) == target_peak_id]
    if target_current.empty:
        raise ValueError(f"Current spectrum-column has no fitted row for peak {target_peak_id!r}.")

    fixed_total = np.zeros_like(x_fit, dtype=float)
    for _idx, peak_row in same.iterrows():
        if str(peak_row["peak_id"]) == target_peak_id:
            continue
        fixed_total += single_lorentzian(
            x_fit,
            float(peak_row["height"]),
            float(peak_row["position_cm-1"]),
            float(peak_row["width_fwhm_cm-1"]),
        )

    target_row = target_current.iloc[0]
    target_y = y_fit - fixed_total
    local_mask = np.isfinite(target_y) & (x_fit >= x_min) & (x_fit <= x_max)
    if int(local_mask.sum()) < 6:
        raise ValueError(f"Peak window {x_min:g}-{x_max:g} cm-1 has too few points for {target_peak_id}.")
    x_local = x_fit[local_mask]
    y_local = target_y[local_mask]

    local_max = float(np.nanmax(y_local)) if y_local.size else float(target_row["height"])
    local_min = float(np.nanmin(y_local)) if y_local.size else 0.0
    local_span = max(local_max - local_min, 1.0)
    height_guess = max(float(target_row["height"]), local_max, 1e-6)
    height_upper = max(height_guess * 4.0, local_span * 8.0, 1.0)
    p0 = np.asarray(
        [
            min(max(height_guess, 1e-9), height_upper * 0.98),
            min(max(float(target_row["position_cm-1"]), x_min + 1e-9), x_max - 1e-9),
            min(max(float(target_row["width_fwhm_cm-1"]), width_min + 1e-9), width_max - 1e-9),
        ],
        dtype=float,
    )
    lower = np.asarray([0.0, x_min, width_min], dtype=float)
    upper = np.asarray([height_upper, x_max, width_max], dtype=float)

    weights = np.ones_like(x_local, dtype=float) * max(float(peak_window_weight), 1.0)
    finite_local = y_local[np.isfinite(y_local)]
    if finite_local.size >= 3:
        floor = float(np.nanpercentile(finite_local, 10.0))
        top = float(np.nanmax(finite_local))
        if math.isfinite(top) and math.isfinite(floor) and top > floor:
            normalized = np.clip((y_local - floor) / (top - floor), 0.0, 1.0)
            weights *= 1.0 + normalized
    sqrt_weights = np.sqrt(weights)

    target_bounds = dict(peak_bounds[target_peak_id])
    target_bounds.update({"x_min": x_min, "x_max": x_max, "width_min": width_min, "width_max": width_max})
    min_center_fraction = float(getattr(args, "trust_required_min_visibility", 0.18))
    if _fit_peak_is_ch_left(target_peak_id, target_bounds):
        min_center_fraction = float(getattr(args, "trust_ch_left_min_visibility", 0.40))

    def single_peak_penalty(params: np.ndarray) -> np.ndarray:
        penalty = float(getattr(args, "trust_anti_burial_penalty", 120.0))
        if penalty <= 0:
            return np.empty(0, dtype=float)
        position = float(params[1])
        target_at_center = float(params[0])
        fixed_at_center = 0.0
        for _idx, peak_row in same.iterrows():
            if str(peak_row["peak_id"]) == target_peak_id:
                continue
            fixed_at_center += float(
                single_lorentzian(
                    np.asarray([position]),
                    float(peak_row["height"]),
                    float(peak_row["position_cm-1"]),
                    float(peak_row["width_fwhm_cm-1"]),
                )[0]
            )
        share = target_at_center / max(target_at_center + fixed_at_center, 1e-12)
        return np.asarray([math.sqrt(penalty) * max(0.0, min_center_fraction - share)], dtype=float)

    def residual(params: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        data_residual = (single_lorentzian(x_local, params[0], params[1], params[2]) - y_target) * sqrt_weights
        penalties = single_peak_penalty(params)
        if penalties.size:
            return np.concatenate([data_residual, penalties])
        return data_residual

    result = least_squares(
        lambda params: residual(params, y_local),
        p0,
        bounds=(lower, upper),
        method="trf",
        loss="linear",
        max_nfev=12000,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    params = result.x
    target_model_fit = single_lorentzian(x_fit, params[0], params[1], params[2])
    total_model = fixed_total + target_model_fit
    raw_residual = y_fit - total_model

    bootstrap_params = []
    local_model = single_lorentzian(x_local, params[0], params[1], params[2])
    local_residual = y_local - local_model
    rng = np.random.default_rng(int(random_seed))
    if bootstrap_runs > 0 and local_residual.size > 3:
        for _idx in range(int(bootstrap_runs)):
            y_boot = local_model + rng.choice(local_residual, size=local_residual.size, replace=True)
            try:
                boot = least_squares(
                    lambda par: residual(par, y_boot),
                    params,
                    bounds=(lower, upper),
                    method="trf",
                    loss="linear",
                    max_nfev=8000,
                    xtol=1e-9,
                    ftol=1e-9,
                    gtol=1e-9,
                )
                if boot.success and np.all(np.isfinite(boot.x)):
                    bootstrap_params.append(boot.x)
            except Exception:
                continue
    boot_array = np.vstack(bootstrap_params) if bootstrap_params else np.empty((0, len(params)))
    boot_std = np.nanstd(boot_array, axis=0, ddof=1) if boot_array.shape[0] > 1 else np.zeros(len(params))

    ss_res = float(np.nansum(raw_residual**2))
    ss_tot = float(np.nansum((y_fit - np.nanmean(y_fit)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(math.sqrt(ss_res / max(len(y_fit), 1)))
    weighted_local = residual(params, y_local)
    refit_row = target_row.copy()
    refit_row["family"] = family
    refit_row["sequence"] = selected_row.get("sequence", "")
    refit_row["temperature_K"] = temperature
    refit_row["file_path"] = selected_row.get("file_path", str(path))
    refit_row["source_txt_path"] = str(path)
    refit_row["file_name"] = path.name
    refit_row["spectrum_in_file"] = selected_row.get("spectrum_in_file", "")
    refit_row["y_column_number"] = selected_row.get("y_column_number", "")
    refit_row["peak_id"] = target_peak_id
    refit_row["peak_label"] = target_bounds.get("label", target_peak_id)
    refit_row["height"] = float(params[0])
    refit_row["position_cm-1"] = float(params[1])
    refit_row["width_fwhm_cm-1"] = float(params[2])
    refit_row["height_std"] = float(boot_std[0]) if boot_std.size else 0.0
    refit_row["position_std_cm-1"] = float(boot_std[1]) if boot_std.size else 0.0
    refit_row["width_fwhm_std_cm-1"] = float(boot_std[2]) if boot_std.size else 0.0
    refit_row["x_bound_min"] = x_min
    refit_row["x_bound_max"] = x_max
    refit_row["width_bound_min"] = width_min
    refit_row["width_bound_max"] = width_max
    refit_row["r_squared"] = r_squared
    refit_row["rmse"] = rmse
    refit_row["weighted_sse"] = float(np.dot(weighted_local, weighted_local))
    refit_row["weighted_sse_data"] = float(np.dot((local_model - y_local) * sqrt_weights, (local_model - y_local) * sqrt_weights))
    refit_row["anti_burial_sse"] = float(np.dot(single_peak_penalty(params), single_peak_penalty(params)))
    refit_row["n_points_fit"] = int(len(x_fit))
    refit_row["fit_x_min"] = float(np.nanmin(x_fit))
    refit_row["fit_x_max"] = float(np.nanmax(x_fit))
    refit_row["bootstrap_requested"] = int(bootstrap_runs)
    refit_row["bootstrap_success"] = int(boot_array.shape[0])
    refit_row["trust_region_success"] = bool(result.success)
    refit_row["trust_region_message"] = str(result.message)
    refit_row["refit_method"] = "single_peak_trust_region_least_squares"
    refit_row["quality_flag"] = "SINGLE_PEAK_TRUST_REGION_REFIT_ACCEPTED"

    meta = {
        "path": str(path),
        "target_peak_id": target_peak_id,
        "success": bool(result.success),
        "message": str(result.message),
        "r_squared": r_squared,
        "rmse": rmse,
        "bootstrap_success": int(boot_array.shape[0]),
        "x_min": x_min,
        "x_max": x_max,
        "width_min": width_min,
        "width_max": width_max,
    }
    return pd.DataFrame([dict(refit_row)]), meta


def append_refit_results(refit_df: pd.DataFrame, output_dir: Path) -> Path:
    folder = output_dir / "TRUST_REGION_REFITS"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "trust_region_refit_results.csv"
    if path.exists():
        old = pd.read_csv(path)
        combined = pd.concat([old, refit_df], ignore_index=True)
    else:
        combined = refit_df
    combined.to_csv(path, index=False)
    return path


def snr_for_peak(
    x: np.ndarray,
    y: np.ndarray,
    peak_id: str,
    peak_bounds: Dict[str, dict],
    fit_x_min: float,
    fit_x_max: float,
) -> Dict[str, float]:
    windows = peak_windows_from_bounds(peak_bounds, margin_cm=6.0)
    noise_mask = noise_mask_from_windows(x, windows, fit_x_min, fit_x_max)
    noise_values = y[noise_mask]
    noise_level = float(np.nanmedian(noise_values)) if np.isfinite(noise_values).any() else 0.0
    sigma = robust_sigma(noise_values)
    bounds = peak_bounds[peak_id]
    peak_mask = np.isfinite(x) & np.isfinite(y) & (x >= float(bounds["x_min"])) & (x <= float(bounds["x_max"]))
    roi_mask = np.isfinite(x) & np.isfinite(y) & (x >= float(fit_x_min)) & (x <= float(fit_x_max))
    global_signal = (
        float(np.nanmax(y[roi_mask]) - noise_level)
        if roi_mask.any()
        else float("nan")
    )
    g_signal = float("nan")
    if "G" in peak_bounds:
        g_bounds = peak_bounds["G"]
        g_mask = (
            np.isfinite(x)
            & np.isfinite(y)
            & (x >= float(g_bounds["x_min"]))
            & (x <= float(g_bounds["x_max"]))
        )
        if g_mask.any():
            g_signal = float(np.nanmax(y[g_mask]) - noise_level)
    if math.isfinite(g_signal) and g_signal > 0:
        whole_reference_signal = g_signal
        whole_reference_source = "G"
    elif math.isfinite(global_signal) and global_signal > 0:
        whole_reference_signal = global_signal
        whole_reference_source = "global_max"
    else:
        whole_reference_signal = float("nan")
        whole_reference_source = "none"
    whole_noise_normalized = (
        sigma / whole_reference_signal
        if math.isfinite(sigma) and sigma > 0 and math.isfinite(whole_reference_signal) and whole_reference_signal > 0
        else float("nan")
    )
    whole_snr = (
        whole_reference_signal / sigma
        if math.isfinite(sigma) and sigma > 0 and math.isfinite(whole_reference_signal) and whole_reference_signal > 0
        else float("nan")
    )
    if not peak_mask.any() or not math.isfinite(sigma) or sigma <= 0:
        return {
            "signal_height": float("nan"),
            "noise_sigma_mad": sigma,
            "noise_median": noise_level,
            "global_signal_height": global_signal,
            "whole_spectrum_reference_signal_height": whole_reference_signal,
            "whole_spectrum_reference_source": whole_reference_source,
            "whole_spectrum_noise_normalized": whole_noise_normalized,
            "whole_spectrum_snr": whole_snr,
            "g_signal_height": g_signal,
            "signal_height_g_normalized": float("nan"),
            "noise_sigma_g_normalized": sigma / g_signal if math.isfinite(g_signal) and g_signal > 0 else float("nan"),
            "g_reference_snr": g_signal / sigma if math.isfinite(g_signal) and g_signal > 0 and sigma > 0 else float("nan"),
            "snr_g_normalized": float("nan"),
            "peak_snr": float("nan"),
            "snr": whole_snr,
        }
    signal = float(np.nanmax(y[peak_mask]) - noise_level)
    if not math.isfinite(g_signal) or g_signal <= 0:
        g_signal = signal if math.isfinite(signal) and signal > 0 else float("nan")
    signal_g_norm = signal / g_signal if math.isfinite(g_signal) and g_signal > 0 else float("nan")
    noise_g_norm = sigma / g_signal if math.isfinite(g_signal) and g_signal > 0 else float("nan")
    peak_snr = signal / sigma if sigma > 0 else float("nan")
    return {
        "signal_height": signal,
        "noise_sigma_mad": sigma,
        "noise_median": noise_level,
        "global_signal_height": global_signal,
        "whole_spectrum_reference_signal_height": whole_reference_signal,
        "whole_spectrum_reference_source": whole_reference_source,
        "whole_spectrum_noise_normalized": whole_noise_normalized,
        "whole_spectrum_snr": whole_snr,
        "g_signal_height": g_signal,
        "signal_height_g_normalized": signal_g_norm,
        "noise_sigma_g_normalized": noise_g_norm,
        "g_reference_snr": g_signal / sigma if math.isfinite(g_signal) and g_signal > 0 and sigma > 0 else float("nan"),
        "snr_g_normalized": signal_g_norm / noise_g_norm if math.isfinite(signal_g_norm) and math.isfinite(noise_g_norm) and noise_g_norm > 0 else float("nan"),
        "peak_snr": peak_snr,
        "snr": whole_snr,
    }


def compute_snr_tables(
    fit_rows: pd.DataFrame,
    spectra_root: Path,
    settings_path: Path,
    output_dir: Path,
    fit_x_min: float,
    fit_x_max: float,
    second_pass_root: Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    settings = load_bounds_settings(settings_path)
    rows = []
    matrix_cache: Dict[Path, pd.DataFrame] = {}
    for _idx, row in fit_rows.iterrows():
        try:
            family = str(row["family"])
            temperature = float(row["temperature_K"])
            peak_id = str(row["peak_id"])
            peak_bounds = bounds_for_family_temperature(settings, family, temperature)
            if peak_id not in peak_bounds:
                continue
            path = locate_spectrum_path(row, spectra_root, second_pass_root)
            if path not in matrix_cache:
                matrix_cache[path] = read_txt_matrix(path)
            matrix = matrix_cache[path]
            y_idx = spectrum_column_index(row, matrix)
            x = pd.to_numeric(matrix.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(matrix.iloc[:, y_idx], errors="coerce").to_numpy(dtype=float)
            snr = snr_for_peak(x, y, peak_id, peak_bounds, fit_x_min, fit_x_max)
            rows.append(
                {
                    "family": family,
                    "family_label": family_label(family),
                    "sequence": row.get("sequence", ""),
                    "sequence_label": sequence_label(row.get("sequence", "")),
                    "temperature_K": temperature,
                    "file_name": row.get("file_name", ""),
                    "file_path": str(path),
                    "spectrum_in_file": row.get("spectrum_in_file", ""),
                    "y_column_number": row.get("y_column_number", ""),
                    "peak_id": peak_id,
                    "fit_row_key": row.get("fit_row_key", fit_row_key(row)),
                    **snr,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "family": row.get("family", ""),
                    "family_label": family_label(row.get("family", "")),
                    "sequence": row.get("sequence", ""),
                    "sequence_label": sequence_label(row.get("sequence", "")),
                    "temperature_K": row.get("temperature_K", np.nan),
                    "file_name": row.get("file_name", ""),
                    "file_path": row.get("file_path", ""),
                    "spectrum_in_file": row.get("spectrum_in_file", ""),
                    "y_column_number": row.get("y_column_number", ""),
                    "peak_id": row.get("peak_id", ""),
                    "fit_row_key": row.get("fit_row_key", fit_row_key(row)),
                    "signal_height": np.nan,
                    "noise_sigma_mad": np.nan,
                    "noise_median": np.nan,
                    "global_signal_height": np.nan,
                    "whole_spectrum_reference_signal_height": np.nan,
                    "whole_spectrum_reference_source": "",
                    "whole_spectrum_noise_normalized": np.nan,
                    "whole_spectrum_snr": np.nan,
                    "g_signal_height": np.nan,
                    "signal_height_g_normalized": np.nan,
                    "noise_sigma_g_normalized": np.nan,
                    "g_reference_snr": np.nan,
                    "snr_g_normalized": np.nan,
                    "peak_snr": np.nan,
                    "snr": np.nan,
                    "error": str(exc),
                }
            )
    snr_rows = pd.DataFrame(rows)
    summary_rows = []
    group_cols = ["family", "family_label", "sequence", "sequence_label", "temperature_K", "peak_id"]
    if not snr_rows.empty:
        for keys, group in snr_rows.groupby(group_cols, dropna=False, sort=False):
            row = dict(zip(group_cols, keys))
            n_for_group = 0
            for metric in (
                "snr",
                "whole_spectrum_snr",
                "whole_spectrum_noise_normalized",
                "whole_spectrum_reference_signal_height",
                "global_signal_height",
                "peak_snr",
                "snr_g_normalized",
                "g_reference_snr",
                "noise_sigma_g_normalized",
                "signal_height_g_normalized",
                "g_signal_height",
            ):
                if metric not in group.columns:
                    continue
                values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
                finite = values[np.isfinite(values)]
                if metric == "snr":
                    n_for_group = int(finite.size)
                row[f"{metric}_mean"] = float(np.nanmean(finite)) if finite.size else float("nan")
                row[f"{metric}_median"] = float(np.nanmedian(finite)) if finite.size else float("nan")
                row[f"{metric}_std"] = float(np.nanstd(finite, ddof=1)) if finite.size > 1 else 0.0
                row[f"{metric}_sem"] = float(row[f"{metric}_std"] / math.sqrt(finite.size)) if finite.size > 1 else 0.0
            row["n_spectra"] = n_for_group
            summary_rows.append(row)
    snr_summary = pd.DataFrame(summary_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "Gamma_T_Comparison_SNR_by_spectrum.csv"
    summary_path = output_dir / "Gamma_T_Comparison_SNR_by_temperature.csv"
    snr_rows.to_csv(rows_path, index=False)
    snr_summary.to_csv(summary_path, index=False)
    print(f"Saved SNR by spectrum to {rows_path}")
    print(f"Saved SNR by family/sequence/temperature to {summary_path}")
    return snr_rows, snr_summary


def temperature_token(value: object) -> str:
    temp = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(temp):
        return safe_filename(value)
    temp = float(temp)
    if math.isclose(temp, round(temp), abs_tol=1e-6):
        return f"{int(round(temp))}K"
    return f"{temp:g}K".replace(".", "p")


def y_column_number_to_index(value: object, matrix: pd.DataFrame) -> int:
    y_column = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(y_column):
        index = int(round(float(y_column))) - 1
        if 1 <= index < matrix.shape[1]:
            return index
    if matrix.shape[1] < 2:
        raise ValueError("TXT matrix has no Y columns to average.")
    return 1


def interpolate_to_reference_x(x_source: np.ndarray, y_source: np.ndarray, x_reference: np.ndarray) -> np.ndarray:
    x_source = np.asarray(x_source, dtype=float)
    y_source = np.asarray(y_source, dtype=float)
    x_reference = np.asarray(x_reference, dtype=float)
    finite = np.isfinite(x_source) & np.isfinite(y_source)
    if int(finite.sum()) < 2:
        return np.full_like(x_reference, np.nan, dtype=float)
    x_clean = x_source[finite]
    y_clean = y_source[finite]
    order = np.argsort(x_clean)
    x_clean = x_clean[order]
    y_clean = y_clean[order]
    unique_x, unique_idx = np.unique(x_clean, return_index=True)
    unique_y = y_clean[unique_idx]
    if unique_x.size < 2:
        return np.full_like(x_reference, np.nan, dtype=float)
    interpolated = np.interp(x_reference, unique_x, unique_y)
    outside = (x_reference < float(unique_x[0])) | (x_reference > float(unique_x[-1]))
    interpolated[outside] = np.nan
    return interpolated


def export_averaged_second_pass_txt(
    manifest: pd.DataFrame,
    spectra_root: Path,
    output_root: Path,
) -> Dict[str, object]:
    if manifest.empty:
        raise ValueError("No spectrum-column manifest rows are available. Save/refresh curation first.")
    required = {"family", "sequence", "temperature_K", "source_txt_path", "y_column_number"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns required for averaging: {missing}")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    all_manifest_path = output_root / "second_pass_average_manifest_all_columns.csv"
    used_manifest_path = output_root / "second_pass_average_manifest_used_columns.csv"
    files_manifest_path = output_root / "second_pass_average_files.csv"
    manifest.to_csv(all_manifest_path, index=False)

    included = manifest[manifest["included_for_all_trend_peaks"].astype(bool)].copy()
    included.to_csv(used_manifest_path, index=False)

    matrix_cache: Dict[Path, pd.DataFrame] = {}
    file_rows: List[Dict[str, object]] = []
    group_cols = ["family", "sequence", "temperature_K"]
    for keys, group in manifest.groupby(group_cols, dropna=False, sort=False):
        family, sequence, temperature = keys
        used = group[group["included_for_all_trend_peaks"].astype(bool)].copy()
        rejected = group[~group["included_for_all_trend_peaks"].astype(bool)].copy()
        if used.empty:
            file_rows.append(
                {
                    "family": family,
                    "sequence": sequence,
                    "temperature_K": temperature,
                    "output_txt": "",
                    "n_used_columns": 0,
                    "n_rejected_columns": int(len(rejected)),
                    "status": "SKIPPED_NO_KEPT_COLUMNS",
                }
            )
            continue

        reference_x: Optional[np.ndarray] = None
        y_columns: List[np.ndarray] = []
        source_labels = []
        for _idx, row in used.iterrows():
            source_text = str(row.get("source_txt_path", "")).strip()
            source_path = Path(source_text) if source_text else None
            if source_path is None or not source_path.exists():
                source_path = locate_spectrum_path(pd.Series(row), spectra_root)
            if source_path not in matrix_cache:
                matrix_cache[source_path] = read_txt_matrix(source_path)
            matrix = matrix_cache[source_path]
            y_index = y_column_number_to_index(row.get("y_column_number", np.nan), matrix)
            x_values = pd.to_numeric(matrix.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
            y_values = pd.to_numeric(matrix.iloc[:, y_index], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(x_values) & np.isfinite(y_values)
            if int(finite.sum()) < 2:
                continue
            if reference_x is None:
                reference_x = x_values[finite].astype(float)
                y_on_ref = y_values[finite].astype(float)
            else:
                if len(x_values) == len(reference_x) and np.allclose(x_values, reference_x, rtol=0.0, atol=1e-8, equal_nan=False):
                    y_on_ref = y_values.astype(float)
                else:
                    y_on_ref = interpolate_to_reference_x(x_values, y_values, reference_x)
            y_columns.append(y_on_ref)
            source_labels.append(f"{source_path.name}:Y{int(y_index + 1):02d}")

        if reference_x is None or not y_columns:
            file_rows.append(
                {
                    "family": family,
                    "sequence": sequence,
                    "temperature_K": temperature,
                    "output_txt": "",
                    "n_used_columns": 0,
                    "n_rejected_columns": int(len(rejected)),
                    "status": "SKIPPED_NO_VALID_NUMERIC_COLUMNS",
                }
            )
            continue

        stack = np.vstack([np.asarray(y, dtype=float) for y in y_columns])
        valid_count = np.sum(np.isfinite(stack), axis=0)
        y_average = np.nanmean(stack, axis=0)
        finite_output = np.isfinite(reference_x) & np.isfinite(y_average) & (valid_count > 0)
        if int(finite_output.sum()) < 2:
            file_rows.append(
                {
                    "family": family,
                    "sequence": sequence,
                    "temperature_K": temperature,
                    "output_txt": "",
                    "n_used_columns": int(len(y_columns)),
                    "n_rejected_columns": int(len(rejected)),
                    "status": "SKIPPED_EMPTY_AVERAGE",
                }
            )
            continue

        family_folder = safe_filename(family)
        sequence_folder = safe_filename(sequence)
        temp_label = temperature_token(temperature)
        output_dir = output_root / family_folder / sequence_folder
        output_dir.mkdir(parents=True, exist_ok=True)
        # Keep names short: the project root is already long enough to hit Windows MAX_PATH.
        output_name = safe_filename(f"TEMP_{temp_label}_AVG.txt", max_len=64)
        output_path = output_dir / output_name
        out_df = pd.DataFrame(
            {
                "Column 1": reference_x[finite_output],
                "Column 2": y_average[finite_output],
            }
        )
        out_df.to_csv(output_path, sep="\t", index=False, float_format="%.10g")
        file_rows.append(
            {
                "family": family,
                "sequence": sequence,
                "temperature_K": temperature,
                "temperature_label": temp_label,
                "output_txt": str(output_path),
                "relative_output_txt": str(output_path.relative_to(output_root)),
                "n_used_columns": int(len(y_columns)),
                "n_rejected_columns": int(len(rejected)),
                "n_points": int(finite_output.sum()),
                "source_columns": "; ".join(source_labels),
                "status": "WROTE_AVERAGED_TXT",
            }
        )

    files_manifest = pd.DataFrame(file_rows)
    files_manifest.to_csv(files_manifest_path, index=False)
    run_manifest = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "source_spectra_root": str(spectra_root),
        "output_root": str(output_root),
        "n_manifest_columns": int(len(manifest)),
        "n_used_columns": int(len(included)),
        "n_written_txt": int((files_manifest.get("status", pd.Series(dtype=str)) == "WROTE_AVERAGED_TXT").sum()),
        "note": "Each TXT is one averaged baseline-corrected spectrum for one family, UP/DOWN sequence, and temperature. Source TXT files were read-only.",
    }
    (output_root / "second_pass_average_run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    return {
        "output_root": output_root,
        "all_manifest": all_manifest_path,
        "used_manifest": used_manifest_path,
        "files_manifest": files_manifest_path,
        "n_written": run_manifest["n_written_txt"],
        "n_used_columns": run_manifest["n_used_columns"],
        "n_total_columns": run_manifest["n_manifest_columns"],
    }


def truthy_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "included"})


def count_csv_rows(path: Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        return 0


def copy_pre_averaging_exam_audit(
    second_pass_root: Path,
    destination_root: Path,
    excluded_families: Set[str],
    excluded_sequences: Set[str],
) -> Dict[str, object]:
    second_pass_root = Path(second_pass_root)
    destination_root = Path(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    manifest_names = [
        "second_pass_average_manifest_all_columns.csv",
        "second_pass_average_manifest_used_columns.csv",
        "second_pass_average_files.csv",
        "second_pass_average_run_manifest.json",
    ]
    copied = []
    for name in manifest_names:
        source = second_pass_root / name
        if source.exists():
            destination = destination_root / name
            shutil.copy2(source, destination)
            copied.append(str(destination))

    all_path = second_pass_root / "second_pass_average_manifest_all_columns.csv"
    used_path = second_pass_root / "second_pass_average_manifest_used_columns.csv"
    files_path = second_pass_root / "second_pass_average_files.csv"
    if not all_path.exists():
        meta = {
            "source_second_pass_root": str(second_pass_root),
            "audit_available": False,
            "copied_files": copied,
            "note": "No second_pass_average_manifest_all_columns.csv was found.",
        }
        (destination_root / "pre_averaging_exam_audit_manifest.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return meta

    all_df = pd.read_csv(all_path)
    used_df = pd.read_csv(used_path) if used_path.exists() else all_df.iloc[0:0].copy()
    files_df = pd.read_csv(files_path) if files_path.exists() else pd.DataFrame()

    all_df["paper_included_current"] = paper_inclusion_mask(all_df, excluded_families, excluded_sequences)
    used_df["paper_included_current"] = paper_inclusion_mask(used_df, excluded_families, excluded_sequences)
    if not files_df.empty and {"family", "sequence"}.issubset(files_df.columns):
        files_df["paper_included_current"] = paper_inclusion_mask(files_df, excluded_families, excluded_sequences)

    if "included_for_all_trend_peaks" in all_df.columns:
        used_mask = truthy_series(all_df["included_for_all_trend_peaks"])
        rejected_df = all_df.loc[~used_mask].copy()
    else:
        identity_cols = [
            col
            for col in ["family", "sequence", "temperature_K", "file_name", "spectrum_in_file", "y_column_number"]
            if col in all_df.columns and col in used_df.columns
        ]
        if identity_cols:
            used_keys = set(used_df[identity_cols].astype(str).agg("\t".join, axis=1))
            all_keys = all_df[identity_cols].astype(str).agg("\t".join, axis=1)
            rejected_df = all_df.loc[~all_keys.isin(used_keys)].copy()
        else:
            rejected_df = all_df.iloc[0:0].copy()

    paper_all = all_df.loc[all_df["paper_included_current"]].copy()
    paper_used = used_df.loc[used_df["paper_included_current"]].copy()
    paper_rejected = rejected_df.loc[rejected_df["paper_included_current"]].copy()
    paper_files = files_df.loc[files_df.get("paper_included_current", pd.Series(False, index=files_df.index))].copy() if not files_df.empty else files_df

    all_df.to_csv(destination_root / "source_column_exam_all_before_paper_exclusion.csv", index=False)
    used_df.to_csv(destination_root / "source_column_exam_used_before_paper_exclusion.csv", index=False)
    rejected_df.to_csv(destination_root / "source_column_exam_removed_before_paper_exclusion.csv", index=False)
    paper_all.to_csv(destination_root / "source_column_exam_all_PAPER_INCLUDED_ONLY.csv", index=False)
    paper_used.to_csv(destination_root / "source_column_exam_used_PAPER_INCLUDED_ONLY.csv", index=False)
    paper_rejected.to_csv(destination_root / "source_column_exam_removed_PAPER_INCLUDED_ONLY.csv", index=False)
    if not files_df.empty:
        files_df.to_csv(destination_root / "source_averaged_txt_files_before_paper_exclusion.csv", index=False)
        paper_files.to_csv(destination_root / "source_averaged_txt_files_PAPER_INCLUDED_ONLY.csv", index=False)

    summary_rows = []
    if {"family", "sequence"}.issubset(all_df.columns):
        groups = sorted(
            set(tuple(x) for x in all_df[["family", "sequence"]].astype(str).to_numpy())
            | set(tuple(x) for x in used_df[["family", "sequence"]].astype(str).to_numpy())
            | set(tuple(x) for x in rejected_df[["family", "sequence"]].astype(str).to_numpy()),
            key=lambda item: (family_sort_key(item[0]), sequence_sort_key(item[1])),
        )
        for family, sequence in groups:
            all_mask = (all_df["family"].astype(str) == family) & (all_df["sequence"].astype(str) == sequence)
            used_mask = (used_df["family"].astype(str) == family) & (used_df["sequence"].astype(str) == sequence)
            rejected_mask = (rejected_df["family"].astype(str) == family) & (rejected_df["sequence"].astype(str) == sequence)
            paper_key = paper_sequence_key(family, sequence)
            summary_rows.append(
                {
                    "family": family,
                    "sequence": sequence,
                    "paper_included_current": bool(family not in excluded_families and paper_key not in excluded_sequences),
                    "source_columns_before_exam": int(all_mask.sum()),
                    "source_columns_kept_for_average": int(used_mask.sum()),
                    "source_columns_removed_before_average": int(rejected_mask.sum()),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(destination_root / "source_column_exam_summary_by_family_sequence.csv", index=False)

    meta = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "source_second_pass_root": str(second_pass_root),
        "audit_available": True,
        "copied_files": copied,
        "n_source_columns_all": int(len(all_df)),
        "n_source_columns_used_for_average": int(len(used_df)),
        "n_source_columns_removed_before_average": int(len(rejected_df)),
        "n_paper_included_source_columns_all": int(len(paper_all)),
        "n_paper_included_source_columns_used_for_average": int(len(paper_used)),
        "n_paper_included_source_columns_removed_before_average": int(len(paper_rejected)),
        "excluded_families_applied_to_paper_included_files": sorted(excluded_families),
        "excluded_sequences_applied_to_paper_included_files": [
            {"family": split_paper_sequence_key(key)[0], "sequence": split_paper_sequence_key(key)[1]}
            for key in sorted(excluded_sequences)
        ],
        "note": (
            "This audit preserves the pre-averaging exam: which original TXT Y-columns were kept or removed "
            "before building the averaged spectra sent to the cluster. PAPER_INCLUDED_ONLY files additionally "
            "remove families/sequences excluded from the paper set."
        ),
    }
    (destination_root / "pre_averaging_exam_audit_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def accepted_refit_identity(row: pd.Series) -> Tuple[str, str, float, str, str, str]:
    return (
        str(row.get("family", "")),
        str(row.get("sequence", "")),
        float(row.get("temperature_K", float("nan"))),
        str(row.get("file_name", "")),
        str(row.get("spectrum_in_file", "")),
        str(row.get("y_column_number", "")),
    )


def match_spectrum_rows(df: pd.DataFrame, row: pd.Series) -> pd.Series:
    def numeric_series_match(series: pd.Series, value: object) -> pd.Series:
        left = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(right):
            return pd.Series(np.isclose(left, float(right), equal_nan=False), index=series.index)
        return series.astype(str) == str(value)

    temp_values = pd.to_numeric(df["temperature_K"], errors="coerce").to_numpy(dtype=float)
    target_temp = pd.to_numeric(pd.Series([row.get("temperature_K", np.nan)]), errors="coerce").iloc[0]
    mask = (
        (df["family"].astype(str) == str(row.get("family", "")))
        & (df["sequence"].astype(str) == str(row.get("sequence", "")))
        & pd.Series(np.isclose(temp_values, float(target_temp), equal_nan=False), index=df.index)
        & (df["file_name"].astype(str) == str(row.get("file_name", "")))
    )
    if "y_column_number" in df.columns and "y_column_number" in row:
        mask &= numeric_series_match(df["y_column_number"], row.get("y_column_number", ""))
    if "spectrum_in_file" in df.columns and "spectrum_in_file" in row:
        mask &= numeric_series_match(df["spectrum_in_file"], row.get("spectrum_in_file", ""))
    return mask


def paper_sequence_key(family: object, sequence: object) -> str:
    return f"{str(family)}\t{str(sequence)}"


def split_paper_sequence_key(key: str) -> Tuple[str, str]:
    if "\t" in str(key):
        family, sequence = str(key).split("\t", 1)
        return family, sequence
    if "::" in str(key):
        family, sequence = str(key).split("::", 1)
        return family, sequence
    return "", str(key)


def paper_inclusion_mask(
    df: pd.DataFrame,
    excluded_families: Set[str],
    excluded_sequences: Set[str],
) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype=bool, index=df.index)
    mask = pd.Series(True, index=df.index)
    if "family" in df.columns and excluded_families:
        mask &= ~df["family"].astype(str).isin(excluded_families)
    if {"family", "sequence"}.issubset(df.columns) and excluded_sequences:
        sequence_keys = [
            paper_sequence_key(family, sequence)
            for family, sequence in zip(df["family"].astype(str), df["sequence"].astype(str))
        ]
        mask &= ~pd.Series(sequence_keys, index=df.index).isin(excluded_sequences)
    return mask


def drop_excluded_temperature_points(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not {"family", "sequence", "temperature_K"}.issubset(df.columns):
        return df.copy()
    filtered = df.copy()
    temps = pd.to_numeric(filtered["temperature_K"], errors="coerce")
    for family, sequence, temperature in EXCLUDED_TEMPERATURE_POINTS:
        keep = ~(
            filtered["family"].astype(str).eq(family)
            & filtered["sequence"].astype(str).eq(sequence)
            & np.isclose(temps, temperature)
        )
        filtered = filtered[keep].copy()
        temps = pd.to_numeric(filtered["temperature_K"], errors="coerce")
    return filtered


def patch_long_results(long_df: pd.DataFrame, accepted_refits: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    patched = long_df.copy()
    n_updates = 0
    object_update_cols = {
        "quality_flag",
        "refit_method",
        "trust_region_success",
        "trust_region_message",
        "source_txt_path",
    }

    def assign_value(mask: pd.Series, col: str, value: object) -> None:
        if col not in patched.columns:
            if col in object_update_cols or isinstance(value, (str, bool)):
                patched[col] = pd.Series([None] * len(patched), dtype="object")
            else:
                patched[col] = np.nan
        if col in object_update_cols or isinstance(value, (str, bool)):
            patched[col] = patched[col].astype("object")
        patched.loc[mask, col] = value

    update_cols = [
        "height",
        "position_cm-1",
        "width_fwhm_cm-1",
        "height_std",
        "position_std_cm-1",
        "width_fwhm_std_cm-1",
        "x_bound_min",
        "x_bound_max",
        "width_bound_min",
        "width_bound_max",
        "r_squared",
        "rmse",
        "weighted_sse",
        "weighted_sse_data",
        "anti_burial_sse",
        "n_points_fit",
        "fit_x_min",
        "fit_x_max",
        "bootstrap_requested",
        "bootstrap_success",
        "quality_flag",
        "refit_method",
        "trust_region_success",
        "trust_region_message",
        "source_txt_path",
    ]
    for _idx, refit in accepted_refits.iterrows():
        mask = match_spectrum_rows(patched, refit) & (patched["peak_id"].astype(str) == str(refit.get("peak_id", "")))
        if not mask.any():
            continue
        for col in update_cols:
            if col in refit.index:
                assign_value(mask, col, refit[col])
        n_updates += int(mask.sum())
    return patched, n_updates


def patch_wide_results(wide_df: pd.DataFrame, accepted_refits: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    patched = wide_df.copy()
    n_updates = 0
    object_update_cols = {
        "quality_flag",
        "refit_method",
        "trust_region_success",
        "trust_region_message",
        "source_txt_path",
    }

    def assign_value(mask: pd.Series, col: str, value: object) -> None:
        if col not in patched.columns:
            if col in object_update_cols or isinstance(value, (str, bool)):
                patched[col] = pd.Series([None] * len(patched), dtype="object")
            else:
                patched[col] = np.nan
        if col in object_update_cols or isinstance(value, (str, bool)):
            patched[col] = patched[col].astype("object")
        patched.loc[mask, col] = value

    base_cols = [
        "r_squared",
        "rmse",
        "weighted_sse",
        "weighted_sse_data",
        "anti_burial_sse",
        "n_points_fit",
        "fit_x_min",
        "fit_x_max",
        "bootstrap_requested",
        "bootstrap_success",
        "quality_flag",
        "refit_method",
        "trust_region_success",
        "trust_region_message",
        "source_txt_path",
    ]
    for _identity, group in accepted_refits.groupby(
        ["family", "sequence", "temperature_K", "file_name", "spectrum_in_file", "y_column_number"], dropna=False
    ):
        first = group.iloc[0]
        mask = match_spectrum_rows(patched, first)
        if not mask.any():
            continue
        for col in base_cols:
            if col in first.index:
                assign_value(mask, col, first[col])
        for _idx, row in group.iterrows():
            peak = str(row["peak_id"])
            for src, suffix in [
                ("height", "height"),
                ("position_cm-1", "position_cm-1"),
                ("width_fwhm_cm-1", "width_fwhm_cm-1"),
                ("height_std", "height_std"),
                ("position_std_cm-1", "position_std_cm-1"),
                ("width_fwhm_std_cm-1", "width_fwhm_std_cm-1"),
                ("x_bound_min", "x_bound_min"),
                ("x_bound_max", "x_bound_max"),
                ("width_bound_min", "width_bound_min"),
                ("width_bound_max", "width_bound_max"),
            ]:
                col = f"{peak}_{suffix}"
                if col in patched.columns and src in row.index:
                    assign_value(mask, col, row[src])
        n_updates += int(mask.sum())
    return patched, n_updates


def family_stats_from_long(long_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def numeric_mean(group: pd.DataFrame, column: str) -> float:
        if column not in group.columns:
            return float("nan")
        return float(pd.to_numeric(group[column], errors="coerce").mean())

    stats_rows = []
    group_cols = ["sequence", "temperature_K", "peak_id", "peak_label"]
    for keys, group in long_df.groupby(group_cols, dropna=False, sort=True):
        row = dict(zip(group_cols, keys))
        row["n_spectra"] = int(len(group))
        row["height_mean"] = float(pd.to_numeric(group["height"], errors="coerce").mean())
        row["height_std"] = float(pd.to_numeric(group["height"], errors="coerce").std(ddof=1)) if len(group) > 1 else 0.0
        row["position_mean_cm_1"] = float(pd.to_numeric(group["position_cm-1"], errors="coerce").mean())
        row["position_std_cm_1"] = (
            float(pd.to_numeric(group["position_cm-1"], errors="coerce").std(ddof=1)) if len(group) > 1 else 0.0
        )
        row["width_fwhm_mean_cm_1"] = float(pd.to_numeric(group["width_fwhm_cm-1"], errors="coerce").mean())
        row["width_fwhm_std_cm_1"] = (
            float(pd.to_numeric(group["width_fwhm_cm-1"], errors="coerce").std(ddof=1)) if len(group) > 1 else 0.0
        )
        row["r_squared_mean"] = numeric_mean(group, "r_squared")
        row["rmse_mean"] = numeric_mean(group, "rmse")
        stats_rows.append(row)
    by_sequence = pd.DataFrame(stats_rows)

    all_rows = []
    group_cols_all = ["temperature_K", "peak_id", "peak_label"]
    for keys, group in long_df.groupby(group_cols_all, dropna=False, sort=True):
        row = dict(zip(group_cols_all, keys))
        row["n_spectra"] = int(len(group))
        row["height_mean"] = float(pd.to_numeric(group["height"], errors="coerce").mean())
        row["height_std"] = float(pd.to_numeric(group["height"], errors="coerce").std(ddof=1)) if len(group) > 1 else 0.0
        row["position_mean_cm_1"] = float(pd.to_numeric(group["position_cm-1"], errors="coerce").mean())
        row["position_std_cm_1"] = (
            float(pd.to_numeric(group["position_cm-1"], errors="coerce").std(ddof=1)) if len(group) > 1 else 0.0
        )
        row["width_fwhm_mean_cm_1"] = float(pd.to_numeric(group["width_fwhm_cm-1"], errors="coerce").mean())
        row["width_fwhm_std_cm_1"] = (
            float(pd.to_numeric(group["width_fwhm_cm-1"], errors="coerce").std(ddof=1)) if len(group) > 1 else 0.0
        )
        row["r_squared_mean"] = numeric_mean(group, "r_squared")
        row["rmse_mean"] = numeric_mean(group, "rmse")
        all_rows.append(row)
    all_sequences = pd.DataFrame(all_rows)
    return by_sequence, all_sequences


def write_family_xlsx(path: Path, long_df: pd.DataFrame, wide_df: pd.DataFrame) -> None:
    stats_by_sequence, stats_all_sequences = family_stats_from_long(long_df)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        long_df.to_excel(writer, sheet_name="all_long", index=False)
        wide_df.to_excel(writer, sheet_name="all_wide", index=False)
        stats_by_sequence.to_excel(writer, sheet_name="stats_by_sequence", index=False)
        stats_all_sequences.to_excel(writer, sheet_name="stats_all_sequences", index=False)
        for temp, group in long_df.groupby("temperature_K", sort=True):
            if not math.isfinite(float(temp)):
                continue
            sheet = f"T_{float(temp):g}K".replace(".", "p")[:31]
            group.to_excel(writer, sheet_name=sheet, index=False)


def export_complete_refitted_folder(
    source_root: Path,
    target_root: Path,
    accepted_refits: pd.DataFrame,
) -> Path:
    if accepted_refits.empty:
        raise ValueError("No accepted PC refits to export.")
    source_root = Path(source_root)
    target_root = Path(target_root)
    if not source_root.exists():
        raise FileNotFoundError(f"Original fitted root does not exist: {source_root}")
    target_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root, target_root, dirs_exist_ok=True)

    manifest_rows = []
    for family, family_refits in accepted_refits.groupby("family", sort=False):
        family_dir = target_root / str(family)
        long_path = family_dir / f"{family}_long_results.csv"
        wide_path = family_dir / f"{family}_wide_results.csv"
        xlsx_path = family_dir / f"{family}_Lorentzian_fit_results.xlsx"
        if not long_path.exists() or not wide_path.exists():
            continue
        long_df = pd.read_csv(long_path)
        wide_df = pd.read_csv(wide_path)
        patched_long, long_updates = patch_long_results(long_df, family_refits)
        patched_wide, wide_updates = patch_wide_results(wide_df, family_refits)
        patched_long.to_csv(long_path, index=False)
        patched_wide.to_csv(wide_path, index=False)
        write_family_xlsx(xlsx_path, patched_long, patched_wide)
        for _idx, row in family_refits.iterrows():
            manifest_rows.append(
                {
                    "family": family,
                    "sequence": row.get("sequence", ""),
                    "temperature_K": row.get("temperature_K", ""),
                    "file_name": row.get("file_name", ""),
                    "spectrum_in_file": row.get("spectrum_in_file", ""),
                    "y_column_number": row.get("y_column_number", ""),
                    "peak_id": row.get("peak_id", ""),
                    "height": row.get("height", ""),
                    "position_cm-1": row.get("position_cm-1", ""),
                    "width_fwhm_cm-1": row.get("width_fwhm_cm-1", ""),
                    "r_squared": row.get("r_squared", ""),
                    "rmse": row.get("rmse", ""),
                    "source_txt_path": row.get("source_txt_path", ""),
                    "long_rows_updated_for_family": long_updates,
                    "wide_rows_updated_for_family": wide_updates,
                }
            )
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = target_root / "pc_refit_accepted_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    meta = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "source_root": str(source_root),
        "target_root": str(target_root),
        "accepted_peak_rows": int(len(accepted_refits)),
        "families": sorted(str(f) for f in accepted_refits["family"].dropna().unique()),
        "note": "Original fitted results were copied, then accepted PC trust-region rows were patched into long/wide CSVs and XLSX sheets. HTML reports are copied from the original run and not regenerated.",
    }
    (target_root / "pc_refit_update_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return target_root


def export_paper_clean_fitted_folder(
    source_root: Path,
    target_root: Path,
    accepted_refits: pd.DataFrame,
    rejected_keys: Set[str],
    excluded_families: Set[str],
    excluded_sequences: Set[str],
) -> Path:
    source_root = Path(source_root)
    if not source_root.exists():
        raise FileNotFoundError(f"Source fitted root does not exist: {source_root}")
    source_refit_manifest = source_root / "pc_refit_accepted_manifest.csv"
    source_refit_rows = count_csv_rows(source_refit_manifest)
    target_root = unique_run_folder(Path(target_root))
    target_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root, target_root)

    accepted_refits = accepted_refits.copy() if accepted_refits is not None else pd.DataFrame()
    if not accepted_refits.empty:
        accepted_refits = accepted_refits.drop_duplicates(
            subset=["family", "sequence", "temperature_K", "file_name", "spectrum_in_file", "y_column_number", "peak_id"],
            keep="last",
        )

    rejected_keys = set(str(key) for key in rejected_keys)
    manifest_rows = []
    for family_dir in sorted([path for path in target_root.iterdir() if path.is_dir()], key=lambda path: path.name):
        family = family_dir.name
        long_candidates = list(family_dir.glob("*_long_results.csv"))
        wide_candidates = list(family_dir.glob("*_wide_results.csv"))
        if family in excluded_families:
            shutil.rmtree(family_dir)
            manifest_rows.append(
                {
                    "family": family,
                    "status": "REMOVED_EXCLUDED_FAMILY",
                    "long_rows_before": 0,
                    "long_rows_after": 0,
                    "wide_rows_before": 0,
                    "wide_rows_after": 0,
                    "staged_refit_rows_applied": 0,
                    "excluded_sequences": "",
                }
            )
            continue
        if not long_candidates or not wide_candidates:
            continue

        long_path = long_candidates[0]
        wide_path = wide_candidates[0]
        xlsx_candidates = list(family_dir.glob("*_Lorentzian_fit_results.xlsx"))
        xlsx_path = xlsx_candidates[0] if xlsx_candidates else family_dir / f"{family}_Lorentzian_fit_results.xlsx"
        long_df = pd.read_csv(long_path)
        wide_df = pd.read_csv(wide_path)
        long_before = int(len(long_df))
        wide_before = int(len(wide_df))

        family_refits = (
            accepted_refits[accepted_refits["family"].astype(str) == family].copy()
            if not accepted_refits.empty and "family" in accepted_refits.columns
            else pd.DataFrame()
        )
        staged_refit_rows = int(len(family_refits))
        if not family_refits.empty:
            long_df, _long_updates = patch_long_results(long_df, family_refits)
            wide_df, _wide_updates = patch_wide_results(wide_df, family_refits)

        excluded_for_family = {
            split_paper_sequence_key(key)[1]
            for key in excluded_sequences
            if split_paper_sequence_key(key)[0] == family
        }

        long_mask = paper_inclusion_mask(long_df, excluded_families, excluded_sequences)
        if rejected_keys:
            long_keys = long_df.apply(fit_row_key, axis=1).astype(str)
            long_mask &= ~long_keys.isin(rejected_keys)
        clean_long = long_df.loc[long_mask].copy()

        wide_mask = paper_inclusion_mask(wide_df, excluded_families, excluded_sequences)
        if rejected_keys and not long_df.empty:
            rejected_rows = long_df[long_df.apply(fit_row_key, axis=1).astype(str).isin(rejected_keys)]
            for _idx, rejected_row in rejected_rows.iterrows():
                try:
                    wide_mask &= ~match_spectrum_rows(wide_df, rejected_row)
                except Exception:
                    continue
        clean_wide = wide_df.loc[wide_mask].copy()

        if clean_long.empty or clean_wide.empty:
            shutil.rmtree(family_dir)
            status = "REMOVED_NO_INCLUDED_ROWS"
        else:
            clean_long.to_csv(long_path, index=False)
            clean_wide.to_csv(wide_path, index=False)
            write_family_xlsx(xlsx_path, clean_long, clean_wide)
            for html_path in family_dir.glob("*.html"):
                try:
                    html_path.unlink()
                except OSError:
                    pass
            status = "KEPT_CLEANED"

        manifest_rows.append(
            {
                "family": family,
                "status": status,
                "long_rows_before": long_before,
                "long_rows_after": int(len(clean_long)),
                "wide_rows_before": wide_before,
                "wide_rows_after": int(len(clean_wide)),
                "staged_refit_rows_applied": staged_refit_rows,
                "excluded_sequences": ";".join(sorted(excluded_for_family, key=sequence_sort_key)),
            }
        )

    for html_path in target_root.glob("*.html"):
        try:
            html_path.unlink()
        except OSError:
            pass

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = target_root / "paper_clean_export_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    sequence_rows = [
        {"family": split_paper_sequence_key(key)[0], "sequence": split_paper_sequence_key(key)[1]}
        for key in sorted(excluded_sequences)
    ]
    pd.DataFrame({"family": sorted(excluded_families)}).to_csv(
        target_root / "paper_clean_excluded_families.csv", index=False
    )
    pd.DataFrame(sequence_rows, columns=["family", "sequence"]).to_csv(
        target_root / "paper_clean_excluded_sequences.csv", index=False
    )
    meta = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "source_root": str(source_root),
        "target_root": str(target_root),
        "rejected_peak_rows_removed": int(len(rejected_keys)),
        "excluded_families": sorted(excluded_families),
        "excluded_sequences": [
            {"family": split_paper_sequence_key(key)[0], "sequence": split_paper_sequence_key(key)[1]}
            for key in sorted(excluded_sequences)
        ],
        "source_pc_refit_peak_rows_already_included": int(source_refit_rows),
        "source_pc_refit_manifest": str(source_refit_manifest) if source_refit_manifest.exists() else "",
        "staged_refit_rows_applied_this_export": int(len(accepted_refits)),
        "note": "Clean paper folder was copied from the source fitted root, staged PC refits were patched if present in this session, excluded families/sequences and rejected spectrum-columns were removed, and XLSX files were regenerated. PC refits already present in the source fitted root remain included. Source fitted data were not modified.",
    }
    (target_root / "paper_clean_export_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return target_root


def expanded_y_limits(values: np.ndarray, base_limits: Tuple[float, float], pad_fraction: float = 0.08) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    ymin, ymax = map(float, base_limits)
    if values.size == 0:
        return ymin, ymax
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    span = max(ymax - ymin, vmax - vmin, 1.0)
    pad = pad_fraction * span
    return min(ymin, vmin - pad), max(ymax, vmax + pad)


def write_bound_hit_reports(
    results: pd.DataFrame,
    output_dir: Path,
    bound_tol: float,
    height_bound_rel_tol: float,
) -> None:
    required = {
        "family",
        "sequence",
        "temperature_K",
        "file_name",
        "y_column_number",
        "peak_id",
        "position_cm-1",
        "width_fwhm_cm-1",
        "x_bound_min",
        "x_bound_max",
        "width_bound_min",
        "width_bound_max",
    }
    missing = sorted(required - set(results.columns))
    if missing:
        print(f"Skipping bound-hit audit; missing columns: {missing}")
        return

    df = results[results["family"].astype(str).isin(TARGET_SAMPLE_FOLDERS)].copy()
    for col in [
        "temperature_K",
        "y_column_number",
        "position_cm-1",
        "width_fwhm_cm-1",
        "x_bound_min",
        "x_bound_max",
        "width_bound_min",
        "width_bound_max",
        "height",
        "height_upper_scaled",
        "scale_used",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    finite_width = (
        np.isfinite(df["width_fwhm_cm-1"])
        & np.isfinite(df["width_bound_min"])
        & np.isfinite(df["width_bound_max"])
    )
    finite_pos = (
        np.isfinite(df["position_cm-1"])
        & np.isfinite(df["x_bound_min"])
        & np.isfinite(df["x_bound_max"])
    )
    df["width_hit_min"] = finite_width & (df["width_fwhm_cm-1"] <= df["width_bound_min"] + bound_tol)
    df["width_hit_max"] = finite_width & (df["width_fwhm_cm-1"] >= df["width_bound_max"] - bound_tol)
    df["width_hit_any"] = df["width_hit_min"] | df["width_hit_max"]
    df["position_hit_min"] = finite_pos & (df["position_cm-1"] <= df["x_bound_min"] + bound_tol)
    df["position_hit_max"] = finite_pos & (df["position_cm-1"] >= df["x_bound_max"] - bound_tol)
    df["position_hit_any"] = df["position_hit_min"] | df["position_hit_max"]
    if {"height", "height_upper_scaled", "scale_used"}.issubset(df.columns):
        df["height_bound_min_raw"] = 0.0
        df["height_bound_max_raw"] = df["height_upper_scaled"] * df["scale_used"]
        finite_height = (
            np.isfinite(df["height"])
            & np.isfinite(df["height_bound_max_raw"])
            & (df["height_bound_max_raw"] > 0)
        )
        raw_tol = np.maximum(np.abs(df["height_bound_max_raw"]) * float(height_bound_rel_tol), 1e-12)
        df["height_hit_min"] = finite_height & (df["height"] <= raw_tol)
        df["height_hit_max"] = finite_height & (df["height"] >= df["height_bound_max_raw"] - raw_tol)
        df["height_hit_any"] = df["height_hit_min"] | df["height_hit_max"]
    else:
        df["height_bound_min_raw"] = np.nan
        df["height_bound_max_raw"] = np.nan
        df["height_hit_min"] = False
        df["height_hit_max"] = False
        df["height_hit_any"] = False
    df["width_bound_side"] = np.select(
        [df["width_hit_min"], df["width_hit_max"]],
        ["MIN", "MAX"],
        default="",
    )
    df["position_bound_side"] = np.select(
        [df["position_hit_min"], df["position_hit_max"]],
        ["MIN", "MAX"],
        default="",
    )
    df["height_bound_side"] = np.select(
        [df["height_hit_min"], df["height_hit_max"]],
        ["MIN", "MAX"],
        default="",
    )

    detail_cols = [
        "family",
        "sequence",
        "temperature_K",
        "file_name",
        "y_column_number",
        "peak_id",
        "width_fwhm_cm-1",
        "width_bound_min",
        "width_bound_max",
        "width_bound_side",
        "position_cm-1",
        "x_bound_min",
        "x_bound_max",
        "position_bound_side",
        "height",
        "height_bound_min_raw",
        "height_bound_max_raw",
        "height_bound_side",
        "r_squared",
        "rmse",
        "quality_flag",
    ]
    detail_cols = [col for col in detail_cols if col in df.columns]
    detail_path = output_dir / "Gamma_T_Comparison_bound_hit_details.csv"
    df[detail_cols].to_csv(detail_path, index=False)

    summary = (
        df.groupby(["family", "peak_id"], dropna=False)
        .agg(
            n_fits=("peak_id", "size"),
            width_min_hits=("width_hit_min", "sum"),
            width_max_hits=("width_hit_max", "sum"),
            width_any_hits=("width_hit_any", "sum"),
            position_min_hits=("position_hit_min", "sum"),
            position_max_hits=("position_hit_max", "sum"),
            position_any_hits=("position_hit_any", "sum"),
            height_min_hits=("height_hit_min", "sum"),
            height_max_hits=("height_hit_max", "sum"),
            height_any_hits=("height_hit_any", "sum"),
            median_width=("width_fwhm_cm-1", "median"),
            median_width_bound_min=("width_bound_min", "median"),
            median_width_bound_max=("width_bound_max", "median"),
            median_position=("position_cm-1", "median"),
            median_x_bound_min=("x_bound_min", "median"),
            median_x_bound_max=("x_bound_max", "median"),
            median_height=("height", "median"),
            median_height_bound_min_raw=("height_bound_min_raw", "median"),
            median_height_bound_max_raw=("height_bound_max_raw", "median"),
        )
        .reset_index()
    )
    summary["width_any_hit_pct"] = 100.0 * summary["width_any_hits"] / summary["n_fits"].clip(lower=1)
    summary["width_min_hit_pct"] = 100.0 * summary["width_min_hits"] / summary["n_fits"].clip(lower=1)
    summary["width_max_hit_pct"] = 100.0 * summary["width_max_hits"] / summary["n_fits"].clip(lower=1)
    summary["position_any_hit_pct"] = 100.0 * summary["position_any_hits"] / summary["n_fits"].clip(lower=1)
    summary["height_any_hit_pct"] = 100.0 * summary["height_any_hits"] / summary["n_fits"].clip(lower=1)
    summary["height_min_hit_pct"] = 100.0 * summary["height_min_hits"] / summary["n_fits"].clip(lower=1)
    summary["height_max_hit_pct"] = 100.0 * summary["height_max_hits"] / summary["n_fits"].clip(lower=1)
    summary["family_key"] = summary["family"].map(family_key)
    summary["_family_order"] = summary["family"].map(lambda name: family_sort_key(str(name))[0])
    summary = summary.sort_values(["_family_order", "peak_id"]).drop(columns=["_family_order"])
    summary_path = output_dir / "Gamma_T_Comparison_bound_hit_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Exported bound-hit details to {detail_path}")
    print(f"Exported bound-hit summary to {summary_path}")
    print("Worst width-bound hit rates:")
    worst = summary.sort_values("width_any_hit_pct", ascending=False).head(12)
    print(
        worst[
            [
                "family",
                "peak_id",
                "n_fits",
                "width_any_hits",
                "width_any_hit_pct",
                "width_min_hits",
                "width_max_hits",
            ]
        ].to_string(index=False)
    )
    print("Worst position-bound hit rates:")
    print(
        summary.sort_values("position_any_hit_pct", ascending=False)
        .head(8)[["family", "peak_id", "n_fits", "position_any_hits", "position_any_hit_pct"]]
        .to_string(index=False)
    )
    print("Worst raw-height-bound hit rates:")
    print(
        summary.sort_values("height_any_hit_pct", ascending=False)
        .head(8)[["family", "peak_id", "n_fits", "height_any_hits", "height_any_hit_pct"]]
        .to_string(index=False)
    )


def omega0_from_low_temperature(df: pd.DataFrame, family: str, peak: str) -> float:
    subset = df[(df["family"].astype(str) == family) & (df["peak_id"].astype(str) == peak)].copy()
    subset = subset[np.isfinite(subset["position_cm-1"])]
    if subset.empty:
        return float("nan")
    min_temp = subset["temperature_K"].min()
    low = subset[np.isclose(subset["temperature_K"], min_temp)]
    return float(np.nanmedian(low["position_cm-1"]))


def prepared_fit_data(
    x: np.ndarray,
    y: np.ndarray,
    sigma: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    data = pd.DataFrame({"x": x[mask], "y": y[mask]})
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        data["sigma"] = sigma[mask]
    if data.empty:
        return np.array([]), np.array([]), None

    agg = {"y": "mean"}
    if "sigma" in data.columns:
        agg["sigma"] = "mean"
    data = data.groupby("x", as_index=False).agg(agg).sort_values("x")
    x_fit = data["x"].to_numpy(dtype=float)
    y_fit = data["y"].to_numpy(dtype=float)
    sigma_fit = None
    if "sigma" in data.columns:
        sigma_fit = data["sigma"].to_numpy(dtype=float)
        positive = sigma_fit[np.isfinite(sigma_fit) & (sigma_fit > 0)]
        if positive.size:
            fallback = float(np.nanmedian(positive))
            sigma_fit = np.where(np.isfinite(sigma_fit) & (sigma_fit > 0), sigma_fit, fallback)
        else:
            sigma_fit = None
    return x_fit, y_fit, sigma_fit


def fit_three_phonon_curve(
    x: np.ndarray,
    y: np.ndarray,
    omega0: float,
    sigma: Optional[np.ndarray] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[float, float]]]:
    x_fit, y_fit, sigma_fit = prepared_fit_data(x, y, sigma)
    if x_fit.size < 3 or not math.isfinite(omega0) or curve_fit is None:
        return None

    t_smooth = np.linspace(float(np.nanmin(x_fit)), float(np.nanmax(x_fit)), 240)
    gamma_min = float(np.nanmin(y_fit))
    gamma_span = float(np.nanmax(y_fit) - np.nanmin(y_fit))
    p0 = [max(0.0, gamma_min - 0.2 * gamma_span), max(0.01, gamma_span * 0.4)]

    def local_model(temp, gamma0, c3):
        return gamma_model(temp, gamma0, c3, omega0)

    try:
        popt, _pcov = curve_fit(
            local_model,
            x_fit,
            y_fit,
            p0=p0,
            sigma=sigma_fit,
            absolute_sigma=False,
            bounds=([0.0, -500.0], [500.0, 500.0]),
            maxfev=20000,
        )
        return t_smooth, local_model(t_smooth, *popt), (float(popt[0]), float(popt[1]))
    except Exception as exc:
        print(f"Three-phonon fit failed: {exc}")
        return None


def fit_quadratic_curve(
    x: np.ndarray,
    y: np.ndarray,
    sigma: Optional[np.ndarray] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[float, float]]]:
    x_fit, y_fit, sigma_fit = prepared_fit_data(x, y, sigma)
    if x_fit.size < 3:
        return None

    t_smooth = np.linspace(float(np.nanmin(x_fit)), float(np.nanmax(x_fit)), 240)
    degree = min(2, x_fit.size - 1)
    if sigma_fit is not None:
        weights = 1.0 / np.clip(sigma_fit, np.nanmedian(sigma_fit[sigma_fit > 0]), np.inf)
        coeff = np.polyfit(x_fit, y_fit, deg=degree, w=weights)
    else:
        coeff = np.polyfit(x_fit, y_fit, deg=degree)
    return t_smooth, np.polyval(coeff, t_smooth), (float("nan"), float("nan"))


def uncertainty_values(subset: pd.DataFrame, source: str) -> Optional[np.ndarray]:
    if source == "none":
        return None
    column = "width_group_sem_cm-1" if source == "sem" else "width_group_std_cm-1"
    if column not in subset.columns:
        return None
    values = pd.to_numeric(subset[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).any() or np.nanmax(values) <= 0:
        return None
    return values


def parameter_uncertainty_values(subset: pd.DataFrame, parameter: str, source: str) -> Optional[np.ndarray]:
    if source == "none":
        return None
    if f"{parameter}_total_{source}" in subset.columns:
        column = f"{parameter}_total_{source}"
    elif source == "sem" and f"{parameter}_group_sem" in subset.columns:
        column = f"{parameter}_group_sem"
    elif source == "std" and f"{parameter}_total_std_visual" in subset.columns:
        column = f"{parameter}_total_std_visual"
    elif source == "std" and f"{parameter}_group_std" in subset.columns:
        column = f"{parameter}_group_std"
    else:
        spec = PARAMETER_SPECS[parameter]
        column = spec["std"]
        if column not in subset.columns:
            return None
    values = pd.to_numeric(subset[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).any() or np.nanmax(values) <= 0:
        return None
    return values


def plot_family_scatter(
    ax,
    df: pd.DataFrame,
    family: str,
    peak: str,
    marker_size: float,
    line_color: str = "black",
    fit_mode: str = "both",
    fit_sigma: str = "sem",
    error_bar: str = "std",
    error_alpha: float = 0.20,
) -> None:
    subset = df[(df["family"].astype(str) == family) & (df["peak_id"].astype(str) == peak)]
    if subset.empty:
        return
    sequences = sorted(subset["sequence"].astype(str).dropna().unique(), key=sequence_sort_key)
    for sequence in sequences:
        seq_subset = subset[subset["sequence"].astype(str) == sequence].sort_values("temperature_K")
        if seq_subset.empty:
            continue
        style = sequence_marker_style(family, sequence, sequences)
        x_values = seq_subset["temperature_K"].to_numpy(dtype=float)
        y_values = seq_subset["width_fwhm_cm-1"].to_numpy(dtype=float)
        yerr = uncertainty_values(seq_subset, error_bar)
        if yerr is not None:
            ax.errorbar(
                x_values,
                y_values,
                yerr=yerr,
                fmt="none",
                ecolor=style["edgecolors"],
                elinewidth=0.85,
                alpha=error_alpha,
                capsize=0,
                zorder=2,
            )
        ax.scatter(
            x_values,
            y_values,
            marker=style["marker"],
            s=marker_size,
            facecolors=style["facecolors"],
            edgecolors=style["edgecolors"],
            linewidths=style["linewidths"],
            alpha=0.9,
            zorder=3,
        )
        omega0 = omega0_from_low_temperature(seq_subset, family, peak)
        fit_uncertainty = uncertainty_values(seq_subset, fit_sigma)
        curve_color = style["edgecolors"] if len(sequences) > 1 else line_color
        if fit_mode in {"three-phonon", "both"}:
            fit = fit_three_phonon_curve(x_values, y_values, omega0, fit_uncertainty)
            if fit is not None:
                t_smooth, gamma_smooth, _params = fit
                ax.plot(t_smooth, gamma_smooth, color=curve_color, linewidth=1.9, zorder=4)
        if fit_mode in {"quadratic", "both"}:
            fit = fit_quadratic_curve(x_values, y_values, fit_uncertainty)
            if fit is not None:
                t_smooth, gamma_smooth, _params = fit
                ax.plot(t_smooth, gamma_smooth, color="#777777", linewidth=1.35, linestyle="--", zorder=4)


def padded_limits(values: np.ndarray, pad_fraction: float = 0.18, fallback: Tuple[float, float] = (0.0, 1.0)) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return fallback
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmin == vmax:
        pad = max(abs(vmin) * 0.05, 0.5)
    else:
        pad = pad_fraction * (vmax - vmin)
    return vmin - pad, vmax + pad


def build_family_legend() -> List[Line2D]:
    handles = []
    for family in sorted(TARGET_SAMPLE_FOLDERS, key=family_sort_key):
        style = marker_style(family)
        handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                linestyle="None",
                label=family_label(family),
                markerfacecolor=style["facecolors"],
                markeredgecolor=style["edgecolors"],
                markeredgewidth=style["linewidths"],
                markersize=7,
            )
        )
    return handles


def plot_g_overlay(df: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    fig, ax = plt.subplots(figsize=(args.overlay_width, args.overlay_height), dpi=args.dpi)
    families = sorted(TARGET_SAMPLE_FOLDERS, key=family_sort_key)
    for family in families:
        key = family_key(family)
        color = SAMPLE_COLORS.get(key, "black")
        plot_family_scatter(
            ax,
            df,
            family,
            "G",
            args.marker_size,
            line_color=color,
            fit_mode=args.fit_mode,
            fit_sigma=args.fit_sigma,
            error_bar=args.error_bar,
            error_alpha=args.error_alpha,
        )

    g_values = df[df["peak_id"].astype(str) == "G"]["width_fwhm_cm-1"].to_numpy(dtype=float)
    ax.set_ylim(*padded_limits(g_values, pad_fraction=0.20, fallback=(0, 20)))
    ax.set_xlim(args.temp_min, args.temp_max)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"G FWHM (cm$^{-1}$)")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.legend(
        handles=build_family_legend(),
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        fontsize=args.legend_font,
        handletextpad=0.45,
    )
    style_single_axis(ax)
    fig.subplots_adjust(left=0.16, right=0.70, bottom=0.17, top=0.96)
    save_figure(fig, output_dir / "Gamma_T_Comparison_G_all_families", args.formats, args.dpi)
    plt.close(fig)


def plot_reference_grid(df: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    peaks = ["RBLM", "D", "G"]
    families = sorted(TARGET_SAMPLE_FOLDERS, key=family_sort_key)
    nrows = len(peaks)
    ncols = len(families)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(args.grid_width, args.grid_height),
        sharex=True,
        squeeze=False,
        dpi=args.dpi,
    )
    fig.subplots_adjust(left=0.080, right=0.985, bottom=0.105, top=0.905, wspace=0.16, hspace=0.12)

    for row_idx, peak in enumerate(peaks):
        for col_idx, family in enumerate(families):
            ax = axes[row_idx, col_idx]
            plot_family_scatter(
                ax,
                df,
                family,
                peak,
                args.grid_marker_size,
                line_color="black",
                fit_mode=args.fit_mode,
                fit_sigma=args.fit_sigma,
                error_bar=args.error_bar,
                error_alpha=args.error_alpha,
            )
            ax.set_xlim(args.temp_min, args.temp_max)
            values = df[
                (df["family"].astype(str) == family) & (df["peak_id"].astype(str) == peak)
            ]["width_fwhm_cm-1"].to_numpy(dtype=float)
            if family in INDEPENDENT_Y_FAMILIES:
                ax.set_ylim(*padded_limits(values, pad_fraction=0.18, fallback=FIXED_NON_ALIGNED_RO_Y_LIMITS[peak]))
            else:
                ax.set_ylim(*expanded_y_limits(values, FIXED_NON_ALIGNED_RO_Y_LIMITS[peak], pad_fraction=0.08))
            style_single_axis(ax)
            if row_idx == 0:
                ax.set_title(family_label(family), fontsize=args.grid_title_font, pad=8)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4 if peak == "G" else 5))
            if col_idx == 0:
                ax.set_ylabel(f"{PEAK_LABELS[peak]} FWHM (cm$^{{-1}}$)")
            else:
                ax.tick_params(labelleft=False)
            if row_idx == nrows - 1:
                ax.set_xlabel("T (K)")
            else:
                ax.tick_params(labelbottom=False)

    axes[0, 0].legend(
        handles=build_family_legend(),
        loc="upper left",
        frameon=False,
        fontsize=max(7.0, args.legend_font - 2.0),
        handletextpad=0.25,
        labelspacing=0.25,
        borderaxespad=0.25,
    )
    axes[0, min(1, ncols - 1)].legend(
        handles=[
            Line2D([0], [0], color="black", lw=1.9, label="3-phonon"),
            Line2D([0], [0], color="#777777", lw=1.35, ls="--", label="quadratic"),
        ],
        loc="upper left",
        frameon=False,
        fontsize=max(7.0, args.legend_font - 2.0),
        handlelength=1.8,
        borderaxespad=0.25,
    )
    save_figure(fig, output_dir / "Gamma_T_Comparison_grid_RBLM_D_G", args.formats, args.dpi)
    plt.close(fig)


def sequence_legend_handles(family: str, sequences: Iterable[object]) -> List[Line2D]:
    handles = []
    ordered = sorted([str(seq) for seq in sequences], key=sequence_sort_key)
    for sequence in ordered:
        style = sequence_marker_style(family, sequence, ordered)
        handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                linestyle="-",
                color=style["edgecolors"],
                markerfacecolor=style["facecolors"],
                markeredgecolor=style["edgecolors"],
                markeredgewidth=style["linewidths"],
                linewidth=1.0,
                markersize=5.8,
                label=sequence_label(sequence),
            )
        )
    return handles


def parameter_y_limits(df: pd.DataFrame, family: str, peak: str, parameter: str) -> Tuple[float, float]:
    spec = PARAMETER_SPECS[parameter]
    values = df[
        (df["family"].astype(str) == family) & (df["peak_id"].astype(str) == peak)
    ][spec["value"]].to_numpy(dtype=float)
    if parameter == "width" and peak in FIXED_NON_ALIGNED_RO_Y_LIMITS:
        if family in INDEPENDENT_Y_FAMILIES:
            return padded_limits(values, pad_fraction=0.20, fallback=FIXED_NON_ALIGNED_RO_Y_LIMITS[peak])
        return expanded_y_limits(values, FIXED_NON_ALIGNED_RO_Y_LIMITS[peak], pad_fraction=0.10)
    fallback = (0.0, 1.0) if parameter == "height" else (float("nan"), float("nan"))
    if parameter == "position":
        finite = values[np.isfinite(values)]
        if finite.size:
            center = float(np.nanmedian(finite))
            span = max(float(np.nanmax(finite) - np.nanmin(finite)), 2.0)
            return center - 0.60 * span, center + 0.60 * span
        fallback = (0.0, 1.0)
    return padded_limits(values, pad_fraction=0.18, fallback=fallback)


def plot_parameter_grid(
    df: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
    parameter: str,
    suffix: str = "",
) -> None:
    spec = PARAMETER_SPECS[parameter]
    peaks = [peak for peak in args.peaks if peak in df["peak_id"].astype(str).unique()]
    families = sorted(TARGET_SAMPLE_FOLDERS, key=family_sort_key)
    if not peaks:
        return

    fig, axes = plt.subplots(
        nrows=len(peaks),
        ncols=len(families),
        figsize=(args.grid_width, args.grid_height),
        sharex=True,
        squeeze=False,
        dpi=args.dpi,
    )
    fig.subplots_adjust(left=0.080, right=0.985, bottom=0.105, top=0.900, wspace=0.18, hspace=0.16)

    for row_idx, peak in enumerate(peaks):
        for col_idx, family in enumerate(families):
            ax = axes[row_idx, col_idx]
            subset = df[(df["family"].astype(str) == family) & (df["peak_id"].astype(str) == peak)].copy()
            sequences = sorted(subset["sequence"].astype(str).unique(), key=sequence_sort_key)
            for sequence in sequences:
                seq_subset = subset[subset["sequence"].astype(str) == sequence].sort_values("temperature_K")
                if seq_subset.empty:
                    continue
                style = sequence_marker_style(family, sequence, sequences)
                x_values = seq_subset["temperature_K"].to_numpy(dtype=float)
                y_values = seq_subset[spec["value"]].to_numpy(dtype=float)
                yerr = parameter_uncertainty_values(seq_subset, parameter, args.error_bar)
                if yerr is not None:
                    ax.errorbar(
                        x_values,
                        y_values,
                        yerr=yerr,
                        fmt="none",
                        ecolor=style["edgecolors"],
                        elinewidth=0.9,
                        alpha=args.error_alpha,
                        capsize=0,
                        zorder=1,
                    )
                ax.plot(
                    x_values,
                    y_values,
                    color=style["edgecolors"],
                    linewidth=1.0,
                    alpha=0.55,
                    zorder=2,
                )
                ax.scatter(
                    x_values,
                    y_values,
                    marker=style["marker"],
                    s=args.grid_marker_size,
                    facecolors=style["facecolors"],
                    edgecolors=style["edgecolors"],
                    linewidths=style["linewidths"],
                    alpha=0.95,
                    zorder=3,
                )

            ax.set_xlim(args.temp_min, args.temp_max)
            ax.set_ylim(*parameter_y_limits(df, family, peak, parameter))
            style_single_axis(ax)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4 if parameter in {"position", "height"} else 5))
            if row_idx == 0:
                ax.set_title(family_label(family), fontsize=args.grid_title_font, pad=8)
                if len(sequences) > 1:
                    ax.legend(
                        handles=sequence_legend_handles(family, sequences),
                        loc="upper left",
                        frameon=False,
                        fontsize=max(6.5, args.legend_font - 3.0),
                        handlelength=1.3,
                        labelspacing=0.15,
                        borderaxespad=0.15,
                    )
            if col_idx == 0:
                ax.set_ylabel(f"{PEAK_LABELS.get(peak, peak)} {spec['label']}")
            else:
                ax.tick_params(labelleft=False)
            if row_idx == len(peaks) - 1:
                ax.set_xlabel("T (K)")
            else:
                ax.tick_params(labelbottom=False)

    fig.legend(
        handles=build_family_legend(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=min(5, len(families)),
        frameon=False,
        fontsize=args.legend_font,
        handletextpad=0.35,
        columnspacing=1.2,
    )
    name = f"Gamma_T_Comparison_grid_{spec['filename']}_RBLM_D_G{suffix}"
    save_figure(fig, output_dir / name, args.formats, args.dpi)
    plt.close(fig)


def plot_all_parameter_grids(df: pd.DataFrame, output_dir: Path, args: argparse.Namespace, suffix: str = "") -> None:
    for parameter in PARAMETER_ORDER:
        plot_parameter_grid(df, output_dir, args, parameter, suffix=suffix)


def row_quality_values(df: pd.DataFrame, mode: str, quality_metrics_by_key: Dict[str, Dict[str, float]]) -> np.ndarray:
    if mode == "Whole-spectrum SNR":
        return np.asarray(
            [
                quality_metrics_by_key.get(str(key), {}).get("whole_spectrum_snr", np.nan)
                for key in df["fit_row_key"].astype(str)
            ],
            dtype=float,
        )
    if mode == "Whole-spectrum normalized noise":
        return np.asarray(
            [
                quality_metrics_by_key.get(str(key), {}).get("whole_spectrum_noise_normalized", np.nan)
                for key in df["fit_row_key"].astype(str)
            ],
            dtype=float,
        )
    if mode == "Peak SNR":
        return np.asarray(
            [
                quality_metrics_by_key.get(str(key), {}).get("peak_snr", np.nan)
                for key in df["fit_row_key"].astype(str)
            ],
            dtype=float,
        )
    if mode == "G-peak SNR":
        return np.asarray(
            [
                quality_metrics_by_key.get(str(key), {}).get("g_reference_snr", np.nan)
                for key in df["fit_row_key"].astype(str)
            ],
            dtype=float,
        )
    if mode == "Relative RMSE":
        rmse = pd.to_numeric(df["rmse"], errors="coerce").to_numpy(dtype=float) if "rmse" in df.columns else np.full(len(df), np.nan)
        g_signal = np.asarray(
            [
                quality_metrics_by_key.get(str(key), {}).get("g_signal_height", np.nan)
                for key in df["fit_row_key"].astype(str)
            ],
            dtype=float,
        )
        return np.divide(
            rmse,
            g_signal,
            out=np.full(len(df), np.nan, dtype=float),
            where=np.isfinite(rmse) & np.isfinite(g_signal) & (g_signal > 0),
        )
    if mode == "R2" and "r_squared" in df.columns:
        return pd.to_numeric(df["r_squared"], errors="coerce").to_numpy(dtype=float)
    return np.full(len(df), np.nan, dtype=float)


def quality_scale_from_values(mode: str, values: np.ndarray):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    vmin, vmax = np.nanpercentile(finite, [5.0, 95.0]) if finite.size > 2 else (float(np.nanmin(finite)), float(np.nanmax(finite)))
    if mode in {"Whole-spectrum SNR", "Peak SNR", "G-peak SNR"}:
        cmap = mpl.colormaps["viridis"]
        label = mode
    elif mode == "Whole-spectrum normalized noise":
        cmap = mpl.colormaps["magma_r"]
        label = "Whole-spectrum noise / G"
    elif mode == "Relative RMSE":
        cmap = mpl.colormaps["magma_r"]
        label = "RMSE / G"
    elif mode == "R2":
        cmap = mpl.colormaps["plasma"]
        label = r"$R^2$"
        vmin = max(0.0, float(vmin))
        vmax = min(1.0, float(vmax))
    else:
        return None
    if not math.isfinite(float(vmin)) or not math.isfinite(float(vmax)) or math.isclose(float(vmin), float(vmax)):
        center = float(finite[0])
        vmin, vmax = center - 0.5, center + 0.5
    return mpl.colors.Normalize(vmin=float(vmin), vmax=float(vmax), clip=True), cmap, label


def plot_curation_review_grids(
    df: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
    rejected_keys: Set[str],
    quality_metrics_by_key: Dict[str, Dict[str, float]],
    suffix: str = "",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    color_modes = [
        "Family / sequence",
        "Whole-spectrum SNR",
        "Whole-spectrum normalized noise",
        "Peak SNR",
        "G-peak SNR",
        "Relative RMSE",
        "R2",
    ]
    families = sorted(TARGET_SAMPLE_FOLDERS, key=family_sort_key)
    available_peaks = [peak for peak in args.peaks if peak in set(df["peak_id"].astype(str))]
    for peak in available_peaks:
        peak_df = df[df["peak_id"].astype(str) == peak].copy()
        if peak_df.empty:
            continue
        for mode in color_modes:
            quality_scale = None
            if mode != "Family / sequence":
                peak_df["_quality_metric"] = row_quality_values(peak_df, mode, quality_metrics_by_key)
                quality_scale = quality_scale_from_values(mode, peak_df["_quality_metric"].to_numpy(dtype=float))
                if quality_scale is None:
                    continue

            fig, axes = plt.subplots(
                nrows=len(PARAMETER_ORDER),
                ncols=len(families),
                figsize=(args.grid_width, args.grid_height),
                sharex=True,
                squeeze=False,
                dpi=args.dpi,
            )
            fig.subplots_adjust(
                left=0.080,
                right=0.925 if quality_scale else 0.985,
                bottom=0.105,
                top=0.905,
                wspace=0.18,
                hspace=0.16,
            )
            for row_idx, parameter in enumerate(PARAMETER_ORDER):
                spec = PARAMETER_SPECS[parameter]
                for col_idx, family in enumerate(families):
                    ax = axes[row_idx, col_idx]
                    subset = peak_df[peak_df["family"].astype(str) == family].copy()
                    sequences = sorted(subset["sequence"].astype(str).unique(), key=sequence_sort_key)
                    for sequence in sequences:
                        seq_subset = subset[subset["sequence"].astype(str) == sequence].sort_values("temperature_K")
                        if seq_subset.empty:
                            continue
                        style = sequence_marker_style(family, sequence, sequences)
                        included = seq_subset[~seq_subset["fit_row_key"].astype(str).isin(rejected_keys)]
                        rejected = seq_subset[seq_subset["fit_row_key"].astype(str).isin(rejected_keys)]
                        for part, is_rejected in ((included, False), (rejected, True)):
                            if part.empty:
                                continue
                            x_values = pd.to_numeric(part["temperature_K"], errors="coerce").to_numpy(dtype=float)
                            y_values = pd.to_numeric(part[spec["value"]], errors="coerce").to_numpy(dtype=float)
                            if is_rejected:
                                ax.scatter(
                                    x_values,
                                    y_values,
                                    marker="x",
                                    s=max(args.grid_marker_size * 0.9, 26.0),
                                    color="#b00020",
                                    linewidths=1.5,
                                    alpha=0.86,
                                    zorder=4,
                                )
                            elif quality_scale:
                                norm, cmap, _label = quality_scale
                                values = pd.to_numeric(part["_quality_metric"], errors="coerce").to_numpy(dtype=float)
                                colors = [cmap(norm(value)) if math.isfinite(float(value)) else (0.7, 0.7, 0.7, 0.8) for value in values]
                                hollow = style["facecolors"] == "none"
                                ax.scatter(
                                    x_values,
                                    y_values,
                                    marker=style["marker"],
                                    s=args.grid_marker_size,
                                    facecolors="none" if hollow else colors,
                                    edgecolors=colors if hollow else style["edgecolors"],
                                    linewidths=style["linewidths"],
                                    alpha=0.94,
                                    zorder=3,
                                )
                            else:
                                ax.scatter(
                                    x_values,
                                    y_values,
                                    marker=style["marker"],
                                    s=args.grid_marker_size,
                                    facecolors=style["facecolors"],
                                    edgecolors=style["edgecolors"],
                                    linewidths=style["linewidths"],
                                    alpha=0.95,
                                    zorder=3,
                                )

                    ax.set_xlim(args.temp_min, args.temp_max)
                    ax.set_ylim(*parameter_y_limits(peak_df, family, peak, parameter))
                    style_single_axis(ax)
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=4 if parameter in {"position", "height"} else 5))
                    if row_idx == 0:
                        ax.set_title(family_label(family), fontsize=args.grid_title_font, pad=8)
                    if col_idx == 0:
                        ax.set_ylabel(f"{PEAK_LABELS.get(peak, peak)} {spec['label']}")
                    else:
                        ax.tick_params(labelleft=False)
                    if row_idx == len(PARAMETER_ORDER) - 1:
                        ax.set_xlabel("T (K)")
                    else:
                        ax.tick_params(labelbottom=False)

            legend_handles = build_family_legend() + [
                Line2D([0], [0], marker="x", linestyle="None", color="#b00020", markeredgewidth=1.5, markersize=7, label="Rejected")
            ]
            fig.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.988),
                ncol=min(6, len(legend_handles)),
                frameon=False,
                fontsize=args.legend_font,
                handletextpad=0.35,
                columnspacing=1.0,
            )
            if quality_scale:
                norm, cmap, label = quality_scale
                sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=axes, location="right", pad=0.012, fraction=0.025, shrink=0.98, aspect=30)
                cbar.set_label(label)
            mode_name = safe_filename(mode, max_len=80)
            save_figure(fig, output_dir / f"Gamma_T_Curation_Review_{peak}_{mode_name}{suffix}", args.formats, args.dpi)
            plt.close(fig)


def style_single_axis(ax) -> None:
    ax.grid(False)
    ax.tick_params(direction="in", length=5.0, width=1.3, top=False, right=False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.35)
        spine.set_color("#222222")


def save_figure(fig, base: Path, formats: Iterable[str], dpi: int) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = base.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved {path}")


def unique_run_folder(base: Path) -> Path:
    if not base.exists():
        return base
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return base.with_name(f"{base.name}_{stamp}")


def archive_existing_output_path(path: Path) -> Optional[Path]:
    path = Path(path)
    if not path.exists():
        return None
    backup_dir = path.parent / "_BACKUPS"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    target = backup_dir / f"{path.name}_{stamp}"
    counter = 1
    while target.exists():
        target = backup_dir / f"{path.name}_{stamp}_{counter}"
        counter += 1
    shutil.move(str(path), str(target))
    return target


def append_cli_values(command: List[str], flag: str, values: Optional[List[str]]) -> None:
    if values:
        command.append(flag)
        command.extend(str(value) for value in values)


def run_pc_bound_refit_if_requested(args: argparse.Namespace) -> Optional[Path]:
    if not args.pc_refit_bound_hits:
        return None
    if not PC_REFIT_SCRIPT.exists():
        raise FileNotFoundError(f"PC refit helper does not exist: {PC_REFIT_SCRIPT}")

    refit_output_root = unique_run_folder(args.pc_refit_output_root)
    bound_report = args.output_dir / "Gamma_T_Comparison_bound_hit_details.csv"
    command = [
        sys.executable,
        str(PC_REFIT_SCRIPT),
        "--source-root",
        str(args.pc_refit_source_root),
        "--fitted-root",
        str(args.fitted_root),
        "--bound-report",
        str(bound_report),
        "--output-root",
        str(refit_output_root),
        "--bound-tol",
        str(args.bound_tol),
        "--height-bound-rel-tol",
        str(args.height_bound_rel_tol),
        "--max-iterations",
        str(args.pc_refit_max_iterations),
        "--width-expand-factor",
        str(args.pc_refit_width_expand_factor),
        "--position-expand-cm",
        str(args.pc_refit_position_expand_cm),
        "--bootstrap-runs",
        str(args.pc_refit_bootstrap_runs),
        "--de-maxiter",
        str(args.pc_refit_de_maxiter),
        "--de-popsize",
        str(args.pc_refit_de_popsize),
        "--de-tol",
        str(args.pc_refit_de_tol),
        "--ls-max-nfev",
        str(args.pc_refit_ls_max_nfev),
        "--fit-x-min",
        str(args.pc_refit_x_min),
        "--fit-x-max",
        str(args.pc_refit_x_max),
        "--peak-window-weight",
        str(args.pc_refit_peak_window_weight),
        "--peak-window-margin",
        str(args.pc_refit_peak_window_margin),
    ]
    append_cli_values(command, "--families", args.pc_refit_families)
    append_cli_values(command, "--peaks", args.pc_refit_peaks or args.peaks)
    append_cli_values(command, "--hit-types", args.pc_refit_hit_types)
    if args.pc_refit_max_spectra is not None:
        command.extend(["--max-spectra", str(args.pc_refit_max_spectra)])
    if args.pc_refit_dry_run:
        command.append("--dry-run")

    print("")
    print("=" * 88)
    print("PC targeted bound-hit refit requested.")
    print("Original fitted results remain read-only.")
    print(f"Refit output root: {refit_output_root}")
    print("Command:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    print("=" * 88)

    process = subprocess.Popen(
        command,
        cwd=str(APP_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(f"[pc-refit] {line}", end="")
    return_code = process.wait()
    print("=" * 88)
    print(f"PC targeted refit finished with exit code {return_code}.")
    print("=" * 88)
    if return_code != 0:
        raise RuntimeError(f"PC targeted refit failed with exit code {return_code}")
    if args.pc_refit_dry_run:
        print("PC refit was a dry run; Gamma plot will use the original fitted root.")
        return None
    if not any(refit_output_root.rglob("*_long_results.csv")):
        print("PC refit produced no updated *_long_results.csv files; using original fitted root.")
        return None
    print(f"Gamma plot will now use updated PC refit results from: {refit_output_root}")
    return refit_output_root


class FitTrendCurationApp:
    def __init__(
        self,
        root,
        df: pd.DataFrame,
        args: argparse.Namespace,
        all_fit_rows: Optional[pd.DataFrame] = None,
    ) -> None:
        import tkinter as tk
        from tkinter import messagebox, ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure

        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.FigureCanvasTkAgg = FigureCanvasTkAgg
        self.NavigationToolbar2Tk = NavigationToolbar2Tk
        self.Figure = Figure
        self.root = root
        self.df = df.copy()
        self.all_fit_rows = all_fit_rows.copy() if all_fit_rows is not None else self.df.copy()
        self.args = args
        self.output_dir = args.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rejections_path = self.output_dir / args.curation_json
        self.rejected_keys: Set[str] = set()
        self.paper_excluded_families: Set[str] = set()
        self.paper_excluded_sequences: Set[str] = set()
        self.artist_keys: Dict[int, List[str]] = {}
        self.tree_iid_to_key: Dict[str, str] = {}
        self.accepted_refits: List[pd.DataFrame] = []
        self.inspect_window = None
        self.inspect_key: Optional[str] = None
        self.inspect_row: Optional[pd.Series] = None
        self.inspect_same_rows: Optional[pd.DataFrame] = None
        self.inspect_bounds_identity: Optional[Tuple[str, ...]] = None
        self.inspect_pending_refit_df: Optional[pd.DataFrame] = None
        self.inspect_pending_refit_meta: Dict[str, object] = {}
        self.inspect_refit_peak_var = tk.StringVar(value="")
        self.inspect_x_min_var = tk.StringVar(value="")
        self.inspect_x_max_var = tk.StringVar(value="")
        self.inspect_w_min_var = tk.StringVar(value="")
        self.inspect_w_max_var = tk.StringVar(value="")
        self.inspect_preview_var = tk.StringVar(value="No pending refit preview.")
        self.snr_by_key: Dict[str, float] = {}
        self.quality_metrics_by_key: Dict[str, Dict[str, float]] = {}
        self.quality_colorbar = None
        self.undo_stack: List[Dict[str, Any]] = []
        self.redo_stack: List[Dict[str, Any]] = []
        self.max_undo_steps = 60

        self.family_by_label = {
            family_label(family): family
            for family in sorted(self.df["family"].astype(str).unique(), key=family_sort_key)
        }
        self.family_options = ["All families"] + list(self.family_by_label.keys())
        peak_values = [peak for peak in args.peaks if peak in set(self.df["peak_id"].astype(str))]
        self.peak_options = peak_values or sorted(self.df["peak_id"].astype(str).unique())

        self.family_var = tk.StringVar(value=self.family_options[1] if len(self.family_options) > 1 else "All families")
        self.sequence_var = tk.StringVar(value="All sequences")
        self.peak_var = tk.StringVar(value="G" if "G" in self.peak_options else (self.peak_options[0] if self.peak_options else ""))
        self.color_mode_var = tk.StringVar(value="Family / sequence")
        self.click_action_var = tk.StringVar(value="Toggle keep/reject")
        self.stage_var = tk.StringVar(value=str(getattr(args, "analysis_stage", "BEFORE")).upper())
        self.show_rejected_var = tk.BooleanVar(value=True)
        self.show_paper_excluded_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="")
        self.staged_refit_var = tk.StringVar(value="Staged PC refits: 0 peak rows. Official files untouched.")

        self.load_rejections()
        self.load_snr_cache()
        root.title("Gamma/T fit trend curator - height, FWHM, position")
        root.geometry("1500x900")

        paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(paned, padding=8)
        right = ttk.Frame(paned, padding=4)
        paned.add(left, weight=0)
        paned.add(right, weight=1)

        controls = ttk.LabelFrame(left, text="Trend curation", padding=8)
        controls.pack(fill=tk.X)
        ttk.Label(controls, text="Family").grid(row=0, column=0, sticky="w")
        self.family_combo = ttk.Combobox(
            controls, textvariable=self.family_var, values=self.family_options, state="readonly", width=28
        )
        self.family_combo.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        ttk.Label(controls, text="UP/DOWN folder").grid(row=1, column=0, sticky="w")
        self.sequence_combo = ttk.Combobox(controls, textvariable=self.sequence_var, state="readonly", width=28)
        self.sequence_combo.grid(row=1, column=1, sticky="ew", padx=(6, 0))
        ttk.Label(controls, text="Peak").grid(row=2, column=0, sticky="w")
        self.peak_combo = ttk.Combobox(
            controls, textvariable=self.peak_var, values=self.peak_options, state="readonly", width=28
        )
        self.peak_combo.grid(row=2, column=1, sticky="ew", padx=(6, 0))
        ttk.Checkbutton(controls, text="Show rejected points", variable=self.show_rejected_var, command=self.refresh).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(5, 0)
        )
        ttk.Label(controls, text="Color by").grid(row=4, column=0, sticky="w", pady=(4, 0))
        self.color_combo = ttk.Combobox(
            controls,
            textvariable=self.color_mode_var,
            values=[
                "Family / sequence",
                "Whole-spectrum SNR",
                "Whole-spectrum normalized noise",
                "Peak SNR",
                "G-peak SNR",
                "Relative RMSE",
                "R2",
            ],
            state="readonly",
            width=28,
        )
        self.color_combo.grid(row=4, column=1, sticky="ew", padx=(6, 0), pady=(4, 0))
        ttk.Label(controls, text="Click action").grid(row=5, column=0, sticky="w", pady=(4, 0))
        self.click_action_combo = ttk.Combobox(
            controls,
            textvariable=self.click_action_var,
            values=["Toggle keep/reject", "Inspect spectrum + fit"],
            state="readonly",
            width=28,
        )
        self.click_action_combo.grid(row=5, column=1, sticky="ew", padx=(6, 0), pady=(4, 0))
        ttk.Label(controls, text="Save stage").grid(row=6, column=0, sticky="w", pady=(4, 0))
        self.stage_combo = ttk.Combobox(
            controls,
            textvariable=self.stage_var,
            values=["BEFORE", "AFTER"],
            state="readonly",
            width=28,
        )
        self.stage_combo.grid(row=6, column=1, sticky="ew", padx=(6, 0), pady=(4, 0))
        controls.columnconfigure(1, weight=1)

        paper_set = ttk.LabelFrame(left, text="Paper set", padding=8)
        paper_set.pack(fill=tk.X, pady=(8, 0))
        ttk.Checkbutton(
            paper_set,
            text="Show paper-excluded families/sequences in app",
            variable=self.show_paper_excluded_var,
            command=self.refresh,
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Button(paper_set, text="Exclude current family", command=self.exclude_current_family_from_paper).grid(
            row=1, column=0, sticky="ew", pady=(5, 0)
        )
        ttk.Button(paper_set, text="Include current family", command=self.include_current_family_in_paper).grid(
            row=1, column=1, sticky="ew", padx=(5, 0), pady=(5, 0)
        )
        ttk.Button(paper_set, text="Exclude current UP/DOWN", command=self.exclude_current_sequence_from_paper).grid(
            row=2, column=0, sticky="ew", pady=(5, 0)
        )
        ttk.Button(paper_set, text="Include current UP/DOWN", command=self.include_current_sequence_in_paper).grid(
            row=2, column=1, sticky="ew", padx=(5, 0), pady=(5, 0)
        )
        self.paper_set_var = tk.StringVar(value="")
        ttk.Label(paper_set, textvariable=self.paper_set_var, wraplength=390).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(5, 0)
        )
        paper_set.columnconfigure(0, weight=1)
        paper_set.columnconfigure(1, weight=1)

        buttons = ttk.LabelFrame(left, text="Outliers", padding=8)
        buttons.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(buttons, text="Reject selected", command=self.reject_selected).grid(row=0, column=0, sticky="ew")
        ttk.Button(buttons, text="Keep selected", command=self.keep_selected).grid(row=0, column=1, sticky="ew", padx=(5, 0))
        ttk.Button(buttons, text="Keep all visible", command=self.keep_visible).grid(row=1, column=0, sticky="ew", pady=(5, 0))
        ttk.Button(buttons, text="Invert visible", command=self.invert_visible).grid(
            row=1, column=1, sticky="ew", padx=(5, 0), pady=(5, 0)
        )
        ttk.Button(buttons, text="Clear all rejected", command=self.clear_all_rejected).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(5, 0)
        )
        ttk.Button(buttons, text="Save curation", command=self.save_curation).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(5, 0)
        )
        self.undo_button = ttk.Button(buttons, text="Undo", command=self.undo_last)
        self.undo_button.grid(row=4, column=0, sticky="ew", pady=(5, 0))
        self.redo_button = ttk.Button(buttons, text="Redo", command=self.redo_last)
        self.redo_button.grid(row=4, column=1, sticky="ew", padx=(5, 0), pady=(5, 0))
        ttk.Button(buttons, text="Save curated plots", command=self.save_curated_plots).grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=(5, 0)
        )
        ttk.Button(buttons, text="Inspect selected spectrum", command=self.inspect_selected_spectrum).grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )
        ttk.Button(buttons, text="Trust-region refit selected", command=self.trust_refit_selected).grid(
            row=7, column=0, columnspan=2, sticky="ew", pady=(5, 0)
        )
        ttk.Button(buttons, text="Export staged refits to new fitted folder", command=self.export_accepted_refit_folder).grid(
            row=8, column=0, columnspan=2, sticky="ew", pady=(5, 0)
        )
        ttk.Label(buttons, textvariable=self.staged_refit_var, wraplength=390).grid(
            row=9, column=0, columnspan=2, sticky="ew", pady=(4, 0)
        )
        ttk.Button(buttons, text="Compute SNR summary", command=self.compute_snr_summary).grid(
            row=10, column=0, columnspan=2, sticky="ew", pady=(5, 0)
        )
        ttk.Button(buttons, text="Save review plots", command=self.save_stage_review_plots).grid(
            row=11, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )
        ttk.Button(buttons, text="Export averaged TXT for cluster", command=self.export_averaged_txt_for_cluster).grid(
            row=12, column=0, columnspan=2, sticky="ew", pady=(5, 0)
        )
        ttk.Button(buttons, text="Export PAPER clean fitted folder", command=self.export_paper_clean_fitted_folder).grid(
            row=13, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )
        ttk.Button(buttons, text="Save ALL GOOD paper package", command=self.export_stage_paper_package).grid(
            row=14, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)

        table_frame = ttk.LabelFrame(
            left,
            text="Fitted rows - double-click toggles; plot click follows Click action; Shift/Ctrl/right-click inspects",
            padding=4,
        )
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        columns = ("keep", "family", "sequence", "t", "peak", "ycol", "height", "width", "position", "snr", "r2", "file")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=20)
        headings = {
            "keep": "Keep",
            "family": "Family",
            "sequence": "Sequence",
            "t": "T",
            "peak": "Peak",
            "ycol": "Y",
            "height": "Height",
            "width": "FWHM",
            "position": "Position",
            "snr": "Global SNR",
            "r2": "R2",
            "file": "File",
        }
        widths = {
            "keep": 52,
            "family": 115,
            "sequence": 125,
            "t": 55,
            "peak": 58,
            "ycol": 45,
            "height": 78,
            "width": 70,
            "position": 82,
            "snr": 62,
            "r2": 62,
            "file": 230,
        }
        for column in columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(column, width=widths[column], anchor="w", stretch=column == "file")
        yscroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        xscroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        ttk.Label(left, textvariable=self.status_var, wraplength=420).pack(fill=tk.X, pady=(6, 0))

        self.figure = Figure(figsize=(10.5, 8.0), dpi=100)
        self.axes = self.figure.subplots(3, 1, sharex=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.toolbar = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("pick_event", self.on_pick)

        self.family_combo.bind("<<ComboboxSelected>>", self.on_family_changed)
        self.sequence_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh())
        self.peak_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh())
        self.color_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh())
        self.tree.bind("<Double-1>", self.on_tree_double_click)
        root.bind_all("<Control-z>", self.on_undo_shortcut)
        root.bind_all("<Control-y>", self.on_redo_shortcut)
        root.bind_all("<Control-Shift-Z>", self.on_redo_shortcut)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.update_sequence_options()
        self.refresh()

    def load_rejections(self) -> None:
        if not self.rejections_path.exists():
            return
        try:
            payload = json.loads(self.rejections_path.read_text(encoding="utf-8"))
            self.rejected_keys = set(str(key) for key in payload.get("rejected_keys", []))
            self.paper_excluded_families = set(str(family) for family in payload.get("paper_excluded_families", []))
            sequence_values = payload.get("paper_excluded_sequences", [])
            excluded_sequences: Set[str] = set()
            for item in sequence_values:
                if isinstance(item, dict):
                    excluded_sequences.add(paper_sequence_key(item.get("family", ""), item.get("sequence", "")))
                else:
                    excluded_sequences.add(str(item))
            self.paper_excluded_sequences = excluded_sequences
            print(
                f"Loaded {len(self.rejected_keys)} manual rejection(s), "
                f"{len(self.paper_excluded_families)} excluded paper family/families, "
                f"{len(self.paper_excluded_sequences)} excluded paper sequence(s) from {self.rejections_path}"
            )
        except Exception as exc:
            print(f"Could not load curation JSON {self.rejections_path}: {exc}")

    def snapshot_state(self, label: str) -> Dict[str, Any]:
        return {
            "label": label,
            "rejected_keys": set(self.rejected_keys),
            "paper_excluded_families": set(self.paper_excluded_families),
            "paper_excluded_sequences": set(self.paper_excluded_sequences),
            "df": self.df.copy(deep=True),
            "all_fit_rows": self.all_fit_rows.copy(deep=True),
            "accepted_refits": [frame.copy(deep=True) for frame in self.accepted_refits],
        }

    def push_undo_state(self, label: str) -> None:
        self.undo_stack.append(self.snapshot_state(label))
        if len(self.undo_stack) > self.max_undo_steps:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self.update_undo_buttons()

    def restore_state(self, state: Dict[str, Any]) -> None:
        self.rejected_keys = set(state["rejected_keys"])
        self.paper_excluded_families = set(state.get("paper_excluded_families", set()))
        self.paper_excluded_sequences = set(state.get("paper_excluded_sequences", set()))
        self.df = state["df"].copy(deep=True)
        if "all_fit_rows" in state:
            self.all_fit_rows = state["all_fit_rows"].copy(deep=True)
        else:
            self.all_fit_rows = self.df.copy(deep=True)
        self.accepted_refits = [frame.copy(deep=True) for frame in state["accepted_refits"]]

    def update_undo_buttons(self) -> None:
        if hasattr(self, "undo_button"):
            self.undo_button.configure(state=("normal" if self.undo_stack else "disabled"))
        if hasattr(self, "redo_button"):
            self.redo_button.configure(state=("normal" if self.redo_stack else "disabled"))

    def undo_last(self) -> None:
        if not self.undo_stack:
            self.status_var.set("Nothing to undo.")
            self.update_undo_buttons()
            return
        current = self.snapshot_state("Redo state")
        previous = self.undo_stack.pop()
        self.redo_stack.append(current)
        self.restore_state(previous)
        self.refresh()
        self.update_staged_refit_status()
        self.status_var.set(f"Undid: {previous.get('label', 'last change')}. Session changes only; export when happy.")
        self.update_undo_buttons()

    def redo_last(self) -> None:
        if not self.redo_stack:
            self.status_var.set("Nothing to redo.")
            self.update_undo_buttons()
            return
        current = self.snapshot_state("Undo state")
        next_state = self.redo_stack.pop()
        self.undo_stack.append(current)
        self.restore_state(next_state)
        self.refresh()
        self.update_staged_refit_status()
        self.status_var.set("Redid the last undone change. Session changes only; export when happy.")
        self.update_undo_buttons()

    def on_undo_shortcut(self, _event=None):
        self.undo_last()
        return "break"

    def on_redo_shortcut(self, _event=None):
        self.redo_last()
        return "break"

    def selected_family(self) -> Optional[str]:
        label = self.family_var.get()
        if label == "All families":
            return None
        return self.family_by_label.get(label)

    def current_sequences(self) -> List[str]:
        df = self.df
        family = self.selected_family()
        if family is not None:
            df = df[df["family"].astype(str) == family]
        return sorted(df["sequence"].astype(str).dropna().unique(), key=sequence_sort_key)

    def selected_sequence(self) -> Optional[str]:
        value = self.sequence_var.get()
        if value == "All sequences":
            return None
        for sequence in self.current_sequences():
            if sequence_label(sequence) == value or str(sequence) == value:
                return str(sequence)
        return value

    def paper_inclusion_mask_for(self, df: pd.DataFrame) -> pd.Series:
        return paper_inclusion_mask(df, self.paper_excluded_families, self.paper_excluded_sequences)

    def paper_included_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[self.paper_inclusion_mask_for(df)].copy()

    def current_paper_sequence_key(self) -> Optional[str]:
        family = self.selected_family()
        sequence = self.selected_sequence()
        if family is None or sequence is None:
            return None
        return paper_sequence_key(family, sequence)

    def update_paper_set_status(self) -> None:
        if not hasattr(self, "paper_set_var"):
            return
        excluded_sequence_labels = []
        for key in sorted(self.paper_excluded_sequences):
            family, sequence = split_paper_sequence_key(key)
            excluded_sequence_labels.append(f"{family_label(family)} / {sequence_label(sequence)}")
        message = (
            f"Paper excluded: {len(self.paper_excluded_families)} family/families, "
            f"{len(self.paper_excluded_sequences)} UP/DOWN folder(s)."
        )
        if self.paper_excluded_families:
            message += " Families: " + ", ".join(family_label(family) for family in sorted(self.paper_excluded_families, key=family_sort_key))
        if excluded_sequence_labels:
            message += " UP/DOWN: " + "; ".join(excluded_sequence_labels[:4])
            if len(excluded_sequence_labels) > 4:
                message += f"; +{len(excluded_sequence_labels) - 4} more"
        self.paper_set_var.set(message)

    def exclude_current_family_from_paper(self) -> None:
        family = self.selected_family()
        if family is None:
            self.status_var.set("Select one family before excluding it from the paper set.")
            return
        if family in self.paper_excluded_families:
            self.status_var.set(f"{family_label(family)} is already excluded from the paper set.")
            return
        self.push_undo_state(f"Exclude {family_label(family)} from paper set")
        self.paper_excluded_families.add(family)
        self.refresh()
        self.status_var.set(f"Excluded {family_label(family)} from the paper set. Save/export when happy.")

    def include_current_family_in_paper(self) -> None:
        family = self.selected_family()
        if family is None:
            self.status_var.set("Select one family before including it in the paper set.")
            return
        sequence_keys = {key for key in self.paper_excluded_sequences if split_paper_sequence_key(key)[0] == family}
        if family not in self.paper_excluded_families and not sequence_keys:
            self.status_var.set(f"{family_label(family)} is already fully included in the paper set.")
            return
        self.push_undo_state(f"Include {family_label(family)} in paper set")
        self.paper_excluded_families.discard(family)
        self.paper_excluded_sequences.difference_update(sequence_keys)
        self.refresh()
        self.status_var.set(f"Included {family_label(family)} in the paper set.")

    def exclude_current_sequence_from_paper(self) -> None:
        key = self.current_paper_sequence_key()
        family = self.selected_family()
        sequence = self.selected_sequence()
        if key is None or family is None or sequence is None:
            self.status_var.set("Select one family and one UP/DOWN folder before excluding it.")
            return
        if key in self.paper_excluded_sequences:
            self.status_var.set(f"{family_label(family)} / {sequence_label(sequence)} is already excluded.")
            return
        self.push_undo_state(f"Exclude {family_label(family)} {sequence_label(sequence)} from paper set")
        self.paper_excluded_sequences.add(key)
        self.refresh()
        self.status_var.set(f"Excluded {family_label(family)} / {sequence_label(sequence)} from the paper set.")

    def include_current_sequence_in_paper(self) -> None:
        key = self.current_paper_sequence_key()
        family = self.selected_family()
        sequence = self.selected_sequence()
        if key is None or family is None or sequence is None:
            self.status_var.set("Select one family and one UP/DOWN folder before including it.")
            return
        if key not in self.paper_excluded_sequences and family not in self.paper_excluded_families:
            self.status_var.set(f"{family_label(family)} / {sequence_label(sequence)} is already included.")
            return
        self.push_undo_state(f"Include {family_label(family)} {sequence_label(sequence)} in paper set")
        self.paper_excluded_families.discard(family)
        self.paper_excluded_sequences.discard(key)
        self.refresh()
        self.status_var.set(f"Included {family_label(family)} / {sequence_label(sequence)} in the paper set.")

    def update_sequence_options(self) -> None:
        sequences = self.current_sequences()
        options = ["All sequences"] + [sequence_label(sequence) for sequence in sequences]
        self.sequence_combo.configure(values=options)
        if self.sequence_var.get() not in options:
            self.sequence_var.set(options[0])

    def on_family_changed(self, _event=None) -> None:
        self.update_sequence_options()
        self.refresh()

    def base_filter_df(self) -> pd.DataFrame:
        df = self.df[self.df["peak_id"].astype(str) == self.peak_var.get()].copy()
        family = self.selected_family()
        if family is not None:
            df = df[df["family"].astype(str) == family]
        sequence = self.selected_sequence()
        if sequence is not None:
            df = df[df["sequence"].astype(str) == sequence]
        if not self.show_paper_excluded_var.get():
            df = self.paper_included_df(df)
        return df

    def visible_df(self) -> pd.DataFrame:
        df = self.base_filter_df()
        if not self.show_rejected_var.get():
            df = df[~df["fit_row_key"].astype(str).isin(self.rejected_keys)]
        return df

    def cache_snr_metrics(self, snr_df: pd.DataFrame) -> None:
        if snr_df.empty or "fit_row_key" not in snr_df.columns:
            self.snr_by_key = {}
            self.quality_metrics_by_key = {}
            return
        metric_columns = (
            "snr",
            "whole_spectrum_snr",
            "whole_spectrum_noise_normalized",
            "whole_spectrum_reference_signal_height",
            "global_signal_height",
            "peak_snr",
            "snr_g_normalized",
            "g_reference_snr",
            "noise_sigma_g_normalized",
            "signal_height_g_normalized",
            "g_signal_height",
        )
        metrics_by_key: Dict[str, Dict[str, float]] = {}
        for _idx, row in snr_df.iterrows():
            key = str(row.get("fit_row_key", ""))
            if not key:
                continue
            metrics: Dict[str, float] = {}
            for column in metric_columns:
                if column not in snr_df.columns:
                    continue
                value = pd.to_numeric(pd.Series([row.get(column, np.nan)]), errors="coerce").iloc[0]
                metrics[column] = float(value) if pd.notna(value) and math.isfinite(float(value)) else float("nan")
            if metrics:
                metrics_by_key[key] = metrics
        self.quality_metrics_by_key = metrics_by_key
        self.snr_by_key = {
            key: metrics.get("whole_spectrum_snr", metrics.get("snr", float("nan")))
            for key, metrics in metrics_by_key.items()
            if math.isfinite(float(metrics.get("whole_spectrum_snr", metrics.get("snr", float("nan")))))
        }

    def load_snr_cache(self) -> None:
        path = self.output_dir / "Gamma_T_Comparison_SNR_by_spectrum.csv"
        if not path.exists():
            return
        try:
            snr_df = pd.read_csv(path)
            self.cache_snr_metrics(snr_df)
            print(f"Loaded {len(self.snr_by_key)} SNR value(s) from {path}")
        except Exception as exc:
            print(f"Could not load SNR cache {path}: {exc}")

    def quality_values_for(self, df: pd.DataFrame) -> np.ndarray:
        mode = self.color_mode_var.get()
        if mode == "Whole-spectrum SNR":
            return np.asarray(
                [
                    self.quality_metrics_by_key.get(str(key), {}).get("whole_spectrum_snr", self.snr_by_key.get(str(key), np.nan))
                    for key in df["fit_row_key"].astype(str)
                ],
                dtype=float,
            )
        if mode == "Whole-spectrum normalized noise":
            return np.asarray(
                [
                    self.quality_metrics_by_key.get(str(key), {}).get("whole_spectrum_noise_normalized", np.nan)
                    for key in df["fit_row_key"].astype(str)
                ],
                dtype=float,
            )
        if mode == "Peak SNR":
            return np.asarray(
                [
                    self.quality_metrics_by_key.get(str(key), {}).get("peak_snr", np.nan)
                    for key in df["fit_row_key"].astype(str)
                ],
                dtype=float,
            )
        if mode == "G-peak SNR":
            return np.asarray(
                [
                    self.quality_metrics_by_key.get(str(key), {}).get("g_reference_snr", np.nan)
                    for key in df["fit_row_key"].astype(str)
                ],
                dtype=float,
            )
        if mode == "Relative RMSE":
            if "rmse" in df.columns:
                rmse_values = pd.to_numeric(df["rmse"], errors="coerce").to_numpy(dtype=float)
            else:
                rmse_values = np.full(len(df), np.nan, dtype=float)
            g_values = np.asarray(
                [
                    self.quality_metrics_by_key.get(str(key), {}).get("g_signal_height", np.nan)
                    for key in df["fit_row_key"].astype(str)
                ],
                dtype=float,
            )
            return np.divide(
                rmse_values,
                g_values,
                out=np.full(len(df), np.nan, dtype=float),
                where=np.isfinite(rmse_values) & np.isfinite(g_values) & (g_values > 0),
            )
        if mode == "R2":
            if "r_squared" in df.columns:
                return pd.to_numeric(df["r_squared"], errors="coerce").to_numpy(dtype=float)
            return np.full(len(df), np.nan, dtype=float)
        return np.full(len(df), np.nan, dtype=float)

    def quality_scale(self, df: pd.DataFrame):
        mode = self.color_mode_var.get()
        if mode not in {"Whole-spectrum SNR", "Whole-spectrum normalized noise", "Peak SNR", "G-peak SNR", "Relative RMSE", "R2"}:
            return None
        values = self.quality_values_for(df)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None
        if mode == "Whole-spectrum SNR":
            vmin, vmax = np.nanpercentile(finite, [5.0, 95.0]) if finite.size > 2 else (float(np.nanmin(finite)), float(np.nanmax(finite)))
            cmap = mpl.colormaps["viridis"]
            label = "Whole-spectrum SNR"
        elif mode == "Peak SNR":
            vmin, vmax = np.nanpercentile(finite, [5.0, 95.0]) if finite.size > 2 else (float(np.nanmin(finite)), float(np.nanmax(finite)))
            cmap = mpl.colormaps["viridis"]
            label = "Peak SNR"
        elif mode == "G-peak SNR":
            vmin, vmax = np.nanpercentile(finite, [5.0, 95.0]) if finite.size > 2 else (float(np.nanmin(finite)), float(np.nanmax(finite)))
            cmap = mpl.colormaps["viridis"]
            label = "G-peak SNR"
        elif mode == "Whole-spectrum normalized noise":
            vmin, vmax = np.nanpercentile(finite, [5.0, 95.0]) if finite.size > 2 else (float(np.nanmin(finite)), float(np.nanmax(finite)))
            cmap = mpl.colormaps["magma_r"]
            label = r"Whole-spectrum noise / G"
        elif mode == "Relative RMSE":
            vmin, vmax = np.nanpercentile(finite, [5.0, 95.0]) if finite.size > 2 else (float(np.nanmin(finite)), float(np.nanmax(finite)))
            cmap = mpl.colormaps["magma_r"]
            label = r"RMSE / G signal"
        else:
            vmin, vmax = np.nanpercentile(finite, [5.0, 95.0]) if finite.size > 2 else (float(np.nanmin(finite)), float(np.nanmax(finite)))
            vmin = max(0.0, float(vmin))
            vmax = min(1.0, float(vmax))
            cmap = mpl.colormaps["plasma"]
            label = r"$R^2$"
        if not math.isfinite(float(vmin)) or not math.isfinite(float(vmax)) or math.isclose(float(vmin), float(vmax)):
            center = float(finite[0]) if finite.size else 0.0
            vmin, vmax = center - 0.5, center + 0.5
        return mpl.colors.Normalize(vmin=float(vmin), vmax=float(vmax), clip=True), cmap, label

    def scatter_with_quality(
        self,
        ax,
        data: pd.DataFrame,
        x_values: np.ndarray,
        y_values: np.ndarray,
        style: Dict[str, object],
        quality_scale,
        is_rejected: bool,
    ):
        if quality_scale is None:
            if is_rejected:
                return ax.scatter(
                    x_values,
                    y_values,
                    marker="x",
                    s=max(self.args.grid_marker_size * 0.7, 20.0),
                    color="#777777",
                    linewidths=1.0,
                    alpha=0.45,
                    zorder=1,
                    picker=6,
                )
            return ax.scatter(
                x_values,
                y_values,
                marker=style["marker"],
                s=self.args.grid_marker_size,
                facecolors=style["facecolors"],
                edgecolors=style["edgecolors"],
                linewidths=style["linewidths"],
                alpha=0.90,
                zorder=3,
                picker=6,
            )

        norm, cmap, _label = quality_scale
        metric = self.quality_values_for(data)
        colors = np.asarray([cmap(norm(value)) if math.isfinite(float(value)) else (0.70, 0.70, 0.70, 0.65) for value in metric])
        if is_rejected:
            return ax.scatter(
                x_values,
                y_values,
                marker="x",
                s=max(self.args.grid_marker_size * 0.75, 22.0),
                c=colors,
                linewidths=1.15,
                alpha=0.75,
                zorder=2,
                picker=6,
            )
        hollow = style["facecolors"] == "none"
        return ax.scatter(
            x_values,
            y_values,
            marker=style["marker"],
            s=self.args.grid_marker_size,
            facecolors="none" if hollow else colors,
            edgecolors=colors if hollow else style["edgecolors"],
            linewidths=style["linewidths"],
            alpha=0.94,
            zorder=3,
            picker=6,
        )

    def refresh(self) -> None:
        self.plot_current()
        self.populate_table()
        total_df = self.base_filter_df()
        total = len(total_df)
        rejected = int(total_df["fit_row_key"].astype(str).isin(self.rejected_keys).sum())
        status = f"{total - rejected} kept / {total} rows for current family/sequence/peak."
        if self.color_mode_var.get() in {"Whole-spectrum SNR", "Whole-spectrum normalized noise", "Peak SNR", "G-peak SNR", "Relative RMSE"} and not self.quality_metrics_by_key:
            status += " Compute SNR summary to color by spectrum quality."
        self.status_var.set(status)
        self.update_staged_refit_status()
        self.update_paper_set_status()
        self.update_undo_buttons()

    def staged_refit_peak_count(self) -> int:
        if not self.accepted_refits:
            return 0
        try:
            accepted = pd.concat(self.accepted_refits, ignore_index=True)
            accepted = accepted.drop_duplicates(
                subset=["family", "sequence", "temperature_K", "file_name", "spectrum_in_file", "y_column_number", "peak_id"],
                keep="last",
            )
            return int(len(accepted))
        except Exception:
            return int(sum(len(frame) for frame in self.accepted_refits))

    def update_staged_refit_status(self) -> None:
        count = self.staged_refit_peak_count()
        if count:
            self.staged_refit_var.set(
                f"Staged PC refits: {count} peak row(s) in this app session/RAM. Official files untouched."
            )
        else:
            self.staged_refit_var.set("Staged PC refits: 0 peak rows. Official files untouched.")

    def plot_current(self) -> None:
        self.artist_keys.clear()
        if self.quality_colorbar is not None:
            try:
                self.quality_colorbar.remove()
            except Exception:
                pass
            self.quality_colorbar = None
        for ax in self.axes:
            ax.clear()
        df = self.visible_df()
        if df.empty:
            for ax, parameter in zip(self.axes, PARAMETER_ORDER):
                ax.set_ylabel(PARAMETER_SPECS[parameter]["label"])
                style_single_axis(ax)
            self.axes[-1].set_xlabel("T (K)")
            self.canvas.draw_idle()
            return

        quality_scale = self.quality_scale(df)
        for axis_idx, parameter in enumerate(PARAMETER_ORDER):
            ax = self.axes[axis_idx]
            spec = PARAMETER_SPECS[parameter]
            for (family, sequence), group in df.groupby(["family", "sequence"], sort=False):
                sequences = sorted(
                    self.df[self.df["family"].astype(str) == str(family)]["sequence"].astype(str).unique(),
                    key=sequence_sort_key,
                )
                style = sequence_marker_style(str(family), str(sequence), sequences)
                included = group[~group["fit_row_key"].astype(str).isin(self.rejected_keys)].sort_values("temperature_K")
                rejected = group[group["fit_row_key"].astype(str).isin(self.rejected_keys)].sort_values("temperature_K")
                for data, is_rejected in ((included, False), (rejected, True)):
                    if data.empty:
                        continue
                    x_values = data["temperature_K"].to_numpy(dtype=float)
                    y_values = data[spec["value"]].to_numpy(dtype=float)
                    keys = data["fit_row_key"].astype(str).tolist()
                    collection = self.scatter_with_quality(
                        ax,
                        data,
                        x_values,
                        y_values,
                        style,
                        quality_scale,
                        is_rejected,
                    )
                    if not is_rejected:
                        means = data.groupby("temperature_K", as_index=False)[spec["value"]].mean().sort_values(
                            "temperature_K"
                        )
                        ax.plot(
                            means["temperature_K"].to_numpy(dtype=float),
                            means[spec["value"]].to_numpy(dtype=float),
                            color=style["edgecolors"],
                            linewidth=1.2,
                            alpha=0.55,
                            zorder=2,
                        )
                    self.artist_keys[id(collection)] = keys

            values = df[spec["value"]].to_numpy(dtype=float)
            if np.isfinite(values).any():
                ax.set_ylim(*padded_limits(values, pad_fraction=0.14))
            ax.set_xlim(self.args.temp_min, self.args.temp_max)
            ax.set_ylabel(spec["label"])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            style_single_axis(ax)
            if axis_idx < len(self.axes) - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("T (K)")
        title_parts = [self.family_var.get(), self.sequence_var.get(), self.peak_var.get()]
        self.axes[0].set_title(" | ".join(part for part in title_parts if part), fontsize=13)
        if quality_scale is not None:
            norm, cmap, label = quality_scale
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            self.quality_colorbar = self.figure.colorbar(
                sm,
                ax=self.axes,
                location="right",
                pad=0.012,
                fraction=0.025,
                shrink=0.98,
                aspect=28,
            )
            self.quality_colorbar.set_label(label)
            if self.color_mode_var.get() in {"Whole-spectrum SNR", "Whole-spectrum normalized noise", "Peak SNR", "G-peak SNR", "Relative RMSE"} and not self.quality_metrics_by_key:
                self.axes[0].text(
                    0.01,
                    0.96,
                    "Compute SNR summary first",
                    transform=self.axes[0].transAxes,
                    ha="left",
                    va="top",
                    fontsize=9.5,
                    color="#8a1f11",
                )
            self.figure.subplots_adjust(right=0.90)
        else:
            self.figure.tight_layout()
        self.canvas.draw_idle()

    def populate_table(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self.tree_iid_to_key.clear()
        df = self.visible_df().sort_values(
            ["_family_order", "_sequence_order", "temperature_K", "file_name", "y_column_number"]
        )
        for idx, row in df.iterrows():
            key = str(row["fit_row_key"])
            iid = f"row_{idx}"
            self.tree_iid_to_key[iid] = key
            paper_included = bool(self.paper_inclusion_mask_for(pd.DataFrame([row])).iloc[0])
            if not paper_included:
                keep = "PAPER-EXCL"
            else:
                keep = "NO" if key in self.rejected_keys else "YES"
            ycol = pd.to_numeric(row.get("y_column_number", np.nan), errors="coerce")
            snr_value = self.snr_by_key.get(key, np.nan)
            values = (
                keep,
                family_label(row["family"]),
                sequence_label(row["sequence"]),
                f"{float(row['temperature_K']):g}K",
                row["peak_id"],
                f"Y{int(ycol):02d}" if math.isfinite(float(ycol)) else "",
                f"{float(row['height']):.4g}",
                f"{float(row['width_fwhm_cm-1']):.4g}",
                f"{float(row['position_cm-1']):.4g}",
                f"{snr_value:.3g}" if math.isfinite(float(snr_value)) else "",
                f"{float(row['r_squared']):.4f}" if "r_squared" in row and math.isfinite(float(row["r_squared"])) else "",
                row.get("file_name", ""),
            )
            self.tree.insert("", "end", iid=iid, values=values)

    def selected_keys(self) -> Set[str]:
        return {self.tree_iid_to_key[iid] for iid in self.tree.selection() if iid in self.tree_iid_to_key}

    def spectrum_column_keys_for_keys(self, keys: Set[str]) -> Set[str]:
        expanded: Set[str] = set()
        for key in keys:
            row = self.row_for_key(str(key))
            if row is None:
                expanded.add(str(key))
            else:
                expanded.update(self.spectrum_column_keys(row))
        return expanded

    def reject_selected(self) -> None:
        keys = self.spectrum_column_keys_for_keys(self.selected_keys())
        changed = keys - self.rejected_keys
        if not changed:
            return
        self.push_undo_state(f"Reject {len(changed)} selected spectrum-column row(s)")
        self.rejected_keys.update(keys)
        self.refresh()

    def keep_selected(self) -> None:
        keys = self.spectrum_column_keys_for_keys(self.selected_keys())
        changed = keys & self.rejected_keys
        if not changed:
            return
        self.push_undo_state(f"Keep {len(changed)} selected spectrum-column row(s)")
        self.rejected_keys.difference_update(keys)
        self.refresh()

    def keep_visible(self) -> None:
        keys = self.spectrum_column_keys_for_keys(set(self.visible_df()["fit_row_key"].astype(str)))
        changed = keys & self.rejected_keys
        if not changed:
            return
        self.push_undo_state(f"Keep {len(changed)} visible spectrum-column row(s)")
        self.rejected_keys.difference_update(keys)
        self.refresh()

    def invert_visible(self) -> None:
        keys = self.spectrum_column_keys_for_keys(set(self.visible_df()["fit_row_key"].astype(str)))
        if not keys:
            return
        self.push_undo_state(f"Invert {len(keys)} visible spectrum-column row(s)")
        for key in keys:
            if key in self.rejected_keys:
                self.rejected_keys.remove(key)
            else:
                self.rejected_keys.add(key)
        self.refresh()

    def clear_all_rejected(self) -> None:
        if not self.rejected_keys:
            self.status_var.set("No rejected rows to clear.")
            return
        count = len(self.rejected_keys)
        self.push_undo_state(f"Clear {count} rejected row(s)")
        self.rejected_keys.clear()
        self.refresh()
        self.status_var.set(f"Cleared {count} rejected row(s). Save curation to write this to disk.")

    def on_tree_double_click(self, _event=None) -> None:
        keys = self.spectrum_column_keys_for_keys(self.selected_keys())
        if not keys:
            return
        self.push_undo_state(f"Toggle {len(keys)} table spectrum-column row(s)")
        for key in keys:
            if key in self.rejected_keys:
                self.rejected_keys.remove(key)
            else:
                self.rejected_keys.add(key)
        self.refresh()

    def toggle_plot_key(self, key: str) -> None:
        row = self.row_for_key(key)
        label = "plotted point"
        keys = {str(key)}
        if row is not None:
            keys = self.spectrum_column_keys(row)
            ycol = pd.to_numeric(pd.Series([row.get("y_column_number", np.nan)]), errors="coerce").iloc[0]
            y_label = f"Y{int(ycol):02d}" if pd.notna(ycol) and math.isfinite(float(ycol)) else "Y?"
            label = (
                f"{family_label(row.get('family', ''))} "
                f"{float(row.get('temperature_K', float('nan'))):g} K "
                f"{y_label}"
            )
        self.push_undo_state(f"Toggle spectrum-column {label}")
        if str(key) in self.rejected_keys:
            self.rejected_keys.difference_update(keys)
            self.refresh()
            self.status_var.set(f"Kept spectrum-column {label}. Save curation to write this to disk.")
        else:
            self.rejected_keys.update(keys)
            self.refresh()
            self.status_var.set(f"Rejected spectrum-column {label}. Save curation to write this to disk.")

    def on_pick(self, event) -> None:
        keys = self.artist_keys.get(id(event.artist), [])
        indices = getattr(event, "ind", None)
        if not keys or indices is None or len(indices) == 0:
            return
        index = int(indices[0])
        if index < 0 or index >= len(keys):
            return
        key = keys[index]
        mouse_event = getattr(event, "mouseevent", None)
        button = getattr(mouse_event, "button", None)
        key_state = str(getattr(mouse_event, "key", "") or "").lower()
        inspect_requested = (
            button == 3
            or "shift" in key_state
            or "control" in key_state
            or "ctrl" in key_state
            or self.click_action_var.get() == "Inspect spectrum + fit"
        )
        if inspect_requested:
            self.inspect_fit_row_key(key)
        else:
            self.toggle_plot_key(key)

    def manual_average_args(self) -> argparse.Namespace:
        avg_args = argparse.Namespace(**vars(self.args))
        avg_args.outlier_method = "none"
        return avg_args

    def curated_all_rows(self) -> pd.DataFrame:
        curated = self.df.copy()
        curated["paper_included"] = self.paper_inclusion_mask_for(curated)
        curated["manually_rejected"] = curated["fit_row_key"].astype(str).isin(self.rejected_keys)
        curated["included"] = curated["paper_included"] & ~curated["manually_rejected"]
        curated["paper_excluded_reason"] = ""
        if "family" in curated.columns:
            family_mask = curated["family"].astype(str).isin(self.paper_excluded_families)
            curated.loc[family_mask, "paper_excluded_reason"] = "EXCLUDED_FAMILY"
        if {"family", "sequence"}.issubset(curated.columns):
            sequence_keys = [
                paper_sequence_key(family, sequence)
                for family, sequence in zip(curated["family"].astype(str), curated["sequence"].astype(str))
            ]
            sequence_mask = pd.Series(sequence_keys, index=curated.index).isin(self.paper_excluded_sequences)
            curated.loc[sequence_mask & (curated["paper_excluded_reason"] == ""), "paper_excluded_reason"] = "EXCLUDED_SEQUENCE"
        return curated

    def spectrum_column_manifest(self) -> pd.DataFrame:
        group_cols = [
            "family",
            "sequence",
            "temperature_K",
            "file_path",
            "file_name",
            "spectrum_in_file",
            "y_column_number",
        ]
        available_cols = [col for col in group_cols if col in self.df.columns]
        rows = []
        for keys, group in self.df.groupby(available_cols, dropna=False, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(available_cols, keys))
            first = group.iloc[0]
            fit_keys = group["fit_row_key"].astype(str)
            rejected_mask = fit_keys.isin(self.rejected_keys)
            paper_included = bool(self.paper_inclusion_mask_for(group).all())
            kept_peaks = sorted(group.loc[~rejected_mask, "peak_id"].astype(str).unique(), key=lambda peak: PEAK_PREVIEW_ORDER.get(str(peak), 100))
            rejected_peaks = sorted(group.loc[rejected_mask, "peak_id"].astype(str).unique(), key=lambda peak: PEAK_PREVIEW_ORDER.get(str(peak), 100))
            snr_values = [
                self.snr_by_key.get(str(key), np.nan)
                for key in fit_keys
            ]
            finite_snr = np.asarray([value for value in snr_values if math.isfinite(float(value))], dtype=float)
            norm_noise_values = [
                self.quality_metrics_by_key.get(str(key), {}).get("noise_sigma_g_normalized", np.nan)
                for key in fit_keys
            ]
            finite_norm_noise = np.asarray(
                [value for value in norm_noise_values if math.isfinite(float(value))],
                dtype=float,
            )
            whole_snr_values = [
                self.quality_metrics_by_key.get(str(key), {}).get("whole_spectrum_snr", np.nan)
                for key in fit_keys
            ]
            finite_whole_snr = np.asarray(
                [value for value in whole_snr_values if math.isfinite(float(value))],
                dtype=float,
            )
            whole_noise_values = [
                self.quality_metrics_by_key.get(str(key), {}).get("whole_spectrum_noise_normalized", np.nan)
                for key in fit_keys
            ]
            finite_whole_noise = np.asarray(
                [value for value in whole_noise_values if math.isfinite(float(value))],
                dtype=float,
            )
            g_ref_snr_values = [
                self.quality_metrics_by_key.get(str(key), {}).get("g_reference_snr", np.nan)
                for key in fit_keys
            ]
            finite_g_ref_snr = np.asarray(
                [value for value in g_ref_snr_values if math.isfinite(float(value))],
                dtype=float,
            )
            try:
                source_txt_path = str(
                    locate_spectrum_path(first, self.args.spectra_root, self.args.second_pass_output_root)
                )
            except Exception:
                source_txt_path = ""
            row.update(
                {
                    "family_label": family_label(first.get("family", "")),
                    "sequence_label": sequence_label(first.get("sequence", "")),
                    "source_txt_path": source_txt_path,
                    "x_column_number": 1,
                    "n_trend_peak_rows": int(len(group)),
                    "n_kept_peak_rows": int((~rejected_mask).sum()),
                    "n_rejected_peak_rows": int(rejected_mask.sum()),
                    "kept_peak_ids": ";".join(kept_peaks),
                    "rejected_peak_ids": ";".join(rejected_peaks),
                    "paper_included": paper_included,
                    "included_for_all_trend_peaks": bool(paper_included and not rejected_mask.any()),
                    "included_for_any_trend_peak": bool(paper_included and (~rejected_mask).any()),
                    "snr_min": float(np.nanmin(finite_snr)) if finite_snr.size else float("nan"),
                    "snr_median": float(np.nanmedian(finite_snr)) if finite_snr.size else float("nan"),
                    "snr_max": float(np.nanmax(finite_snr)) if finite_snr.size else float("nan"),
                    "whole_spectrum_snr_min": float(np.nanmin(finite_whole_snr)) if finite_whole_snr.size else float("nan"),
                    "whole_spectrum_snr_median": float(np.nanmedian(finite_whole_snr)) if finite_whole_snr.size else float("nan"),
                    "whole_spectrum_snr_max": float(np.nanmax(finite_whole_snr)) if finite_whole_snr.size else float("nan"),
                    "whole_spectrum_noise_normalized_min": float(np.nanmin(finite_whole_noise)) if finite_whole_noise.size else float("nan"),
                    "whole_spectrum_noise_normalized_median": float(np.nanmedian(finite_whole_noise)) if finite_whole_noise.size else float("nan"),
                    "whole_spectrum_noise_normalized_max": float(np.nanmax(finite_whole_noise)) if finite_whole_noise.size else float("nan"),
                    "noise_sigma_g_normalized_min": float(np.nanmin(finite_norm_noise)) if finite_norm_noise.size else float("nan"),
                    "noise_sigma_g_normalized_median": float(np.nanmedian(finite_norm_noise)) if finite_norm_noise.size else float("nan"),
                    "noise_sigma_g_normalized_max": float(np.nanmax(finite_norm_noise)) if finite_norm_noise.size else float("nan"),
                    "g_reference_snr_min": float(np.nanmin(finite_g_ref_snr)) if finite_g_ref_snr.size else float("nan"),
                    "g_reference_snr_median": float(np.nanmedian(finite_g_ref_snr)) if finite_g_ref_snr.size else float("nan"),
                    "g_reference_snr_max": float(np.nanmax(finite_g_ref_snr)) if finite_g_ref_snr.size else float("nan"),
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    def selected_rows(self) -> pd.DataFrame:
        keys = self.selected_keys()
        if not keys:
            return self.df.iloc[0:0].copy()
        return self.df[self.df["fit_row_key"].astype(str).isin(keys)].copy()

    def row_for_key(self, key: str) -> Optional[pd.Series]:
        rows = self.df[self.df["fit_row_key"].astype(str) == str(key)]
        if rows.empty:
            return None
        return rows.iloc[0]

    def select_tree_key(self, key: str) -> None:
        for iid, row_key in self.tree_iid_to_key.items():
            if row_key == key:
                self.tree.selection_set(iid)
                self.tree.see(iid)
                return

    def spectrum_column_keys(self, row: pd.Series) -> Set[str]:
        same = self.df.loc[spectrum_identity_mask(self.df, row)].copy()
        return set(same["fit_row_key"].astype(str))

    def set_rejection_for_keys(self, keys: Set[str], reject: bool, label: str) -> None:
        if not keys:
            self.status_var.set("No matching trend rows for that action.")
            return
        if reject:
            changed = keys - self.rejected_keys
        else:
            changed = keys & self.rejected_keys
        if not changed:
            self.status_var.set("No change needed for that selection.")
            return
        self.push_undo_state(label)
        if reject:
            self.rejected_keys.update(keys)
        else:
            self.rejected_keys.difference_update(keys)
        self.refresh()
        if self.inspect_key:
            row = self.row_for_key(self.inspect_key)
            if row is not None:
                self.inspect_row = row
                self.update_inspection_plot(row)

    def inspect_selected_spectrum(self) -> None:
        rows = self.selected_rows()
        if rows.empty:
            self.messagebox.showwarning("No row selected", "Select a fitted row first.")
            return
        self.inspect_fit_row_key(str(rows.iloc[0]["fit_row_key"]))

    def inspect_fit_row_key(self, key: str) -> None:
        row = self.row_for_key(key)
        if row is None:
            self.status_var.set("Could not inspect that point because its row is not visible in the trend table.")
            return
        if self.inspect_key is not None and str(key) != str(self.inspect_key):
            self.clear_pending_inspection_refit(redraw=False)
        self.inspect_key = str(key)
        self.inspect_row = row
        self.select_tree_key(str(key))
        if self.inspect_window is None or not self.inspect_window.winfo_exists():
            self.create_inspection_window()
        self.update_inspection_plot(row)
        self.inspect_window.deiconify()
        self.inspect_window.lift()

    def create_inspection_window(self) -> None:
        self.inspect_window = self.tk.Toplevel(self.root)
        self.inspect_window.title("Inspect fitted spectrum-column")
        self.inspect_window.geometry("1200x820")
        self.inspect_window.protocol("WM_DELETE_WINDOW", self.inspect_window.withdraw)

        top = self.ttk.Frame(self.inspect_window, padding=6)
        top.pack(side=self.tk.TOP, fill=self.tk.X)
        self.inspect_info_var = self.tk.StringVar(value="")
        self.ttk.Label(top, textvariable=self.inspect_info_var, wraplength=980).grid(
            row=0, column=0, columnspan=4, sticky="ew", pady=(0, 5)
        )
        self.ttk.Button(top, text="Reject selected peak", command=self.reject_inspected_peak).grid(
            row=1, column=0, sticky="ew", padx=(0, 5)
        )
        self.ttk.Button(top, text="Keep selected peak", command=self.keep_inspected_peak).grid(
            row=1, column=1, sticky="ew", padx=(0, 5)
        )
        self.ttk.Button(top, text="Reject whole spectrum-column", command=self.reject_inspected_spectrum).grid(
            row=1, column=2, sticky="ew", padx=(0, 5)
        )
        self.ttk.Button(top, text="Keep whole spectrum-column", command=self.keep_inspected_spectrum).grid(
            row=1, column=3, sticky="ew"
        )
        for col in range(4):
            top.columnconfigure(col, weight=1)

        refit = self.ttk.LabelFrame(self.inspect_window, text="Single-peak trust-region refit", padding=6)
        refit.pack(side=self.tk.TOP, fill=self.tk.X, padx=6, pady=(0, 4))
        self.ttk.Label(refit, text="Peak").grid(row=0, column=0, sticky="w")
        self.inspect_peak_combo = self.ttk.Combobox(
            refit,
            textvariable=self.inspect_refit_peak_var,
            values=[],
            state="readonly",
            width=10,
        )
        self.inspect_peak_combo.grid(row=0, column=1, sticky="ew", padx=(4, 8))
        self.inspect_peak_combo.bind("<<ComboboxSelected>>", lambda _event: self.on_inspection_refit_peak_changed())
        self.ttk.Label(refit, text="X min").grid(row=0, column=2, sticky="w")
        self.ttk.Entry(refit, textvariable=self.inspect_x_min_var, width=10).grid(
            row=0, column=3, sticky="ew", padx=(4, 8)
        )
        self.ttk.Label(refit, text="X max").grid(row=0, column=4, sticky="w")
        self.ttk.Entry(refit, textvariable=self.inspect_x_max_var, width=10).grid(
            row=0, column=5, sticky="ew", padx=(4, 8)
        )
        self.ttk.Label(refit, text="W min").grid(row=0, column=6, sticky="w")
        self.ttk.Entry(refit, textvariable=self.inspect_w_min_var, width=10).grid(
            row=0, column=7, sticky="ew", padx=(4, 8)
        )
        self.ttk.Label(refit, text="W max").grid(row=0, column=8, sticky="w")
        self.ttk.Entry(refit, textvariable=self.inspect_w_max_var, width=10).grid(
            row=0, column=9, sticky="ew", padx=(4, 8)
        )
        self.ttk.Button(refit, text="Load bounds", command=self.load_inspection_peak_bounds).grid(
            row=0, column=10, sticky="ew", padx=(4, 0)
        )
        self.ttk.Button(refit, text="Use zoom as X bounds", command=self.use_inspection_zoom_as_bounds).grid(
            row=0, column=11, sticky="ew", padx=(4, 0)
        )
        self.ttk.Button(refit, text="Refit this peak only", command=self.refit_inspected_single_peak).grid(
            row=0, column=12, sticky="ew", padx=(4, 0)
        )
        self.ttk.Button(refit, text="Stage previewed refit", command=self.stage_inspection_refit_preview).grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0), padx=(0, 4)
        )
        self.ttk.Button(refit, text="Discard preview", command=self.discard_inspection_refit_preview).grid(
            row=1, column=3, columnspan=3, sticky="ew", pady=(6, 0), padx=(0, 4)
        )
        self.ttk.Label(refit, textvariable=self.inspect_preview_var, wraplength=780).grid(
            row=1, column=6, columnspan=7, sticky="ew", pady=(6, 0)
        )
        for col in (1, 3, 5, 7, 9):
            refit.columnconfigure(col, weight=1)

        self.inspect_figure = self.Figure(figsize=(10.8, 7.2), dpi=100)
        self.inspect_axes = self.inspect_figure.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3.2, 1.0]})
        self.inspect_canvas = self.FigureCanvasTkAgg(self.inspect_figure, master=self.inspect_window)
        self.inspect_toolbar = self.NavigationToolbar2Tk(self.inspect_canvas, self.inspect_window, pack_toolbar=False)
        self.inspect_toolbar.update()
        self.inspect_toolbar.pack(side=self.tk.BOTTOM, fill=self.tk.X)
        self.inspect_canvas.get_tk_widget().pack(side=self.tk.TOP, fill=self.tk.BOTH, expand=True)

    def inspected_selected_row(self) -> Optional[pd.Series]:
        if self.inspect_key is None:
            return None
        return self.row_for_key(self.inspect_key)

    def reject_inspected_peak(self) -> None:
        row = self.inspected_selected_row()
        if row is None:
            return
        self.set_rejection_for_keys({str(row["fit_row_key"])}, True, "Reject inspected peak")

    def keep_inspected_peak(self) -> None:
        row = self.inspected_selected_row()
        if row is None:
            return
        self.set_rejection_for_keys({str(row["fit_row_key"])}, False, "Keep inspected peak")

    def reject_inspected_spectrum(self) -> None:
        row = self.inspected_selected_row()
        if row is None:
            return
        self.set_rejection_for_keys(
            self.spectrum_column_keys(row),
            True,
            "Reject inspected spectrum-column",
        )

    def keep_inspected_spectrum(self) -> None:
        row = self.inspected_selected_row()
        if row is None:
            return
        self.set_rejection_for_keys(
            self.spectrum_column_keys(row),
            False,
            "Keep inspected spectrum-column",
        )

    @staticmethod
    def _fmt_bound(value: object) -> str:
        try:
            f = float(value)
        except Exception:
            return ""
        if not math.isfinite(f):
            return ""
        return f"{f:.6g}"

    def sync_inspection_peak_controls(self, row: pd.Series, same: pd.DataFrame) -> None:
        self.inspect_same_rows = same.copy()
        if not hasattr(self, "inspect_peak_combo"):
            return
        peaks = sorted(same["peak_id"].astype(str).unique(), key=lambda peak: PEAK_PREVIEW_ORDER.get(str(peak), 100))
        self.inspect_peak_combo.configure(values=peaks)
        selected_peak = str(row.get("peak_id", ""))
        if selected_peak not in peaks and peaks:
            selected_peak = peaks[0]
        identity = (
            str(row.get("family", "")),
            str(row.get("sequence", "")),
            str(row.get("temperature_K", "")),
            str(row.get("file_name", "")),
            str(row.get("y_column_number", "")),
            selected_peak,
        )
        if self.inspect_refit_peak_var.get() not in peaks or self.inspect_bounds_identity != identity:
            self.inspect_refit_peak_var.set(selected_peak)
            self.inspect_bounds_identity = identity
            self.load_inspection_peak_bounds()

    def load_inspection_peak_bounds(self) -> None:
        row = self.inspected_selected_row()
        peak_id = str(self.inspect_refit_peak_var.get() or (row.get("peak_id", "") if row is not None else ""))
        if not peak_id:
            return
        peak_row = None
        if self.inspect_same_rows is not None and not self.inspect_same_rows.empty:
            rows = self.inspect_same_rows[self.inspect_same_rows["peak_id"].astype(str) == peak_id]
            if not rows.empty:
                peak_row = rows.iloc[0]
        bounds = {}
        if row is not None:
            try:
                settings = load_bounds_settings(self.args.bounds_json)
                peak_bounds = bounds_for_family_temperature(settings, str(row["family"]), float(row["temperature_K"]))
                bounds = dict(peak_bounds.get(peak_id, {}))
            except Exception:
                bounds = {}
        source = peak_row if peak_row is not None else pd.Series(dtype=object)

        def finite_or_fallback(source_key: str, bounds_key: str) -> object:
            value = source.get(source_key, np.nan)
            try:
                numeric = float(value)
            except Exception:
                numeric = float("nan")
            if math.isfinite(numeric):
                return numeric
            return bounds.get(bounds_key, "")

        self.inspect_x_min_var.set(self._fmt_bound(finite_or_fallback("x_bound_min", "x_min")))
        self.inspect_x_max_var.set(self._fmt_bound(finite_or_fallback("x_bound_max", "x_max")))
        self.inspect_w_min_var.set(self._fmt_bound(finite_or_fallback("width_bound_min", "width_min")))
        self.inspect_w_max_var.set(self._fmt_bound(finite_or_fallback("width_bound_max", "width_max")))

    def on_inspection_refit_peak_changed(self) -> None:
        self.clear_pending_inspection_refit(redraw=False)
        self.load_inspection_peak_bounds()
        row = self.inspected_selected_row()
        if row is not None:
            self.update_inspection_plot(row)

    def use_inspection_zoom_as_bounds(self) -> None:
        if not hasattr(self, "inspect_axes"):
            return
        left, right = self.inspect_axes[0].get_xlim()
        self.inspect_x_min_var.set(self._fmt_bound(min(left, right)))
        self.inspect_x_max_var.set(self._fmt_bound(max(left, right)))
        self.status_var.set("Copied current inspection X zoom into single-peak X bounds.")

    def _inspection_float(self, var, label: str) -> float:
        text = str(var.get()).strip().replace(",", ".")
        try:
            value = float(text)
        except Exception as exc:
            raise ValueError(f"{label} must be a number, got {text!r}.") from exc
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite.")
        return value

    def pending_inspection_refit_matches(self, row: pd.Series) -> bool:
        pending = self.inspect_pending_refit_df
        if pending is None or pending.empty:
            return False
        try:
            return bool(spectrum_identity_mask(pending, row).any())
        except Exception:
            return False

    def clear_pending_inspection_refit(self, redraw: bool = False) -> None:
        self.inspect_pending_refit_df = None
        self.inspect_pending_refit_meta = {}
        if hasattr(self, "inspect_preview_var"):
            self.inspect_preview_var.set("No pending refit preview.")
        if redraw and self.inspect_key:
            row = self.row_for_key(self.inspect_key)
            if row is not None:
                self.update_inspection_plot(row)

    def discard_inspection_refit_preview(self) -> None:
        if self.inspect_pending_refit_df is None or self.inspect_pending_refit_df.empty:
            self.status_var.set("No pending refit preview to discard.")
            return
        self.clear_pending_inspection_refit(redraw=True)
        self.status_var.set("Discarded pending refit preview. No session or disk changes were made.")

    def stage_inspection_refit_preview(self) -> None:
        if self.inspect_pending_refit_df is None or self.inspect_pending_refit_df.empty:
            self.messagebox.showwarning("No preview", "Run 'Refit this peak only' first, then inspect the preview.")
            return
        refit_df = self.inspect_pending_refit_df.copy()
        peak_id = str(refit_df.iloc[0].get("peak_id", ""))
        refit_df["accepted_in_app"] = True
        self.push_undo_state(f"Stage single-peak refit for {peak_id}")
        self.apply_accepted_refits_to_session(refit_df)
        self.accepted_refits.append(refit_df.copy())
        self.clear_pending_inspection_refit(redraw=False)
        self.refresh()
        self.update_staged_refit_status()

        updated_rows = self.df.loc[
            spectrum_identity_mask(self.df, refit_df.iloc[0])
            & (self.df["peak_id"].astype(str) == str(peak_id))
        ]
        if not updated_rows.empty:
            updated = updated_rows.iloc[0]
            self.inspect_key = str(updated.get("fit_row_key", ""))
            self.inspect_row = updated
            self.select_tree_key(self.inspect_key)
            self.update_inspection_plot(updated)
        self.status_var.set(
            f"Staged single-peak refit for {peak_id} in this session. Trends updated; official files untouched."
        )

    def refit_inspected_single_peak(self) -> None:
        row = self.inspected_selected_row()
        if row is None:
            self.messagebox.showwarning("No inspected row", "Inspect/select a fitted row first.")
            return
        peak_id = str(self.inspect_refit_peak_var.get() or row.get("peak_id", ""))
        try:
            x_min = self._inspection_float(self.inspect_x_min_var, "X min")
            x_max = self._inspection_float(self.inspect_x_max_var, "X max")
            w_min = self._inspection_float(self.inspect_w_min_var, "W min")
            w_max = self._inspection_float(self.inspect_w_max_var, "W max")
            self.status_var.set(f"Single-peak refitting {peak_id} with trust-region bounds...")
            self.root.update_idletasks()
            refit_df, meta = trust_region_refit_single_peak(
                row,
                self.all_fit_rows,
                self.args.spectra_root,
                self.args.bounds_json,
                self.args,
                target_peak_id=peak_id,
                x_min=x_min,
                x_max=x_max,
                width_min=w_min,
                width_max=w_max,
                fit_x_min=self.args.trust_fit_x_min,
                fit_x_max=self.args.trust_fit_x_max,
                peak_window_weight=self.args.trust_peak_window_weight,
                bootstrap_runs=self.args.trust_bootstrap_runs,
                random_seed=self.args.trust_random_seed,
            )
            refit_df["refit_timestamp"] = dt.datetime.now().isoformat(timespec="seconds")
            refit_df["selected_peak_id"] = peak_id
            refit_df["selected_fit_row_key"] = row.get("fit_row_key", "")
            refit_df["accepted_in_app"] = False
            self.inspect_pending_refit_df = refit_df.copy()
            self.inspect_pending_refit_meta = dict(meta)
            self.inspect_preview_var.set(
                f"Preview ready for {peak_id}: R2={meta['r_squared']:.5f}, RMSE={meta['rmse']:.4g}, "
                f"bootstrap {meta['bootstrap_success']}/{self.args.trust_bootstrap_runs}. "
                "Blue overlay is proposed; press Stage previewed refit only if it looks better."
            )
            self.update_inspection_plot(row)
            self.status_var.set(
                f"Previewing single-peak refit for {peak_id}. Inspect the blue overlay, then stage or discard."
            )
        except Exception as exc:
            self.messagebox.showerror("Single-peak refit failed", str(exc))
            self.status_var.set(f"Single-peak refit failed: {exc}")

    def update_inspection_plot(self, row: pd.Series) -> None:
        try:
            path = locate_spectrum_path(row, self.args.spectra_root, self.args.second_pass_output_root)
            matrix = read_txt_matrix(path)
            y_index = spectrum_column_index(row, matrix)
            x = pd.to_numeric(matrix.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(matrix.iloc[:, y_index], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            if int(finite.sum()) < 3:
                raise ValueError(f"Not enough finite points in TXT column: {path}")
            x = x[finite]
            y = y[finite]
            order = np.argsort(x)
            x = x[order]
            y = y[order]

            same = self.all_fit_rows.loc[spectrum_identity_mask(self.all_fit_rows, row)].copy()
            if same.empty:
                same = self.df.loc[spectrum_identity_mask(self.df, row)].copy()
            same = same.drop_duplicates("peak_id", keep="last")
            same = same[
                np.isfinite(pd.to_numeric(same["height"], errors="coerce"))
                & np.isfinite(pd.to_numeric(same["position_cm-1"], errors="coerce"))
                & np.isfinite(pd.to_numeric(same["width_fwhm_cm-1"], errors="coerce"))
            ].copy()
            same["_preview_order"] = same["peak_id"].map(lambda peak: PEAK_PREVIEW_ORDER.get(str(peak), 100))
            same = same.sort_values(["_preview_order", "position_cm-1", "peak_id"])
            self.sync_inspection_peak_controls(row, same)

            preview_same = None
            preview_peak_id = ""
            preview_meta = {}
            if self.pending_inspection_refit_matches(row):
                pending = self.inspect_pending_refit_df.copy()
                preview_same = same.copy()
                for _pending_idx, refit_row in pending.iterrows():
                    preview_peak_id = str(refit_row.get("peak_id", ""))
                    peak_mask = preview_same["peak_id"].astype(str) == preview_peak_id
                    if peak_mask.any():
                        for col in [
                            "height",
                            "position_cm-1",
                            "width_fwhm_cm-1",
                            "height_std",
                            "position_std_cm-1",
                            "width_fwhm_std_cm-1",
                            "r_squared",
                            "rmse",
                            "x_bound_min",
                            "x_bound_max",
                            "width_bound_min",
                            "width_bound_max",
                        ]:
                            if col in refit_row.index:
                                preview_same.loc[peak_mask, col] = refit_row[col]
                    else:
                        preview_same = pd.concat([preview_same, refit_row.to_frame().T], ignore_index=True)
                preview_same["_preview_order"] = preview_same["peak_id"].map(
                    lambda peak: PEAK_PREVIEW_ORDER.get(str(peak), 100)
                )
                preview_same = preview_same.sort_values(["_preview_order", "position_cm-1", "peak_id"])
                preview_meta = dict(self.inspect_pending_refit_meta)

            ax, residual_ax = self.inspect_axes
            ax.clear()
            residual_ax.clear()

            family = str(row.get("family", ""))
            sequence = str(row.get("sequence", ""))
            sequences = sorted(
                self.all_fit_rows[self.all_fit_rows["family"].astype(str) == family]["sequence"].astype(str).unique(),
                key=sequence_sort_key,
            )
            style = sequence_marker_style(family, sequence, sequences)
            ax.plot(x, y, color="#555555", linewidth=0.95, alpha=0.78, label="Baseline-corrected spectrum", zorder=1)
            ax.scatter(
                x,
                y,
                marker=style["marker"],
                s=max(self.args.grid_marker_size * 0.45, 12.0),
                facecolors=style["facecolors"],
                edgecolors=style["edgecolors"],
                linewidths=max(style["linewidths"] * 0.65, 0.8),
                alpha=0.82,
                label=family_label(family),
                zorder=2,
            )

            total_data = np.zeros_like(x, dtype=float)
            if not same.empty:
                x_smooth = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), max(2400, len(x) * 3))
                total_smooth = np.zeros_like(x_smooth, dtype=float)
                selected_peak = str(row.get("peak_id", ""))
                for _idx, fit in same.iterrows():
                    peak = str(fit["peak_id"])
                    height = float(fit["height"])
                    position = float(fit["position_cm-1"])
                    width = float(fit["width_fwhm_cm-1"])
                    component_data = single_lorentzian(x, height, position, width)
                    component_smooth = single_lorentzian(x_smooth, height, position, width)
                    total_data += component_data
                    total_smooth += component_smooth
                    color = PEAK_COLORS.get(peak, "#777777")
                    ax.plot(
                        x_smooth,
                        component_smooth,
                        linestyle="--",
                        linewidth=2.2 if peak == selected_peak else 1.45,
                        color=color,
                        alpha=0.96 if peak == selected_peak else 0.78,
                        label=PEAK_LABELS.get(peak, peak),
                        zorder=4 if peak == selected_peak else 3,
                    )
                ax.plot(
                    x_smooth,
                    total_smooth,
                    color="#e02b2b",
                    linewidth=2.2,
                    label="Current total fit",
                    zorder=5,
                )
                if preview_same is not None and not preview_same.empty:
                    preview_total_data = np.zeros_like(x, dtype=float)
                    preview_total_smooth = np.zeros_like(x_smooth, dtype=float)
                    preview_target_smooth = None
                    for _idx, fit in preview_same.iterrows():
                        peak = str(fit["peak_id"])
                        height = float(fit["height"])
                        position = float(fit["position_cm-1"])
                        width = float(fit["width_fwhm_cm-1"])
                        component_data = single_lorentzian(x, height, position, width)
                        component_smooth = single_lorentzian(x_smooth, height, position, width)
                        preview_total_data += component_data
                        preview_total_smooth += component_smooth
                        if peak == preview_peak_id:
                            preview_target_smooth = component_smooth
                    if preview_target_smooth is not None:
                        ax.plot(
                            x_smooth,
                            preview_target_smooth,
                            color="#1f5eff",
                            linestyle=":",
                            linewidth=3.0,
                            alpha=0.95,
                            label=f"Proposed {PEAK_LABELS.get(preview_peak_id, preview_peak_id)}",
                            zorder=7,
                        )
                    ax.plot(
                        x_smooth,
                        preview_total_smooth,
                        color="#1f5eff",
                        linewidth=2.6,
                        alpha=0.96,
                        label="Proposed total fit",
                        zorder=6,
                    )
                    preview_residual = y - preview_total_data
                    residual_ax.plot(
                        x,
                        preview_residual,
                        color="#1f5eff",
                        linewidth=1.15,
                        alpha=0.95,
                        label="Proposed residual",
                    )
                    r2_preview = preview_meta.get("r_squared", np.nan)
                    rmse_preview = preview_meta.get("rmse", np.nan)
                    try:
                        preview_text = f"Proposed {preview_peak_id}: R2={float(r2_preview):.5f}, RMSE={float(rmse_preview):.4g}"
                    except Exception:
                        preview_text = f"Proposed {preview_peak_id} preview"
                    ax.text(
                        0.01,
                        0.90,
                        preview_text,
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=8.5,
                        color="#1f5eff",
                    )
                try:
                    active_peak = str(self.inspect_refit_peak_var.get() or selected_peak)
                    active_x_min = self._inspection_float(self.inspect_x_min_var, "X min")
                    active_x_max = self._inspection_float(self.inspect_x_max_var, "X max")
                    left, right = sorted([active_x_min, active_x_max])
                    bound_color = PEAK_COLORS.get(active_peak, "#777777")
                    for axis in (ax, residual_ax):
                        axis.axvspan(left, right, color=bound_color, alpha=0.08, linewidth=0, zorder=0)
                        axis.axvline(left, color=bound_color, linestyle=":", linewidth=1.0, alpha=0.85, zorder=0)
                        axis.axvline(right, color=bound_color, linestyle=":", linewidth=1.0, alpha=0.85, zorder=0)
                    ax.text(
                        0.01,
                        0.97,
                        f"Refit window: {active_peak}  X={left:.3g}-{right:.3g} cm$^{{-1}}$",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=8.5,
                        color=bound_color,
                    )
                except Exception:
                    pass
                residual = y - total_data
                residual_ax.axhline(0.0, color="#888888", linestyle="--", linewidth=1.0)
                residual_ax.plot(x, residual, color="#333333", linewidth=0.9, alpha=0.62, label="Current residual")
                if preview_same is not None:
                    residual_ax.legend(loc="upper right", fontsize=8.0, frameon=False)
            else:
                residual_ax.text(0.5, 0.5, "No fitted peaks found for this spectrum-column.", transform=residual_ax.transAxes)

            selected_key = str(row.get("fit_row_key", ""))
            rejected = selected_key in self.rejected_keys
            ycol = pd.to_numeric(row.get("y_column_number", np.nan), errors="coerce")
            y_label = f"Y{int(ycol):02d}" if math.isfinite(float(ycol)) else str(row.get("y_column_number", ""))
            title = (
                f"{family_label(family)} | {sequence_label(sequence)} | {float(row['temperature_K']):g} K | "
                f"{Path(str(path)).name} | {y_label} | selected {row.get('peak_id', '')}"
            )
            ax.set_title(title, fontsize=11.5)
            ax.set_ylabel("Intensity (a.u.)")
            residual_ax.set_ylabel("Residual")
            residual_ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
            r2 = row.get("r_squared", np.nan)
            rmse = row.get("rmse", np.nan)
            info = (
                f"{'REJECTED' if rejected else 'KEPT'} selected trend row | "
                f"R2={float(r2):.5f} | RMSE={float(rmse):.4g} | "
                f"{len(same)} fitted peak(s) shown | {path}"
            )
            self.inspect_info_var.set(info)

            ax.legend(loc="upper right", fontsize=8.5, frameon=False, ncol=2)
            for axis in self.inspect_axes:
                style_single_axis(axis)
                axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
            self.inspect_figure.tight_layout()
            self.inspect_canvas.draw_idle()
            self.status_var.set(f"Inspecting {family_label(family)} {float(row['temperature_K']):g} K {y_label}.")
        except Exception as exc:
            self.messagebox.showerror("Inspect spectrum failed", str(exc))
            self.status_var.set(f"Inspect spectrum failed: {exc}")

    def trust_refit_selected(self) -> None:
        selected = self.selected_rows()
        if selected.empty:
            self.messagebox.showwarning("No row selected", "Select one fitted row/spectrum first.")
            return
        unique_rows = []
        seen = set()
        for _idx, row in selected.iterrows():
            ident = (
                str(row.get("family", "")),
                str(row.get("sequence", "")),
                f"{float(row.get('temperature_K', float('nan'))):.6g}",
                str(row.get("file_name", "")),
                str(row.get("spectrum_in_file", "")),
                str(row.get("y_column_number", "")),
            )
            if ident not in seen:
                seen.add(ident)
                unique_rows.append(row)
        all_refits = []
        messages = []
        try:
            self.status_var.set(f"Trust-region refitting {len(unique_rows)} selected spectrum-column(s)...")
            self.root.update_idletasks()
            for row in unique_rows:
                refit_df, meta = trust_region_refit_spectrum(
                    row,
                    self.df,
                    self.args.spectra_root,
                    self.args.bounds_json,
                    self.args,
                    fit_x_min=self.args.trust_fit_x_min,
                    fit_x_max=self.args.trust_fit_x_max,
                    peak_window_weight=self.args.trust_peak_window_weight,
                    bootstrap_runs=self.args.trust_bootstrap_runs,
                    random_seed=self.args.trust_random_seed,
                )
                refit_df["refit_timestamp"] = dt.datetime.now().isoformat(timespec="seconds")
                refit_df["selected_peak_id"] = row.get("peak_id", "")
                refit_df["selected_fit_row_key"] = row.get("fit_row_key", "")
                refit_df["accepted_in_app"] = False
                all_refits.append(refit_df)
                messages.append(
                    f"{Path(str(meta['path'])).name}: R2={meta['r_squared']:.5f}, "
                    f"RMSE={meta['rmse']:.4g}, bootstrap {meta['bootstrap_success']}/{self.args.trust_bootstrap_runs}"
                )
            combined = pd.concat(all_refits, ignore_index=True) if all_refits else pd.DataFrame()
            if not combined.empty:
                accept = self.messagebox.askyesno(
                    "Stage trust-region refit?",
                    "\n".join(messages[:12])
                    + "\n\nStage these PC trust-region values into the current app session/RAM?\n"
                    + "Trend plots update immediately. Official CSV/XLSX fitted-results files stay untouched until export.",
                )
                if accept:
                    combined["accepted_in_app"] = True
                    self.push_undo_state(f"Accept trust-region refit for {len(unique_rows)} spectrum-column(s)")
                    self.apply_accepted_refits_to_session(combined)
                    self.accepted_refits.append(combined.copy())
                    self.refresh()
                    self.update_staged_refit_status()
                    self.status_var.set(
                        f"Staged {len(combined)} peak row(s) in this session. Trends updated; official files untouched."
                    )
                else:
                    self.status_var.set("Trust-region refit was not staged; no session or disk changes were made.")
            else:
                self.refresh()
        except Exception as exc:
            self.messagebox.showerror("Trust-region refit failed", str(exc))
            self.status_var.set(f"Trust-region refit failed: {exc}")

    def apply_accepted_refits_to_session(self, refit_df: pd.DataFrame) -> None:
        def apply_to_table(table: pd.DataFrame) -> pd.DataFrame:
            updated = table.copy()
            for _identity, group in refit_df.groupby(
                ["family", "sequence", "temperature_K", "file_name", "spectrum_in_file", "y_column_number"], dropna=False
            ):
                first = group.iloc[0]
                spectrum_mask = spectrum_identity_mask(updated, first)
                for _rid, fit_row in group.iterrows():
                    peak_mask = spectrum_mask & (updated["peak_id"].astype(str) == str(fit_row["peak_id"]))
                    for col in [
                        "height",
                        "position_cm-1",
                        "width_fwhm_cm-1",
                        "height_std",
                        "position_std_cm-1",
                        "width_fwhm_std_cm-1",
                        "x_bound_min",
                        "x_bound_max",
                        "width_bound_min",
                        "width_bound_max",
                        "r_squared",
                        "rmse",
                        "weighted_sse",
                        "weighted_sse_data",
                        "anti_burial_sse",
                        "n_points_fit",
                        "fit_x_min",
                        "fit_x_max",
                        "bootstrap_requested",
                        "bootstrap_success",
                        "quality_flag",
                        "refit_method",
                        "source_txt_path",
                        "trust_region_success",
                        "trust_region_message",
                    ]:
                        if col in fit_row.index:
                            if col not in updated.columns:
                                updated[col] = np.nan
                            if isinstance(fit_row[col], (str, bool)):
                                updated[col] = updated[col].astype("object")
                            updated.loc[peak_mask, col] = fit_row[col]
            return updated

        self.df = apply_to_table(self.df)
        self.all_fit_rows = apply_to_table(self.all_fit_rows)

    def export_accepted_refit_folder(self) -> None:
        if not self.accepted_refits:
            self.messagebox.showwarning("No accepted refits", "No accepted trust-region refits are staged yet.")
            return
        accepted = pd.concat(self.accepted_refits, ignore_index=True)
        accepted = accepted.drop_duplicates(
            subset=["family", "sequence", "temperature_K", "file_name", "spectrum_in_file", "y_column_number", "peak_id"],
            keep="last",
        )
        try:
            self.status_var.set(f"Exporting complete accepted refit folder to {self.args.accepted_refit_root}...")
            self.root.update_idletasks()
            target = export_complete_refitted_folder(self.args.fitted_root, self.args.accepted_refit_root, accepted)
            self.status_var.set(f"Exported complete accepted refit folder: {target}")
            self.messagebox.showinfo(
                "Accepted refit folder exported",
                f"Created/updated complete fitted-results folder:\n{target}\n\n"
                "Original cluster fitted folder was not modified.",
            )
        except Exception as exc:
            self.messagebox.showerror("Export accepted refit folder failed", str(exc))
            self.status_var.set(f"Export failed: {exc}")

    def staged_refits_dataframe(self) -> pd.DataFrame:
        if not self.accepted_refits:
            return pd.DataFrame()
        accepted = pd.concat(self.accepted_refits, ignore_index=True)
        return accepted.drop_duplicates(
            subset=["family", "sequence", "temperature_K", "file_name", "spectrum_in_file", "y_column_number", "peak_id"],
            keep="last",
        )

    def export_paper_clean_fitted_folder(self) -> None:
        try:
            self.save_curation()
            staged = self.staged_refits_dataframe()
            self.status_var.set(f"Exporting paper-clean fitted folder to {self.args.paper_clean_root}...")
            self.root.update_idletasks()
            target = export_paper_clean_fitted_folder(
                self.args.fitted_root,
                self.args.paper_clean_root,
                staged,
                self.rejected_keys,
                self.paper_excluded_families,
                self.paper_excluded_sequences,
            )
            self.status_var.set(f"Exported paper-clean fitted folder: {target}")
            self.messagebox.showinfo(
                "Paper-clean fitted folder exported",
                f"Created paper-clean fitted-results folder:\n{target}\n\n"
                "It includes only paper-included families/UP-DOWN folders and kept spectrum-columns.\n"
                "Staged PC refits were applied. Original fitted folders were not modified.",
            )
        except Exception as exc:
            self.messagebox.showerror("Export paper-clean fitted folder failed", str(exc))
            self.status_var.set(f"Paper-clean export failed: {exc}")

    def export_stage_paper_package(self) -> None:
        try:
            stage_dir = self.stage_output_dir()
            stage_dir.mkdir(parents=True, exist_ok=True)
            self.status_var.set(f"Building complete {self.current_stage()} paper package...")
            self.root.update_idletasks()

            archived_paths = []
            review_dir = stage_dir / "CURATION_REVIEW_PLOTS_WITH_REJECTED"
            mean_dir = stage_dir / "CURATED_TEMPERATURE_MEAN_GRIDS"
            curated_plot_dir = stage_dir / "CURATED_PARAMETER_GRIDS"
            for output_path in (review_dir, mean_dir, curated_plot_dir):
                archived = archive_existing_output_path(output_path)
                if archived is not None:
                    archived_paths.append(str(archived))

            self.save_curation()
            self.ensure_quality_metrics()
            self.save_stage_review_plots(show_message=False, save_first=False)
            averaged_for_plots = average_parameters_by_temperature(
                self.paper_included_df(self.df),
                self.manual_average_args(),
                self.rejected_keys,
            )
            curated_plot_dir.mkdir(parents=True, exist_ok=True)
            plot_all_parameter_grids(
                averaged_for_plots,
                curated_plot_dir,
                self.args,
                suffix=f"_{self.current_stage().lower()}_curated",
            )

            tables_dir = stage_dir / "CURATED_TABLES_AND_MANIFESTS"
            pre_averaging_audit_dir = stage_dir / "PRE_AVERAGING_SOURCE_COLUMN_EXAM"
            averaged_txt_dir = stage_dir / "AVERAGED_BASELINE_CORRECTED_TXT_FOR_CLUSTER"
            clean_fitted_dir = stage_dir / "FITTED_RESULTS_PAPER_CLEAN"
            for output_path in (tables_dir, pre_averaging_audit_dir, averaged_txt_dir, clean_fitted_dir):
                archived = archive_existing_output_path(output_path)
                if archived is not None:
                    archived_paths.append(str(archived))

            tables_dir.mkdir(parents=True, exist_ok=True)
            table_names = [
                self.args.curation_json,
                "Gamma_T_Comparison_curated_all_fit_rows.csv",
                "Gamma_T_Comparison_curated_kept_fit_rows.csv",
                "Gamma_T_Comparison_curated_temperature_means.csv",
                "Gamma_T_Comparison_spectrum_column_manifest_all.csv",
                "Gamma_T_Comparison_clean_spectrum_columns_for_averaging.csv",
                "Gamma_T_Comparison_SNR_by_spectrum.csv",
                "Gamma_T_Comparison_SNR_by_temperature.csv",
                "Gamma_T_Comparison_parameter_rows.csv",
                "Gamma_T_Comparison_bound_hit_details.csv",
                "Gamma_T_Comparison_bound_hit_summary.csv",
            ]
            copied_tables = []
            for name in table_names:
                source = self.output_dir / name
                if source.exists():
                    destination = tables_dir / source.name
                    shutil.copy2(source, destination)
                    copied_tables.append(str(destination))

            source_refit_manifest = Path(self.args.fitted_root) / "pc_refit_accepted_manifest.csv"
            source_refit_manifest_rows = count_csv_rows(source_refit_manifest)
            if source_refit_manifest.exists():
                shutil.copy2(source_refit_manifest, tables_dir / source_refit_manifest.name)
            source_refit_meta = Path(self.args.fitted_root) / "pc_refit_update_manifest.json"
            if source_refit_meta.exists():
                shutil.copy2(source_refit_meta, tables_dir / source_refit_meta.name)

            pre_averaging_audit = copy_pre_averaging_exam_audit(
                self.args.second_pass_output_root,
                pre_averaging_audit_dir,
                self.paper_excluded_families,
                self.paper_excluded_sequences,
            )

            manifest = self.spectrum_column_manifest()
            averaged_result = export_averaged_second_pass_txt(
                manifest,
                self.args.spectra_root,
                averaged_txt_dir,
            )

            staged = self.staged_refits_dataframe()
            clean_fitted_root = export_paper_clean_fitted_folder(
                self.args.fitted_root,
                clean_fitted_dir,
                staged,
                self.rejected_keys,
                self.paper_excluded_families,
                self.paper_excluded_sequences,
            )

            curated = self.curated_all_rows()
            included_rows = curated[curated["included"]].copy()
            package_manifest = {
                "created": dt.datetime.now().isoformat(timespec="seconds"),
                "stage": self.current_stage(),
                "package_root": str(stage_dir),
                "source_fitted_root_read_only": str(self.args.fitted_root),
                "source_spectra_root_read_only": str(self.args.spectra_root),
                "curated_tables_dir": str(tables_dir),
                "pre_averaging_source_column_exam_dir": str(pre_averaging_audit_dir),
                "averaged_baseline_corrected_txt_dir": str(averaged_result["output_root"]),
                "clean_fitted_results_dir": str(clean_fitted_root),
                "review_plots_dir": str(stage_dir / "CURATION_REVIEW_PLOTS_WITH_REJECTED"),
                "temperature_mean_plots_dir": str(stage_dir / "CURATED_TEMPERATURE_MEAN_GRIDS"),
                "curated_parameter_plots_dir": str(stage_dir / "CURATED_PARAMETER_GRIDS"),
                "n_fit_peak_rows_total": int(len(curated)),
                "n_fit_peak_rows_included": int(len(included_rows)),
                "n_fit_peak_rows_rejected_or_excluded": int(len(curated) - len(included_rows)),
                "n_spectrum_columns_total": int(len(manifest)),
                "n_averaged_txt_written": int(averaged_result["n_written"]),
                "n_source_columns_used_for_averages": int(averaged_result["n_used_columns"]),
                "n_pre_averaging_source_columns_all": int(pre_averaging_audit.get("n_source_columns_all", 0)),
                "n_pre_averaging_source_columns_used_for_average": int(
                    pre_averaging_audit.get("n_source_columns_used_for_average", 0)
                ),
                "n_pre_averaging_source_columns_removed_before_average": int(
                    pre_averaging_audit.get("n_source_columns_removed_before_average", 0)
                ),
                "source_pc_refit_peak_rows_already_included": int(source_refit_manifest_rows),
                "source_pc_refit_manifest": str(source_refit_manifest) if source_refit_manifest.exists() else "",
                "staged_refit_peak_rows_applied_this_export": int(len(staged)),
                "paper_excluded_families": sorted(self.paper_excluded_families, key=family_sort_key),
                "paper_excluded_sequences": [
                    {"family": split_paper_sequence_key(key)[0], "sequence": split_paper_sequence_key(key)[1]}
                    for key in sorted(self.paper_excluded_sequences)
                ],
                "archived_previous_outputs": archived_paths,
                "note": (
                    "This folder is the self-contained paper package for the selected stage. "
                    "It contains kept averaged baseline-corrected TXT files, clean fitted CSV/XLSX files, "
                    "curated/SNR tables, and review figures. Source fitted folders and source TXT folders were not modified."
                ),
            }
            manifest_path = stage_dir / "PAPER_READY_PACKAGE_MANIFEST.json"
            manifest_path.write_text(json.dumps(package_manifest, indent=2), encoding="utf-8")
            pd.DataFrame([package_manifest]).to_csv(stage_dir / "PAPER_READY_PACKAGE_MANIFEST.csv", index=False)

            self.status_var.set(f"Saved complete {self.current_stage()} paper package to {stage_dir}")
            self.messagebox.showinfo(
                "ALL GOOD paper package saved",
                f"Saved complete {self.current_stage()} package in:\n{stage_dir}\n\n"
                f"Clean averaged TXT:\n{averaged_result['output_root']}\n\n"
                f"Clean fitted CSV/XLSX:\n{clean_fitted_root}\n\n"
                f"Pre-averaging keep/reject audit:\n{pre_averaging_audit_dir}\n\n"
                "The source fitted/TXT folders were not modified. If you delete this stage package later, "
                "the app still has the source fitted folder to rebuild it.",
            )
        except Exception as exc:
            self.messagebox.showerror("Save ALL GOOD paper package failed", str(exc))
            self.status_var.set(f"ALL GOOD package failed: {exc}")

    def compute_snr_summary(self) -> None:
        try:
            self.status_var.set("Computing SNR by spectrum and by family/sequence/temperature...")
            self.root.update_idletasks()
            snr_rows, snr_summary = compute_snr_tables(
                self.df,
                self.args.spectra_root,
                self.args.bounds_json,
                self.output_dir,
                self.args.trust_fit_x_min,
                self.args.trust_fit_x_max,
                self.args.second_pass_output_root,
            )
            self.cache_snr_metrics(snr_rows)
            self.color_mode_var.set("Whole-spectrum SNR")
            self.refresh()
            self.status_var.set(
                f"SNR complete: {len(snr_rows)} spectrum-peak rows, {len(snr_summary)} grouped temperature rows."
            )
            self.messagebox.showinfo(
                "SNR complete",
                "Saved:\n"
                f"{self.output_dir / 'Gamma_T_Comparison_SNR_by_spectrum.csv'}\n"
                f"{self.output_dir / 'Gamma_T_Comparison_SNR_by_temperature.csv'}",
            )
        except Exception as exc:
            self.messagebox.showerror("SNR failed", str(exc))
            self.status_var.set(f"SNR failed: {exc}")

    def quality_metrics_have_global_snr(self) -> bool:
        for metrics in self.quality_metrics_by_key.values():
            value = metrics.get("whole_spectrum_snr", np.nan)
            value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.notna(value) and math.isfinite(float(value)):
                return True
        return False

    def ensure_quality_metrics(self) -> None:
        if self.quality_metrics_by_key and self.quality_metrics_have_global_snr():
            return
        self.status_var.set("Computing whole-spectrum SNR metrics for review plots...")
        self.root.update_idletasks()
        snr_rows, _snr_summary = compute_snr_tables(
            self.df,
            self.args.spectra_root,
            self.args.bounds_json,
            self.output_dir,
            self.args.trust_fit_x_min,
            self.args.trust_fit_x_max,
            self.args.second_pass_output_root,
        )
        self.cache_snr_metrics(snr_rows)

    def current_stage(self) -> str:
        stage = str(self.stage_var.get() or "BEFORE").strip().upper()
        return "AFTER" if stage == "AFTER" else "BEFORE"

    def stage_output_dir(self) -> Path:
        return self.output_dir / self.current_stage()

    def save_curation(self) -> None:
        paper_sequence_payload = [
            {"family": split_paper_sequence_key(key)[0], "sequence": split_paper_sequence_key(key)[1]}
            for key in sorted(self.paper_excluded_sequences)
        ]
        payload = {
            "updated": dt.datetime.now().isoformat(timespec="seconds"),
            "rejected_count": len(self.rejected_keys),
            "rejected_keys": sorted(self.rejected_keys),
            "paper_excluded_families": sorted(self.paper_excluded_families, key=family_sort_key),
            "paper_excluded_sequences": paper_sequence_payload,
            "note": "Each key is one fitted spectrum-column/peak row; rejecting it removes height, FWHM, and position together. Paper exclusions remove whole families or family/UP-DOWN folders from paper-clean outputs.",
        }
        self.rejections_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        curated = self.curated_all_rows()
        all_path = self.output_dir / "Gamma_T_Comparison_curated_all_fit_rows.csv"
        kept_path = self.output_dir / "Gamma_T_Comparison_curated_kept_fit_rows.csv"
        avg_path = self.output_dir / "Gamma_T_Comparison_curated_temperature_means.csv"
        manifest_all_path = self.output_dir / "Gamma_T_Comparison_spectrum_column_manifest_all.csv"
        manifest_clean_path = self.output_dir / "Gamma_T_Comparison_clean_spectrum_columns_for_averaging.csv"
        curated.to_csv(all_path, index=False)
        curated[curated["included"]].to_csv(kept_path, index=False)
        averaged = average_parameters_by_temperature(self.paper_included_df(self.df), self.manual_average_args(), self.rejected_keys)
        averaged.to_csv(avg_path, index=False)
        manifest = self.spectrum_column_manifest()
        manifest.to_csv(manifest_all_path, index=False)
        manifest[manifest["included_for_all_trend_peaks"]].to_csv(manifest_clean_path, index=False)
        self.status_var.set(f"Saved curation JSON and curated CSVs to {self.output_dir}")
        print(f"Saved {self.rejections_path}")
        print(f"Saved {all_path}")
        print(f"Saved {kept_path}")
        print(f"Saved {avg_path}")
        print(f"Saved {manifest_all_path}")
        print(f"Saved {manifest_clean_path}")

    def save_curated_plots(self) -> None:
        self.save_curation()
        averaged = average_parameters_by_temperature(self.paper_included_df(self.df), self.manual_average_args(), self.rejected_keys)
        curated_dir = self.stage_output_dir() / "CURATED_PARAMETER_GRIDS"
        curated_dir.mkdir(parents=True, exist_ok=True)
        plot_all_parameter_grids(averaged, curated_dir, self.args, suffix=f"_{self.current_stage().lower()}_curated")
        self.messagebox.showinfo("Curated plots saved", f"Saved curated parameter grids in:\n{curated_dir}")

    def save_stage_review_plots(self, show_message: bool = True, save_first: bool = True) -> Path:
        if save_first:
            self.save_curation()
        self.ensure_quality_metrics()
        stage_dir = self.stage_output_dir()
        review_dir = stage_dir / "CURATION_REVIEW_PLOTS_WITH_REJECTED"
        mean_dir = stage_dir / "CURATED_TEMPERATURE_MEAN_GRIDS"
        self.status_var.set(f"Saving {self.current_stage()} review plots...")
        self.root.update_idletasks()
        plot_curation_review_grids(
            self.paper_included_df(self.df),
            review_dir,
            self.args,
            self.rejected_keys,
            self.quality_metrics_by_key,
            suffix=f"_{self.current_stage().lower()}",
        )
        averaged = average_parameters_by_temperature(self.paper_included_df(self.df), self.manual_average_args(), self.rejected_keys)
        plot_all_parameter_grids(
            averaged,
            mean_dir,
            self.args,
            suffix=f"_{self.current_stage().lower()}_kept_temperature_means",
        )
        self.status_var.set(f"Saved {self.current_stage()} review plots to {stage_dir}")
        if show_message:
            self.messagebox.showinfo(
                "Review plots saved",
                f"Saved rejected-point review plots and kept-temperature mean grids in:\n{stage_dir}",
            )
        return stage_dir

    def export_averaged_txt_for_cluster(self) -> None:
        try:
            self.save_curation()
            self.save_stage_review_plots(show_message=False, save_first=False)
            manifest = self.spectrum_column_manifest()
            result = export_averaged_second_pass_txt(
                manifest,
                self.args.spectra_root,
                self.args.second_pass_output_root,
            )
            self.status_var.set(
                f"Exported {result['n_written']} averaged TXT files using "
                f"{result['n_used_columns']} kept columns out of {result['n_total_columns']}."
            )
            self.messagebox.showinfo(
                "Averaged TXT export complete",
                "Saved second-pass cluster input:\n"
                f"{result['output_root']}\n\n"
                f"Averaged TXT files: {result['n_written']}\n"
                f"Kept source columns used: {result['n_used_columns']} / {result['n_total_columns']}\n\n"
                "Also saved manifests and BEFORE/AFTER review plots.",
            )
        except Exception as exc:
            self.messagebox.showerror("Averaged TXT export failed", str(exc))
            self.status_var.set(f"Averaged TXT export failed: {exc}")

    def on_close(self) -> None:
        try:
            self.save_curation()
        finally:
            self.root.destroy()


def launch_curation_app(
    df: pd.DataFrame,
    args: argparse.Namespace,
    all_fit_rows: Optional[pd.DataFrame] = None,
) -> None:
    import tkinter as tk

    root = tk.Tk()
    FitTrendCurationApp(root, df, args, all_fit_rows=all_fit_rows)
    root.mainloop()


def load_rejection_keys(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        keys = set(str(key) for key in payload.get("rejected_keys", []))
        print(f"Loaded {len(keys)} manual rejection(s) from {path}")
        return keys
    except Exception as exc:
        print(f"Could not load curation JSON {path}: {exc}")
        return set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot and curate temperature trends from ETH cluster Lorentzian fits.")
    parser.add_argument("--fitted-root", type=Path, default=DEFAULT_FITTED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--spectra-root",
        type=Path,
        default=DEFAULT_SPECTRA_ROOT,
        help="Baseline-corrected TXT source used for SNR and local trust-region refits.",
    )
    parser.add_argument(
        "--bounds-json",
        type=Path,
        default=DEFAULT_BOUNDS_JSON,
        help="Temperature/family peak-bound JSON used as trust-region bounds for local refits.",
    )
    parser.add_argument(
        "--accepted-refit-root",
        type=Path,
        default=DEFAULT_ACCEPTED_REFIT_ROOT,
        help="Complete copied fitted-results folder written after accepting PC trust-region refits.",
    )
    parser.add_argument(
        "--paper-clean-root",
        type=Path,
        default=DEFAULT_PAPER_CLEAN_ROOT,
        help="Fresh fitted-results folder written with only paper-included families/sequences and kept spectrum-columns.",
    )
    parser.add_argument(
        "--second-pass-output-root",
        type=Path,
        default=DEFAULT_SECOND_PASS_AVERAGED_ROOT,
        help="Folder written by the app when exporting kept spectra averaged by family/sequence/temperature.",
    )
    parser.add_argument(
        "--analysis-stage",
        choices=["BEFORE", "AFTER"],
        default=DEFAULT_ANALYSIS_STAGE,
        help="Stage subfolder used for saved review plots. Use BEFORE before second-pass refitting and AFTER after new fits.",
    )
    parser.add_argument("--app", action="store_true", help="Launch the interactive outlier-curation app.")
    parser.add_argument(
        "--curation-json",
        default="Gamma_T_Comparison_outlier_rejections.json",
        help="JSON file name, inside --output-dir, used to persist manual outlier rejections.",
    )
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--peaks", nargs="+", default=["RBLM", "D", "G"], choices=["RBLM", "D", "G"])
    parser.add_argument("--min-r2", type=float, default=None, help="Optional minimum cluster-fit R2 filter.")
    parser.add_argument("--temp-min", type=float, default=50.0)
    parser.add_argument("--temp-max", type=float, default=310.0)
    parser.add_argument("--marker-size", type=float, default=56.0)
    parser.add_argument("--grid-marker-size", type=float, default=40.0)
    parser.add_argument("--overlay-width", type=float, default=8.4)
    parser.add_argument("--overlay-height", type=float, default=5.4)
    parser.add_argument("--grid-width", type=float, default=12.8)
    parser.add_argument("--grid-height", type=float, default=15.0)
    parser.add_argument("--grid-title-font", type=float, default=11.5)
    parser.add_argument("--legend-font", type=float, default=10.5)
    parser.add_argument("--legend-loc", default="best")
    parser.add_argument(
        "--raw-points",
        action="store_true",
        help="Plot every individual fitted spectrum instead of cleaned temperature means.",
    )
    parser.add_argument(
        "--outlier-method",
        choices=["mad", "iqr", "none"],
        default="none",
        help="Outlier rule used before averaging repeated spectra at the same family/peak/temperature.",
    )
    parser.add_argument("--mad-threshold", type=float, default=3.5)
    parser.add_argument("--iqr-factor", type=float, default=1.5)
    parser.add_argument(
        "--fit-mode",
        choices=["three-phonon", "quadratic", "both", "none"],
        default="both",
        help="Linewidth-vs-temperature guide fit: physical three-phonon, empirical quadratic, both, or none.",
    )
    parser.add_argument(
        "--fit-sigma",
        choices=["sem", "std", "none"],
        default="sem",
        help="Uncertainty used as fit weights for averaged temperature points.",
    )
    parser.add_argument(
        "--error-bar",
        choices=["std", "sem", "none"],
        default="std",
        help="Vertical uncertainty shown on averaged temperature points.",
    )
    parser.add_argument("--error-alpha", type=float, default=0.22)
    parser.add_argument(
        "--bound-tol",
        type=float,
        default=0.05,
        help="Tolerance in cm^-1 for deciding whether a fitted width/position is pinned to a bound.",
    )
    parser.add_argument(
        "--height-bound-rel-tol",
        type=float,
        default=0.01,
        help="Relative tolerance for raw-height upper-bound hit detection.",
    )
    parser.add_argument(
        "--pc-refit-bound-hits",
        action="store_true",
        help="Before plotting, run the local PC targeted refit helper on bound-hit spectra, then plot from its updated output.",
    )
    parser.add_argument("--pc-refit-source-root", type=Path, default=DEFAULT_PC_REFIT_SOURCE_ROOT)
    parser.add_argument("--pc-refit-output-root", type=Path, default=DEFAULT_PC_REFIT_OUTPUT_ROOT)
    parser.add_argument(
        "--pc-refit-families",
        nargs="*",
        default=None,
        help="Optional family folder names to refit. Omit to allow all families in the bound-hit report.",
    )
    parser.add_argument(
        "--pc-refit-peaks",
        nargs="*",
        default=None,
        help="Optional peak IDs to refit. Omit to reuse --peaks.",
    )
    parser.add_argument(
        "--pc-refit-hit-types",
        nargs="+",
        choices=["width", "position", "height"],
        default=["width", "position"],
        help="Which bound-hit types select spectra for PC refitting.",
    )
    parser.add_argument("--pc-refit-max-spectra", type=int, default=None, help="Optional cap for testing.")
    parser.add_argument("--pc-refit-dry-run", action="store_true", help="Show selected refit targets without fitting.")
    parser.add_argument("--pc-refit-max-iterations", type=int, default=4)
    parser.add_argument("--pc-refit-width-expand-factor", type=float, default=1.35)
    parser.add_argument("--pc-refit-position-expand-cm", type=float, default=3.0)
    parser.add_argument("--pc-refit-bootstrap-runs", type=int, default=100)
    parser.add_argument("--pc-refit-de-maxiter", type=int, default=100000)
    parser.add_argument("--pc-refit-de-popsize", type=int, default=200)
    parser.add_argument("--pc-refit-de-tol", type=float, default=1e-7)
    parser.add_argument("--pc-refit-ls-max-nfev", type=int, default=20000)
    parser.add_argument("--pc-refit-x-min", type=float, default=200.0)
    parser.add_argument("--pc-refit-x-max", type=float, default=2000.0)
    parser.add_argument("--pc-refit-peak-window-weight", type=float, default=6.0)
    parser.add_argument("--pc-refit-peak-window-margin", type=float, default=0.0)
    parser.add_argument(
        "--trust-fit-x-min",
        type=float,
        default=200.0,
        help="Lower Raman-shift bound used for app trust-region refits and SNR noise estimation.",
    )
    parser.add_argument(
        "--trust-fit-x-max",
        type=float,
        default=2000.0,
        help="Upper Raman-shift bound used for app trust-region refits and SNR noise estimation.",
    )
    parser.add_argument(
        "--trust-peak-window-weight",
        type=float,
        default=6.0,
        help="Residual weight assigned inside the JSON peak windows during app trust-region refits.",
    )
    parser.add_argument(
        "--trust-bootstrap-runs",
        type=int,
        default=100,
        help="Residual bootstrap runs for accepted app trust-region refit uncertainties.",
    )
    parser.add_argument("--trust-random-seed", type=int, default=12345)
    parser.add_argument("--trust-anti-burial-penalty", type=float, default=120.0)
    parser.add_argument("--trust-required-min-visibility", type=float, default=0.18)
    parser.add_argument("--trust-ch-left-min-visibility", type=float, default=0.40)
    parser.add_argument("--trust-required-min-area-share", type=float, default=0.10)
    parser.add_argument("--trust-ch-left-min-area-share", type=float, default=0.30)
    parser.add_argument("--trust-anti-burial-optional", action="store_true")
    parser.add_argument("--trust-ch-left-master", choices=["CH_L1", "CH_L2", "none"], default="CH_L2")
    parser.add_argument("--trust-ch-left-identity-penalty", type=float, default=60.0)
    parser.add_argument("--trust-ch-left-slave-min-area-ratio", type=float, default=0.45)
    parser.add_argument("--trust-ch-left-slave-max-area-ratio", type=float, default=0.90)
    parser.add_argument("--trust-ch-left-center-dominance-penalty", type=float, default=90.0)
    parser.add_argument("--trust-ch-left-own-center-share", type=float, default=0.55)
    parser.add_argument("--trust-ch-left-width-ratio-penalty", type=float, default=80.0)
    parser.add_argument("--trust-ch-left-width-ratio-min", type=float, default=0.75)
    parser.add_argument("--trust-ch-left-width-ratio-max", type=float, default=1.30)
    parser.add_argument(
        "--compute-snr",
        action="store_true",
        help="Write SNR-by-spectrum and SNR-by-temperature CSVs from the baseline-corrected TXT files.",
    )
    parser.add_argument(
        "--skip-parameter-grids",
        action="store_true",
        help="Do not save the new Height/FWHM/Position sequence-aware parameter grids in batch mode.",
    )
    parser.add_argument("--only-g", action="store_true", help="Only make the all-family G-mode overlay.")
    parser.add_argument("--only-grid", action="store_true", help="Only make the RBLM/D/G grid.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_paper_style()
    stage = str(getattr(args, "analysis_stage", "BEFORE")).strip().upper()
    if not args.app and args.output_dir.name.upper() not in {"BEFORE", "AFTER"}:
        args.output_dir = args.output_dir / stage
    print(f"Using fitted root: {args.fitted_root}")
    print(f"Saving {stage} outputs to: {args.output_dir}")
    results = load_cluster_results(args.fitted_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_bound_hit_reports(results, args.output_dir, args.bound_tol, args.height_bound_rel_tol)

    updated_fitted_root = run_pc_bound_refit_if_requested(args)
    if updated_fitted_root is not None:
        args.fitted_root = updated_fitted_root
        results = load_cluster_results(args.fitted_root)
        write_bound_hit_reports(results, args.output_dir, args.bound_tol, args.height_bound_rel_tol)

    parameter_df = fit_parameter_table(results, args.peaks, args.min_r2)
    parameter_df = drop_excluded_temperature_points(parameter_df)
    if parameter_df.empty:
        raise ValueError("No finite height/width/position fit rows were found after filtering.")
    all_peak_ids = sorted(
        {str(peak) for peak in results["peak_id"].dropna().unique()},
        key=lambda peak: PEAK_PREVIEW_ORDER.get(str(peak), 100),
    )
    all_parameter_df = drop_excluded_temperature_points(fit_parameter_table(results, all_peak_ids, args.min_r2))

    parameter_rows_path = args.output_dir / "Gamma_T_Comparison_parameter_rows.csv"
    parameter_export_cols = [
        "family",
        "family_label",
        "sequence",
        "sequence_label",
        "temperature_K",
        "file_name",
        "spectrum_in_file",
        "y_column_number",
        "peak_id",
        "height",
        "height_std",
        "position_cm-1",
        "position_std_cm-1",
        "width_fwhm_cm-1",
        "width_fwhm_std_cm-1",
        "r_squared",
        "rmse",
        "quality_flag",
        "fit_row_key",
        "_source_csv",
    ]
    parameter_df[[col for col in parameter_export_cols if col in parameter_df.columns]].to_csv(
        parameter_rows_path, index=False
    )
    print(f"Exported raw height/FWHM/position fit rows to {parameter_rows_path}")

    if args.compute_snr:
        compute_snr_tables(
            parameter_df,
            args.spectra_root,
            args.bounds_json,
            args.output_dir,
            args.trust_fit_x_min,
            args.trust_fit_x_max,
            args.second_pass_output_root,
        )

    if args.app:
        launch_curation_app(parameter_df, args, all_fit_rows=all_parameter_df)
        return

    rejected_keys = load_rejection_keys(args.output_dir / args.curation_json)
    if rejected_keys:
        parameter_df = parameter_df[~parameter_df["fit_row_key"].astype(str).isin(rejected_keys)].copy()
        print(f"Batch plots will ignore {len(rejected_keys)} manually rejected fitted row(s).")

    df = linewidth_table(results, args.peaks, args.min_r2)
    df = drop_excluded_temperature_points(df)
    if df.empty:
        raise ValueError("No finite linewidth rows were found after filtering.")
    df["fit_row_key"] = df.apply(fit_row_key, axis=1)
    if rejected_keys:
        df = df[~df["fit_row_key"].astype(str).isin(rejected_keys)].copy()

    summary_path = args.output_dir / "Gamma_T_Comparison_linewidth_rows.csv"
    export_cols = [
        "family",
        "family_label",
        "sequence",
        "temperature_K",
        "file_name",
        "spectrum_in_file",
        "y_column_number",
        "peak_id",
        "position_cm-1",
        "width_fwhm_cm-1",
        "width_fwhm_std_cm-1",
        "r_squared",
        "rmse",
        "quality_flag",
        "_source_csv",
    ]
    available = [col for col in export_cols if col in df.columns]
    df[available].to_csv(summary_path, index=False)
    print(f"Exported plotted linewidth rows to {summary_path}")
    print("Raw rows per family/peak:")
    print(df.groupby(["family", "peak_id"]).size().unstack(fill_value=0).to_string())

    if args.raw_points:
        plot_df = df
        print("Plotting every individual fitted spectrum (--raw-points).")
    else:
        plot_df = average_linewidth_by_temperature(df, args)
        if plot_df.empty:
            raise ValueError("No averaged linewidth rows were found after outlier filtering.")
        avg_path = args.output_dir / "Gamma_T_Comparison_averaged_linewidth_rows.csv"
        plot_df.to_csv(avg_path, index=False)
        removed = int(plot_df.get("n_outliers_removed", pd.Series(dtype=int)).sum())
        print(f"Exported cleaned temperature means to {avg_path}")
        print(f"Plotting cleaned temperature means; removed {removed} linewidth outlier(s).")
        print("Averaged rows per family/peak:")
        print(plot_df.groupby(["family", "peak_id"]).size().unstack(fill_value=0).to_string())

    if args.raw_points:
        parameter_plot_df = parameter_df
        print("Parameter grids will show every individual fitted spectrum (--raw-points).")
    else:
        parameter_plot_df = average_parameters_by_temperature(parameter_df, args, rejected_keys)
        parameter_avg_path = args.output_dir / "Gamma_T_Comparison_averaged_parameter_rows.csv"
        parameter_plot_df.to_csv(parameter_avg_path, index=False)
        print(f"Exported sequence-aware height/FWHM/position temperature means to {parameter_avg_path}")
        print("Averaged parameter rows per family/sequence/peak:")
        print(
            parameter_plot_df.groupby(["family", "sequence", "peak_id"]).size().unstack(fill_value=0).to_string()
        )

    if not args.only_grid:
        plot_g_overlay(plot_df, args.output_dir, args)
    if not args.only_g:
        plot_reference_grid(plot_df, args.output_dir, args)
    if not args.skip_parameter_grids:
        plot_all_parameter_grids(parameter_plot_df, args.output_dir, args)


if __name__ == "__main__":
    main()
