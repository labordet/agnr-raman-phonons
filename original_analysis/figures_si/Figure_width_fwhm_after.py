#!/usr/bin/env python3
"""
Paper-style FWHM-vs-temperature figures from the final AFTER curated data.

Inputs
------
GAMMA_T_COMPARISON/AFTER/CURATED_TABLES_AND_MANIFESTS/
    Gamma_T_Comparison_curated_temperature_means.csv

Outputs
-------
GAMMA_T_COMPARISON/AFTER/WIDHT_FIGURES/
    Fig_S9_Linewidth_Trends_3Phonon_All_Configurations.png
    Fig_S10_Aligned_Au_High_Coverage_Thermal_Cycle_Linewidths.png
    WIDTH_after_fit_parameters.csv

The folder name intentionally follows the requested spelling: WIDHT_FIGURES.
The legacy image filenames are retained for document compatibility, although
the figures now show only the physically motivated 3-phonon model.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit


APP_DIR = Path(__file__).resolve().parent
AFTER_DIR = APP_DIR / "GAMMA_T_COMPARISON" / "AFTER"
INPUT_CSV = AFTER_DIR / "CURATED_TABLES_AND_MANIFESTS" / "Gamma_T_Comparison_curated_temperature_means.csv"
OUTPUT_DIR = AFTER_DIR / "WIDHT_FIGURES"

PEAKS = ["RBLM", "D", "G"]
PEAK_LABELS = {"RBLM": "RBLM", "D": "D", "G": "G"}

FAMILY_ORDER = [
    "Aligned_Au_3A",
    "Aligned_Au_8A",
    "MIRA_Au_unaligned_8A",
    "MIRA_RO_unaligned_8A",
    "Aligned_RO_8A",
]

FAMILY_LABELS = {
    "Aligned_Au_3A": "Aligned Au, low coverage",
    "Aligned_Au_8A": "Aligned Au, high coverage",
    "MIRA_Au_unaligned_8A": "Unaligned Au, high coverage",
    "MIRA_RO_unaligned_8A": "Unaligned RO, high coverage",
    "Aligned_RO_8A": "Aligned RO, high coverage",
}

FAMILY_KEYS = {
    "Aligned_Au_3A": "aligned_au_3a",
    "Aligned_Au_8A": "aligned_au_8a",
    "MIRA_Au_unaligned_8A": "unaligned_au_8a",
    "MIRA_RO_unaligned_8A": "unaligned_ro_8a",
    "Aligned_RO_8A": "aligned_ro_8a",
}

SAMPLE_COLORS = {
    "aligned_au_3a": "#feb715",
    "aligned_au_8a": "#f08228",
    "aligned_ro_8a": "#00c8c8",
    "unaligned_au_8a": "#ae540b",
    "unaligned_ro_8a": "#008686",
}

# Darker companions keep fit curves and uncertainty bars legible when printed.
FAMILY_FIT_COLORS = {
    "Aligned_Au_3A": "#a86f00",
    "Aligned_Au_8A": "#c85b12",
    "MIRA_Au_unaligned_8A": "#743700",
    "MIRA_RO_unaligned_8A": "#006a6a",
    "Aligned_RO_8A": "#007f8c",
}

MAIN_SEQUENCE_FILTERS = {
    "Aligned_Au_3A": {"Spikes_Removed"},
    "Aligned_Au_8A": {"Spikes_Removed_UP_1"},
    "MIRA_Au_unaligned_8A": {"Spikes_Removed_UP_1"},
    "MIRA_RO_unaligned_8A": {"Spikes_Removed_UP_1"},
    "Aligned_RO_8A": {"Spikes_Removed_DOWN_1"},
}

AU8A_NESTED = ["Spikes_Removed_UP_1", "Spikes_Removed_DOWN_1", "Spikes_Removed_UP_2"]
AU8A_NESTED_LABELS = {
    "Spikes_Removed_UP_1": "Aligned Au, high coverage: UP 1",
    "Spikes_Removed_DOWN_1": "Aligned Au, high coverage: DOWN 1",
    "Spikes_Removed_UP_2": "Aligned Au, high coverage: UP 2",
}

AU8A_CYCLE_LABELS = {
    "Spikes_Removed_UP_1": "Heating 1 (UP 1)",
    "Spikes_Removed_DOWN_1": "Cooling 1 (DOWN 1)",
    "Spikes_Removed_UP_2": "Heating 2 (UP 2)",
}

AU8A_CYCLE_MARKERS = {
    "Spikes_Removed_UP_1": "^",
    "Spikes_Removed_DOWN_1": "v",
    "Spikes_Removed_UP_2": "D",
}

AU8A_CYCLE_FIT_COLORS = {
    "Spikes_Removed_UP_1": "#c85b12",
    "Spikes_Removed_DOWN_1": "#743700",
    "Spikes_Removed_UP_2": "#c97818",
}

FIXED_Y_LIMITS = {
    "RBLM": (10.0, 43.0),
    "D": (11.0, 18.0),
    "G": (11.0, 15.5),
}

INDEPENDENT_Y_FAMILIES = {"Aligned_RO_8A"}
EXCLUDED_POINTS = {("Aligned_Au_3A", "Spikes_Removed", 300.0)}
ERRORBAR_LIMIT_PAD_FRACTION = 0.045

# Source-font sizes are intentionally larger for the wide multi-panel canvases:
# both figures are reduced to approximately one text width in SI_FINAL.tex.
S9_TICK_LABEL_SIZE = 22.0
S9_AXIS_LABEL_SIZE = 25.0
S9_COLUMN_TITLE_SIZE = 20.0
S9_LEGEND_SIZE = 18.0
S10_TICK_LABEL_SIZE = 22.0
S10_AXIS_LABEL_SIZE = 25.0
S10_LEGEND_SIZE = 19.0

HC_OVER_KB = 1.4387769  # cm K


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 15,
            "axes.labelsize": 18,
            "axes.titlesize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 15,
            "axes.linewidth": 1.4,
            "xtick.major.width": 1.3,
            "ytick.major.width": 1.3,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.grid": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "rm",
        }
    )


def family_key(family: str) -> str:
    return FAMILY_KEYS.get(str(family), str(family).lower())


def family_label(family: str) -> str:
    return FAMILY_LABELS.get(str(family), str(family).replace("_", " "))


def marker_style(family: str, color: Optional[str] = None) -> Dict[str, object]:
    key = family_key(family)
    color = color or SAMPLE_COLORS.get(key, "#333333")
    hollow = key in {"unaligned_au_8a", "unaligned_ro_8a"}
    return {
        "marker": "o" if key == "aligned_au_3a" else "D",
        "facecolors": "none" if hollow else color,
        "edgecolors": color,
        "linewidths": 1.8 if hollow else 0.9,
        "color": color,
    }


def sequence_color(sequence: str) -> str:
    colors = {
        "Spikes_Removed_UP_1": "#f08228",
        "Spikes_Removed_DOWN_1": "#b85b1f",
        "Spikes_Removed_UP_2": "#ffad4c",
    }
    return colors.get(sequence, "#f08228")


def sequence_label(sequence: str) -> str:
    return AU8A_NESTED_LABELS.get(sequence, sequence.replace("Spikes_Removed_", "").replace("_", " "))


def style_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(1.35)
    ax.tick_params(axis="both", which="major", length=5.5, width=1.25, top=False, right=False)
    ax.grid(False)


def n_be(omega_cm1: float, temperature: np.ndarray) -> np.ndarray:
    temperature = np.asarray(temperature, dtype=float)
    clipped = np.clip(temperature, 1e-9, None)
    x = HC_OVER_KB * float(omega_cm1) / clipped
    return 1.0 / (np.exp(x) - 1.0)


def gamma_model(temperature: np.ndarray, gamma0: float, c3: float, omega_cm1: float) -> np.ndarray:
    return gamma0 + c3 * (1.0 + 2.0 * n_be(float(omega_cm1) / 2.0, temperature))


def prepared_fit_data(
    x: np.ndarray,
    y: np.ndarray,
    sigma: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        mask &= np.isfinite(sigma)
    if not np.any(mask):
        return np.array([]), np.array([]), None
    frame = pd.DataFrame({"x": x[mask], "y": y[mask]})
    if sigma is not None:
        frame["sigma"] = sigma[mask]
    agg = {"y": "mean"}
    if "sigma" in frame.columns:
        agg["sigma"] = "mean"
    frame = frame.groupby("x", as_index=False).agg(agg).sort_values("x")
    sigma_fit = None
    if "sigma" in frame.columns:
        sigma_fit = frame["sigma"].to_numpy(dtype=float)
        positive = sigma_fit[np.isfinite(sigma_fit) & (sigma_fit > 0)]
        if positive.size:
            fallback = float(np.nanmedian(positive))
            sigma_fit = np.where(np.isfinite(sigma_fit) & (sigma_fit > 0), sigma_fit, fallback)
        else:
            sigma_fit = None
    return frame["x"].to_numpy(dtype=float), frame["y"].to_numpy(dtype=float), sigma_fit


def fit_three_phonon_curve(
    x: np.ndarray,
    y: np.ndarray,
    omega0: float,
    sigma: Optional[np.ndarray],
) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[float, float]]]:
    x_fit, y_fit, sigma_fit = prepared_fit_data(x, y, sigma)
    if x_fit.size < 3 or not math.isfinite(omega0):
        return None
    t_smooth = np.linspace(float(np.nanmin(x_fit)), float(np.nanmax(x_fit)), 300)
    gamma_min = float(np.nanmin(y_fit))
    gamma_span = max(float(np.nanmax(y_fit) - np.nanmin(y_fit)), 0.1)
    p0 = [max(0.0, gamma_min - 0.2 * gamma_span), max(0.01, gamma_span * 0.4)]

    def local_model(temp, gamma0, c3):
        return gamma_model(temp, gamma0, c3, omega0)

    try:
        popt, _ = curve_fit(
            local_model,
            x_fit,
            y_fit,
            p0=p0,
            sigma=sigma_fit,
            absolute_sigma=False,
            bounds=([0.0, -500.0], [500.0, 500.0]),
            maxfev=30000,
        )
    except Exception:
        return None
    return t_smooth, local_model(t_smooth, *popt), (float(popt[0]), float(popt[1]))


def width_sigma(subset: pd.DataFrame) -> Optional[np.ndarray]:
    for column in ("width_total_sem", "width_group_sem", "width_bootstrap_sem", "width_total_std_visual"):
        if column in subset.columns:
            values = pd.to_numeric(subset[column], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(values).any() and np.nanmax(values) > 0:
                return values
    return None


def width_values_with_error_extent(subset: pd.DataFrame) -> np.ndarray:
    if subset.empty:
        return np.array([], dtype=float)
    y = pd.to_numeric(subset["width_fwhm_cm-1"], errors="coerce").to_numpy(dtype=float)
    values = [y]
    sigma = width_sigma(subset)
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        valid = np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
        if np.any(valid):
            values.extend([y[valid] - sigma[valid], y[valid] + sigma[valid]])
    return np.concatenate([np.asarray(v, dtype=float).ravel() for v in values])


def limits_with_visible_error_gap(
    values: Iterable[float],
    base_limits: Tuple[float, float],
    pad_fraction: float = ERRORBAR_LIMIT_PAD_FRACTION,
) -> Tuple[float, float]:
    low, high = base_limits
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return low, high
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    span = max(high - low, vmax - vmin, 1e-9)
    gap = max(pad_fraction * span, 0.08)
    return min(low, vmin - gap), max(high, vmax + gap)


def omega0_for_width_model(subset: pd.DataFrame) -> float:
    if "position_cm-1" not in subset.columns or subset.empty:
        return float("nan")
    finite = subset[np.isfinite(pd.to_numeric(subset["position_cm-1"], errors="coerce"))].copy()
    if finite.empty:
        return float("nan")
    t_min = float(finite["temperature_K"].min())
    low = finite[np.isclose(finite["temperature_K"], t_min)]
    return float(pd.to_numeric(low["position_cm-1"], errors="coerce").median())


def padded_limits(values: Iterable[float], fallback: Tuple[float, float], pad_fraction: float = 0.18) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return fallback
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if math.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.05, 0.5)
    else:
        pad = pad_fraction * (vmax - vmin)
    return vmin - pad, vmax + pad


def expanded_y_limits(values: Iterable[float], fixed: Tuple[float, float], pad_fraction: float = 0.08) -> Tuple[float, float]:
    low, high = fixed
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size:
        span = high - low
        low = min(low, float(np.nanmin(arr)) - pad_fraction * span)
        high = max(high, float(np.nanmax(arr)) + pad_fraction * span)
    return low, high


def load_after_means() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    required = {"family", "sequence", "peak_id", "temperature_K", "width_fwhm_cm-1", "position_cm-1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {sorted(missing)}")
    for column in ["temperature_K", "width_fwhm_cm-1", "position_cm-1"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df[df["peak_id"].astype(str).isin(PEAKS)].copy()
    for family, sequence, temp in EXCLUDED_POINTS:
        df = df[
            ~(
                (df["family"].astype(str) == family)
                & (df["sequence"].astype(str) == sequence)
                & np.isclose(df["temperature_K"], temp)
            )
        ].copy()
    return df


def filter_main_rows(df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for family in FAMILY_ORDER:
        seqs = MAIN_SEQUENCE_FILTERS[family]
        part = df[(df["family"].astype(str) == family) & (df["sequence"].astype(str).isin(seqs))].copy()
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True) if pieces else df.iloc[0:0].copy()


def family_legend_handles(families: Iterable[str]) -> List[Line2D]:
    handles = []
    for family in families:
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
                markersize=7.0,
            )
        )
    return handles


def fit_legend_handles() -> List[Line2D]:
    return [Line2D([0], [0], color="black", lw=2.0, label="3-phonon")]


def draw_subset(
    ax: plt.Axes,
    subset: pd.DataFrame,
    family: str,
    color: str,
    label: Optional[str],
    marker_size: float,
    fit_rows: List[Dict[str, object]],
) -> None:
    subset = subset.sort_values("temperature_K").copy()
    if subset.empty:
        return
    style = marker_style(family, color=color)
    x = subset["temperature_K"].to_numpy(dtype=float)
    y = subset["width_fwhm_cm-1"].to_numpy(dtype=float)
    sigma = width_sigma(subset)

    if sigma is not None:
        ax.errorbar(
            x,
            y,
            yerr=sigma,
            fmt="none",
            ecolor=style["edgecolors"],
            elinewidth=1.30,
            alpha=0.58,
            capsize=2.5,
            capthick=1.10,
            zorder=1,
        )
    ax.scatter(
        x,
        y,
        marker=style["marker"],
        s=marker_size,
        facecolors=style["facecolors"],
        edgecolors=style["edgecolors"],
        linewidths=style["linewidths"],
        alpha=0.95,
        label=label,
        zorder=3,
    )

    omega0 = omega0_for_width_model(subset)
    fit = fit_three_phonon_curve(x, y, omega0, sigma)
    if fit is not None:
        t, curve, params = fit
        ax.plot(t, curve, color="black", lw=2.0, zorder=4)
        fit_rows.append(
            {
                "figure": "width",
                "family": family,
                "sequence": str(subset["sequence"].iloc[0]),
                "peak_id": str(subset["peak_id"].iloc[0]),
                "model": "3-phonon",
                "omega0_position_cm-1": omega0,
                "gamma0": params[0],
                "c3": params[1],
                "n_points": len(subset),
            }
        )


def save_png(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved {path}")


def draw_si_width_series(
    ax: plt.Axes,
    subset: pd.DataFrame,
    family: str,
    marker_color: str,
    fit_color: str,
    marker: str,
    hollow: bool,
    label: str,
    fit_rows: List[Dict[str, object]],
    series_kind: str,
) -> None:
    """Draw a print-legible linewidth series without altering the fit model."""
    subset = subset.sort_values("temperature_K").copy()
    if subset.empty:
        return
    x = pd.to_numeric(subset["temperature_K"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(subset["width_fwhm_cm-1"], errors="coerce").to_numpy(dtype=float)
    sigma = width_sigma(subset)

    if sigma is not None:
        ax.errorbar(
            x,
            y,
            yerr=sigma,
            fmt="none",
            ecolor=fit_color,
            elinewidth=1.55,
            capsize=3.2,
            capthick=1.30,
            alpha=0.74,
            zorder=1,
        )
    ax.scatter(
        x,
        y,
        marker=marker,
        s=62.0,
        facecolors="white" if hollow else marker_color,
        edgecolors=fit_color,
        linewidths=1.35,
        label=label,
        zorder=3,
    )

    omega0 = omega0_for_width_model(subset)
    fit = fit_three_phonon_curve(x, y, omega0, sigma)
    if fit is None:
        return
    t_smooth, curve, params = fit
    ax.plot(t_smooth, curve, color=fit_color, lw=2.20, zorder=4)
    fit_rows.append(
        {
            "figure": "SI_combined_widths",
            "series_kind": series_kind,
            "series_label": label,
            "family": family,
            "sequence": str(subset["sequence"].iloc[0]),
            "peak_id": str(subset["peak_id"].iloc[0]),
            "model": "3-phonon",
            "omega0_position_cm-1": omega0,
            "gamma0": params[0],
            "c3": params[1],
            "n_points": len(subset),
        }
    )


def style_si_width_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        top=False,
        right=False,
        length=6.0,
        width=1.25,
        labelsize=16.5,
    )
    ax.set_xlim(60.0, 310.0)
    ax.set_xticks([100, 200, 300])
    ax.grid(False)


def _legend_marker_handle(
    marker: str,
    marker_color: str,
    edge_color: str,
    label: str,
    hollow: bool = False,
) -> Line2D:
    return Line2D(
        [0],
        [0],
        marker=marker,
        linestyle="None",
        markerfacecolor="white" if hollow else marker_color,
        markeredgecolor=edge_color,
        markeredgewidth=1.35,
        markersize=8.0,
        label=label,
    )


def plot_si_combined_width_figure(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path, Path]:
    """Create the SI-ready all-configuration and thermal-cycle linewidth figure."""
    output_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    main = filter_main_rows(df)
    nested = df[
        (df["family"].astype(str) == "Aligned_Au_8A")
        & (df["sequence"].astype(str).isin(AU8A_NESTED))
    ].copy()
    fit_rows: List[Dict[str, object]] = []

    fig, axes = plt.subplots(
        len(PEAKS),
        2,
        figsize=(12.8, 10.8),
        sharex="col",
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        bottom=0.085,
        top=0.775,
        wspace=0.24,
        hspace=0.13,
    )

    panel_letters = iter("abcdef")
    for row_idx, peak in enumerate(PEAKS):
        ax_all = axes[row_idx, 0]
        ax_cycles = axes[row_idx, 1]

        for family in FAMILY_ORDER:
            subset = main[
                (main["family"].astype(str) == family)
                & (main["peak_id"].astype(str) == peak)
            ].copy()
            key = family_key(family)
            style = marker_style(family)
            draw_si_width_series(
                ax_all,
                subset,
                family=family,
                marker_color=SAMPLE_COLORS[key],
                fit_color=FAMILY_FIT_COLORS[family],
                marker=str(style["marker"]),
                hollow=key in {"unaligned_au_8a", "unaligned_ro_8a"},
                label=family_label(family),
                fit_rows=fit_rows,
                series_kind="configuration",
            )

        for sequence in AU8A_NESTED:
            subset = nested[
                (nested["sequence"].astype(str) == sequence)
                & (nested["peak_id"].astype(str) == peak)
            ].copy()
            draw_si_width_series(
                ax_cycles,
                subset,
                family="Aligned_Au_8A",
                marker_color=sequence_color(sequence),
                fit_color=AU8A_CYCLE_FIT_COLORS[sequence],
                marker=AU8A_CYCLE_MARKERS[sequence],
                hollow=sequence == "Spikes_Removed_UP_2",
                label=AU8A_CYCLE_LABELS[sequence],
                fit_rows=fit_rows,
                series_kind="thermal_cycle",
            )

        all_values = width_values_with_error_extent(
            main[main["peak_id"].astype(str) == peak]
        )
        cycle_values = width_values_with_error_extent(
            nested[nested["peak_id"].astype(str) == peak]
        )
        ax_all.set_ylim(*padded_limits(all_values, FIXED_Y_LIMITS[peak], pad_fraction=0.08))
        ax_cycles.set_ylim(*padded_limits(cycle_values, FIXED_Y_LIMITS[peak], pad_fraction=0.10))

        for ax in (ax_all, ax_cycles):
            style_si_width_axis(ax)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            letter = next(panel_letters)
            ax.text(
                0.018,
                0.955,
                f"({letter})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=14.0,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.0),
                zorder=10,
            )

        ax_all.set_ylabel(
            f"{PEAK_LABELS[peak]} FWHM (cm$^{{-1}}$)",
            fontsize=18.5,
        )
        if row_idx < len(PEAKS) - 1:
            ax_all.tick_params(labelbottom=False)
            ax_cycles.tick_params(labelbottom=False)
        else:
            ax_all.set_xlabel("Temperature (K)", fontsize=18.5)
            ax_cycles.set_xlabel("Temperature (K)", fontsize=18.5)

    left_position = axes[0, 0].get_position()
    right_position = axes[0, 1].get_position()
    left_center = 0.5 * (left_position.x0 + left_position.x1)
    right_center = 0.5 * (right_position.x0 + right_position.x1)
    fig.text(
        left_center,
        0.982,
        "Five sample configurations",
        ha="center",
        va="top",
        fontsize=18.5,
        fontweight="bold",
    )
    fig.text(
        right_center,
        0.982,
        "Aligned Au, high coverage: thermal cycles",
        ha="center",
        va="top",
        fontsize=18.5,
        fontweight="bold",
    )

    family_handles = []
    for family in FAMILY_ORDER:
        key = family_key(family)
        style = marker_style(family)
        family_handles.append(
            _legend_marker_handle(
                marker=str(style["marker"]),
                marker_color=SAMPLE_COLORS[key],
                edge_color=FAMILY_FIT_COLORS[family],
                label=family_label(family),
                hollow=key in {"unaligned_au_8a", "unaligned_ro_8a"},
            )
        )
    cycle_handles = [
        _legend_marker_handle(
            marker=AU8A_CYCLE_MARKERS[sequence],
            marker_color=sequence_color(sequence),
            edge_color=AU8A_CYCLE_FIT_COLORS[sequence],
            label=AU8A_CYCLE_LABELS[sequence],
            hollow=sequence == "Spikes_Removed_UP_2",
        )
        for sequence in AU8A_NESTED
    ]
    fig.legend(
        handles=family_handles,
        loc="upper center",
        bbox_to_anchor=(left_center, 0.947),
        ncol=2,
        frameon=False,
        fontsize=14.5,
        handletextpad=0.45,
        columnspacing=1.2,
        labelspacing=0.38,
    )
    fig.legend(
        handles=cycle_handles,
        loc="upper center",
        bbox_to_anchor=(right_center, 0.947),
        ncol=1,
        frameon=False,
        fontsize=14.5,
        handletextpad=0.45,
        labelspacing=0.38,
    )
    fig.text(
        0.5,
        0.802,
        r"Solid curves: 3-phonon fits; error bars: $\pm 1$ SEM",
        ha="center",
        va="center",
        fontsize=14.5,
    )

    stem = "Fig_S9_Linewidths_All_Configurations_and_Aligned_Au_Thermal_Cycles"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fit_path = output_dir / f"{stem}_Fit_Parameters.csv"
    fig.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    pd.DataFrame(fit_rows).to_csv(fit_path, index=False)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    print(f"Saved {fit_path}")
    return png_path, pdf_path, fit_path


def plot_main_grid(df: pd.DataFrame, fit_rows: List[Dict[str, object]]) -> Path:
    main = filter_main_rows(df)
    fig, axes = plt.subplots(
        len(PEAKS),
        len(FAMILY_ORDER),
        figsize=(15.2, 19.2),
        sharex=True,
        squeeze=False,
    )
    fig.subplots_adjust(left=0.082, right=0.985, bottom=0.085, top=0.935, wspace=0.17, hspace=0.13)

    for row_idx, peak in enumerate(PEAKS):
        for col_idx, family in enumerate(FAMILY_ORDER):
            ax = axes[row_idx, col_idx]
            subset = main[(main["family"].astype(str) == family) & (main["peak_id"].astype(str) == peak)].copy()
            color = SAMPLE_COLORS[family_key(family)]
            draw_subset(ax, subset, family, color, None, 54.0, fit_rows)

            ax.set_xlim(50, 310)
            values = width_values_with_error_extent(subset)
            if family in INDEPENDENT_Y_FAMILIES:
                ax.set_ylim(*padded_limits(values, FIXED_Y_LIMITS[peak], pad_fraction=0.20))
            else:
                ax.set_ylim(*limits_with_visible_error_gap(values, FIXED_Y_LIMITS[peak]))
            style_axis(ax)
            ax.tick_params(axis="both", which="major", labelsize=S9_TICK_LABEL_SIZE)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5 if peak != "G" else 4))
            if row_idx == 0:
                ax.set_title(
                    family_label(family).replace(", ", ",\n"),
                    fontsize=S9_COLUMN_TITLE_SIZE,
                    pad=9.0,
                )
            if col_idx == 0:
                ax.set_ylabel(
                    f"{PEAK_LABELS[peak]} FWHM (cm$^{{-1}}$)",
                    fontsize=S9_AXIS_LABEL_SIZE,
                )
            else:
                ax.tick_params(labelleft=False)
            if row_idx == len(PEAKS) - 1:
                ax.set_xlabel("T (K)", fontsize=S9_AXIS_LABEL_SIZE)
            else:
                ax.tick_params(labelbottom=False)

    fit_handle = Line2D(
        [0],
        [0],
        color="black",
        linewidth=2.0,
        label="3-phonon fit",
    )
    axes[0, 0].legend(
        handles=[fit_handle],
        loc="upper left",
        bbox_to_anchor=(0.025, 0.975),
        frameon=False,
        fontsize=S9_LEGEND_SIZE,
        handlelength=2.4,
        handletextpad=0.55,
        borderaxespad=0.0,
    )

    out = OUTPUT_DIR / "Fig_S9_Linewidth_Trends_3Phonon_All_Configurations.png"
    save_png(fig, out)
    plt.close(fig)
    return out


def plot_au8a_nested_grid(df: pd.DataFrame, fit_rows: List[Dict[str, object]]) -> Path:
    nested = df[(df["family"].astype(str) == "Aligned_Au_8A") & (df["sequence"].astype(str).isin(AU8A_NESTED))].copy()
    fig, axes = plt.subplots(
        len(PEAKS),
        len(AU8A_NESTED),
        figsize=(10.2, 15.2),
        sharex=True,
        squeeze=False,
    )
    fig.subplots_adjust(left=0.125, right=0.985, bottom=0.085, top=0.985, wspace=0.18, hspace=0.12)

    for row_idx, peak in enumerate(PEAKS):
        row_values = width_values_with_error_extent(nested[nested["peak_id"].astype(str) == peak])
        row_ylim = limits_with_visible_error_gap(row_values, FIXED_Y_LIMITS[peak])
        for col_idx, sequence in enumerate(AU8A_NESTED):
            ax = axes[row_idx, col_idx]
            subset = nested[
                (nested["sequence"].astype(str) == sequence) & (nested["peak_id"].astype(str) == peak)
            ].copy()
            draw_subset(ax, subset, "Aligned_Au_8A", sequence_color(sequence), None, 58.0, fit_rows)

            ax.set_xlim(50, 310)
            ax.set_xticks([100, 200, 300])
            ax.set_ylim(*row_ylim)
            style_axis(ax)
            ax.tick_params(axis="both", which="major", labelsize=S10_TICK_LABEL_SIZE)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5 if peak != "G" else 4))
            if col_idx == 0:
                ax.set_ylabel(
                    f"{PEAK_LABELS[peak]} FWHM (cm$^{{-1}}$)",
                    fontsize=S10_AXIS_LABEL_SIZE,
                )
            else:
                ax.tick_params(labelleft=False)
            if row_idx == len(PEAKS) - 1:
                ax.set_xlabel("T (K)", fontsize=S10_AXIS_LABEL_SIZE)
            else:
                ax.tick_params(labelbottom=False)

    cycle_legend_labels = {
        "Spikes_Removed_UP_1": "Heating 1",
        "Spikes_Removed_DOWN_1": "Cooling 1",
        "Spikes_Removed_UP_2": "Heating 2",
    }
    cycle_handles = [
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            markerfacecolor=sequence_color(sequence),
            markeredgecolor=sequence_color(sequence),
            markeredgewidth=1.0,
            markersize=8.0,
            label=cycle_legend_labels[sequence],
        )
        for sequence in AU8A_NESTED
    ]
    cycle_handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.0,
            label="3-phonon fit",
        )
    )
    axes[0, 0].legend(
        handles=cycle_handles,
        loc="upper left",
        bbox_to_anchor=(0.025, 0.975),
        ncol=1,
        frameon=False,
        fontsize=S10_LEGEND_SIZE,
        handlelength=2.2,
        handletextpad=0.55,
        labelspacing=0.32,
        borderaxespad=0.0,
    )

    out = OUTPUT_DIR / "Fig_S10_Aligned_Au_High_Coverage_Thermal_Cycle_Linewidths.png"
    save_png(fig, out)
    plt.close(fig)
    return out
def main() -> None:
    setup_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_after_means()
    fit_rows: List[Dict[str, object]] = []
    print(f"Reading final AFTER width data from {INPUT_CSV}")
    out_main = plot_main_grid(df, fit_rows)
    out_nested = plot_au8a_nested_grid(df, fit_rows)
    fit_table = pd.DataFrame(fit_rows)
    fit_path = OUTPUT_DIR / "WIDTH_after_fit_parameters.csv"
    fit_table.to_csv(fit_path, index=False)
    print(f"Saved fit parameters: {fit_path}")
    print("Done.")
    print(f"Main figure: {out_main}")
    print(f"Nested Aligned Au 8A figure: {out_nested}")


if __name__ == "__main__":
    main()
