#!/usr/bin/env python3
"""Build corrected thermal-expansion inputs for the Raman temperature model.

The legacy TEC preparation forced alpha(0 K) = 0 by subtracting the fitted
zero-temperature intercept from the entire Au and sapphire curves. That
changes values inside the temperature ranges supported by the source data.

This script preserves source-supported values and changes only temperatures
below the corresponding source range:

* Au: fourth-order fit to the measured 40--300 K lattice expansion; the
  analytical derivative is used from 40 K upward. A C1 cubic continuation
  joins alpha(0) = 0 to the fitted value and slope at 40 K.
* Al2O3 (sapphire): the NIST alpha correlation is used from 5 to 75 K, a
  smooth 75--80 K transition joins it to the analytical derivative of the
  NIST linear-expansion correlation, and that derivative is used from 80 to
  300 K. A T^3 continuation is used only below 5 K.
* CNT: the published axial-CTE points up to 500 K are fitted as in the legacy
  analysis. A C1 cubic continuation joins alpha(0) = 0 to the fit at the
  lowest digitized source temperature; the source-range fit is unchanged.

Only derived TEC tables and figures are written. Source workbooks are read-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.interpolate import CubicHermiteSpline


APP_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = APP_DIR.parent
DEFAULT_OUTPUT_DIR = (
    APP_DIR
    / "THERMAL_EXPANSION_CORRECTION_20260711"
    / "FIXED"
    / "TEC_INPUTS"
)

GOLD_SOURCE = "T-Dep_Lattice_Gold.xlsx"
CNT_SOURCE = "ALPHA_CNT_TO_PROCESS.xlsx"

GOLD_DOI = "https://doi.org/10.1107/S1600576718002248"
CNT_DOI = "https://doi.org/10.1103/PhysRevB.80.205429"
SAPPHIRE_NIST_URL = (
    "https://trc.nist.gov/cryogenics/materials/Sapphire/Sapphire_rev.htm"
)

SAPPHIRE_ALPHA_COEFFICIENTS = np.array(
    [
        10.97236,
        -97.23540,
        240.2436,
        -294.9933,
        195.9244,
        -66.89247,
        9.19921,
        0.0,
        0.0,
    ],
    dtype=float,
)
SAPPHIRE_EXPANSION_COEFFICIENTS = {
    "a": -7.8850e1,
    "b": -2.2346e-2,
    "c": 1.0185e-4,
    "d": 5.5594e-6,
    "e": -8.5422e-9,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build corrected Au, Al2O3, and axial-CNT TEC inputs."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def sapphire_alpha_nist(temperature: np.ndarray) -> np.ndarray:
    """NIST expansion-coefficient correlation, valid from 5 to 80 K."""
    temperature = np.asarray(temperature, dtype=float)
    log_t = np.log10(temperature)
    exponent = np.zeros_like(log_t)
    for power, coefficient in enumerate(SAPPHIRE_ALPHA_COEFFICIENTS):
        exponent += coefficient * log_t**power
    return np.power(10.0, exponent) * 1e-6


def sapphire_alpha_from_expansion(temperature: np.ndarray) -> np.ndarray:
    """Derivative of the NIST 15--300 K linear-expansion polynomial."""
    temperature = np.asarray(temperature, dtype=float)
    c = SAPPHIRE_EXPANSION_COEFFICIENTS
    return (
        c["b"]
        + 2.0 * c["c"] * temperature
        + 3.0 * c["d"] * temperature**2
        + 4.0 * c["e"] * temperature**3
    ) * 1e-5


def smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    return value**2 * (3.0 - 2.0 * value)


def build_sapphire_curve(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    alpha = np.empty_like(grid, dtype=float)

    below_5 = grid < 5.0
    alpha_5 = float(sapphire_alpha_nist(np.array([5.0]))[0])
    alpha[below_5] = alpha_5 * (grid[below_5] / 5.0) ** 3

    nist_low = (grid >= 5.0) & (grid <= 75.0)
    alpha[nist_low] = sapphire_alpha_nist(grid[nist_low])

    transition = (grid > 75.0) & (grid < 80.0)
    transition_fraction = smoothstep((grid[transition] - 75.0) / 5.0)
    alpha[transition] = (
        (1.0 - transition_fraction) * sapphire_alpha_nist(grid[transition])
        + transition_fraction * sapphire_alpha_from_expansion(grid[transition])
    )

    nist_high = grid >= 80.0
    alpha[nist_high] = sapphire_alpha_from_expansion(grid[nist_high])

    metadata = {
        "source": "NIST Cryogenic Material Properties Database: Sapphire",
        "url": SAPPHIRE_NIST_URL,
        "direct_alpha_equation_range_K": [5.0, 80.0],
        "linear_expansion_equation_range_K": [15.0, 300.0],
        "transition_K": [75.0, 80.0],
        "below_source_continuation": "alpha(5 K) * (T / 5 K)^3",
    }
    return alpha, metadata


def build_gold_curve(
    grid: np.ndarray,
    source_dir: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    source_path = source_dir / GOLD_SOURCE
    source = pd.read_excel(source_path, engine="openpyxl")
    temperature = source["TEMPERATURE (K)"].to_numpy(dtype=float)
    lattice = source["Lattice parameter (a)"].to_numpy(dtype=float)

    lattice_293 = float(np.interp(293.0, temperature, lattice))
    relative_expansion = (lattice - lattice_293) / lattice_293
    expansion_coefficients = np.polyfit(temperature, relative_expansion, 4)
    alpha_coefficients = np.polyder(expansion_coefficients)
    alpha_slope_coefficients = np.polyder(alpha_coefficients)

    source_min = float(np.min(temperature))
    alpha_at_min = float(np.polyval(alpha_coefficients, source_min))
    slope_at_min = float(np.polyval(alpha_slope_coefficients, source_min))
    continuation = CubicHermiteSpline(
        [0.0, source_min],
        [0.0, alpha_at_min],
        [0.0, slope_at_min],
    )

    alpha = np.polyval(alpha_coefficients, grid)
    low = grid < source_min
    alpha[low] = continuation(grid[low])

    metadata = {
        "source": GOLD_SOURCE,
        "source_path": str(source_path.resolve()),
        "doi": GOLD_DOI,
        "source_temperature_range_K": [float(np.min(temperature)), float(np.max(temperature))],
        "fit": "degree-4 polynomial in relative lattice expansion; analytical derivative",
        "relative_expansion_reference_K": 293.0,
        "relative_expansion_polynomial_high_to_low": expansion_coefficients.tolist(),
        "below_source_continuation": (
            "C1 cubic Hermite continuation through (0 K, 0) matching fitted "
            "alpha and d(alpha)/dT at 40 K"
        ),
    }
    return alpha, metadata


def build_cnt_curve(
    grid: np.ndarray,
    source_dir: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    source_path = source_dir / CNT_SOURCE
    source = pd.read_excel(source_path, engine="openpyxl")
    temperature = source.iloc[:, 0].to_numpy(dtype=float)
    alpha_source = source.iloc[:, 1].to_numpy(dtype=float) * 1e-6
    fit_mask = (temperature >= 0.0) & (temperature <= 500.0)
    fit_temperature = temperature[fit_mask]
    fit_alpha = alpha_source[fit_mask]

    coefficients = np.polyfit(fit_temperature, fit_alpha, 4)
    slope_coefficients = np.polyder(coefficients)
    source_min = float(np.min(fit_temperature))
    alpha_at_min = float(np.polyval(coefficients, source_min))
    slope_at_min = float(np.polyval(slope_coefficients, source_min))
    continuation = CubicHermiteSpline(
        [0.0, source_min],
        [0.0, alpha_at_min],
        [0.0, slope_at_min],
    )

    alpha = np.polyval(coefficients, grid)
    low = grid < source_min
    alpha[low] = continuation(grid[low])

    metadata = {
        "source": CNT_SOURCE,
        "source_path": str(source_path.resolve()),
        "doi": CNT_DOI,
        "quantity": "axial linear thermal-expansion coefficient of a single-walled CNT",
        "fit_temperature_range_K": [float(np.min(fit_temperature)), float(np.max(fit_temperature))],
        "fit": "degree-4 polynomial in alpha(T)",
        "alpha_polynomial_high_to_low": coefficients.tolist(),
        "below_source_continuation": (
            "C1 cubic Hermite continuation through (0 K, 0) matching fitted "
            "alpha and d(alpha)/dT at the lowest digitized temperature"
        ),
    }
    return alpha, metadata


def validate_curve(name: str, grid: np.ndarray, alpha: np.ndarray) -> None:
    if grid.shape != alpha.shape:
        raise ValueError(f"{name}: grid and alpha shapes differ")
    if not np.all(np.isfinite(alpha)):
        raise ValueError(f"{name}: non-finite alpha values")
    if abs(float(alpha[0])) > 1e-18:
        raise ValueError(f"{name}: alpha(0 K) is not zero")
    if np.nanmin(alpha) < -1e-14:
        raise ValueError(f"{name}: negative alpha below numerical tolerance")


def write_curve(path: Path, grid: np.ndarray, alpha: np.ndarray) -> None:
    pd.DataFrame({"T (K)": grid, "alpha (1/K)": alpha}).to_csv(path, index=False)


def plot_curves(
    output_dir: Path,
    combined: pd.DataFrame,
    source_dir: Path,
    gold_metadata: dict[str, object],
) -> Path:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 15,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 16,
            "axes.linewidth": 1.2,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.top": False,
            "ytick.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(7.0, 6.5))
    temperature = combined["T (K)"].to_numpy(dtype=float)
    series = (
        ("Au alpha (1/K)", "#d62728", "Au", 40.0, "o"),
        ("Al2O3 alpha (1/K)", "#1f77b4", r"Al$_2$O$_3$", 5.0, "s"),
        ("axial CNT alpha (1/K)", "#2ca02c", "axial CNT", 15.96348, "^") ,
    )

    # Mark every source-temperature value available on the plotted TEC axis.
    # Au is reported as lattice parameter, so its TEC markers are the fitted
    # analytical derivative evaluated at all 22 measurement temperatures.
    gold_source = pd.read_excel(source_dir / GOLD_SOURCE, engine="openpyxl")
    gold_t = gold_source["TEMPERATURE (K)"].to_numpy(dtype=float)
    gold_expansion_coefficients = np.asarray(
        gold_metadata["relative_expansion_polynomial_high_to_low"],
        dtype=float,
    )
    gold_alpha = np.polyval(np.polyder(gold_expansion_coefficients), gold_t)

    # Sapphire is supplied by continuous NIST correlations rather than a local
    # point table.  Five-kelvin correlation samples make that source-supported
    # curve explicit without representing them as additional measurements.
    sapphire_t = np.arange(5.0, 300.0 + 0.001, 5.0)
    sapphire_alpha = np.interp(
        sapphire_t,
        temperature,
        combined["Al2O3 alpha (1/K)"].to_numpy(dtype=float),
    )

    cnt_source = pd.read_excel(source_dir / CNT_SOURCE, engine="openpyxl")
    cnt_t_all = cnt_source.iloc[:, 0].to_numpy(dtype=float)
    cnt_alpha_all = cnt_source.iloc[:, 1].to_numpy(dtype=float) * 1e-6
    cnt_visible = np.isfinite(cnt_t_all) & np.isfinite(cnt_alpha_all) & (cnt_t_all <= 300.0)
    source_markers = {
        "Au": (gold_t, gold_alpha),
        r"Al$_2$O$_3$": (sapphire_t, sapphire_alpha),
        "axial CNT": (cnt_t_all[cnt_visible], cnt_alpha_all[cnt_visible]),
    }

    material_handles = []
    for column, color, label, source_minimum, marker in series:
        alpha = combined[column].to_numpy(dtype=float)
        continuation = temperature < source_minimum
        source_supported = temperature >= source_minimum
        ax.plot(
            temperature[continuation],
            alpha[continuation],
            color=color,
            lw=3.0,
            linestyle=(0, (4, 2.2)),
        )
        ax.plot(
            temperature[source_supported],
            alpha[source_supported],
            color=color,
            lw=3.0,
        )
        marker_t, marker_alpha = source_markers[label]
        ax.scatter(
            marker_t,
            marker_alpha,
            marker=marker,
            s=34.0,
            facecolor=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=5,
        )
        material_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=3.0,
                marker=marker,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.55,
                markersize=7.0,
                label=label,
            )
        )

    continuation_handle = Line2D(
        [0],
        [0],
        color="#333333",
        lw=2.6,
        linestyle=(0, (4, 2.2)),
        label="low-$T$ continuation",
    )
    ax.set_xlim(-5.0, 305.0)
    ax.set_xticks(np.arange(50.0, 301.0, 50.0))
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Linear TEC $\alpha$ (K$^{-1}$)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-5, -5), useMathText=True)
    material_legend = ax.legend(
        handles=material_handles,
        frameon=False,
        loc="upper left",
        handlelength=2.8,
    )
    ax.add_artist(material_legend)
    ax.legend(
        handles=[continuation_handle],
        frameon=False,
        loc="center right",
        bbox_to_anchor=(0.98, 0.56),
        handlelength=2.6,
        fontsize=14,
    )
    ax.grid(False)
    fig.tight_layout()
    path = output_dir / "Fig_S6_Thermal_Expansion_Coefficients_FIXED.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def comparison_table(
    grid: np.ndarray,
    curves: dict[str, np.ndarray],
    source_dir: Path,
) -> pd.DataFrame:
    selected_temperatures = np.array([0.0, 5.0, 15.0, 40.0, 70.0, 75.0, 80.0, 100.0, 200.0, 300.0])
    rows = []
    legacy_paths = {
        "Au": source_dir / "Gold_CTE_α_SI.csv",
        "Al2O3": source_dir / "Sapphire_CTE_α_SI.csv",
        "axial CNT": source_dir / "CNT_CTE_alpha_SI.csv",
    }
    for material, alpha in curves.items():
        old = pd.read_csv(legacy_paths[material])
        old_t = old["T (K)"].to_numpy(dtype=float)
        old_alpha = old["alpha (1/K)"].to_numpy(dtype=float)
        for temperature in selected_temperatures:
            fixed_value = float(np.interp(temperature, grid, alpha))
            old_value = float(np.interp(temperature, old_t, old_alpha))
            rows.append(
                {
                    "material": material,
                    "temperature_K": temperature,
                    "alpha_OLD_1_per_K": old_value,
                    "alpha_FIXED_1_per_K": fixed_value,
                    "FIXED_minus_OLD_1_per_K": fixed_value - old_value,
                    "relative_change_percent": (
                        100.0 * (fixed_value - old_value) / old_value
                        if old_value != 0.0
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = np.arange(0.0, 300.0 + 0.0001, 0.1)
    alpha_gold, gold_metadata = build_gold_curve(grid, source_dir)
    alpha_sapphire, sapphire_metadata = build_sapphire_curve(grid)
    alpha_cnt, cnt_metadata = build_cnt_curve(grid, source_dir)

    curves = {
        "Au": alpha_gold,
        "Al2O3": alpha_sapphire,
        "axial CNT": alpha_cnt,
    }
    for name, alpha in curves.items():
        validate_curve(name, grid, alpha)

    write_curve(output_dir / "Gold_CTE_alpha_SI.csv", grid, alpha_gold)
    write_curve(output_dir / "Sapphire_CTE_alpha_SI.csv", grid, alpha_sapphire)
    write_curve(output_dir / "CNT_CTE_alpha_SI.csv", grid, alpha_cnt)

    combined = pd.DataFrame(
        {
            "T (K)": grid,
            "Au alpha (1/K)": alpha_gold,
            "Al2O3 alpha (1/K)": alpha_sapphire,
            "axial CNT alpha (1/K)": alpha_cnt,
        }
    )
    combined.to_csv(output_dir / "CTE_Au_Al2O3_axial_CNT_FIXED.csv", index=False)
    combined.to_excel(output_dir / "CTE_Au_Al2O3_axial_CNT_FIXED.xlsx", index=False)

    summary = comparison_table(grid, curves, source_dir)
    summary.to_csv(output_dir / "TEC_OLD_vs_FIXED_selected_temperatures.csv", index=False)

    metadata = {
        "temperature_grid_K": {"start": 0.0, "stop": 300.0, "step": 0.1},
        "Au": gold_metadata,
        "Al2O3": sapphire_metadata,
        "axial_CNT": cnt_metadata,
        "legacy_vertical_shift_removed": True,
    }
    (output_dir / "TEC_FIXED_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    figure_path = plot_curves(output_dir, combined, source_dir, gold_metadata)

    print(f"Wrote corrected TEC inputs to {output_dir}")
    print(f"Wrote TEC figure: {figure_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
