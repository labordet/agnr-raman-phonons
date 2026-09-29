#!/usr/bin/env python3
"""
Figure 3 main-paper temperature-dependent Raman shift plot from AFTER data.

Reads the curated AFTER table written by Gamma_T_Comparison.py:
    GAMMA_T_COMPARISON/AFTER/CURATED_TABLES_AND_MANIFESTS/
        Gamma_T_Comparison_curated_temperature_means.csv

The plotted values are the fitted Lorentzian peak positions for RBLM, D, and G
after the final paper-cleaning step. No old Excel workbook is read.

For the paper figure, sequence selection is explicit:
    Aligned_Au_8A uses Spikes_Removed_UP_1 only.
    Aligned_RO_8A uses Spikes_Removed_DOWN_1 only.
    The unaligned families use their UP_1 sequence.

Outputs are saved to:
    GAMMA_T_COMPARISON/AFTER/FIGURE_3_PAPER
"""

from __future__ import annotations

import argparse
import math
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, MultipleLocator
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import curve_fit


APP_DIR = Path(__file__).resolve().parent
DEFAULT_AFTER_DIR = APP_DIR / "GAMMA_T_COMPARISON" / "AFTER"
DEFAULT_INPUT_CSV = (
    DEFAULT_AFTER_DIR
    / "CURATED_TABLES_AND_MANIFESTS"
    / "Gamma_T_Comparison_curated_temperature_means.csv"
)
DEFAULT_OUT_DIR = DEFAULT_AFTER_DIR / "FIGURE_3_PAPER"
DEFAULT_CTE_DIR = APP_DIR.parent

PEAKS = [
    {"id": "RBLM", "target": 312.0, "abs_ylim": (303.5, 318.5)},
    {"id": "D", "target": 1340.0, "abs_ylim": (1331.0, 1346.0)},
    {"id": "G", "target": 1597.0, "abs_ylim": (1587.5, 1603.0)},
]

SAMPLES_PLOT = [
    "Aligned_Au_3A",
    "Aligned_Au_8A",
    "MIRA_Au_unaligned_8A",
    "MIRA_RO_unaligned_8A",
    "Aligned_RO_8A",
]

SAMPLE_CONFIG: Dict[str, Dict[str, str]] = {
    "Aligned_Au_3A": {
        "family_label": "Aligned Au, low coverage",
        "legend_label": "Aligned Au, low coverage",
        "cte_kind": "gold",
        "color": "#feb715",
        "marker": "o",
        "face": "filled",
    },
    "Aligned_Au_8A": {
        "family_label": "Aligned Au, high coverage",
        "legend_label": "Aligned Au, high coverage",
        "cte_kind": "gold",
        "color": "#f08228",
        "marker": "D",
        "face": "filled",
    },
    "Aligned_Au_8A_UP1": {
        "family_label": "Aligned Au, high coverage, heating 1",
        "legend_label": "Heating 1",
        "cte_kind": "gold",
        "color": "#f08228",
        "marker": "D",
        "face": "filled",
        "source_family": "Aligned_Au_8A",
    },
    "Aligned_Au_8A_DOWN1": {
        "family_label": "Aligned Au, high coverage, cooling",
        "legend_label": "Cooling",
        "cte_kind": "gold",
        "color": "#b85b1f",
        "marker": "D",
        "face": "filled",
        "source_family": "Aligned_Au_8A",
    },
    "Aligned_Au_8A_UP2": {
        "family_label": "Aligned Au, high coverage, reheating",
        "legend_label": "Reheating",
        "cte_kind": "gold",
        "color": "#ffad4c",
        "marker": "D",
        "face": "filled",
        "source_family": "Aligned_Au_8A",
    },
    "MIRA_Au_unaligned_8A": {
        "family_label": "Unaligned Au, high coverage",
        "legend_label": "Unaligned Au, high coverage",
        "cte_kind": "gold",
        "color": "#ae540b",
        "marker": "D",
        "face": "hollow",
    },
    "MIRA_RO_unaligned_8A": {
        "family_label": "Unaligned RO, high coverage",
        "legend_label": "Unaligned RO, high coverage",
        "cte_kind": "sapphire",
        "color": "#008686",
        "marker": "D",
        "face": "hollow",
    },
    "Aligned_RO_8A": {
        "family_label": "Aligned RO, high coverage",
        "legend_label": "Aligned RO, high coverage",
        "cte_kind": "sapphire",
        "color": "#00c8c8",
        "marker": "D",
        "face": "filled",
    },
}

DEFAULT_SEQUENCE_FILTERS = {
    "Aligned_Au_3A": {"Spikes_Removed"},
    "Aligned_Au_8A": {"Spikes_Removed_UP_1"},
    "MIRA_Au_unaligned_8A": {"Spikes_Removed_UP_1"},
    "MIRA_RO_unaligned_8A": {"Spikes_Removed_UP_1"},
    "Aligned_RO_8A": {"Spikes_Removed_DOWN_1"},
}

FIXED_TEMPERATURE_COMPARISON_K = 100.0
FIXED_TEMPERATURE_CATEGORY_LABELS = {
    "Aligned_Au_3A": "Aligned Au\nlow coverage",
    "Aligned_Au_8A": "Aligned Au\nhigh coverage",
    "MIRA_Au_unaligned_8A": "Unaligned Au\nhigh coverage",
    "Aligned_RO_8A": "Aligned RO\nhigh coverage",
    "MIRA_RO_unaligned_8A": "Unaligned RO\nhigh coverage",
}
FIXED_TEMPERATURE_SAMPLE_ORDER = [
    "Aligned_Au_3A",
    "Aligned_Au_8A",
    "MIRA_Au_unaligned_8A",
    "Aligned_RO_8A",
    "MIRA_RO_unaligned_8A",
]
FIXED_TEMPERATURE_YLIMS = {
    "RBLM": (304.0, 318.5),
    "D": (1329.5, 1348.0),
    "G": (1590.0, 1606.5),
}
FIXED_TEMPERATURE_YTICKS = {
    "RBLM": [306.0, 310.0, 314.0, 318.0],
    "D": [1330.0, 1335.0, 1340.0, 1345.0],
    "G": [1592.0, 1596.0, 1600.0, 1604.0],
}
FIXED_TEMPERATURE_YMINOR_TICKS = {
    "RBLM": [308.0, 312.0, 316.0],
    "D": [1332.5, 1337.5, 1342.5],
    "G": [1594.0, 1598.0, 1602.0],
}
FIXED_TEMPERATURE_MODE_COLORS = {
    "RBLM": "#c96500",
    "D": "#009b63",
    "G": "#0074b8",
}
FIXED_TEMPERATURE_MODE_BACKGROUNDS = {
    "RBLM": "#fff4df",
    "D": "#f2fae9",
    "G": "#e7f5fc",
}
FIXED_TEMPERATURE_STYLE = {
    "figsize": (6.6, 6.9),
    "tick": 13,
    "axis_label": 17,
    "mode_label": 15,
    "temperature_label": 14,
    "xtick": 12,
    "marker_filled": 48,
    "marker_hollow": 58,
    "marker_halo_extra": 30,
    "marker_lw_filled": 1.2,
    "marker_lw_hollow": 2.0,
    "error_lw": 2.2,
    "error_halo_lw": 5.0,
    "error_capsize": 6.0,
    "error_capthick": 2.0,
}
TEMPERATURE_SPREAD_STYLE = {
    "bar_lw": 5.0,
    "summary_marker_halo_extra": 36,
    "summary_marker_halo_lw": 3.8,
    "colorbar_width": 0.028,
    "colorbar_pad": 0.018,
    "colorbar_tick": 11,
    "colorbar_label": 13,
}
TEMPERATURE_SPREAD_CMAP = "turbo"
TEMPERATURE_SPREAD_NORM = mpl.colors.Normalize(vmin=70.0, vmax=300.0)
TEMPERATURE_SPREAD_TICKS = [100, 150, 200, 250]

ALIGNED_AU_8A_SEQUENCE_VARIANTS = [
    ("AlignedAu8A_UP1", "Spikes_Removed_UP_1"),
    ("AlignedAu8A_DOWN1", "Spikes_Removed_DOWN_1"),
    ("AlignedAu8A_UP2", "Spikes_Removed_UP_2"),
]

ALIGNED_AU_8A_NESTED_SAMPLE_ORDER = [
    "Aligned_Au_8A_UP1",
    "Aligned_Au_8A_DOWN1",
    "Aligned_Au_8A_UP2",
]

ALIGNED_AU_8A_NESTED_SEQUENCE_FILTERS = {
    "Aligned_Au_8A_UP1": {"Spikes_Removed_UP_1"},
    "Aligned_Au_8A_DOWN1": {"Spikes_Removed_DOWN_1"},
    "Aligned_Au_8A_UP2": {"Spikes_Removed_UP_2"},
}

EXCLUDED_TEMPERATURE_POINTS = {
    ("Aligned_Au_3A", "Spikes_Removed", 300.0),
}

NORM_SCALE = 1000.0
Y_PAD_FRAC = 0.10
ABS_TIGHT_Y_PAD_FRAC = 0.035
NORM_Y_PAD_FRAC = 0.16
INCLUDE_ERRORBARS = False
FIT_WITH_ERRORBARS = False
INCLUDE_MODEL_BAND = False
DISPLAY_FIGURES = False
MODEL_BAND_SAMPLES = 500
MODEL_BAND_RANDOM_SEED = 20260706

STYLE = {
    "font_family": "Arial",
    "font_size": 24,
    "axis_label": 30,
    "title": 30,
    "tick": 26,
    "legend": 18,
    "line_width": 3.5,
    "te_width": 2.8,
    "anh_width": 3.6,
    "spine_width": 1.0,
    "tick_length": 6,
    "tick_width": 1,
    "scatter_alpha": 0.9,
    "figsize": (15, 26.37),
}

H_PLANCK = 6.62607015e-34
C_LIGHT = 2.99792458e8
KB = 1.380649e-23
HC_CM = H_PLANCK * C_LIGHT * 100.0


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [STYLE["font_family"], "DejaVu Sans", "Arial"],
            "font.size": STYLE["font_size"],
            "axes.labelsize": STYLE["axis_label"],
            "axes.titlesize": STYLE["title"],
            "xtick.labelsize": STYLE["tick"],
            "ytick.labelsize": STYLE["tick"],
            "legend.fontsize": STYLE["legend"],
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.grid": False,
            "axes.unicode_minus": False,
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "regular",
        }
    )


def safe_name(path: Path) -> str:
    return str(path).encode("ascii", "backslashreplace").decode("ascii")


def find_cte_file(cte_dir: Path, kind: str) -> Path:
    candidates = list(cte_dir.glob("*.csv"))
    if kind == "gold":
        preferred = [
            p for p in candidates
            if "gold" in p.name.lower() and "cte" in p.name.lower() and "si" in p.name.lower()
        ]
    elif kind == "sapphire":
        preferred = [
            p for p in candidates
            if "sapphire" in p.name.lower() and "cte" in p.name.lower() and "si" in p.name.lower()
        ]
    elif kind == "cnt":
        preferred = [
            p for p in candidates
            if "cnt" in p.name.lower() and "cte" in p.name.lower() and "si" in p.name.lower()
        ]
    else:
        raise ValueError(f"Unknown CTE kind: {kind}")

    if not preferred:
        raise FileNotFoundError(f"No {kind} CTE CSV found in {cte_dir}")

    # Prefer the single-material alpha files if present.
    preferred = sorted(
        preferred,
        key=lambda p: (
            "70-300" in p.name,
            p.name.lower().startswith("cte_"),
            len(p.name),
        ),
    )
    return preferred[0]


def load_cte(cte_dir: Path, kind: str) -> pd.DataFrame:
    path = find_cte_file(cte_dir, kind)
    df = pd.read_csv(path)
    if "T (K)" not in df.columns:
        raise ValueError(f"{path} has no 'T (K)' column.")

    alpha_col = None
    for col in df.columns:
        if str(col).strip() == "alpha (1/K)":
            alpha_col = col
            break
    if alpha_col is None:
        for col in df.columns:
            text = str(col).lower()
            if "alpha" in text or "α" in text:
                alpha_col = col
                break
    if alpha_col is None:
        raise ValueError(f"{path} has no alpha column.")

    out = df[["T (K)", alpha_col]].copy()
    out.columns = ["T (K)", "alpha (1/K)"]
    out = out.dropna().sort_values("T (K)")
    print(f"Using {kind} CTE: {safe_name(path.resolve())}")
    return out


def n_be(nu_cm: float, temperature):
    temperature = np.asarray(temperature, dtype=float)
    with np.errstate(divide="ignore"):
        x = HC_CM * nu_cm / (KB * temperature)
    x = np.clip(x, None, 700)
    with np.errstate(divide="ignore", over="ignore"):
        n = 1.0 / np.expm1(x)
    n = np.asarray(n, dtype=float)
    n[temperature == 0.0] = 0.0
    return n


def sample_style(sample: str) -> Dict[str, Any]:
    cfg = SAMPLE_CONFIG[sample]
    color = cfg["color"]
    hollow = cfg["face"] == "hollow"
    return {
        "marker": cfg["marker"],
        "size": 120 if hollow else 100,
        "lw": 2.0 if hollow else 1.6,
        "face": "none" if hollow else color,
        "edge": color,
        "color": color,
    }


def fixed_temperature_sample_style(sample: str) -> Dict[str, Any]:
    st = sample_style(sample)
    hollow = SAMPLE_CONFIG[sample]["face"] == "hollow"
    st = dict(st)
    st["size"] = (
        FIXED_TEMPERATURE_STYLE["marker_hollow"]
        if hollow
        else FIXED_TEMPERATURE_STYLE["marker_filled"]
    )
    st["lw"] = (
        FIXED_TEMPERATURE_STYLE["marker_lw_hollow"]
        if hollow
        else FIXED_TEMPERATURE_STYLE["marker_lw_filled"]
    )
    return st


def filter_paper_rows(
    df: pd.DataFrame,
    sequence_filters: Dict[str, set[str]],
    sample_order: List[str],
) -> pd.DataFrame:
    rows = []
    for sample in sample_order:
        source_family = SAMPLE_CONFIG[sample].get("source_family", sample)
        allowed = sequence_filters[sample]
        chunk = df[
            df["family"].astype(str).eq(source_family)
            & df["sequence"].astype(str).isin(allowed)
            & df["peak_id"].astype(str).isin([p["id"] for p in PEAKS])
        ].copy()
        if chunk.empty:
            raise ValueError(f"No AFTER rows for {sample} / source {source_family} with sequences {sorted(allowed)}")
        chunk["plot_sample"] = sample
        rows.append(chunk)
    out = pd.concat(rows, ignore_index=True)
    out = out[np.isfinite(pd.to_numeric(out["temperature_K"], errors="coerce"))].copy()
    out["temperature_K"] = pd.to_numeric(out["temperature_K"], errors="coerce")
    out["position_cm-1"] = pd.to_numeric(out["position_cm-1"], errors="coerce")
    out["position_total_sem"] = pd.to_numeric(out.get("position_total_sem", np.nan), errors="coerce")
    out["position_total_std_visual"] = pd.to_numeric(
        out.get("position_total_std_visual", out.get("position_total_sem", np.nan)),
        errors="coerce",
    )
    out = out[np.isfinite(out["position_cm-1"])].copy()
    for ex_family, ex_sequence, ex_temp in EXCLUDED_TEMPERATURE_POINTS:
        keep = ~(
            out["family"].astype(str).eq(ex_family)
            & out["sequence"].astype(str).eq(ex_sequence)
            & np.isclose(out["temperature_K"], ex_temp)
        )
        out = out[keep].copy()
    return out


def linear_intercept_uncertainty(
    t_exp: np.ndarray,
    w_exp: np.ndarray,
    yerr: np.ndarray,
    mask: np.ndarray,
) -> Tuple[float, int, float, float]:
    """Estimate uncertainty of the same unweighted linear intercept used for omega0."""
    x = np.asarray(t_exp, dtype=float)[mask]
    y = np.asarray(w_exp, dtype=float)[mask]
    err = np.asarray(yerr, dtype=float)[mask]
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    err = err[valid]
    n = int(x.size)
    if n < 2:
        return np.nan, n, np.nan, np.nan

    design = np.column_stack([x, np.ones_like(x)])
    xtx_inv = np.linalg.pinv(design.T @ design)
    beta = xtx_inv @ design.T @ y
    residual = y - design @ beta

    variances = []
    finite_err = np.isfinite(err) & (err > 0)
    if np.any(finite_err):
        err_filled = err.copy()
        fallback = float(np.nanmedian(err[finite_err]))
        err_filled[~finite_err] = fallback
        sigma = np.diag(err_filled**2)
        cov_measured = xtx_inv @ design.T @ sigma @ design @ xtx_inv
        if np.isfinite(cov_measured[1, 1]) and cov_measured[1, 1] >= 0:
            variances.append(float(cov_measured[1, 1]))

    if n > 2:
        dof = n - 2
        residual_var = float(np.nansum(residual**2) / dof)
        cov_residual = residual_var * xtx_inv
        if np.isfinite(cov_residual[1, 1]) and cov_residual[1, 1] >= 0:
            variances.append(float(cov_residual[1, 1]))

    if not variances:
        return np.nan, n, float(np.nanmin(x)), float(np.nanmax(x))

    # The measured-position propagation and the residual-scatter estimate are
    # two views of the same extrapolation uncertainty. Use the larger one to
    # avoid unrealistically tiny two-point/intercept bars.
    return math.sqrt(max(variances)), n, float(np.nanmin(x)), float(np.nanmax(x))


def fit_temperature_model(
    data: pd.DataFrame,
    sample: str,
    peak_id: str,
    cte_sub: pd.DataFrame,
    cte_cnt: pd.DataFrame,
) -> Dict[str, Any]:
    sample_col = "plot_sample" if "plot_sample" in data.columns else "family"
    d = data[(data[sample_col].eq(sample)) & (data["peak_id"].eq(peak_id))].copy()
    if d.empty:
        raise ValueError(f"No data for {sample} / {peak_id}")
    d = d.sort_values("temperature_K")

    t_exp = d["temperature_K"].to_numpy(float)
    w_exp = d["position_cm-1"].to_numpy(float)
    yerr_fit = d["position_total_sem"].to_numpy(float) if "position_total_sem" in d else np.full_like(t_exp, np.nan)
    if "position_total_std_visual" in d:
        yerr_visual = d["position_total_std_visual"].to_numpy(float)
    else:
        yerr_visual = yerr_fit.copy()
    sequence = ", ".join(sorted(d["sequence"].astype(str).unique()))

    t_max = max(float(np.nanmax(t_exp)), 300.0)
    t_grid = np.arange(0.0, math.ceil(t_max) + 1.0, 1.0)

    alpha_sub = np.interp(t_grid, cte_sub["T (K)"], cte_sub["alpha (1/K)"])
    alpha_cnt = np.interp(t_grid, cte_cnt["T (K)"], cte_cnt["alpha (1/K)"])
    i_grid = cumulative_trapezoid(alpha_sub - alpha_cnt, t_grid, initial=0.0)

    def i_t(t):
        return np.interp(t, t_grid, i_grid)

    if peak_id == "RBLM":
        mask_omega0 = np.ones_like(t_exp, dtype=bool)
        omega0_method = "original_RBLM_linear_extrapolation_all_temperatures"
    else:
        mask_omega0 = t_exp <= 110.0
        omega0_method = "original_linear_extrapolation_T_le_110K"
        if mask_omega0.sum() < 2:
            mask_omega0 = np.ones_like(t_exp, dtype=bool)
            omega0_method = "original_linear_extrapolation_all_temperatures_fallback"

    if mask_omega0.sum() >= 2:
        _slope, omega0 = np.polyfit(t_exp[mask_omega0], w_exp[mask_omega0], 1)
    else:
        omega0 = float(np.nanmedian(w_exp))
        omega0_method = "median_fallback"
    omega0_stderr, omega0_n_points, omega0_tmin, omega0_tmax = linear_intercept_uncertainty(
        t_exp,
        w_exp,
        yerr_fit,
        mask_omega0,
    )

    delta_exp = w_exp - omega0

    def parts(t, a3, gamma):
        delta_anh = a3 * (1.0 + 2.0 * n_be(omega0 / 2.0, t))
        delta_te = omega0 * (np.exp(-gamma * i_t(t)) - 1.0)
        return delta_anh, delta_te, delta_anh + delta_te

    try:
        sigma = None
        if FIT_WITH_ERRORBARS and np.all(np.isfinite(yerr_fit)) and np.all(yerr_fit > 0):
            sigma = yerr_fit
        (a3, gamma), pcov = curve_fit(
            lambda t, a3, gamma: parts(t, a3, gamma)[2],
            t_exp,
            delta_exp,
            p0=(-2.0, 1.0),
            sigma=sigma,
            absolute_sigma=bool(sigma is not None),
            maxfev=20000,
        )
        perr = np.sqrt(np.diag(pcov)) if pcov.shape == (2, 2) else np.array([np.nan, np.nan])
        fit_status = "OK"
    except Exception as exc:
        a3, gamma = np.nan, np.nan
        perr = np.array([np.nan, np.nan])
        fit_status = f"FIT_FAILED: {exc}"
        pcov = np.full((2, 2), np.nan)

    if np.isfinite(a3) and np.isfinite(gamma):
        delta_anh_raw, delta_te, delta_sum_raw = parts(t_grid, a3, gamma)
        delta_fit_exp = parts(t_exp, a3, gamma)[2]
    else:
        delta_anh_raw = np.full_like(t_grid, np.nan)
        delta_te = np.full_like(t_grid, np.nan)
        delta_sum_raw = np.full_like(t_grid, np.nan)
        delta_fit_exp = np.full_like(t_exp, np.nan)

    offset0 = float(delta_sum_raw[0]) if np.isfinite(delta_sum_raw[0]) else 0.0
    delta_exp_norm = delta_exp - offset0
    delta_anh_norm = delta_anh_raw - offset0
    delta_sum_norm = delta_sum_raw - offset0
    delta_fit_exp_norm = delta_fit_exp - offset0

    band_lo = np.full_like(t_grid, np.nan)
    band_hi = np.full_like(t_grid, np.nan)
    if (
        np.isfinite(a3)
        and np.isfinite(gamma)
        and np.asarray(pcov).shape == (2, 2)
        and np.all(np.isfinite(pcov))
        and MODEL_BAND_SAMPLES > 0
    ):
        try:
            seed_offset = zlib.crc32(f"{sample}|{peak_id}".encode("utf-8"))
            rng_seed = MODEL_BAND_RANDOM_SEED + seed_offset
            rng = np.random.default_rng(rng_seed)
            draws = rng.multivariate_normal(
                mean=np.array([a3, gamma], dtype=float),
                cov=np.asarray(pcov, dtype=float),
                size=MODEL_BAND_SAMPLES,
                check_valid="ignore",
            )
            curves = []
            for a3_draw, gamma_draw in draws:
                if not np.isfinite(a3_draw) or not np.isfinite(gamma_draw):
                    continue
                _draw_anh, _draw_te, draw_sum = parts(t_grid, a3_draw, gamma_draw)
                draw_offset = float(draw_sum[0]) if np.isfinite(draw_sum[0]) else 0.0
                draw_norm = draw_sum - draw_offset
                if np.all(np.isfinite(draw_norm)):
                    curves.append(draw_norm)
            if len(curves) >= 20:
                curve_stack = np.vstack(curves)
                band_lo, band_hi = np.nanpercentile(curve_stack, [16.0, 84.0], axis=0)
        except Exception:
            band_lo = np.full_like(t_grid, np.nan)
            band_hi = np.full_like(t_grid, np.nan)

    resid = delta_fit_exp_norm - delta_exp_norm
    rmse = float(np.sqrt(np.nanmean(resid**2))) if np.any(np.isfinite(resid)) else np.nan
    ss_res = float(np.nansum(resid**2))
    ss_tot = float(np.nansum((delta_exp_norm - np.nanmean(delta_exp_norm)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    cfg = SAMPLE_CONFIG[sample]
    st = sample_style(sample)
    return {
        "sample": sample,
        "sequence": sequence,
        "peak_id": peak_id,
        "label": cfg["legend_label"],
        "color": cfg["color"],
        "marker": st["marker"],
        "msize": st["size"],
        "mface": st["face"],
        "medge": st["edge"],
        "mlw": st["lw"],
        "T_exp": t_exp,
        "w_exp": w_exp,
        "yerr": yerr_visual,
        "yerr_fit": yerr_fit,
        "T_grid": t_grid,
        "omega0": float(omega0),
        "omega0_stderr": float(omega0_stderr),
        "omega0_n_points": int(omega0_n_points),
        "omega0_temperature_min": float(omega0_tmin),
        "omega0_temperature_max": float(omega0_tmax),
        "omega0_method": omega0_method,
        "delta_exp": delta_exp_norm,
        "delta_sum": delta_sum_norm,
        "delta_sum_band_lo": band_lo,
        "delta_sum_band_hi": band_hi,
        "delta_anh": delta_anh_norm,
        "delta_te": delta_te,
        "delta_sum_raw": delta_sum_raw,
        "delta_anh_raw": delta_anh_raw,
        "delta_offset0": offset0,
        "A3": float(a3),
        "A3_err": float(perr[0]),
        "gamma": float(gamma),
        "gamma_err": float(perr[1]),
        "rmse": rmse,
        "r2": r2,
        "fit_status": fit_status,
    }


def build_legend_handles(sample_order: List[str], include_model_band: bool = False) -> List[Any]:
    handles: List[Any] = []
    for sample in sample_order:
        cfg = SAMPLE_CONFIG[sample]
        st = sample_style(sample)
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker=st["marker"],
                markersize=10,
                markerfacecolor=st["face"],
                markeredgecolor=st["edge"],
                markeredgewidth=st["lw"],
                label=cfg["legend_label"],
            )
        )
    handles.extend(
        [
            Line2D([0], [0], color="black", lw=4.0, linestyle="-", label="Total fit"),
            Line2D([0], [0], color="black", lw=2.8, linestyle=(0, (10, 4)), label="TE contribution"),
            Line2D([0], [0], color="black", lw=3.6, linestyle=":", label="Anharmonic contribution"),
        ]
    )
    if include_model_band:
        handles.append(Patch(facecolor="black", alpha=0.16, edgecolor="none", label="Fit 68% band"))
    return handles


def save_legend(out_path: Path, sample_order: List[str], include_model_band: bool = False) -> None:
    handles = build_legend_handles(sample_order, include_model_band=include_model_band)
    labels = [h.get_label() for h in handles]
    fig, ax = plt.subplots(figsize=(15, 3.7))
    ax.axis("off")
    ax.legend(
        handles,
        labels,
        loc="center",
        ncol=4,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.2,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def normalized_ylim(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    sample_order: List[str],
    peak_id: str | None = None,
) -> Tuple[float, float]:
    vals = []
    peak_ids = [peak_id] if peak_id else [p["id"] for p in PEAKS]
    for pid in peak_ids:
        for sample in sample_order:
            d = results[pid][sample]
            omega0 = d["omega0"]
            y_plot = NORM_SCALE * (d["w_exp"] - omega0) / omega0
            yerr = NORM_SCALE * np.asarray(d["yerr"], dtype=float) / omega0
            vals.append(y_plot)
            valid_err = np.isfinite(yerr) & (yerr > 0) & np.isfinite(y_plot)
            if np.any(valid_err):
                vals.append(y_plot[valid_err] - yerr[valid_err])
                vals.append(y_plot[valid_err] + yerr[valid_err])
            mask = (d["T_grid"] >= np.nanmin(d["T_exp"])) & (d["T_grid"] <= np.nanmax(d["T_exp"]))
            vals.append(NORM_SCALE * d["delta_sum_raw"][mask] / omega0)
    all_vals = np.concatenate([np.asarray(v, dtype=float) for v in vals])
    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))
    span = vmax - vmin if vmax > vmin else 1.0
    pad = NORM_Y_PAD_FRAC * span
    if peak_id == "RBLM":
        pad = max(pad, 0.12)
    return vmin - pad, vmax + pad


def absolute_tight_ylim(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    sample_order: List[str],
    peak_id: str,
) -> Tuple[float, float]:
    vals = []
    for sample in sample_order:
        d = results[peak_id][sample]
        y_plot = np.asarray(d["w_exp"], dtype=float)
        yerr = np.asarray(d["yerr"], dtype=float)
        vals.append(y_plot)
        valid_err = np.isfinite(yerr) & (yerr > 0) & np.isfinite(y_plot)
        if np.any(valid_err):
            vals.append(y_plot[valid_err] - yerr[valid_err])
            vals.append(y_plot[valid_err] + yerr[valid_err])
        mask = (d["T_grid"] >= np.nanmin(d["T_exp"])) & (d["T_grid"] <= np.nanmax(d["T_exp"]))
        vals.append(d["omega0"] + d["delta_sum_raw"][mask])

    all_vals = np.concatenate([np.asarray(v, dtype=float) for v in vals])
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        return (0.0, 1.0)
    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))
    span = vmax - vmin if vmax > vmin else 1.0
    pad = max(ABS_TIGHT_Y_PAD_FRAC * span, 0.04)
    return vmin - pad, vmax + pad


def make_combined_figure(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    sample_order: List[str],
    mode: str,
    out_fig: Path,
    out_legend: Path,
) -> None:
    if mode not in {"absolute", "absolute_tight", "normalized", "normalized_free"}:
        raise ValueError("mode must be absolute, absolute_tight, normalized, or normalized_free")

    n_peaks = len(PEAKS)
    n_samples = len(sample_order)
    is_nested_au8a = sample_order == ALIGNED_AU_8A_NESTED_SAMPLE_ORDER
    fig = plt.figure(figsize=STYLE["figsize"])
    gs_main = GridSpec(
        nrows=n_peaks,
        ncols=2,
        width_ratios=[1.0, 1.0],
        hspace=0.06,
        wspace=0.21,
        left=0.12,
        right=0.93,
        top=0.96,
        bottom=0.08,
    )

    axes_freq = [fig.add_subplot(gs_main[i, 0]) for i in range(n_peaks)]
    axes_bands: List[List[plt.Axes]] = []
    for row_idx in range(n_peaks):
        gs_row = GridSpecFromSubplotSpec(
            nrows=n_samples,
            ncols=1,
            subplot_spec=gs_main[row_idx, 1],
            hspace=0.02,
        )
        row_axes = []
        for idx in range(n_samples):
            sharex = row_axes[0] if idx else None
            row_axes.append(fig.add_subplot(gs_row[idx], sharex=sharex))
        axes_bands.append(row_axes)

    norm_ylim_common = normalized_ylim(results, sample_order) if mode == "normalized" else None
    norm_ylim_by_peak = {
        p["id"]: normalized_ylim(results, sample_order, p["id"]) for p in PEAKS
    } if mode == "normalized_free" else {}

    for row_idx, peak in enumerate(PEAKS):
        peak_id = peak["id"]
        ax = axes_freq[row_idx]
        for sample in sample_order:
            d = results[peak_id][sample]
            st = sample_style(sample)
            mask_fit = (d["T_grid"] >= np.nanmin(d["T_exp"])) & (d["T_grid"] <= np.nanmax(d["T_exp"]))

            if mode in {"absolute", "absolute_tight"}:
                y_plot = d["w_exp"]
                y_fit = d["omega0"] + d["delta_sum_raw"][mask_fit]
                ylabel = "Frequency (cm$^{-1}$)"
            else:
                omega0 = d["omega0"]
                y_plot = NORM_SCALE * (d["w_exp"] - omega0) / omega0
                y_fit = NORM_SCALE * d["delta_sum_raw"][mask_fit] / omega0
                ylabel = r"$\Delta\omega / \omega_0$ ($\times 10^{-3}$)"

            ax.scatter(
                d["T_exp"],
                y_plot,
                s=st["size"],
                marker=st["marker"],
                facecolors=st["face"],
                edgecolors=st["edge"],
                linewidths=st["lw"],
                alpha=STYLE["scatter_alpha"],
                zorder=3,
            )
            if INCLUDE_ERRORBARS:
                yerr_plot = np.asarray(d["yerr"], dtype=float).copy()
                if mode not in {"absolute", "absolute_tight"}:
                    yerr_plot = NORM_SCALE * yerr_plot / d["omega0"]
                valid_err = np.isfinite(yerr_plot) & (yerr_plot > 0) & np.isfinite(y_plot)
                if np.any(valid_err):
                    ax.errorbar(
                        d["T_exp"][valid_err],
                        y_plot[valid_err],
                        yerr=yerr_plot[valid_err],
                        fmt="none",
                        ecolor="white",
                        elinewidth=2.7,
                        capsize=3.8,
                        capthick=2.0,
                        alpha=0.60,
                        zorder=5.0,
                    )
                    ax.errorbar(
                        d["T_exp"][valid_err],
                        y_plot[valid_err],
                        yerr=yerr_plot[valid_err],
                        fmt="none",
                        ecolor=st["edge"],
                        elinewidth=1.35,
                        capsize=2.7,
                        capthick=1.05,
                        alpha=0.72,
                        zorder=5.2,
                    )
            ax.plot(d["T_grid"][mask_fit], y_fit, color=d["color"], lw=STYLE["line_width"], zorder=2)

        style_data_axis(ax)
        if mode == "absolute":
            ax.set_ylim(*peak["abs_ylim"])
            ax.yaxis.set_major_locator(MultipleLocator(4.0))
        elif mode == "absolute_tight":
            ax.set_ylim(*absolute_tight_ylim(results, sample_order, peak_id))
        elif mode == "normalized":
            ax.set_ylim(*norm_ylim_common)
        else:
            ax.set_ylim(*norm_ylim_by_peak[peak_id])
        ax.set_ylabel(ylabel)
        if is_nested_au8a:
            ax.text(
                0.50,
                0.965,
                peak_id,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=28,
                fontweight="normal",
                zorder=10,
            )

    for ax in axes_freq[:-1]:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)
    axes_freq[-1].set_xlabel("Temperature (K)")

    for row_idx, peak in enumerate(PEAKS):
        peak_id = peak["id"]
        row_axes = axes_bands[row_idx]
        vals = []
        for sample in sample_order:
            d = results[peak_id][sample]
            vals.extend([d["delta_exp"], d["delta_sum"], d["delta_anh"], d["delta_te"]])
        all_vals = np.concatenate([np.asarray(v, dtype=float) for v in vals])
        vmin = float(np.nanmin(all_vals))
        vmax = max(float(np.nanmax(all_vals)), 0.0)
        span = vmax - vmin if vmax > vmin else 1.0
        ylim = (vmin - Y_PAD_FRAC * span, vmax + Y_PAD_FRAC * span)

        for band_idx, sample in enumerate(sample_order):
            ax = row_axes[band_idx]
            d = results[peak_id][sample]
            ax.scatter(
                d["T_exp"],
                d["delta_exp"],
                s=d["msize"],
                marker=d["marker"],
                facecolors=d["mface"],
                edgecolors=d["medge"],
                linewidths=d["mlw"],
                alpha=1,
                zorder=4,
            )
            if INCLUDE_MODEL_BAND:
                band_ok = np.isfinite(d["delta_sum_band_lo"]) & np.isfinite(d["delta_sum_band_hi"])
                if np.any(band_ok):
                    ax.fill_between(
                        d["T_grid"][band_ok],
                        d["delta_sum_band_lo"][band_ok],
                        d["delta_sum_band_hi"][band_ok],
                        color=d["color"],
                        alpha=0.16,
                        linewidth=0,
                        zorder=1.4,
                    )
            ax.plot(d["T_grid"], d["delta_sum"], color=d["color"], lw=4.0, zorder=3)
            ax.plot(d["T_grid"], d["delta_te"], color="black", lw=STYLE["te_width"], linestyle=(0, (10, 4)), zorder=5)
            ax.plot(d["T_grid"], d["delta_anh"], color="black", lw=STYLE["anh_width"], linestyle=":", zorder=2)
            ax.set_ylim(*ylim)
            style_data_axis(ax)

            annotate_nested = is_nested_au8a and band_idx == 0
            annotate_main = (not is_nested_au8a) and sample == "Aligned_Au_8A"
            if mode == "absolute" and row_idx == 1 and (annotate_nested or annotate_main):
                te_target_t = 235.0
                anh_target_t = 150.0
                te_target_y = float(np.interp(te_target_t, d["T_grid"], d["delta_te"]))
                anh_target_y = float(np.interp(anh_target_t, d["T_grid"], d["delta_anh"]))
                if annotate_nested:
                    te_text = (0.08, 0.18)
                    anh_text = (0.62, 0.72)
                    annotation_fontsize = 20
                    arrow_width = 2.4
                else:
                    te_text = (0.015, 0.14)
                    anh_text = (0.66, 0.78)
                    annotation_fontsize = 20
                    arrow_width = 2.2
                anh_arrowprops = None if annotate_main else dict(
                    arrowstyle="-|>", color="black", lw=arrow_width
                )
                ax.annotate(
                    "Thermal expansion",
                    xy=(te_target_t, te_target_y),
                    xycoords="data",
                    xytext=te_text,
                    textcoords="axes fraction",
                    fontsize=annotation_fontsize,
                    fontweight="semibold",
                    ha="left",
                    va="center",
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=arrow_width),
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.94, pad=1.8),
                    zorder=10,
                )
                ax.annotate(
                    "Anharmonicity",
                    xy=(anh_target_t, anh_target_y),
                    xycoords="data",
                    xytext=anh_text,
                    textcoords="axes fraction",
                    fontsize=annotation_fontsize,
                    fontweight="semibold",
                    ha="center",
                    va="center",
                    arrowprops=anh_arrowprops,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.94, pad=1.8),
                    zorder=10,
                )

            if band_idx == len(sample_order) // 2:
                ax.set_ylabel(r"$\Delta\omega$ (cm$^{-1}$)")
            else:
                ax.set_ylabel("")

            is_last = row_idx == len(PEAKS) - 1 and band_idx == len(sample_order) - 1
            if not is_last:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.set_xlabel("Temperature (K)")

    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {safe_name(out_fig.resolve())}")
    if DISPLAY_FIGURES:
        plt.show()
    plt.close(fig)

    save_legend(out_legend, sample_order, include_model_band=INCLUDE_MODEL_BAND)
    print(f"Saved legend: {safe_name(out_legend.resolve())}")


def style_data_axis(ax: plt.Axes) -> None:
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        length=STYLE["tick_length"],
        width=STYLE["tick_width"],
    )
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(STYLE["spine_width"])


def save_params_table(params: pd.DataFrame, out_csv: Path, out_png: Path) -> None:
    params.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved fit parameters: {safe_name(out_csv.resolve())}")

    disp = params.copy()
    numeric_cols = [
        "omega0_cm-1",
        "omega0_stderr",
        "omega0_temperature_min_K",
        "omega0_temperature_max_K",
        "A3_cm-1",
        "A3_stderr",
        "gamma_parallel",
        "gamma_stderr",
        "RMSE_cm-1",
        "R2",
    ]
    for col in numeric_cols:
        if col in disp:
            disp[col] = disp[col].map(lambda v: "" if not np.isfinite(v) else f"{v:.4g}")

    show_cols = [
        "peak_id",
        "family_label",
        "sequence",
        "omega0_cm-1",
        "omega0_stderr",
        "omega0_n_points",
        "omega0_temperature_min_K",
        "omega0_temperature_max_K",
        "omega0_method",
        "A3_cm-1",
        "gamma_parallel",
        "RMSE_cm-1",
        "R2",
        "fit_status",
    ]
    fig, ax = plt.subplots(figsize=(17, 0.65 + 0.42 * len(disp)))
    ax.axis("off")
    table = ax.table(
        cellText=disp[show_cols].values,
        colLabels=show_cols,
        cellLoc="center",
        loc="upper center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.35)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved fit-parameter table image: {safe_name(out_png.resolve())}")
    plt.close(fig)


def _interpolate_at_temperature(subset: pd.DataFrame, column: str, target_temp: float) -> Tuple[float, str]:
    temp = pd.to_numeric(subset["temperature_K"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(subset[column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(temp) & np.isfinite(values)
    temp = temp[valid]
    values = values[valid]
    if temp.size == 0:
        return np.nan, "missing"

    order = np.argsort(temp)
    temp = temp[order]
    values = values[order]
    unique = pd.DataFrame({"temp": temp, "value": values}).groupby("temp", as_index=False).mean()
    temp = unique["temp"].to_numpy(dtype=float)
    values = unique["value"].to_numpy(dtype=float)
    if temp.size == 1:
        return float(values[0]), f"single_nearest_{temp[0]:g}K"
    if target_temp < temp.min() or target_temp > temp.max():
        idx = int(np.argmin(np.abs(temp - target_temp)))
        return float(values[idx]), f"nearest_{temp[idx]:g}K"
    value = float(np.interp(target_temp, temp, values))
    if np.any(np.isclose(temp, target_temp)):
        return value, "exact"
    lo = float(temp[temp < target_temp].max())
    hi = float(temp[temp > target_temp].min())
    return value, f"interpolated_{lo:g}-{hi:g}K"


def fixed_temperature_comparison_table(
    source: pd.DataFrame,
    target_temp: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for sample in FIXED_TEMPERATURE_SAMPLE_ORDER:
        cfg = SAMPLE_CONFIG[sample]
        source_family = cfg.get("source_family", sample)
        allowed_sequences = DEFAULT_SEQUENCE_FILTERS[sample]
        for peak in PEAKS:
            peak_id = peak["id"]
            subset = source[
                source["family"].astype(str).eq(source_family)
                & source["sequence"].astype(str).isin(allowed_sequences)
                & source["peak_id"].astype(str).eq(peak_id)
            ].copy()
            if subset.empty:
                continue
            position, method = _interpolate_at_temperature(subset, "position_cm-1", target_temp)
            err_col = "position_total_std_visual"
            if err_col not in subset.columns:
                err_col = "position_total_sem" if "position_total_sem" in subset.columns else ""
            if err_col:
                yerr, err_method = _interpolate_at_temperature(subset, err_col, target_temp)
            else:
                yerr, err_method = np.nan, "missing"
            rows.append(
                {
                    "sample": sample,
                    "family": source_family,
                    "family_label": cfg["family_label"],
                    "legend_label": cfg["legend_label"],
                    "sequence": ", ".join(sorted(str(item) for item in allowed_sequences)),
                    "peak_id": peak_id,
                    "target_temperature_K": float(target_temp),
                    "position_cm-1": position,
                    "position_error_cm-1": abs(float(yerr)) if np.isfinite(yerr) else np.nan,
                    "position_method": method,
                    "error_method": err_method,
                    "source_csv_temperature_count": int(subset["temperature_K"].nunique()),
                }
            )
    table = pd.DataFrame(rows)
    table = table[np.isfinite(pd.to_numeric(table["position_cm-1"], errors="coerce"))].copy()
    return table


def _draw_axis_break_marks(ax: plt.Axes, where: str) -> None:
    d = 0.012
    kwargs = {"transform": ax.transAxes, "color": "black", "clip_on": False, "lw": 1.15}
    if where in {"top", "both"}:
        ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    if where in {"bottom", "both"}:
        ax.plot((-d, +d), (-d, +d), **kwargs)
        ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)


def apply_fixed_comparison_y_ticks(ax: plt.Axes, peak_id: str) -> None:
    """Apply the same sparse interior tick positions to every fixed-comparison plot."""
    ax.set_yticks(FIXED_TEMPERATURE_YTICKS[peak_id])
    ax.set_yticks(FIXED_TEMPERATURE_YMINOR_TICKS[peak_id], minor=True)


def save_fixed_temperature_position_comparison(
    source: pd.DataFrame,
    out_dir: Path,
    target_temp: float = FIXED_TEMPERATURE_COMPARISON_K,
) -> None:
    table = fixed_temperature_comparison_table(source, target_temp)
    out_csv = out_dir / f"FIG3_fixed_temperature_peak_positions_{target_temp:g}K.csv"
    table.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved fixed-temperature comparison table: {safe_name(out_csv.resolve())}")
    if table.empty:
        print("No fixed-temperature comparison rows were available.")
        return

    def draw_one(colored_bands: bool, suffix: str, also_legacy_name: bool = False) -> None:
        fig, axes = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=FIXED_TEMPERATURE_STYLE["figsize"],
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.0, 1.0], "hspace": 0.16},
        )
        peak_order = ["G", "D", "RBLM"]
        x_positions = np.arange(len(FIXED_TEMPERATURE_SAMPLE_ORDER), dtype=float)

        for ax, peak_id in zip(axes, peak_order):
            ax.set_facecolor(FIXED_TEMPERATURE_MODE_BACKGROUNDS[peak_id] if colored_bands else "white")
            ax.set_ylim(*FIXED_TEMPERATURE_YLIMS[peak_id])
            ax.set_xlim(-0.55, len(FIXED_TEMPERATURE_SAMPLE_ORDER) - 0.45)
            apply_fixed_comparison_y_ticks(ax, peak_id)
            ax.grid(axis="x", color="#bfc7cc", alpha=0.55, lw=0.8)
            ax.tick_params(
                axis="both",
                which="major",
                direction="in",
                top=True,
                right=False,
                length=5,
                width=1.1,
                labelsize=FIXED_TEMPERATURE_STYLE["tick"],
            )
            ax.tick_params(
                axis="both",
                which="minor",
                direction="in",
                top=True,
                right=False,
                length=2.7,
                width=0.85,
            )
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)

            for idx, sample in enumerate(FIXED_TEMPERATURE_SAMPLE_ORDER):
                row = table[(table["sample"].eq(sample)) & (table["peak_id"].eq(peak_id))]
                if row.empty:
                    continue
                rec = row.iloc[0]
                st = fixed_temperature_sample_style(sample)
                y = float(rec["position_cm-1"])
                yerr = float(rec["position_error_cm-1"]) if np.isfinite(rec["position_error_cm-1"]) else np.nan
                ax.scatter(
                    [idx],
                    [y],
                    s=st["size"] + FIXED_TEMPERATURE_STYLE["marker_halo_extra"],
                    marker=st["marker"],
                    facecolors="white",
                    edgecolors="white",
                    linewidths=3.6,
                    zorder=5,
                )
                ax.scatter(
                    [idx],
                    [y],
                    s=st["size"],
                    marker=st["marker"],
                    facecolors=st["face"],
                    edgecolors=st["edge"],
                    linewidths=st["lw"],
                    zorder=6,
                )
                if np.isfinite(yerr) and yerr > 0:
                    ax.errorbar(
                        [idx],
                        [y],
                        yerr=[yerr],
                        fmt="none",
                        ecolor="white",
                        elinewidth=FIXED_TEMPERATURE_STYLE["error_halo_lw"],
                        capsize=FIXED_TEMPERATURE_STYLE["error_capsize"] + 1.4,
                        capthick=FIXED_TEMPERATURE_STYLE["error_capthick"] + 1.6,
                        alpha=0.90,
                        zorder=20,
                    )
                    ax.errorbar(
                        [idx],
                        [y],
                        yerr=[yerr],
                        fmt="none",
                        ecolor=st["edge"],
                        elinewidth=FIXED_TEMPERATURE_STYLE["error_lw"],
                        capsize=FIXED_TEMPERATURE_STYLE["error_capsize"],
                        capthick=FIXED_TEMPERATURE_STYLE["error_capthick"],
                        alpha=0.98,
                        zorder=21,
                    )

            ax.text(
                0.965,
                0.78,
                f"{peak_id} mode",
                transform=ax.transAxes,
                ha="right",
                va="center",
                color=FIXED_TEMPERATURE_MODE_COLORS[peak_id] if colored_bands else "black",
                fontsize=FIXED_TEMPERATURE_STYLE["mode_label"],
                fontweight="bold",
            )

        labels = [FIXED_TEMPERATURE_CATEGORY_LABELS[sample] for sample in FIXED_TEMPERATURE_SAMPLE_ORDER]
        axes[-1].set_xticks(x_positions, labels=labels, rotation=35, ha="right", rotation_mode="anchor")
        for tick, sample in zip(axes[-1].get_xticklabels(), FIXED_TEMPERATURE_SAMPLE_ORDER):
            tick.set_color(fixed_temperature_sample_style(sample)["edge"])
            tick.set_fontsize(FIXED_TEMPERATURE_STYLE["xtick"])

        fig.text(
            0.05,
            0.50,
            r"Peak position $\omega$ (cm$^{-1}$)",
            rotation=90,
            va="center",
            ha="center",
            fontsize=FIXED_TEMPERATURE_STYLE["axis_label"],
        )
        axes[0].legend(
            [Line2D([0], [0], linestyle="none", label=f"T = {target_temp:g} K")],
            [f"T = {target_temp:g} K"],
            loc="upper left",
            frameon=False,
            handlelength=0,
            handletextpad=0,
            fontsize=FIXED_TEMPERATURE_STYLE["temperature_label"],
        )

        fig.subplots_adjust(left=0.17, right=0.96, bottom=0.23, top=0.96)
        out_png = out_dir / f"FIG3_fixed_temperature_peak_positions_{target_temp:g}K_{suffix}.png"
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"Saved fixed-temperature comparison figure: {safe_name(out_png.resolve())}")
        if also_legacy_name:
            legacy_png = out_dir / f"FIG3_fixed_temperature_peak_positions_{target_temp:g}K.png"
            fig.savefig(legacy_png, dpi=300, bbox_inches="tight")
            print(f"Saved fixed-temperature comparison figure: {safe_name(legacy_png.resolve())}")
        plt.close(fig)

    draw_one(colored_bands=True, suffix="colored_bands", also_legacy_name=True)
    draw_one(colored_bands=False, suffix="white_background")


def omega0_comparison_table_from_results(
    results: Dict[str, Dict[str, Dict[str, Any]]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for sample in FIXED_TEMPERATURE_SAMPLE_ORDER:
        cfg = SAMPLE_CONFIG[sample]
        for peak in PEAKS:
            peak_id = peak["id"]
            if peak_id not in results or sample not in results[peak_id]:
                continue
            rec = results[peak_id][sample]
            rows.append(
                {
                    "sample": sample,
                    "family": cfg.get("source_family", sample),
                    "family_label": cfg["family_label"],
                    "legend_label": cfg["legend_label"],
                    "sequence": rec["sequence"],
                    "peak_id": peak_id,
                    "target_temperature_K": 0.0,
                    "omega0_cm-1": rec["omega0"],
                    "omega0_stderr": rec["omega0_stderr"],
                    "omega0_n_points": rec["omega0_n_points"],
                    "omega0_temperature_min_K": rec["omega0_temperature_min"],
                    "omega0_temperature_max_K": rec["omega0_temperature_max"],
                    "omega0_method": rec["omega0_method"],
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["omega0_cm-1"] = pd.to_numeric(table["omega0_cm-1"], errors="coerce")
    table["omega0_stderr"] = pd.to_numeric(table["omega0_stderr"], errors="coerce")
    return table[np.isfinite(table["omega0_cm-1"])].copy()


def save_zero_temperature_omega0_comparison(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    out_dir: Path,
) -> None:
    table = omega0_comparison_table_from_results(results)
    out_csv = out_dir / "FIG3_extrapolated_omega0_peak_positions_0K.csv"
    table.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved extrapolated omega0 comparison table: {safe_name(out_csv.resolve())}")
    if table.empty:
        print("No omega0 comparison rows were available.")
        return

    def draw_one(colored_bands: bool, suffix: str, also_legacy_name: bool = False) -> None:
        fig, axes = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=FIXED_TEMPERATURE_STYLE["figsize"],
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.0, 1.0], "hspace": 0.16},
        )
        peak_order = ["G", "D", "RBLM"]
        x_positions = np.arange(len(FIXED_TEMPERATURE_SAMPLE_ORDER), dtype=float)

        for ax, peak_id in zip(axes, peak_order):
            ax.set_facecolor(FIXED_TEMPERATURE_MODE_BACKGROUNDS[peak_id] if colored_bands else "white")
            ax.set_ylim(*FIXED_TEMPERATURE_YLIMS[peak_id])
            ax.set_xlim(-0.55, len(FIXED_TEMPERATURE_SAMPLE_ORDER) - 0.45)
            apply_fixed_comparison_y_ticks(ax, peak_id)
            ax.grid(axis="x", color="#bfc7cc", alpha=0.55, lw=0.8)
            ax.tick_params(
                axis="both",
                which="major",
                direction="in",
                top=True,
                right=False,
                length=5,
                width=1.1,
                labelsize=FIXED_TEMPERATURE_STYLE["tick"],
            )
            ax.tick_params(
                axis="both",
                which="minor",
                direction="in",
                top=True,
                right=False,
                length=2.7,
                width=0.85,
            )
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)

            for idx, sample in enumerate(FIXED_TEMPERATURE_SAMPLE_ORDER):
                row = table[(table["sample"].eq(sample)) & (table["peak_id"].eq(peak_id))]
                if row.empty:
                    continue
                rec = row.iloc[0]
                st = fixed_temperature_sample_style(sample)
                y = float(rec["omega0_cm-1"])
                yerr = float(rec["omega0_stderr"]) if np.isfinite(rec["omega0_stderr"]) else np.nan
                ax.scatter(
                    [idx],
                    [y],
                    s=st["size"] + FIXED_TEMPERATURE_STYLE["marker_halo_extra"],
                    marker=st["marker"],
                    facecolors="white",
                    edgecolors="white",
                    linewidths=3.6,
                    zorder=5,
                )
                ax.scatter(
                    [idx],
                    [y],
                    s=st["size"],
                    marker=st["marker"],
                    facecolors=st["face"],
                    edgecolors=st["edge"],
                    linewidths=st["lw"],
                    zorder=6,
                )
                if np.isfinite(yerr) and yerr > 0:
                    ax.errorbar(
                        [idx],
                        [y],
                        yerr=[yerr],
                        fmt="none",
                        ecolor="white",
                        elinewidth=FIXED_TEMPERATURE_STYLE["error_halo_lw"],
                        capsize=FIXED_TEMPERATURE_STYLE["error_capsize"] + 1.4,
                        capthick=FIXED_TEMPERATURE_STYLE["error_capthick"] + 1.6,
                        alpha=0.90,
                        zorder=20,
                    )
                    ax.errorbar(
                        [idx],
                        [y],
                        yerr=[yerr],
                        fmt="none",
                        ecolor=st["edge"],
                        elinewidth=FIXED_TEMPERATURE_STYLE["error_lw"],
                        capsize=FIXED_TEMPERATURE_STYLE["error_capsize"],
                        capthick=FIXED_TEMPERATURE_STYLE["error_capthick"],
                        alpha=0.98,
                        zorder=21,
                    )

            ax.text(
                0.965,
                0.78,
                f"{peak_id} mode",
                transform=ax.transAxes,
                ha="right",
                va="center",
                color=FIXED_TEMPERATURE_MODE_COLORS[peak_id] if colored_bands else "black",
                fontsize=FIXED_TEMPERATURE_STYLE["mode_label"],
                fontweight="bold",
            )

        labels = [FIXED_TEMPERATURE_CATEGORY_LABELS[sample] for sample in FIXED_TEMPERATURE_SAMPLE_ORDER]
        axes[-1].set_xticks(x_positions, labels=labels, rotation=35, ha="right", rotation_mode="anchor")
        for tick, sample in zip(axes[-1].get_xticklabels(), FIXED_TEMPERATURE_SAMPLE_ORDER):
            tick.set_color(fixed_temperature_sample_style(sample)["edge"])
            tick.set_fontsize(FIXED_TEMPERATURE_STYLE["xtick"])

        fig.text(
            0.05,
            0.50,
            r"Extrapolated $\omega_0$ (cm$^{-1}$)",
            rotation=90,
            va="center",
            ha="center",
            fontsize=FIXED_TEMPERATURE_STYLE["axis_label"],
        )
        axes[0].legend(
            [Line2D([0], [0], linestyle="none", label="T = 0 K")],
            ["T = 0 K"],
            loc="upper left",
            frameon=False,
            handlelength=0,
            handletextpad=0,
            fontsize=FIXED_TEMPERATURE_STYLE["temperature_label"],
        )

        fig.subplots_adjust(left=0.17, right=0.96, bottom=0.23, top=0.96)
        out_png = out_dir / f"FIG3_extrapolated_omega0_peak_positions_0K_{suffix}.png"
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"Saved extrapolated omega0 comparison figure: {safe_name(out_png.resolve())}")
        if also_legacy_name:
            legacy_png = out_dir / "FIG3_extrapolated_omega0_peak_positions_0K.png"
            fig.savefig(legacy_png, dpi=300, bbox_inches="tight")
            print(f"Saved extrapolated omega0 comparison figure: {safe_name(legacy_png.resolve())}")
        plt.close(fig)

    draw_one(colored_bands=True, suffix="colored_bands", also_legacy_name=True)
    draw_one(colored_bands=False, suffix="white_background")


def temperature_spread_summary_table(
    source: pd.DataFrame,
    summary_method: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if summary_method not in {"mean", "median"}:
        raise ValueError("summary_method must be 'mean' or 'median'")

    filtered = filter_paper_rows(
        source,
        sequence_filters=DEFAULT_SEQUENCE_FILTERS,
        sample_order=FIXED_TEMPERATURE_SAMPLE_ORDER,
    )
    summary_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    for sample in FIXED_TEMPERATURE_SAMPLE_ORDER:
        cfg = SAMPLE_CONFIG[sample]
        for peak in PEAKS:
            peak_id = peak["id"]
            subset = filtered[
                filtered["plot_sample"].astype(str).eq(sample)
                & filtered["peak_id"].astype(str).eq(peak_id)
            ].copy()
            if subset.empty:
                continue
            subset = subset.sort_values("temperature_K")
            temp = subset["temperature_K"].to_numpy(float)
            pos = subset["position_cm-1"].to_numpy(float)
            valid = np.isfinite(temp) & np.isfinite(pos)
            temp = temp[valid]
            pos = pos[valid]
            if temp.size == 0:
                continue
            if summary_method == "mean":
                center = float(np.nanmean(pos))
                label = "mean"
            else:
                center = float(np.nanmedian(pos))
                label = "median"

            summary_rows.append(
                {
                    "sample": sample,
                    "family": cfg.get("source_family", sample),
                    "family_label": cfg["family_label"],
                    "legend_label": cfg["legend_label"],
                    "sequence": ", ".join(sorted(subset["sequence"].astype(str).unique())),
                    "peak_id": peak_id,
                    "summary_method": label,
                    "summary_position_cm-1": center,
                    "position_min_cm-1": float(np.nanmin(pos)),
                    "position_max_cm-1": float(np.nanmax(pos)),
                    "temperature_min_K": float(np.nanmin(temp)),
                    "temperature_max_K": float(np.nanmax(temp)),
                    "n_temperatures": int(np.unique(temp).size),
                }
            )
            for t_val, p_val in zip(temp, pos):
                trace_rows.append(
                    {
                        "sample": sample,
                        "family": cfg.get("source_family", sample),
                        "family_label": cfg["family_label"],
                        "sequence": ", ".join(sorted(subset["sequence"].astype(str).unique())),
                        "peak_id": peak_id,
                        "temperature_K": float(t_val),
                        "position_cm-1": float(p_val),
                        "summary_method": label,
                    }
                )

    return pd.DataFrame(summary_rows), pd.DataFrame(trace_rows)


def draw_temperature_position_bar(
    ax: plt.Axes,
    x: float,
    temperatures: np.ndarray,
    positions: np.ndarray,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
) -> None:
    temp = np.asarray(temperatures, dtype=float)
    pos = np.asarray(positions, dtype=float)
    valid = np.isfinite(temp) & np.isfinite(pos)
    temp = temp[valid]
    pos = pos[valid]
    if temp.size == 0:
        return
    order = np.argsort(temp)
    temp = temp[order]
    pos = pos[order]
    if temp.size == 1:
        ax.scatter([x], [pos[0]], s=18, color=cmap(norm(temp[0])), edgecolors="none", zorder=9)
        return

    segments = []
    colors = []
    for i in range(temp.size - 1):
        if not np.isfinite(pos[i]) or not np.isfinite(pos[i + 1]):
            continue
        segments.append([(x, pos[i]), (x, pos[i + 1])])
        colors.append(0.5 * (temp[i] + temp[i + 1]))
    if segments:
        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidths=TEMPERATURE_SPREAD_STYLE["bar_lw"],
            capstyle="round",
            zorder=10,
        )
        lc.set_array(np.asarray(colors, dtype=float))
        ax.add_collection(lc)

    ax.scatter(
        np.full_like(temp, x, dtype=float),
        pos,
        s=9,
        c=temp,
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        zorder=11,
    )


def save_temperature_spread_position_summary(
    source: pd.DataFrame,
    out_dir: Path,
    summary_method: str,
) -> None:
    summary, traces = temperature_spread_summary_table(source, summary_method)
    title_word = "Mean" if summary_method == "mean" else "Median"
    stem_word = "mean" if summary_method == "mean" else "median"
    out_summary = out_dir / f"FIG3_temperature_spread_peak_positions_{stem_word}_summary.csv"
    out_traces = out_dir / f"FIG3_temperature_spread_peak_positions_{stem_word}_traces.csv"
    summary.to_csv(out_summary, index=False, encoding="utf-8-sig")
    traces.to_csv(out_traces, index=False, encoding="utf-8-sig")
    print(f"Saved temperature-spread summary table: {safe_name(out_summary.resolve())}")
    print(f"Saved temperature-spread trace table: {safe_name(out_traces.resolve())}")
    if summary.empty:
        print(f"No rows available for {summary_method} temperature-spread summary.")
        return

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=FIXED_TEMPERATURE_STYLE["figsize"],
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0], "hspace": 0.16},
    )
    fig.subplots_adjust(left=0.17, right=0.88, bottom=0.23, top=0.96)
    cmap = mpl.colormaps[TEMPERATURE_SPREAD_CMAP]
    norm = TEMPERATURE_SPREAD_NORM
    peak_order = ["G", "D", "RBLM"]
    x_positions = np.arange(len(FIXED_TEMPERATURE_SAMPLE_ORDER), dtype=float)

    for ax, peak_id in zip(axes, peak_order):
        ax.set_facecolor("white")
        ax.set_ylim(*FIXED_TEMPERATURE_YLIMS[peak_id])
        ax.set_xlim(-0.55, len(FIXED_TEMPERATURE_SAMPLE_ORDER) - 0.45)
        apply_fixed_comparison_y_ticks(ax, peak_id)
        ax.grid(axis="x", color="#bfc7cc", alpha=0.55, lw=0.8)
        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=False,
            length=5,
            width=1.1,
            labelsize=FIXED_TEMPERATURE_STYLE["tick"],
        )
        ax.tick_params(
            axis="both",
            which="minor",
            direction="in",
            top=True,
            right=False,
            length=2.7,
            width=0.85,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        for idx, sample in enumerate(FIXED_TEMPERATURE_SAMPLE_ORDER):
            row = summary[(summary["sample"].eq(sample)) & (summary["peak_id"].eq(peak_id))]
            trace = traces[(traces["sample"].eq(sample)) & (traces["peak_id"].eq(peak_id))]
            if row.empty or trace.empty:
                continue
            rec = row.iloc[0]
            st = fixed_temperature_sample_style(sample)
            draw_temperature_position_bar(
                ax,
                float(idx),
                trace["temperature_K"].to_numpy(float),
                trace["position_cm-1"].to_numpy(float),
                cmap,
                norm,
            )
            y = float(rec["summary_position_cm-1"])
            ax.scatter(
                [idx],
                [y],
                s=st["size"],
                marker=st["marker"],
                facecolors=st["face"],
                edgecolors=st["edge"],
                linewidths=st["lw"],
                zorder=30,
            )

        ax.text(
            0.965,
            0.78,
            f"{peak_id} mode",
            transform=ax.transAxes,
            ha="right",
            va="center",
            color="black",
            fontsize=FIXED_TEMPERATURE_STYLE["mode_label"],
            fontweight="bold",
        )

    labels = [FIXED_TEMPERATURE_CATEGORY_LABELS[sample] for sample in FIXED_TEMPERATURE_SAMPLE_ORDER]
    axes[-1].set_xticks(x_positions, labels=labels, rotation=35, ha="right", rotation_mode="anchor")
    for tick, sample in zip(axes[-1].get_xticklabels(), FIXED_TEMPERATURE_SAMPLE_ORDER):
        tick.set_color(fixed_temperature_sample_style(sample)["edge"])
        tick.set_fontsize(FIXED_TEMPERATURE_STYLE["xtick"])

    fig.text(
        0.05,
        0.50,
        f"{title_word} peak position $\\omega$ (cm$^{{-1}}$)",
        rotation=90,
        va="center",
        ha="center",
        fontsize=FIXED_TEMPERATURE_STYLE["axis_label"],
    )
    axes[0].legend(
        [Line2D([0], [0], linestyle="none", label=f"{title_word} over measured T")],
        [f"{title_word} over measured T"],
        loc="upper left",
        frameon=False,
        handlelength=0,
        handletextpad=0,
        fontsize=FIXED_TEMPERATURE_STYLE["temperature_label"],
    )

    bottom = axes[-1].get_position().y0
    top = axes[0].get_position().y1
    right = axes[0].get_position().x1
    cax = fig.add_axes(
        [
            right + TEMPERATURE_SPREAD_STYLE["colorbar_pad"],
            bottom,
            TEMPERATURE_SPREAD_STYLE["colorbar_width"],
            top - bottom,
        ]
    )
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Temperature (K)", fontsize=TEMPERATURE_SPREAD_STYLE["colorbar_label"])
    cbar.set_ticks(TEMPERATURE_SPREAD_TICKS)
    cbar.ax.tick_params(labelsize=TEMPERATURE_SPREAD_STYLE["colorbar_tick"], direction="out", length=3.5, width=0.9)
    cbar.outline.set_linewidth(1.0)

    out_png = out_dir / f"FIG3_temperature_spread_peak_positions_{stem_word}.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved temperature-spread comparison figure: {safe_name(out_png.resolve())}")
    plt.close(fig)


def build_results(
    input_csv: Path,
    cte_dir: Path,
    sequence_filters: Dict[str, set[str]] | None = None,
    sample_order: List[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Dict[str, Any]]], pd.DataFrame]:
    source = pd.read_csv(input_csv)
    required = {"family", "sequence", "peak_id", "temperature_K", "position_cm-1"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"{input_csv} is missing columns: {missing}")

    if sequence_filters is None:
        sequence_filters = DEFAULT_SEQUENCE_FILTERS
    if sample_order is None:
        sample_order = SAMPLES_PLOT

    data = filter_paper_rows(source, sequence_filters=sequence_filters, sample_order=sample_order)
    data = data.sort_values(["family", "sequence", "peak_id", "temperature_K"])

    cte_gold = load_cte(cte_dir, "gold")
    cte_sapphire = load_cte(cte_dir, "sapphire")
    cte_cnt = load_cte(cte_dir, "cnt")

    results: Dict[str, Dict[str, Dict[str, Any]]] = {p["id"]: {} for p in PEAKS}
    params_rows: List[Dict[str, Any]] = []
    for peak in PEAKS:
        peak_id = peak["id"]
        for sample in sample_order:
            cfg = SAMPLE_CONFIG[sample]
            cte_sub = cte_gold if cfg["cte_kind"] == "gold" else cte_sapphire
            res = fit_temperature_model(data, sample, peak_id, cte_sub, cte_cnt)
            results[peak_id][sample] = res
            params_rows.append(
                {
                    "peak_id": peak_id,
                    "peak_target_cm-1": peak["target"],
                    "family": sample,
                    "family_label": cfg["family_label"],
                    "sequence": res["sequence"],
                    "omega0_cm-1": res["omega0"],
                    "omega0_stderr": res["omega0_stderr"],
                    "omega0_n_points": res["omega0_n_points"],
                    "omega0_temperature_min_K": res["omega0_temperature_min"],
                    "omega0_temperature_max_K": res["omega0_temperature_max"],
                    "omega0_method": res["omega0_method"],
                    "A3_cm-1": res["A3"],
                    "A3_stderr": res["A3_err"],
                    "gamma_parallel": res["gamma"],
                    "gamma_stderr": res["gamma_err"],
                    "RMSE_cm-1": res["rmse"],
                    "R2": res["r2"],
                    "fit_status": res["fit_status"],
                    "source_csv": str(input_csv),
                }
            )
    params = pd.DataFrame(params_rows)
    return data, results, params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Figure 3 Raman-shift plots from AFTER curated CSV data.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cte-dir", type=Path, default=DEFAULT_CTE_DIR)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--with-errorbars",
        action="store_true",
        help="Also save the left-column error-bar variant. Kept for compatibility; variants are saved by default.",
    )
    parser.add_argument(
        "--no-errorbar-copy",
        action="store_true",
        help="Save only the original no-error-bar figures.",
    )
    parser.add_argument(
        "--no-model-band-copy",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--with-model-band-copy",
        action="store_true",
        help="Also save an experimental variant with right-column fitted trend bands.",
    )
    parser.add_argument(
        "--fit-with-errorbars",
        action="store_true",
        help="Use position SEM as sigma in the model fit. Off by default so visual uncertainty does not alter the curves.",
    )
    parser.add_argument("--model-band-samples", type=int, default=MODEL_BAND_SAMPLES)
    return parser.parse_args()


def main() -> None:
    global INCLUDE_ERRORBARS, FIT_WITH_ERRORBARS, INCLUDE_MODEL_BAND, DISPLAY_FIGURES, MODEL_BAND_SAMPLES
    args = parse_args()
    INCLUDE_ERRORBARS = False
    FIT_WITH_ERRORBARS = bool(args.fit_with_errorbars)
    INCLUDE_MODEL_BAND = False
    DISPLAY_FIGURES = bool(args.show)
    MODEL_BAND_SAMPLES = max(0, int(args.model_band_samples))

    apply_style()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading AFTER curated positions: {safe_name(args.input_csv.resolve())}")

    source_for_fixed_temperature = pd.read_csv(args.input_csv)
    save_fixed_temperature_position_comparison(
        source_for_fixed_temperature,
        out_dir,
        target_temp=FIXED_TEMPERATURE_COMPARISON_K,
    )
    save_temperature_spread_position_summary(source_for_fixed_temperature, out_dir, summary_method="mean")
    save_temperature_spread_position_summary(source_for_fixed_temperature, out_dir, summary_method="median")

    figure_specs = [
        ("absolute", "FIG3_Combination1_AbsFreq_AFTER"),
        ("normalized", "FIG3_Combination2_NormalizedLeft_AFTER"),
        ("normalized_free", "FIG3_Combination3_NormalizedLeft_FreeY_AFTER"),
        ("absolute_tight", "FIG3_Combination4_AbsFreq_TightY_AFTER"),
    ]
    variants = [(False, False, "")]
    if not args.no_errorbar_copy or args.with_errorbars:
        variants.append((True, False, "_WITH_ERRORBARS"))
    if args.with_model_band_copy and not args.no_errorbar_copy and not args.no_model_band_copy:
        variants.append((True, True, "_WITH_ERRORBARS_MODELBAND"))

    for au8a_label, au8a_sequence in ALIGNED_AU_8A_SEQUENCE_VARIANTS:
        sequence_filters = {sample: set(sequences) for sample, sequences in DEFAULT_SEQUENCE_FILTERS.items()}
        sequence_filters["Aligned_Au_8A"] = {au8a_sequence}
        print(f"Building Figure 3 with Aligned_Au_8A sequence: {au8a_sequence}")

        data, results, params = build_results(args.input_csv, args.cte_dir, sequence_filters=sequence_filters)
        filtered_csv = out_dir / f"FIG3_input_positions_AFTER_filtered_{au8a_label}.csv"
        data.to_csv(filtered_csv, index=False, encoding="utf-8-sig")
        print(f"Saved filtered Figure 3 input table: {safe_name(filtered_csv.resolve())}")
        if au8a_label == "AlignedAu8A_UP1":
            save_zero_temperature_omega0_comparison(results, out_dir)

        for plot_errorbars, plot_model_band, suffix in variants:
            INCLUDE_ERRORBARS = plot_errorbars
            INCLUDE_MODEL_BAND = plot_model_band
            for mode, stem in figure_specs:
                for fmt in args.formats:
                    make_combined_figure(
                        results=results,
                        sample_order=SAMPLES_PLOT,
                        mode=mode,
                        out_fig=out_dir / f"{stem}_{au8a_label}{suffix}.{fmt}",
                        out_legend=out_dir / f"{stem}_{au8a_label}{suffix}_LEGEND.{fmt}",
                    )

        save_params_table(
            params,
            out_dir / f"FIG3_fit_parameters_AFTER_{au8a_label}.csv",
            out_dir / f"FIG3_fit_parameters_AFTER_{au8a_label}.png",
        )

    nested_label = "AlignedAu8A_NestedSequences"
    print("Building Figure 3 nested-only plot for Aligned_Au_8A UP1 / DOWN1 / UP2")
    nested_data, nested_results, nested_params = build_results(
        args.input_csv,
        args.cte_dir,
        sequence_filters=ALIGNED_AU_8A_NESTED_SEQUENCE_FILTERS,
        sample_order=ALIGNED_AU_8A_NESTED_SAMPLE_ORDER,
    )
    nested_filtered_csv = out_dir / f"FIG3_input_positions_AFTER_filtered_{nested_label}.csv"
    nested_data.to_csv(nested_filtered_csv, index=False, encoding="utf-8-sig")
    print(f"Saved filtered Figure 3 input table: {safe_name(nested_filtered_csv.resolve())}")

    for plot_errorbars, plot_model_band, suffix in variants:
        INCLUDE_ERRORBARS = plot_errorbars
        INCLUDE_MODEL_BAND = plot_model_band
        for mode, stem in figure_specs:
            nested_stem = stem.replace("FIG3_", f"FIG3_{nested_label}_")
            for fmt in args.formats:
                make_combined_figure(
                    results=nested_results,
                    sample_order=ALIGNED_AU_8A_NESTED_SAMPLE_ORDER,
                    mode=mode,
                    out_fig=out_dir / f"{nested_stem}{suffix}.{fmt}",
                    out_legend=out_dir / f"{nested_stem}{suffix}_LEGEND.{fmt}",
                )

    save_params_table(
        nested_params,
        out_dir / f"FIG3_fit_parameters_AFTER_{nested_label}.csv",
        out_dir / f"FIG3_fit_parameters_AFTER_{nested_label}.png",
    )
    print(f"Done. Outputs in: {safe_name(out_dir.resolve())}")


if __name__ == "__main__":
    main()
