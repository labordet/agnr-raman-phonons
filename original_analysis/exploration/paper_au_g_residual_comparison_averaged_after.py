"""
Batch paper figures comparing averaged G-peak Lorentzian residuals for aligned Au samples.

Panel a) is Aligned_Au_8A and panel b) is Aligned_Au_3A. Each panel shows the
averaged baseline-corrected spectrum in the selected G-region, the Lorentzian
sum fit, and the residual below it. It also creates Aligned_Au_8A temperature
self-comparisons within each UP/DOWN/UP sequence. TXT files are read as X in
column 1 and one or more Y spectra in columns 2..N. Input spectra and
fitted-result files are read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


APP_DIR = Path(__file__).resolve().parent
AFTER_DIR = APP_DIR / "GAMMA_T_COMPARISON" / "AFTER"
DEFAULT_FITTED_ROOT = AFTER_DIR / "FITTED_RESULTS_PAPER_CLEAN"
DEFAULT_SPECTRA_ROOT = AFTER_DIR / "AVERAGED_BASELINE_CORRECTED_TXT_FOR_CLUSTER"
SECOND_PASS_AVERAGED_ROOT = APP_DIR / "Fitting_Cluster_ETH_Second_Pass_After_Deselecting_Bad_Spectra"
DEFAULT_OUTPUT_DIR = APP_DIR / "FIGURES SI" / "PAPER_AU_G_RESIDUAL_COMPARISON_AVERAGED_AFTER"

FAMILY_A = "Aligned_Au_8A"
FAMILY_B = "Aligned_Au_3A"

FAMILY_LABELS = {
    "Aligned_Au_8A": "Aligned Au 8A",
    "Aligned_Au_3A": "Aligned Au 3A",
}

FAMILY_MARKERS = {
    "Aligned_Au_8A": {"marker": "D", "facecolors": "#f08228", "edgecolors": "#f08228"},
    "Aligned_Au_3A": {"marker": "o", "facecolors": "#feb715", "edgecolors": "#feb715"},
}

PEAK_ORDER = ["RBLM", "CH_L1", "CH_L2", "CH_MID", "D", "G"]
PEAK_LABELS = {
    "RBLM": "RBLM",
    "CH_L1": "CH L1",
    "CH_L2": "CH L2",
    "CH_MID": "CH MID",
    "D": "D",
    "G": "G",
}
PEAK_COLORS = {
    "G": "#17becf",
}

SPECTRUM_GROUP_COLS = [
    "family",
    "sequence",
    "temperature_K",
    "file_path",
    "file_name",
    "spectrum_in_file",
    "y_column_number",
    "y_column_name",
]


@dataclass(frozen=True)
class SpectrumRecord:
    family: str
    sequence: str
    temperature_K: float
    file_path: str
    file_name: str
    spectrum_in_file: int
    y_column_number: int
    y_column_name: str
    r_squared: float
    rmse: float
    quality_flag: str

    @property
    def key(self) -> Tuple[object, ...]:
        return (
            self.family,
            self.sequence,
            self.temperature_K,
            self.file_path,
            self.file_name,
            self.spectrum_in_file,
            self.y_column_number,
            self.y_column_name,
        )


def setup_paper_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Nimbus Sans L"],
            "axes.titlesize": 14,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 11,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "axes.unicode_minus": False,
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "rm",
        }
    )


def safe_filename(text: object, max_len: int = 160) -> str:
    value = re.sub(r"[<>:\"/\\|?*\x00-\x1f]+", "_", str(text))
    value = re.sub(r"\s+", "_", value).strip("._ ")
    return (value[:max_len] or "item").strip("._ ")


def parse_float(value: object, default: float) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        if math.isfinite(result):
            return result
    except (TypeError, ValueError):
        pass
    return default


def parse_int(value: object, default: int) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def natural_key(text: object) -> List[object]:
    parts = re.split(r"(\d+)", str(text))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def lorentzian(x: np.ndarray, height: float, pos: float, width: float) -> np.ndarray:
    width = max(float(width), 1e-12)
    return float(height) / (1.0 + ((x - float(pos)) / (0.5 * width)) ** 2)


def read_txt_spectra(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Read TXT with first column as X and all remaining columns as Y spectra."""
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first_line = ""
        for line in handle:
            if line.strip():
                first_line = line.rstrip("\n")
                break
    if not first_line:
        raise ValueError(f"Empty TXT file: {path}")

    if "\t" in first_line:
        sep = "\t"
        first_parts = first_line.split("\t")
    elif "," in first_line:
        sep = ","
        first_parts = first_line.split(",")
    elif ";" in first_line:
        sep = ";"
        first_parts = first_line.split(";")
    else:
        sep = r"\s+"
        first_parts = re.split(r"\s+", first_line.strip())

    def all_numeric(parts: Iterable[str]) -> bool:
        for part in parts:
            try:
                float(part)
            except ValueError:
                return False
        return True

    header = None if all_numeric(first_parts) else 0
    df = pd.read_csv(path, sep=sep, header=header, engine="python", comment="#")
    if header is None:
        df.columns = [f"Column {idx + 1}" for idx in range(df.shape[1])]
    else:
        df.columns = [str(col).strip() or f"Column {idx + 1}" for idx, col in enumerate(df.columns)]

    numeric = df.apply(pd.to_numeric, errors="coerce")
    if numeric.shape[1] < 2:
        raise ValueError(f"Expected X plus at least one Y column in {path}")

    x = numeric.iloc[:, 0].to_numpy(dtype=float)
    y = numeric.iloc[:, 1:].to_numpy(dtype=float)
    finite_x = np.isfinite(x)
    return x[finite_x], y[finite_x, :], [str(col) for col in df.columns[1:]]


class SpectraCache:
    def __init__(self, spectra_root: Path) -> None:
        self.spectra_root = spectra_root
        self._cache: Dict[Path, Tuple[np.ndarray, np.ndarray, List[str]]] = {}
        self.fallback_roots: List[Path] = []
        if SECOND_PASS_AVERAGED_ROOT.exists() and SECOND_PASS_AVERAGED_ROOT.resolve() != spectra_root.resolve():
            self.fallback_roots.append(SECOND_PASS_AVERAGED_ROOT)

    def resolve(self, record: SpectrumRecord) -> Path:
        rel = Path(record.file_path)
        candidates = [
            self.spectra_root / rel,
            self.spectra_root / record.family / record.sequence / record.file_name,
            APP_DIR / rel,
            APP_DIR / record.family / record.sequence / record.file_name,
        ]
        for fallback_root in self.fallback_roots:
            candidates.extend(
                [
                    fallback_root / rel,
                    fallback_root / record.family / record.sequence / record.file_name,
                ]
            )
        for candidate in candidates:
            if candidate.exists():
                return candidate

        family_roots = [self.spectra_root / record.family]
        family_roots.extend(fallback_root / record.family for fallback_root in self.fallback_roots)
        for family_root in family_roots:
            if not family_root.exists():
                continue
            matches = [
                match
                for match in family_root.rglob(record.file_name)
                if " - Copy" not in str(match.relative_to(family_root))
            ]
            if matches:
                return sorted(matches, key=lambda p: len(p.parts))[0]

        raise FileNotFoundError(
            f"Could not find averaged TXT for {record.file_path}. "
            f"Checked {self.spectra_root} and fallback {SECOND_PASS_AVERAGED_ROOT}."
        )

    def load_y_column(self, record: SpectrumRecord) -> Tuple[np.ndarray, np.ndarray, Path]:
        path = self.resolve(record)
        if path not in self._cache:
            self._cache[path] = read_txt_spectra(path)
        x, y_matrix, _names = self._cache[path]
        y_index = int(record.y_column_number) - 2
        if y_index < 0 or y_index >= y_matrix.shape[1]:
            raise IndexError(
                f"{path.name} has {y_matrix.shape[1]} Y columns, but fitted result asks for TXT column "
                f"{record.y_column_number}."
            )
        y = y_matrix[:, y_index]
        finite = np.isfinite(x) & np.isfinite(y)
        return x[finite], y[finite], path


def load_all_fitted_results(fitted_root: Path) -> Tuple[pd.DataFrame, List[Path]]:
    """Match the main plotter app: recursively load all cluster *_long_results.csv files."""
    if not fitted_root.exists():
        raise FileNotFoundError(f"Fitted result folder not found: {fitted_root}")

    csv_paths = sorted(
        path
        for path in fitted_root.rglob("*_long_results.csv")
        if "_checkpoints" not in {part.lower() for part in path.parts}
    )
    if not csv_paths:
        raise FileNotFoundError(f"No *_long_results.csv files found under: {fitted_root}")

    required = set(SPECTRUM_GROUP_COLS + ["peak_id", "height", "position_cm-1", "width_fwhm_cm-1"])
    frames = []
    skipped = []
    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            skipped.append(f"{csv_path}: could not read CSV ({exc})")
            continue
        missing = required - set(df.columns)
        if missing:
            skipped.append(f"{csv_path}: missing columns {sorted(missing)}")
            continue
        df["_source_csv"] = str(csv_path)
        frames.append(df)

    if not frames:
        details = "\n".join(skipped[:12])
        raise ValueError(f"No usable fitted long-result CSV files found in {fitted_root}.\n{details}")

    if skipped:
        print(f"Skipped {len(skipped)} fitted CSV(s) that were not usable:")
        for message in skipped[:8]:
            print(f"  - {message}")

    results = pd.concat(frames, ignore_index=True)
    for col in ["temperature_K", "spectrum_in_file", "y_column_number", "r_squared", "rmse"]:
        if col in results.columns:
            results[col] = pd.to_numeric(results[col], errors="coerce")
    return results, csv_paths


def filter_family_results(results: pd.DataFrame, family: str) -> pd.DataFrame:
    df = results[results["family"].astype(str) == family].copy()
    if df.empty:
        available = sorted(results["family"].astype(str).dropna().unique().tolist())
        raise ValueError(f"Family {family!r} not found in fitted results. Available families: {available}")
    return df


def make_record(group: pd.DataFrame) -> SpectrumRecord:
    row = group.iloc[0]
    return SpectrumRecord(
        family=str(row["family"]),
        sequence=str(row["sequence"]),
        temperature_K=float(row["temperature_K"]),
        file_path=str(row["file_path"]),
        file_name=str(row["file_name"]),
        spectrum_in_file=parse_int(row.get("spectrum_in_file"), 1),
        y_column_number=parse_int(row.get("y_column_number"), 2),
        y_column_name=str(row.get("y_column_name", "")),
        r_squared=parse_float(row.get("r_squared"), float("nan")),
        rmse=parse_float(row.get("rmse"), float("nan")),
        quality_flag=str(row.get("quality_flag", "")),
    )


def build_records(df: pd.DataFrame) -> Tuple[List[SpectrumRecord], Dict[Tuple[object, ...], pd.DataFrame]]:
    records: List[SpectrumRecord] = []
    peak_rows_by_key: Dict[Tuple[object, ...], pd.DataFrame] = {}
    order_map = {peak: idx for idx, peak in enumerate(PEAK_ORDER)}

    for key, group in df.groupby(SPECTRUM_GROUP_COLS, dropna=False, sort=False):
        peak_ids = set(group["peak_id"].astype(str))
        if "G" not in peak_ids:
            continue
        record = make_record(group)
        group = group.copy()
        group["_peak_order"] = group["peak_id"].map(lambda peak: order_map.get(str(peak), 999))
        group = group.sort_values("_peak_order")
        records.append(record)
        peak_rows_by_key[record.key] = group

    records.sort(
        key=lambda rec: (
            natural_key(rec.sequence),
            rec.temperature_K,
            natural_key(rec.file_name),
            rec.y_column_number,
        )
    )
    return records, peak_rows_by_key


@dataclass
class PanelData:
    record: SpectrumRecord
    x_data: np.ndarray
    y_norm: np.ndarray
    x_fit: np.ndarray
    total_norm: np.ndarray
    g_norm: Optional[np.ndarray]
    residual_norm: np.ndarray
    scale: float
    r_squared_local: float
    rms_residual: float
    max_abs_residual: float
    path: Path


def reconstruct_total(x: np.ndarray, peak_rows: pd.DataFrame) -> np.ndarray:
    total = np.zeros_like(x, dtype=float)
    for _, row in peak_rows.iterrows():
        height = parse_float(row.get("height"), float("nan"))
        pos = parse_float(row.get("position_cm-1"), float("nan"))
        width = parse_float(row.get("width_fwhm_cm-1"), float("nan"))
        if math.isfinite(height) and math.isfinite(pos) and math.isfinite(width) and width > 0:
            total += lorentzian(x, height, pos, width)
    return total


def reconstruct_peak(x: np.ndarray, peak_rows: pd.DataFrame, peak_id: str) -> Optional[np.ndarray]:
    rows = peak_rows[peak_rows["peak_id"].astype(str) == peak_id]
    if rows.empty:
        return None
    total = np.zeros_like(x, dtype=float)
    used = False
    for _, row in rows.iterrows():
        height = parse_float(row.get("height"), float("nan"))
        pos = parse_float(row.get("position_cm-1"), float("nan"))
        width = parse_float(row.get("width_fwhm_cm-1"), float("nan"))
        if math.isfinite(height) and math.isfinite(pos) and math.isfinite(width) and width > 0:
            total += lorentzian(x, height, pos, width)
            used = True
    return total if used else None


def make_panel_data(
    record: SpectrumRecord,
    peak_rows: pd.DataFrame,
    cache: SpectraCache,
    x_min: float,
    x_max: float,
    smooth_points: int,
) -> PanelData:
    x, y, path = cache.load_y_column(record)
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    mask = (x >= x_min) & (x <= x_max)
    if not np.any(mask):
        raise ValueError(f"{record.file_name} column {record.y_column_number} has no points in X range.")

    x_data = x[mask]
    y_data = y[mask]
    order = np.argsort(x_data)
    x_data = x_data[order]
    y_data = y_data[order]

    x_fit = np.linspace(x_min, x_max, smooth_points)
    fit_data = reconstruct_total(x_data, peak_rows)
    total_fit = reconstruct_total(x_fit, peak_rows)
    g_fit = reconstruct_peak(x_fit, peak_rows, "G")

    scale = float(np.nanmax(total_fit)) if total_fit.size else float("nan")
    if not math.isfinite(scale) or abs(scale) < 1e-12:
        fallback = float(np.nanmax(np.abs(y_data))) if y_data.size else 1.0
        scale = fallback if math.isfinite(fallback) and fallback > 0 else 1.0

    y_norm = y_data / scale
    fit_norm_data = fit_data / scale
    total_norm = total_fit / scale
    g_norm = None if g_fit is None else g_fit / scale
    residual_norm = y_norm - fit_norm_data
    finite_residual = residual_norm[np.isfinite(residual_norm)]
    rms_residual = float(np.sqrt(np.mean(finite_residual**2))) if finite_residual.size else float("nan")
    max_abs_residual = float(np.nanmax(np.abs(finite_residual))) if finite_residual.size else float("nan")
    finite_fit = np.isfinite(y_norm) & np.isfinite(fit_norm_data)
    if np.count_nonzero(finite_fit) >= 2:
        y_valid = y_norm[finite_fit]
        fit_valid = fit_norm_data[finite_fit]
        ss_res = float(np.sum((y_valid - fit_valid) ** 2))
        ss_tot = float(np.sum((y_valid - np.mean(y_valid)) ** 2))
        r_squared_local = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    else:
        r_squared_local = float("nan")

    return PanelData(
        record=record,
        x_data=x_data,
        y_norm=y_norm,
        x_fit=x_fit,
        total_norm=total_norm,
        g_norm=g_norm,
        residual_norm=residual_norm,
        scale=scale,
        r_squared_local=r_squared_local,
        rms_residual=rms_residual,
        max_abs_residual=max_abs_residual,
        path=path,
    )


def marker_kwargs(family: str, size: float, edge_width: float) -> Dict[str, object]:
    style = dict(FAMILY_MARKERS[family])
    style.update({"s": size, "linewidths": edge_width, "alpha": 0.96, "zorder": 5})
    return style


def format_temp(temp: float) -> str:
    return f"{temp:g}K"


def panel_title(panel: PanelData) -> str:
    rec = panel.record
    return f"{FAMILY_LABELS.get(rec.family, rec.family)} | {rec.sequence} | {format_temp(rec.temperature_K)} | Y{rec.y_column_number:02d}"


def plot_panel(
    ax_top,
    ax_res,
    panel: PanelData,
    panel_letter: str,
    top_ylim: Tuple[float, float],
    residual_ylim: Tuple[float, float],
    args: argparse.Namespace,
) -> None:
    rec = panel.record
    family_color = FAMILY_MARKERS[rec.family]["edgecolors"]

    ax_top.plot(
        panel.x_data,
        panel.y_norm,
        color="#3f3f3f",
        linewidth=args.data_line_width,
        alpha=0.9,
        zorder=2,
    )
    ax_top.scatter(
        panel.x_data,
        panel.y_norm,
        label=FAMILY_LABELS.get(rec.family, rec.family),
        **marker_kwargs(rec.family, args.scatter_size, args.marker_edge),
    )
    ax_top.plot(
        panel.x_fit,
        panel.total_norm,
        color="#d62728",
        linewidth=args.fit_line_width,
        label="Lorentzian fit",
        zorder=6,
        solid_capstyle="round",
    )
    if panel.g_norm is not None and args.show_g_component:
        ax_top.plot(
            panel.x_fit,
            panel.g_norm,
            color=PEAK_COLORS["G"],
            linestyle="--",
            linewidth=args.component_line_width,
            label="G",
            zorder=7,
            dash_capstyle="round",
        )

    ax_res.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0, zorder=1)
    ax_res.plot(
        panel.x_data,
        panel.residual_norm,
        color=family_color,
        linewidth=args.residual_line_width,
        marker=FAMILY_MARKERS[rec.family]["marker"],
        markersize=max(2.5, math.sqrt(args.scatter_size) * 0.45),
        markerfacecolor=FAMILY_MARKERS[rec.family]["facecolors"],
        markeredgecolor=FAMILY_MARKERS[rec.family]["edgecolors"],
        markeredgewidth=args.marker_edge * 0.75,
        zorder=4,
    )

    ax_top.set_title(panel_title(panel), loc="left", pad=10)
    ax_top.text(
        0.01,
        0.98,
        f"{panel_letter})",
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=args.panel_label_font,
        fontweight="bold",
    )
    if args.show_stats:
        ax_top.text(
            0.035,
            0.82,
            f"R$^2$={panel.r_squared_local:.5f}\nRMSE={panel.rms_residual:.4f}",
            transform=ax_top.transAxes,
            ha="left",
            va="top",
            fontsize=args.stats_font,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=2.2),
        )

    ax_top.set_ylim(*top_ylim)
    ax_res.set_ylim(*residual_ylim)
    ax_res.set_xlim(args.x_min, args.x_max)
    ax_top.set_xlim(args.x_min, args.x_max)
    ax_top.tick_params(labelbottom=False)
    ax_top.legend(loc=args.legend_loc, frameon=False, ncol=1, handlelength=2.2)


def style_axes(fig, axes, args: argparse.Namespace) -> None:
    for ax in axes:
        ax.grid(False)
        ax.tick_params(
            axis="both",
            labelsize=args.tick_font,
            width=args.tick_width,
            length=args.tick_length,
            direction="out",
            top=False,
            right=False,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(args.spine_width)
            spine.set_color("#333333")
    fig.supxlabel(r"Raman shift (cm$^{-1}$)", fontsize=args.axis_font)


def interior_ticks(vmin: float, vmax: float, max_ticks: int = 5) -> np.ndarray:
    """Return restrained major ticks without placing ticks on either axis corner."""
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.asarray([], dtype=float)
    locator = MaxNLocator(nbins=max_ticks, steps=[1, 2, 2.5, 5, 10])
    ticks = np.asarray(locator.tick_values(vmin, vmax), dtype=float)
    tol = max(abs(vmax - vmin) * 1e-9, 1e-12)
    return ticks[(ticks > vmin + tol) & (ticks < vmax - tol)]


def apply_consistent_interior_ticks(
    top_axes,
    residual_axes,
    x_limits: Tuple[float, float],
    top_limits: Tuple[float, float],
    residual_limits: Tuple[float, float],
) -> None:
    """Use identical, sparse ticks for corresponding panels and omit endpoints."""
    x_ticks = interior_ticks(*x_limits, max_ticks=5)
    top_ticks = interior_ticks(*top_limits, max_ticks=4)
    residual_ticks = interior_ticks(*residual_limits, max_ticks=4)
    for ax in top_axes:
        ax.set_xticks(x_ticks)
        ax.set_yticks(top_ticks)
    for ax in residual_axes:
        ax.set_xticks(x_ticks)
        ax.set_yticks(residual_ticks)


def plot_comparison(panel_a: PanelData, panel_b: PanelData, output_base: Optional[Path], args: argparse.Namespace) -> None:
    top_max = np.nanmax(
        [
            np.nanmax(panel_a.y_norm),
            np.nanmax(panel_a.total_norm),
            np.nanmax(panel_b.y_norm),
            np.nanmax(panel_b.total_norm),
            1.0,
        ]
    )
    top_min = np.nanmin(
        [
            np.nanmin(panel_a.y_norm),
            np.nanmin(panel_a.total_norm),
            np.nanmin(panel_b.y_norm),
            np.nanmin(panel_b.total_norm),
            0.0,
        ]
    )
    top_ylim = (min(-0.05, top_min - 0.04), max(1.08, top_max + 0.06))

    residual_abs = np.nanmax(
        [
            np.nanmax(np.abs(panel_a.residual_norm)),
            np.nanmax(np.abs(panel_b.residual_norm)),
            0.03,
        ]
    )
    residual_ylim = (-residual_abs * 1.12, residual_abs * 1.12)

    fig = plt.figure(figsize=(args.width, args.height), dpi=args.dpi)
    gs = fig.add_gridspec(2, 2, height_ratios=(3.0, 1.35), hspace=0.05, wspace=0.23)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    res_a = fig.add_subplot(gs[1, 0], sharex=ax_a)
    res_b = fig.add_subplot(gs[1, 1], sharex=ax_b, sharey=res_a)

    plot_panel(ax_a, res_a, panel_a, "a", top_ylim, residual_ylim, args)
    plot_panel(ax_b, res_b, panel_b, "b", top_ylim, residual_ylim, args)

    ax_a.set_ylabel("Intensity (a.u.)", fontsize=args.axis_font)
    res_a.set_ylabel("Residual intensity (a.u.)", fontsize=args.axis_font)
    ax_b.tick_params(labelleft=False)
    res_b.tick_params(labelleft=False)
    style_axes(fig, [ax_a, ax_b, res_a, res_b], args)
    apply_consistent_interior_ticks(
        [ax_a, ax_b],
        [res_a, res_b],
        (args.x_min, args.x_max),
        top_ylim,
        residual_ylim,
    )

    if args.hide_y_tick_labels:
        for ax in [ax_a, ax_b, res_a, res_b]:
            ax.set_yticklabels([])

    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.105, top=0.92, wspace=0.23, hspace=0.06)
    if output_base is None:
        plt.close(fig)
        return

    output_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats:
        fig.savefig(output_base.with_suffix(f".{fmt}"), dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def select_combo_record(
    records: List[SpectrumRecord],
    temperature: float,
    sequence: Optional[str] = None,
) -> SpectrumRecord:
    candidates = [
        rec
        for rec in records
        if math.isclose(float(rec.temperature_K), float(temperature), abs_tol=1e-6)
        and (sequence is None or rec.sequence == sequence)
    ]
    if not candidates:
        sequence_note = f" in {sequence}" if sequence else ""
        raise ValueError(f"No averaged spectrum found at {temperature:g} K{sequence_note}.")
    candidates.sort(key=lambda rec: (rec.y_column_number, natural_key(rec.file_name), natural_key(rec.file_path)))
    return candidates[0]


def plot_four_panel_temperature_comparison(
    panels: List[PanelData],
    output_base: Path,
    args: argparse.Namespace,
) -> None:
    """Plot 3A/8A at low/high temperature as four fit-plus-residual cells."""
    if len(panels) != 4:
        raise ValueError("The four-panel comparison requires exactly four spectra.")

    top_max = max(
        1.0,
        *[float(np.nanmax(panel.y_norm)) for panel in panels],
        *[float(np.nanmax(panel.total_norm)) for panel in panels],
    )
    top_min = min(
        0.0,
        *[float(np.nanmin(panel.y_norm)) for panel in panels],
        *[float(np.nanmin(panel.total_norm)) for panel in panels],
    )
    top_ylim = (min(-0.05, top_min - 0.04), max(1.08, top_max + 0.06))
    residual_abs = max(0.03, *[float(np.nanmax(np.abs(panel.residual_norm))) for panel in panels])
    residual_ylim = (-residual_abs * 1.12, residual_abs * 1.12)

    fig = plt.figure(figsize=(args.combo_width, args.combo_height), dpi=args.dpi)
    outer = fig.add_gridspec(2, 2, hspace=0.20, wspace=0.18)
    top_axes = []
    residual_axes = []
    letters = ["a", "b", "c", "d"]

    # Input order is top-left, top-right, bottom-left, bottom-right.
    for index, (panel, letter) in enumerate(zip(panels, letters)):
        row, col = divmod(index, 2)
        inner = outer[row, col].subgridspec(2, 1, height_ratios=(3.0, 1.15), hspace=0.04)
        ax_top = fig.add_subplot(inner[0, 0])
        ax_res = fig.add_subplot(inner[1, 0], sharex=ax_top)
        plot_panel(ax_top, ax_res, panel, letter, top_ylim, residual_ylim, args)
        ax_top.set_title(
            f"{FAMILY_LABELS.get(panel.record.family, panel.record.family)} | {panel.record.temperature_K:g} K",
            loc="left",
            pad=7,
        )
        if row == 0:
            ax_res.tick_params(labelbottom=False)
        if row == 1 and ax_top.get_legend() is not None:
            ax_top.get_legend().remove()
        if col == 1:
            ax_top.tick_params(labelleft=False)
            ax_res.tick_params(labelleft=False)
        top_axes.append(ax_top)
        residual_axes.append(ax_res)

    top_axes[0].set_ylabel("Normalized intensity", fontsize=args.axis_font)
    top_axes[2].set_ylabel("Normalized intensity", fontsize=args.axis_font)
    residual_axes[0].set_ylabel("Residual", fontsize=args.axis_font)
    residual_axes[2].set_ylabel("Residual", fontsize=args.axis_font)

    style_axes(fig, top_axes + residual_axes, args)
    apply_consistent_interior_ticks(
        top_axes,
        residual_axes,
        (args.x_min, args.x_max),
        top_ylim,
        residual_ylim,
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.085, top=0.955)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats:
        fig.savefig(output_base.with_suffix(f".{fmt}"), dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def records_by_temperature(records: List[SpectrumRecord]) -> Dict[float, List[SpectrumRecord]]:
    by_temp: Dict[float, List[SpectrumRecord]] = {}
    for rec in records:
        by_temp.setdefault(float(rec.temperature_K), []).append(rec)
    for temp_records in by_temp.values():
        temp_records.sort(key=lambda rec: (natural_key(rec.sequence), rec.y_column_number, natural_key(rec.file_name)))
    return by_temp


def paired_records(
    records_a: List[SpectrumRecord],
    records_b: List[SpectrumRecord],
    same_y_column_only: bool,
) -> List[Tuple[SpectrumRecord, SpectrumRecord]]:
    by_temp_b = records_by_temperature(records_b)
    pairs: List[Tuple[SpectrumRecord, SpectrumRecord]] = []
    for rec_a in records_a:
        candidates = by_temp_b.get(float(rec_a.temperature_K), [])
        if same_y_column_only:
            candidates = [rec_b for rec_b in candidates if rec_b.y_column_number == rec_a.y_column_number]
        for rec_b in candidates:
            if math.isclose(float(rec_a.temperature_K), float(rec_b.temperature_K), abs_tol=1e-6):
                pairs.append((rec_a, rec_b))
    return pairs


def output_base_for_pair(output_dir: Path, rec_a: SpectrumRecord, rec_b: SpectrumRecord, x_min: float, x_max: float) -> Path:
    token = hashlib.sha1(
        f"{rec_a.file_path}|{rec_a.y_column_number}|{rec_b.file_path}|{rec_b.y_column_number}|{x_min:g}|{x_max:g}".encode(
            "utf-8", errors="replace"
        )
    ).hexdigest()[:10]
    temp = format_temp(rec_a.temperature_K)
    file_name = safe_filename(
        f"{temp}_8A_Y{rec_a.y_column_number:02d}_3A_Y{rec_b.y_column_number:02d}_{token}_G_residuals",
        max_len=90,
    )
    return (
        output_dir
        / "Au8A_vs_Au3A"
        / safe_filename(rec_a.sequence)
        / temp
        / file_name
    )


def temperature_comparison_pairs(
    records: List[SpectrumRecord],
    target_temps: List[float],
    reference_temps: Optional[List[float]],
) -> List[Tuple[SpectrumRecord, SpectrumRecord]]:
    target_temps = [float(temp) for temp in target_temps]
    reference_set = None if reference_temps is None else {float(temp) for temp in reference_temps}
    pairs: List[Tuple[SpectrumRecord, SpectrumRecord]] = []

    for rec_ref in records:
        ref_temp = float(rec_ref.temperature_K)
        if reference_set is not None and not any(math.isclose(ref_temp, temp, abs_tol=1e-6) for temp in reference_set):
            continue
        if reference_set is None and any(math.isclose(ref_temp, temp, abs_tol=1e-6) for temp in target_temps):
            continue

        for target_temp in target_temps:
            for rec_target in records:
                if rec_target.family != rec_ref.family:
                    continue
                if rec_target.sequence != rec_ref.sequence:
                    continue
                if rec_target.y_column_number != rec_ref.y_column_number:
                    continue
                if not math.isclose(float(rec_target.temperature_K), target_temp, abs_tol=1e-6):
                    continue
                if math.isclose(ref_temp, float(rec_target.temperature_K), abs_tol=1e-6):
                    continue
                pairs.append((rec_ref, rec_target))
    pairs.sort(
        key=lambda pair: (
            natural_key(pair[0].sequence),
            pair[0].temperature_K,
            pair[1].temperature_K,
            pair[0].y_column_number,
        )
    )
    return pairs


def output_base_for_temperature_pair(
    output_dir: Path,
    rec_ref: SpectrumRecord,
    rec_target: SpectrumRecord,
    x_min: float,
    x_max: float,
) -> Path:
    ref_temp = format_temp(rec_ref.temperature_K)
    target_temp = format_temp(rec_target.temperature_K)
    token = hashlib.sha1(
        f"{rec_ref.file_path}|{rec_ref.y_column_number}|{rec_target.file_path}|{rec_target.y_column_number}|{x_min:g}|{x_max:g}".encode(
            "utf-8", errors="replace"
        )
    ).hexdigest()[:10]
    file_name = safe_filename(f"{ref_temp}_vs_{target_temp}_Y{rec_ref.y_column_number:02d}_{token}_G", max_len=55)
    return (
        output_dir
        / "Au8A_Temp_vs_Temp"
        / safe_filename(rec_ref.sequence)
        / file_name
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create paired a)/b) G-peak Lorentzian residual figures from averaged AFTER spectra "
            "for Aligned Au 8A vs Aligned Au 3A and Au 8A temperature self-comparisons."
        )
    )
    parser.add_argument("--fitted-root", type=Path, default=DEFAULT_FITTED_ROOT)
    parser.add_argument("--spectra-root", type=Path, default=DEFAULT_SPECTRA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--x-min", type=float, default=1550.0)
    parser.add_argument("--x-max", type=float, default=1650.0)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf", "svg"], choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--width", type=float, default=10.5)
    parser.add_argument("--height", type=float, default=7.5)
    parser.add_argument("--smooth-points", type=int, default=1200)
    parser.add_argument(
        "--comparison-set",
        choices=["all", "au8-vs-3a", "au8-temperature", "four-panel"],
        default="all",
        help=(
            "Which figure set to make. Default also creates the four-cell 3A/8A, "
            "low/high-temperature SI comparison."
        ),
    )
    parser.add_argument("--combo-low-temp", type=float, default=80.0)
    parser.add_argument("--combo-high-temp", type=float, default=300.0)
    parser.add_argument("--combo-au8-sequence", default="Spikes_Removed_UP_1")
    parser.add_argument("--combo-width", type=float, default=11.0)
    parser.add_argument("--combo-height", type=float, default=10.0)
    parser.add_argument(
        "--temperature-targets",
        nargs="+",
        type=float,
        default=[300.0, 290.0, 280.0],
        help="Target temperatures for Au 8A self-comparisons, e.g. 300 290 280.",
    )
    parser.add_argument(
        "--temperature-reference-temps",
        nargs="+",
        type=float,
        default=None,
        help="Optional reference temperatures for Au 8A self-comparisons. Default is every non-target temperature.",
    )
    parser.add_argument("--scatter-size", type=float, default=45.0)
    parser.add_argument("--marker-edge", type=float, default=1.2)
    parser.add_argument("--data-line-width", type=float, default=1.1)
    parser.add_argument("--fit-line-width", type=float, default=2.5)
    parser.add_argument("--component-line-width", type=float, default=1.5)
    parser.add_argument("--residual-line-width", type=float, default=1.2)
    parser.add_argument("--spine-width", type=float, default=1.3)
    parser.add_argument("--tick-width", type=float, default=1.2)
    parser.add_argument("--tick-length", type=float, default=5.0)
    parser.add_argument("--axis-font", type=float, default=18.0)
    parser.add_argument("--tick-font", type=float, default=14.0)
    parser.add_argument("--panel-label-font", type=float, default=18.0)
    parser.add_argument("--stats-font", type=float, default=12.0)
    parser.add_argument("--legend-loc", default="best")
    parser.add_argument("--hide-y-tick-labels", action="store_true")
    parser.add_argument("--show-stats", action="store_true", default=True)
    parser.add_argument("--no-stats", dest="show_stats", action="store_false")
    parser.add_argument("--show-g-component", action="store_true", default=True)
    parser.add_argument("--no-g-component", dest="show_g_component", action="store_false")
    parser.add_argument(
        "--same-y-column-only",
        action="store_true",
        help="Only compare 8A and 3A spectra with the same TXT Y-column number. By default, every 3A column at the same temperature is compared.",
    )
    parser.add_argument("--max-figures", type=int, default=0, help="Limit number of figures for testing. 0 means all.")
    parser.add_argument("--dry-run", action="store_true", help="Build pair list and render nothing/safe no-output test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_paper_style()
    args.smooth_points = max(300, min(5000, int(args.smooth_points)))

    print(f"Reading fitted Lorentzian cluster results from: {args.fitted_root}")
    all_results, csv_paths = load_all_fitted_results(args.fitted_root)
    print(f"Loaded {len(all_results)} fitted peak rows from {len(csv_paths)} *_long_results.csv file(s).")
    df_a = filter_family_results(all_results, FAMILY_A)
    df_b = filter_family_results(all_results, FAMILY_B)
    records_a, peaks_a = build_records(df_a)
    records_b, peaks_b = build_records(df_b)

    pairs_same_temp: List[Tuple[SpectrumRecord, SpectrumRecord]] = []
    if args.comparison_set in {"all", "au8-vs-3a"}:
        pairs_same_temp = paired_records(records_a, records_b, args.same_y_column_only)
        if args.max_figures > 0:
            pairs_same_temp = pairs_same_temp[: args.max_figures]

    pairs_au8_temp: List[Tuple[SpectrumRecord, SpectrumRecord]] = []
    if args.comparison_set in {"all", "au8-temperature"}:
        pairs_au8_temp = temperature_comparison_pairs(
            records_a,
            target_temps=args.temperature_targets,
            reference_temps=args.temperature_reference_temps,
        )
        if args.max_figures > 0:
            pairs_au8_temp = pairs_au8_temp[: args.max_figures]

    combo_records: List[SpectrumRecord] = []
    if args.comparison_set in {"all", "four-panel"}:
        rec_3a_low = select_combo_record(records_b, args.combo_low_temp)
        rec_3a_high = select_combo_record(records_b, args.combo_high_temp)
        rec_8a_low = select_combo_record(records_a, args.combo_low_temp, args.combo_au8_sequence)
        rec_8a_high = select_combo_record(records_a, args.combo_high_temp, args.combo_au8_sequence)
        combo_records = [rec_3a_low, rec_8a_low, rec_3a_high, rec_8a_high]

    print(f"Found {len(records_a)} {FAMILY_A} spectra columns.")
    print(f"Found {len(records_b)} {FAMILY_B} spectra columns.")
    print(f"Prepared {len(pairs_same_temp)} same-temperature Au 8A vs Au 3A comparison figure(s).")
    print(
        f"Prepared {len(pairs_au8_temp)} Au 8A temperature comparison figure(s) "
        f"against targets {', '.join(format_temp(temp) for temp in args.temperature_targets)}."
    )
    if combo_records:
        print(
            "Prepared four-panel SI comparison: "
            f"3A and 8A ({args.combo_au8_sequence}) at "
            f"{format_temp(args.combo_low_temp)} and {format_temp(args.combo_high_temp)}."
        )
    if args.dry_run:
        for rec_a, rec_b in pairs_same_temp[:10]:
            print(
                f"DRY same-temp {rec_a.sequence} {format_temp(rec_a.temperature_K)} "
                f"8A Y{rec_a.y_column_number:02d} vs 3A Y{rec_b.y_column_number:02d}"
            )
        for rec_ref, rec_target in pairs_au8_temp[:10]:
            print(
                f"DRY Au8-temp {rec_ref.sequence} Y{rec_ref.y_column_number:02d} "
                f"{format_temp(rec_ref.temperature_K)} vs {format_temp(rec_target.temperature_K)}"
            )
        for rec in combo_records:
            print(
                f"DRY four-panel {rec.family} {rec.sequence} "
                f"{format_temp(rec.temperature_K)} Y{rec.y_column_number:02d}"
            )
        return

    cache = SpectraCache(args.spectra_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    failures = []

    for idx, (rec_a, rec_b) in enumerate(pairs_same_temp, start=1):
        try:
            panel_a = make_panel_data(rec_a, peaks_a[rec_a.key], cache, args.x_min, args.x_max, args.smooth_points)
            panel_b = make_panel_data(rec_b, peaks_b[rec_b.key], cache, args.x_min, args.x_max, args.smooth_points)
            output_base = output_base_for_pair(args.output_dir, rec_a, rec_b, args.x_min, args.x_max)
            plot_comparison(panel_a, panel_b, output_base, args)
            manifest_rows.append(
                {
                    "comparison_type": "au8_vs_3a_same_temperature",
                    "output_base": str(output_base),
                    "temperature_K": rec_a.temperature_K,
                    "family_a": rec_a.family,
                    "sequence_a": rec_a.sequence,
                    "file_a": rec_a.file_name,
                    "y_column_a": rec_a.y_column_number,
                    "r2_a": rec_a.r_squared,
                    "rmse_a": rec_a.rmse,
                    "r2_plotted_a": panel_a.r_squared_local,
                    "rmse_plotted_a": panel_a.rms_residual,
                    "residual_rms_a": panel_a.rms_residual,
                    "residual_max_abs_a": panel_a.max_abs_residual,
                    "family_b": rec_b.family,
                    "sequence_b": rec_b.sequence,
                    "file_b": rec_b.file_name,
                    "y_column_b": rec_b.y_column_number,
                    "r2_b": rec_b.r_squared,
                    "rmse_b": rec_b.rmse,
                    "r2_plotted_b": panel_b.r_squared_local,
                    "rmse_plotted_b": panel_b.rms_residual,
                    "residual_rms_b": panel_b.rms_residual,
                    "residual_max_abs_b": panel_b.max_abs_residual,
                    "x_min": args.x_min,
                    "x_max": args.x_max,
                }
            )
            if idx % 25 == 0 or idx == len(pairs_same_temp):
                print(f"Saved {idx}/{len(pairs_same_temp)} same-temperature comparison figure(s)...")
        except Exception as exc:
            msg = (
                f"{rec_a.sequence} {format_temp(rec_a.temperature_K)} "
                f"8A Y{rec_a.y_column_number:02d} vs 3A Y{rec_b.y_column_number:02d}: {exc}"
            )
            failures.append(msg)
            print(f"FAILED {msg}")

    for idx, (rec_ref, rec_target) in enumerate(pairs_au8_temp, start=1):
        try:
            panel_ref = make_panel_data(
                rec_ref, peaks_a[rec_ref.key], cache, args.x_min, args.x_max, args.smooth_points
            )
            panel_target = make_panel_data(
                rec_target, peaks_a[rec_target.key], cache, args.x_min, args.x_max, args.smooth_points
            )
            output_base = output_base_for_temperature_pair(
                args.output_dir, rec_ref, rec_target, args.x_min, args.x_max
            )
            plot_comparison(panel_ref, panel_target, output_base, args)
            manifest_rows.append(
                {
                    "comparison_type": "au8_temperature_comparison",
                    "output_base": str(output_base),
                    "temperature_K": rec_ref.temperature_K,
                    "target_temperature_K": rec_target.temperature_K,
                    "family_a": rec_ref.family,
                    "sequence_a": rec_ref.sequence,
                    "file_a": rec_ref.file_name,
                    "y_column_a": rec_ref.y_column_number,
                    "r2_a": rec_ref.r_squared,
                    "rmse_a": rec_ref.rmse,
                    "r2_plotted_a": panel_ref.r_squared_local,
                    "rmse_plotted_a": panel_ref.rms_residual,
                    "residual_rms_a": panel_ref.rms_residual,
                    "residual_max_abs_a": panel_ref.max_abs_residual,
                    "family_b": rec_target.family,
                    "sequence_b": rec_target.sequence,
                    "file_b": rec_target.file_name,
                    "y_column_b": rec_target.y_column_number,
                    "r2_b": rec_target.r_squared,
                    "rmse_b": rec_target.rmse,
                    "r2_plotted_b": panel_target.r_squared_local,
                    "rmse_plotted_b": panel_target.rms_residual,
                    "residual_rms_b": panel_target.rms_residual,
                    "residual_max_abs_b": panel_target.max_abs_residual,
                    "x_min": args.x_min,
                    "x_max": args.x_max,
                }
            )
            if idx % 25 == 0 or idx == len(pairs_au8_temp):
                print(f"Saved {idx}/{len(pairs_au8_temp)} Au 8A temperature comparison figure(s)...")
        except Exception as exc:
            msg = (
                f"{rec_ref.sequence} Y{rec_ref.y_column_number:02d} "
                f"{format_temp(rec_ref.temperature_K)} vs {format_temp(rec_target.temperature_K)}: {exc}"
            )
            failures.append(msg)
            print(f"FAILED {msg}")

    if combo_records:
        try:
            combo_panels = []
            for rec in combo_records:
                peak_lookup = peaks_a if rec.family == FAMILY_A else peaks_b
                combo_panels.append(
                    make_panel_data(
                        rec,
                        peak_lookup[rec.key],
                        cache,
                        args.x_min,
                        args.x_max,
                        args.smooth_points,
                    )
                )
            combo_name = safe_filename(
                f"Aligned_Au_3A_Aligned_Au_8A_{args.combo_au8_sequence}_"
                f"{format_temp(args.combo_low_temp)}_{format_temp(args.combo_high_temp)}_G_fit_residuals_4panel"
            )
            combo_base = args.output_dir / "FOUR_PANEL_80K_300K" / combo_name
            plot_four_panel_temperature_comparison(combo_panels, combo_base, args)
            manifest_rows.append(
                {
                    "comparison_type": "four_panel_3a_8a_low_high_temperature",
                    "output_base": str(combo_base),
                    "temperature_K": args.combo_low_temp,
                    "target_temperature_K": args.combo_high_temp,
                    "family_a": FAMILY_B,
                    "sequence_a": combo_records[0].sequence,
                    "file_a": combo_records[0].file_name,
                    "y_column_a": combo_records[0].y_column_number,
                    "family_b": FAMILY_A,
                    "sequence_b": args.combo_au8_sequence,
                    "file_b": combo_records[1].file_name,
                    "y_column_b": combo_records[1].y_column_number,
                    "x_min": args.x_min,
                    "x_max": args.x_max,
                }
            )
            print(f"Saved four-panel SI comparison to: {combo_base}")
        except Exception as exc:
            msg = f"four-panel 3A/8A low/high-temperature comparison: {exc}"
            failures.append(msg)
            print(f"FAILED {msg}")

    if args.comparison_set == "four-panel":
        manifest_path = args.output_dir / "FOUR_PANEL_80K_300K" / "four_panel_manifest.csv"
        failure_path = args.output_dir / "FOUR_PANEL_80K_300K" / "four_panel_failures.txt"
    else:
        manifest_path = args.output_dir / "au_g_residual_comparison_manifest.csv"
        failure_path = args.output_dir / "au_g_residual_comparison_failures.txt"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    if failures:
        failure_path.write_text("\n".join(failures), encoding="utf-8")
        print(f"Finished with {len(failures)} failure(s). See {failure_path}")
    elif failure_path.exists():
        failure_path.unlink()
    print(f"Manifest: {manifest_path}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
