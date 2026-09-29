#!/usr/bin/env python3
"""Build the final local SI figure set from read-only Raman inputs.

The script does not modify spectra, fitted CSV/XLSX files, or Overleaf.  It
creates publication-ready PNG files in ``SI FIGURES FINAL`` using:

* original paired RAW/spike-removed spectra for the spike-removal comparison;
* matched spike-removed and ALS-corrected spectra for the baseline comparison;
* the curated AFTER averaged TXT and Lorentzian results for fit figures; and
* already approved AFTER figures for the nested-sequence and linewidth panels.

All waterfall normalizations are display-only.  A single scale is applied to
each before/after pair, so processing differences are not hidden by separate
normalization.
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import paper_spike_before_after_plotter as spike_app


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "SI FIGURES FINAL"
AFTER_DIR = ROOT / "GAMMA_T_COMPARISON" / "AFTER"
AVERAGED_TXT_ROOT = AFTER_DIR / "AVERAGED_BASELINE_CORRECTED_TXT_FOR_CLUSTER"
SECOND_PASS_TXT_ROOT = ROOT / "Fitting_Cluster_ETH_Second_Pass_After_Deselecting_Bad_Spectra"
FITTED_ROOT = AFTER_DIR / "FITTED_RESULTS_PAPER_CLEAN"
ALS_ROOT = ROOT / "ALS_BASELINE_CORRECTED"
EXAMPLE_MANIFEST_DIR = ROOT / "FIGURES SI" / "ALS_PAPER_EXAMPLE_FIGURES"

TEMPERATURE_CMAP = mpl.colormaps["turbo"]
TEMPERATURE_NORM = Normalize(vmin=70.0, vmax=300.0)


@dataclass(frozen=True)
class FamilySpec:
    key: str
    folder: str
    fit_family: str
    sequence: str
    label: str
    color: str
    marker: str
    hollow: bool


FAMILIES: Tuple[FamilySpec, ...] = (
    FamilySpec(
        "aligned_au_3a",
        "Aligned_Au_3A",
        "Aligned_Au_3A",
        "Spikes_Removed",
        "Aligned Au, low coverage",
        "#feb715",
        "o",
        False,
    ),
    FamilySpec(
        "aligned_au_8a",
        "Aligned_Au_8A",
        "Aligned_Au_8A",
        "Spikes_Removed_UP_1",
        "Aligned Au, high coverage",
        "#f08228",
        "D",
        False,
    ),
    FamilySpec(
        "unaligned_au_8a",
        "MIRA_Au_unaligned_8A",
        "MIRA_Au_unaligned_8A",
        "Spikes_Removed_UP_1",
        "Unaligned Au, high coverage",
        "#ae540b",
        "D",
        True,
    ),
    FamilySpec(
        "unaligned_ro_8a",
        "MIRA_RO_unaligned_8A",
        "MIRA_RO_unaligned_8A",
        "Spikes_Removed_UP_1",
        "Unaligned RO, high coverage",
        "#008686",
        "D",
        True,
    ),
    FamilySpec(
        "aligned_ro_8a",
        "Aligned_RO_8A",
        "Aligned_RO_8A",
        "Spikes_Removed_DOWN_1",
        "Aligned RO, high coverage",
        "#00c8c8",
        "D",
        False,
    ),
)


@dataclass(frozen=True)
class ExampleSelection:
    family: str
    sequence: str
    temperature: float
    clean_name: str
    curve_index: int
    x_start: float
    x_end: float
    als_lambda: float
    als_p: float
    als_iterations: int


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 13,
            "axes.linewidth": 1.15,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.top": False,
            "ytick.right": False,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "mathtext.default": "regular",
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(top=False, right=False, direction="out")
    for spine in ax.spines.values():
        spine.set_linewidth(1.15)


def marker_kwargs(spec: FamilySpec, color: str, size: float = 3.0) -> Dict[str, object]:
    return {
        "marker": spec.marker,
        "markersize": size,
        "markerfacecolor": "none" if spec.hollow else color,
        "markeredgecolor": color,
        "markeredgewidth": 0.8 if spec.hollow else 0.35,
    }


def save_figure(fig: plt.Figure, name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved {path}", flush=True)
    return path


def _sorted_unique_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    if x.size == 0:
        return x, y
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x_unique, unique_index = np.unique(x, return_index=True)
    return x_unique, y[unique_index]


def interpolate_curve(x: np.ndarray, y: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    x, y = _sorted_unique_xy(x, y)
    if x.size < 2:
        return np.full_like(x_ref, np.nan, dtype=float)
    out = np.interp(x_ref, x, y)
    out[(x_ref < x[0]) | (x_ref > x[-1])] = np.nan
    return out


def mean_spectrum(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = spike_app.read_spectrum_file(path)
    if not data.curves:
        raise ValueError(f"No Y spectra in {path}")
    x_ref = np.asarray(data.curves[0].x, dtype=float)
    ys = [interpolate_curve(curve.x, curve.y, x_ref) for curve in data.curves]
    return x_ref, np.nanmean(np.vstack(ys), axis=0)


def paired_file_curves(
    path_a: Path,
    path_b: Path,
    x_min: float = 200.0,
    x_max: float = 2000.0,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    data_a = spike_app.read_spectrum_file(path_a)
    data_b = spike_app.read_spectrum_file(path_b)
    count = min(len(data_a.curves), len(data_b.curves))
    if count == 0:
        raise ValueError(f"No paired Y spectra in {path_a} and {path_b}")
    x_ref = np.asarray(data_b.curves[0].x, dtype=float)
    domain = np.isfinite(x_ref) & (x_ref >= x_min) & (x_ref <= x_max)
    x_ref = x_ref[domain]
    a_curves: List[np.ndarray] = []
    b_curves: List[np.ndarray] = []
    for index in range(count):
        a = data_a.curves[index]
        b = data_b.curves[index]
        a_curves.append(interpolate_curve(a.x, a.y, x_ref))
        b_curves.append(interpolate_curve(b.x, b.y, x_ref))
    return x_ref, a_curves, b_curves


def average_paired_records(
    pairs: Sequence[Tuple[Path, Path]],
    x_min: float = 200.0,
    x_max: float = 2000.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_ref: Optional[np.ndarray] = None
    left: List[np.ndarray] = []
    right: List[np.ndarray] = []
    for path_left, path_right in pairs:
        x_local, left_local, right_local = paired_file_curves(
            path_left, path_right, x_min=x_min, x_max=x_max
        )
        if x_ref is None:
            x_ref = x_local
        for y_left, y_right in zip(left_local, right_local):
            if not np.array_equal(x_local, x_ref):
                y_left = interpolate_curve(x_local, y_left, x_ref)
                y_right = interpolate_curve(x_local, y_right, x_ref)
            left.append(y_left)
            right.append(y_right)
    if x_ref is None or not left:
        raise ValueError("No paired spectra were available for averaging")
    return x_ref, np.nanmean(np.vstack(left), axis=0), np.nanmean(np.vstack(right), axis=0)


def most_spike_affected_pair(
    pairs: Sequence[Tuple[Path, Path]],
    x_min: float = 200.0,
    x_max: float = 2000.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the correctly paired Y column with the clearest removed spike.

    Averaging line-scan columns can dilute a spike that occurred in only one
    acquisition.  The SI comparison therefore shows, at each temperature, the
    paired column with the largest robust positive RAW-minus-clean excursion.
    The RAW and spike-removed traces always come from that same column.
    """

    best: Optional[Tuple[float, np.ndarray, np.ndarray, np.ndarray]] = None
    for raw_path, clean_path in pairs:
        x_local, raw_curves, clean_curves = paired_file_curves(
            raw_path, clean_path, x_min=x_min, x_max=x_max
        )
        for raw_y, clean_y in zip(raw_curves, clean_curves):
            baseline, scale = _display_scale(clean_y)
            del baseline
            positive_change = np.maximum(raw_y - clean_y, 0.0) / scale
            finite = positive_change[np.isfinite(positive_change)]
            score = float(np.nanpercentile(finite, 99.8)) if finite.size else -np.inf
            if best is None or score > best[0]:
                best = (score, x_local, raw_y, clean_y)
    if best is None:
        raise ValueError("No paired spectra were available for spike comparison")
    return best[1], best[2], best[3]


def scan_original_records() -> Dict[str, List[spike_app.SpectrumRecord]]:
    records: Dict[str, List[spike_app.SpectrumRecord]] = {}
    for spec in FAMILIES:
        scanned, warnings = spike_app.scan_sample_folder(ROOT / spec.folder)
        selected = [
            row
            for row in scanned
            if row.has_pair
            and row.clean_rel_dir.startswith("Spikes_Removed")
            and row.temperature is not None
        ]
        if not selected:
            detail = "; ".join(warnings[:3])
            raise RuntimeError(f"No paired records for {spec.label} ({spec.sequence}). {detail}")
        records[spec.key] = selected
        print(
            f"Mapped {spec.label}: {len(selected)} paired files across original spike folders",
            flush=True,
        )
    return records


def load_example_selections() -> Dict[str, ExampleSelection]:
    selections: Dict[str, ExampleSelection] = {}
    manifests = sorted(EXAMPLE_MANIFEST_DIR.glob("*_paper_example_manifest.csv"))
    for manifest in manifests:
        frame = pd.read_csv(manifest)
        if frame.empty:
            continue
        row = frame.iloc[0]
        family = str(row["family"])
        selections[family] = ExampleSelection(
            family=family,
            sequence=str(row["spike_folder"]),
            temperature=float(row["temperature_K"]),
            clean_name=Path(str(row["input_path"])).name,
            curve_index=max(int(row["y_column_index"]) - 1, 0),
            x_start=float(row["x_start_saved"]),
            x_end=float(row["x_end_saved"]),
            als_lambda=float(row["lambda"]),
            als_p=float(row["p"]),
            als_iterations=int(row["iterations"]),
        )
    missing = [spec.fit_family for spec in FAMILIES if spec.fit_family not in selections]
    if missing:
        raise RuntimeError(f"Missing ALS example manifests for: {', '.join(missing)}")
    return selections


def selected_curve(path: Path, curve_index: int) -> Tuple[np.ndarray, np.ndarray]:
    data = spike_app.read_spectrum_file(path)
    if not data.curves:
        raise ValueError(f"No Y spectra in {path}")
    index = min(max(int(curve_index), 0), len(data.curves) - 1)
    curve = data.curves[index]
    return _sorted_unique_xy(np.asarray(curve.x, dtype=float), np.asarray(curve.y, dtype=float))


def selected_curve_pair(
    left_path: Path,
    right_path: Path,
    curve_index: int,
    x_start: float,
    x_end: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_right, y_right = selected_curve(right_path, curve_index)
    domain = (x_right >= x_start) & (x_right <= x_end)
    x_ref = x_right[domain]
    y_right = y_right[domain]
    x_left, y_left = selected_curve(left_path, curve_index)
    y_left = interpolate_curve(x_left, y_left, x_ref)
    return x_ref, y_left, y_right


def record_for_example(
    spec: FamilySpec,
    selection: ExampleSelection,
    records: Sequence[spike_app.SpectrumRecord],
) -> spike_app.SpectrumRecord:
    candidates = [
        row
        for row in records
        if row.has_pair
        and row.clean_rel_dir == selection.sequence
        and row.clean_path is not None
        and row.clean_path.name == selection.clean_name
        and row.temperature is not None
        and abs(float(row.temperature) - selection.temperature) < 1e-6
    ]
    if not candidates:
        raise RuntimeError(
            f"Could not map example {spec.label}, {selection.temperature:g} K, "
            f"{selection.sequence}, {selection.clean_name} to a RAW/spike pair"
        )
    return candidates[0]


def clearest_spike_pair(
    records: Sequence[spike_app.SpectrumRecord],
    x_start: float,
    x_end: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, spike_app.SpectrumRecord, int, float]:
    best: Optional[
        Tuple[float, np.ndarray, np.ndarray, np.ndarray, spike_app.SpectrumRecord, int]
    ] = None
    for record in records:
        if record.raw_path is None or record.clean_path is None:
            continue
        try:
            x, raw_curves, clean_curves = paired_file_curves(
                record.raw_path,
                record.clean_path,
                x_min=x_start,
                x_max=x_end,
            )
        except (OSError, ValueError):
            continue
        score_domain = (
            np.isfinite(x)
            & (x >= max(x_start, 350.0))
            & (x <= min(x_end, 1900.0))
        )
        for low, high in ((1180.0, 1380.0), (1540.0, 1640.0)):
            score_domain &= ~((x >= low) & (x <= high))
        for curve_index, (raw_y, clean_y) in enumerate(zip(raw_curves, clean_curves)):
            _, scale = _display_scale(clean_y)
            positive_change = np.maximum(raw_y - clean_y, 0.0) / scale
            finite = positive_change[np.isfinite(positive_change) & score_domain]
            score = float(np.nanpercentile(finite, 99.8)) if finite.size else -np.inf
            if best is None or score > best[0]:
                best = (score, x, raw_y, clean_y, record, curve_index)
    if best is None:
        raise RuntimeError("No paired RAW/spike spectrum was available")
    return best[1], best[2], best[3], best[4], best[5], best[0]


def grouped_record_pairs(
    records: Sequence[spike_app.SpectrumRecord],
    right_path_for_record,
) -> List[Tuple[float, List[Tuple[Path, Path]]]]:
    grouped: Dict[float, List[Tuple[Path, Path]]] = defaultdict(list)
    for row in records:
        if row.raw_path is None or row.clean_path is None or row.temperature is None:
            continue
        right = right_path_for_record(row)
        if right is None or not right.exists():
            continue
        grouped[float(row.temperature)].append((row.raw_path, right))
    return sorted(grouped.items(), key=lambda item: item[0])


def _display_scale(y_reference: np.ndarray) -> Tuple[float, float]:
    finite = y_reference[np.isfinite(y_reference)]
    if finite.size == 0:
        return 0.0, 1.0
    baseline = float(np.nanpercentile(finite, 5.0))
    top = float(np.nanpercentile(finite, 99.5))
    scale = max(top - baseline, np.nanstd(finite), 1e-12)
    return baseline, scale


def _temperature_color(temperature: float) -> Tuple[float, float, float, float]:
    return TEMPERATURE_CMAP(TEMPERATURE_NORM(np.clip(temperature, 70.0, 300.0)))


def _configure_waterfall_row(
    axes: Sequence[plt.Axes],
    row_index: int,
    total_rows: int,
    x_limits: Tuple[float, float] = (250.0, 2000.0),
) -> None:
    for ax in axes:
        style_axis(ax)
        ax.set_xlim(*x_limits)
        ax.set_yticks([])
        if row_index < total_rows - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
            ax.set_xticks([400, 800, 1200, 1600, 2000])


def add_temperature_colorbar(fig: plt.Figure, cax: plt.Axes) -> None:
    sm = mpl.cm.ScalarMappable(norm=TEMPERATURE_NORM, cmap=TEMPERATURE_CMAP)
    colorbar = fig.colorbar(sm, cax=cax)
    colorbar.set_label("Temperature (K)")
    colorbar.set_ticks([70, 116, 162, 208, 254, 300])
    colorbar.outline.set_linewidth(1.0)


def plot_raw_vs_spike_removed_waterfall(
    record_map: Dict[str, List[spike_app.SpectrumRecord]],
) -> Path:
    fig = plt.figure(figsize=(12.5, 12.8))
    grid = GridSpec(
        len(FAMILIES),
        3,
        figure=fig,
        width_ratios=[1.0, 1.0, 0.035],
        left=0.075,
        right=0.96,
        bottom=0.065,
        top=0.99,
        hspace=0.08,
        wspace=0.06,
    )
    axes: List[Tuple[plt.Axes, plt.Axes]] = []
    for row_index, spec in enumerate(FAMILIES):
        ax_raw = fig.add_subplot(grid[row_index, 0])
        ax_clean = fig.add_subplot(grid[row_index, 1], sharex=ax_raw)
        axes.append((ax_raw, ax_clean))
        records = record_map[spec.key]
        grouped: Dict[float, List[Tuple[Path, Path]]] = defaultdict(list)
        for row in records:
            if row.raw_path and row.clean_path and row.temperature is not None:
                grouped[float(row.temperature)].append((row.raw_path, row.clean_path))
        temperatures = sorted(grouped)
        offset_step = 0.22
        max_display = 1.35
        for offset_index, temperature in enumerate(temperatures):
            x, raw_y, clean_y = most_spike_affected_pair(
                grouped[temperature], 250.0, 2000.0
            )
            baseline, scale = _display_scale(clean_y)
            raw_display = (raw_y - baseline) / scale
            clean_display = (clean_y - baseline) / scale
            max_display = max(
                max_display,
                float(np.nanpercentile(raw_display[np.isfinite(raw_display)], 99.7)),
            )
            offset = offset_index * offset_step
            color = _temperature_color(temperature)
            kwargs = marker_kwargs(spec, color, size=2.5)
            ax_raw.plot(
                x,
                raw_display + offset,
                color=color,
                linewidth=0.8,
                markevery=14,
                zorder=2,
                **kwargs,
            )
            ax_clean.plot(
                x,
                clean_display + offset,
                color=color,
                linewidth=0.8,
                markevery=14,
                zorder=2,
                **kwargs,
            )
        top = offset_step * max(len(temperatures) - 1, 0) + min(max_display, 2.8) + 0.12
        for ax in (ax_raw, ax_clean):
            ax.set_ylim(-0.18, top)
        _configure_waterfall_row((ax_raw, ax_clean), row_index, len(FAMILIES))
        ax_raw.text(
            0.015,
            0.91,
            spec.label,
            transform=ax_raw.transAxes,
            color=spec.color,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0),
        )
        if row_index == 0:
            ax_raw.text(
                0.985,
                0.91,
                "Before spike removal",
                transform=ax_raw.transAxes,
                ha="right",
                va="top",
                fontsize=11,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0),
            )
            ax_clean.text(
                0.985,
                0.91,
                "After spike removal",
                transform=ax_clean.transAxes,
                ha="right",
                va="top",
                fontsize=11,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0),
            )
    cax = fig.add_subplot(grid[:, 2])
    add_temperature_colorbar(fig, cax)
    fig.text(0.017, 0.52, "Normalized intensity + offset", rotation=90, va="center", fontsize=13)
    return save_figure(fig, "Fig_S1_RAW_vs_Spike_Removed_All_Families.png")


def plot_als_baseline_removal_waterfall(
    record_map: Dict[str, List[spike_app.SpectrumRecord]],
) -> Path:
    fig = plt.figure(figsize=(12.5, 12.8))
    grid = GridSpec(
        len(FAMILIES),
        3,
        figure=fig,
        width_ratios=[1.0, 1.0, 0.035],
        left=0.075,
        right=0.96,
        bottom=0.065,
        top=0.99,
        hspace=0.08,
        wspace=0.06,
    )
    for row_index, spec in enumerate(FAMILIES):
        ax_before = fig.add_subplot(grid[row_index, 0])
        ax_after = fig.add_subplot(grid[row_index, 1], sharex=ax_before)
        grouped: Dict[float, List[Tuple[Path, Path]]] = defaultdict(list)
        for row in record_map[spec.key]:
            if row.clean_path is None or row.temperature is None:
                continue
            corrected = ALS_ROOT / spec.folder / spec.sequence / row.clean_path.name
            if corrected.exists():
                grouped[float(row.temperature)].append((row.clean_path, corrected))
        temperatures = sorted(grouped)
        if not temperatures:
            raise RuntimeError(f"No matched ALS files for {spec.label}")
        offset_step = 0.22
        for offset_index, temperature in enumerate(temperatures):
            x, before_y, corrected_y = average_paired_records(
                grouped[temperature], 250.0, 2000.0
            )
            als_baseline = before_y - corrected_y
            baseline_zero = float(np.nanpercentile(als_baseline, 5.0))
            corrected_zero = float(np.nanmedian(corrected_y[x >= 1800])) if np.any(x >= 1800) else 0.0
            corrected_centered = corrected_y - corrected_zero
            peak_scale = max(
                float(np.nanpercentile(corrected_centered, 99.5) - np.nanpercentile(corrected_centered, 5.0)),
                1e-12,
            )
            before_display = (before_y - baseline_zero) / peak_scale
            baseline_display = (als_baseline - baseline_zero) / peak_scale
            corrected_display = corrected_centered / peak_scale
            offset = offset_index * offset_step
            color = _temperature_color(temperature)
            kwargs = marker_kwargs(spec, color, size=2.4)
            ax_before.plot(
                x,
                before_display + offset,
                color=color,
                linewidth=0.78,
                markevery=14,
                zorder=3,
                **kwargs,
            )
            ax_before.plot(
                x,
                baseline_display + offset,
                color="#d62728",
                linewidth=0.75,
                alpha=0.48,
                zorder=2,
            )
            ax_after.plot(
                x,
                corrected_display + offset,
                color=color,
                linewidth=0.8,
                markevery=14,
                zorder=3,
                **kwargs,
            )
        top = offset_step * max(len(temperatures) - 1, 0) + 1.5
        for ax in (ax_before, ax_after):
            ax.set_ylim(-0.22, top)
        _configure_waterfall_row((ax_before, ax_after), row_index, len(FAMILIES))
        ax_before.text(
            0.015,
            0.91,
            spec.label,
            transform=ax_before.transAxes,
            color=spec.color,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0),
        )
        if row_index == 0:
            ax_before.text(
                0.985,
                0.91,
                "Spike removed + ALS baseline",
                transform=ax_before.transAxes,
                ha="right",
                va="top",
                fontsize=11,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0),
            )
            ax_after.text(
                0.985,
                0.91,
                "Baseline corrected",
                transform=ax_after.transAxes,
                ha="right",
                va="top",
                fontsize=11,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0),
            )
    cax = fig.add_subplot(grid[:, 2])
    add_temperature_colorbar(fig, cax)
    fig.text(0.017, 0.52, "Normalized intensity + offset", rotation=90, va="center", fontsize=13)
    return save_figure(fig, "Fig_S2_ALS_Baseline_Removal_All_Families.png")


def _example_axes_grid() -> Tuple[plt.Figure, List[Tuple[plt.Axes, plt.Axes]]]:
    fig, axes = plt.subplots(
        len(FAMILIES),
        2,
        figsize=(12.2, 12.2),
        sharex=True,
        gridspec_kw={
            "left": 0.075,
            "right": 0.985,
            "bottom": 0.065,
            "top": 0.985,
            "hspace": 0.11,
            "wspace": 0.08,
        },
    )
    rows: List[Tuple[plt.Axes, plt.Axes]] = []
    for row in range(len(FAMILIES)):
        left = axes[row, 0]
        right = axes[row, 1]
        rows.append((left, right))
        for ax in (left, right):
            style_axis(ax)
            ax.set_xlim(200.0, 2000.0)
            ax.set_yticks([])
            if row < len(FAMILIES) - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
                ax.set_xticks([400, 800, 1200, 1600, 2000])
    return fig, rows


def _plot_example_trace(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    spec: FamilySpec,
    label: Optional[str] = None,
    marker_size: float = 3.2,
    markevery: Optional[int] = 10,
) -> None:
    ax.plot(x, y, color="#444444", linewidth=1.1, zorder=2)
    kwargs = marker_kwargs(spec, spec.color, size=marker_size)
    ax.plot(
        x,
        y,
        linestyle="none",
        markevery=markevery,
        zorder=4,
        label=label,
        **kwargs,
    )


def _set_example_ylim(ax: plt.Axes, arrays: Iterable[np.ndarray]) -> None:
    finite_arrays = [np.asarray(values)[np.isfinite(values)] for values in arrays]
    finite_arrays = [values for values in finite_arrays if values.size]
    if not finite_arrays:
        return
    values = np.concatenate(finite_arrays)
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    span = max(high - low, abs(high), 1.0)
    ax.set_ylim(low - 0.07 * span, high + 0.10 * span)


def _label_example_family(ax: plt.Axes, spec: FamilySpec) -> None:
    ax.text(
        0.018,
        0.90,
        spec.label,
        transform=ax.transAxes,
        color=spec.color,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.1),
        zorder=10,
    )


def plot_raw_vs_spike_removed(
    record_map: Dict[str, List[spike_app.SpectrumRecord]],
    examples: Dict[str, ExampleSelection],
) -> Path:
    fig, axes = _example_axes_grid()
    for row_index, spec in enumerate(FAMILIES):
        selection = examples[spec.fit_family]
        x, raw_y, clean_y, record, curve_index, score = clearest_spike_pair(
            record_map[spec.key],
            selection.x_start,
            selection.x_end,
        )
        print(
            f"S1 {spec.label}: {record.temperature_label}, {record.clean_rel_dir}, "
            f"Y{curve_index + 1:02d}, spike score={score:.3g}",
            flush=True,
        )
        ax_raw, ax_clean = axes[row_index]
        baseline, scale = _display_scale(clean_y)
        raw_display = (raw_y - baseline) / scale
        clean_display = (clean_y - baseline) / scale
        _plot_example_trace(
            ax_raw,
            x,
            raw_display,
            spec,
            marker_size=3.2,
            markevery=1,
        )
        _plot_example_trace(
            ax_clean,
            x,
            clean_display,
            spec,
            marker_size=3.2,
            markevery=1,
        )

        combined = np.concatenate([raw_display, clean_display])
        combined = combined[np.isfinite(combined)]
        raw_finite = raw_display[np.isfinite(raw_display)]
        clean_finite = clean_display[np.isfinite(clean_display)]
        low = max(float(np.nanpercentile(combined, 0.5)) - 0.05, -0.25)
        raw_high = float(np.nanmax(raw_finite))
        clean_high = float(np.nanmax(clean_finite))
        raw_span = max(raw_high - low, 1e-12)
        clean_span = max(clean_high - low, 1e-12)
        raw_padding = 0.28 if row_index == 0 else 0.12
        ax_raw.set_ylim(low, raw_high + raw_padding * raw_span)
        ax_clean.set_ylim(low, clean_high + 0.12 * clean_span)
        _label_example_family(ax_raw, spec)
        if row_index == 0:
            ax_raw.text(
                0.98,
                0.90,
                "Before spike removal",
                transform=ax_raw.transAxes,
                ha="right",
                va="top",
                fontsize=11,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.1),
            )
            ax_clean.text(
                0.98,
                0.90,
                "After spike removal",
                transform=ax_clean.transAxes,
                ha="right",
                va="top",
                fontsize=11,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.1),
            )
    fig.text(
        0.017,
        0.52,
        "Intensity (a.u.)",
        rotation=90,
        va="center",
        fontsize=15,
    )
    return save_figure(fig, "Fig_S1_RAW_vs_Spike_Removed_All_Families.png")


def plot_als_baseline_removal(examples: Dict[str, ExampleSelection]) -> Path:
    fig, axes = _example_axes_grid()
    for row_index, spec in enumerate(FAMILIES):
        selection = examples[spec.fit_family]
        source = ROOT / spec.folder / selection.sequence / selection.clean_name
        corrected = ALS_ROOT / spec.folder / selection.sequence / selection.clean_name
        if not source.exists() or not corrected.exists():
            raise FileNotFoundError(f"Missing matched ALS example: {source} | {corrected}")
        x, source_y, corrected_y = selected_curve_pair(
            source,
            corrected,
            selection.curve_index,
            selection.x_start,
            selection.x_end,
        )
        baseline = source_y - corrected_y
        ax_before, ax_after = axes[row_index]
        # Show every measured spectral point in Figure S2.  The smaller marker
        # size keeps the 647-point traces legible without subsampling them.
        _plot_example_trace(
            ax_before,
            x,
            source_y,
            spec,
            label="Spike-removed spectrum",
            marker_size=1.8,
            markevery=1,
        )
        ax_before.plot(
            x,
            baseline,
            color="#df2b2f",
            linewidth=1.8,
            zorder=5,
            label="ALS baseline",
        )
        _plot_example_trace(
            ax_after,
            x,
            corrected_y,
            spec,
            label="Baseline-corrected spectrum",
            marker_size=1.8,
            markevery=1,
        )
        _set_example_ylim(ax_before, (source_y, baseline))
        _set_example_ylim(ax_after, (corrected_y,))
        _label_example_family(ax_before, spec)
        if row_index == 0:
            ax_before.legend(
                handles=[
                    Line2D(
                        [0],
                        [0],
                        color="#df2b2f",
                        linewidth=1.8,
                        label="ALS baseline",
                    )
                ],
                loc="upper left",
                bbox_to_anchor=(0.018, 0.81),
                frameon=False,
                handlelength=2.6,
                borderaxespad=0.0,
            )
            ax_before.text(
                0.98,
                0.90,
                "Spectrum and ALS baseline",
                transform=ax_before.transAxes,
                ha="right",
                va="top",
                fontsize=11,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.1),
            )
            ax_after.text(
                0.02,
                0.90,
                "Baseline corrected",
                transform=ax_after.transAxes,
                ha="left",
                va="top",
                fontsize=11,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.1),
            )
    fig.text(0.017, 0.52, "Intensity (a.u.)", rotation=90, va="center", fontsize=15)
    return save_figure(fig, "Fig_S2_ALS_Baseline_Removal_All_Families.png")


def lorentzian(x: np.ndarray, height: float, position: float, width: float) -> np.ndarray:
    width = max(float(width), 1e-12)
    return float(height) / (1.0 + ((x - float(position)) / (0.5 * width)) ** 2)


def load_fit_rows() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for csv_path in sorted(FITTED_ROOT.glob("*/*_long_results.csv")):
        frame = pd.read_csv(csv_path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No long-results CSV files under {FITTED_ROOT}")
    data = pd.concat(frames, ignore_index=True)
    for column in (
        "temperature_K",
        "height",
        "position_cm-1",
        "width_fwhm_cm-1",
        "r_squared",
        "rmse",
    ):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def spectrum_path_for_group(group: pd.DataFrame) -> Path:
    relative = Path(str(group.iloc[0]["file_path"]))
    candidates = (AVERAGED_TXT_ROOT / relative, SECOND_PASS_TXT_ROOT / relative)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find fitted spectrum in either read-only averaged TXT source: "
        + " | ".join(str(path) for path in candidates)
    )


def fit_sum_safe(x: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
    total = np.zeros_like(np.asarray(x, dtype=float))
    for _, row in rows.iterrows():
        total += lorentzian(
            x,
            float(row["height"]),
            float(row["position_cm-1"]),
            float(row["width_fwhm_cm-1"]),
        )
    return total


def fit_groups_for_family_sequence(
    data: pd.DataFrame,
    family: str,
    sequence: str,
) -> List[pd.DataFrame]:
    subset = data[
        (data["family"] == family) & (data["sequence"] == sequence)
    ].copy()
    if subset.empty:
        raise RuntimeError(f"No fitted rows for {family}/{sequence}")
    keys = ["temperature_K", "file_path", "spectrum_in_file", "y_column_number"]
    groups = [group.copy() for _, group in subset.groupby(keys, sort=True, dropna=False)]
    groups.sort(key=lambda group: float(group.iloc[0]["temperature_K"]))
    return groups


def selected_fit_groups(data: pd.DataFrame, spec: FamilySpec) -> List[pd.DataFrame]:
    return fit_groups_for_family_sequence(data, spec.fit_family, spec.sequence)


def example_fit_group(
    data: pd.DataFrame,
    selection: ExampleSelection,
) -> pd.DataFrame:
    groups = fit_groups_for_family_sequence(data, selection.family, selection.sequence)
    return min(
        groups,
        key=lambda group: abs(float(group.iloc[0]["temperature_K"]) - selection.temperature),
    )


def load_group_spectrum(group: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    path = spectrum_path_for_group(group)
    data = spike_app.read_spectrum_file(path)
    column_number = int(float(group.iloc[0].get("y_column_number", 2)))
    curve_index = max(column_number - 2, 0)
    curve_index = min(curve_index, len(data.curves) - 1)
    curve = data.curves[curve_index]
    return np.asarray(curve.x, dtype=float), np.asarray(curve.y, dtype=float)


def _broken_axis_marks(ax: plt.Axes, side: str, size: float = 0.018) -> None:
    kwargs = dict(transform=ax.transAxes, color="black", clip_on=False, linewidth=1.0)
    if side == "left":
        ax.plot((-size, +size), (-size, +size), **kwargs)
        ax.plot((-size, +size), (1 - size, 1 + size), **kwargs)
    else:
        ax.plot((1 - size, 1 + size), (-size, +size), **kwargs)
        ax.plot((1 - size, 1 + size), (1 - size, 1 + size), **kwargs)


def plot_global_lorentzian_fits_waterfall(fit_rows: pd.DataFrame) -> Path:
    windows = ((280.0, 340.0), (1180.0, 1370.0), (1545.0, 1630.0))
    width_ratios = [0.78, 2.45, 1.05, 0.04]
    fig = plt.figure(figsize=(13.0, 13.2))
    grid = GridSpec(
        len(FAMILIES),
        4,
        figure=fig,
        width_ratios=width_ratios,
        left=0.07,
        right=0.965,
        bottom=0.065,
        top=0.955,
        hspace=0.08,
        wspace=0.055,
    )
    for row_index, spec in enumerate(FAMILIES):
        row_axes = [fig.add_subplot(grid[row_index, col]) for col in range(3)]
        groups = selected_fit_groups(fit_rows, spec)
        offset_step = 0.22
        for offset_index, group in enumerate(groups):
            temperature = float(group.iloc[0]["temperature_K"])
            x_data, y_data = load_group_spectrum(group)
            fit_on_data = fit_sum_safe(x_data, group)
            union = np.zeros_like(x_data, dtype=bool)
            for low, high in windows:
                union |= (x_data >= low) & (x_data <= high)
            scale = max(float(np.nanmax(fit_on_data[union])), 1e-12)
            baseline = float(np.nanmedian(y_data[(x_data >= 1750) & (x_data <= 1950)])) if np.any((x_data >= 1750) & (x_data <= 1950)) else 0.0
            data_norm = (y_data - baseline) / scale
            offset = offset_index * offset_step
            color = _temperature_color(temperature)
            for ax, (low, high) in zip(row_axes, windows):
                mask = (x_data >= low) & (x_data <= high)
                if not np.any(mask):
                    continue
                kwargs = marker_kwargs(spec, color, size=2.7)
                ax.plot(
                    x_data[mask],
                    data_norm[mask] + offset,
                    color=color,
                    linewidth=0.7,
                    markevery=2,
                    zorder=3,
                    **kwargs,
                )
                x_dense = np.linspace(low, high, 500)
                fit_dense = fit_sum_safe(x_dense, group) / scale + offset
                ax.plot(x_dense, fit_dense, color="#222222", linewidth=1.0, zorder=4)
        top = offset_step * max(len(groups) - 1, 0) + 1.35
        for col_index, (ax, limits) in enumerate(zip(row_axes, windows)):
            style_axis(ax)
            ax.set_xlim(*limits)
            ax.set_ylim(-0.18, top)
            ax.set_yticks([])
            if row_index < len(FAMILIES) - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
            if col_index > 0:
                ax.spines["left"].set_visible(False)
            if col_index < 2:
                ax.spines["right"].set_visible(False)
            if col_index == 0:
                _broken_axis_marks(ax, "right")
            elif col_index == 1:
                _broken_axis_marks(ax, "left")
                _broken_axis_marks(ax, "right")
            else:
                _broken_axis_marks(ax, "left")
        row_axes[0].text(
            0.02,
            0.90,
            spec.label,
            transform=row_axes[0].transAxes,
            color=spec.color,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0),
        )
    fig.legend(
        handles=[
            Line2D([0], [0], color="#555555", marker="o", markersize=4, linewidth=0.7, label="Baseline-corrected spectrum"),
            Line2D([0], [0], color="#222222", linewidth=1.2, label="Total Lorentzian fit"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.52, 0.995),
        frameon=False,
        ncol=2,
        handlelength=2.6,
        columnspacing=1.8,
    )
    cax = fig.add_subplot(grid[:, 3])
    add_temperature_colorbar(fig, cax)
    fig.text(0.016, 0.52, "Normalized intensity + offset", rotation=90, va="center", fontsize=13)
    return save_figure(fig, "Fig_S4_Global_Lorentzian_Fits_All_Families.png")


COMPONENT_STYLES: Dict[str, Tuple[str, str]] = {
    "RBLM": ("#2c7fb8", "RBLM"),
    "CH_L1": ("#31a354", "CH L1"),
    "CH_L2": ("#8c6bb1", "CH L2"),
    "CH_MID": ("#de77ae", "CH middle"),
    "D": ("#b5bd00", "D"),
    "G": ("#12b8c4", "G"),
}


def plot_global_lorentzian_fits(
    fit_rows: pd.DataFrame,
    examples: Dict[str, ExampleSelection],
) -> Path:
    x_min, x_max = 200.0, 2000.0
    fig, axes = plt.subplots(
        len(FAMILIES),
        1,
        figsize=(12.7, 11.8),
        sharex=True,
        gridspec_kw={
            "left": 0.07,
            "right": 0.99,
            "bottom": 0.07,
            "top": 0.985,
            "hspace": 0.10,
        },
    )

    for row_index, spec in enumerate(FAMILIES):
        selection = examples[spec.fit_family]
        group = example_fit_group(fit_rows, selection)
        x_data, y_data = load_group_spectrum(group)
        total_on_data = fit_sum_safe(x_data, group)
        ax = axes[row_index]
        visible = (x_data >= x_min) & (x_data <= x_max)
        visible_values = np.concatenate([y_data[visible], total_on_data[visible], [0.0]])
        visible_values = visible_values[np.isfinite(visible_values)]
        low_y = float(np.nanmin(visible_values)) if visible_values.size else 0.0
        high_y = float(np.nanmax(visible_values)) if visible_values.size else 1.0
        span = max(high_y - low_y, abs(high_y), 1.0)
        y_limits = (low_y - 0.06 * span, high_y + 0.12 * span)

        style_axis(ax)
        mask = (x_data >= x_min) & (x_data <= x_max)
        kwargs = marker_kwargs(spec, spec.color, size=4.8)
        ax.plot(x_data[mask], y_data[mask], color="#4b4b4b", linewidth=0.95, zorder=2)
        ax.plot(
            x_data[mask],
            y_data[mask],
            linestyle="none",
            markevery=1,
            zorder=5,
            **kwargs,
        )
        x_dense = np.linspace(x_min, x_max, 4000)
        ax.plot(
            x_dense,
            fit_sum_safe(x_dense, group),
            color="#e52b2f",
            linewidth=2.25,
            zorder=6,
        )
        for peak_id, (color, _) in COMPONENT_STYLES.items():
            peak_rows = group[group["peak_id"].astype(str).str.upper() == peak_id]
            if peak_rows.empty:
                continue
            peak = peak_rows.iloc[0]
            component = lorentzian(
                x_dense,
                float(peak["height"]),
                float(peak["position_cm-1"]),
                float(peak["width_fwhm_cm-1"]),
            )
            ax.plot(
                x_dense,
                component,
                color=color,
                linewidth=1.65,
                linestyle="--",
                zorder=9,
            )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(*y_limits)
        ax.set_yticks([])
        ax.set_xticks([400, 800, 1200, 1600, 2000])
        if row_index < len(FAMILIES) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
        _label_example_family(ax, spec)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#e52b2f",
            linewidth=2.25,
            linestyle="-",
            label="Total Lorentzian fit",
        )
    ] + [
        Line2D(
            [0],
            [0],
            color=COMPONENT_STYLES[peak_id][0],
            linewidth=1.65,
            linestyle="--",
            label=COMPONENT_STYLES[peak_id][1],
        )
        for peak_id in ("RBLM", "D", "G")
    ]
    axes[0].legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        ncol=1,
        handlelength=2.5,
        borderaxespad=0.8,
    )
    fig.text(0.017, 0.49, "Intensity (a.u.)", rotation=90, va="center", fontsize=13)
    return save_figure(fig, "Fig_S4_Global_Lorentzian_Fits_All_Families.png")


def nearest_fit_group(
    fit_rows: pd.DataFrame,
    spec: FamilySpec,
    temperature: float,
) -> pd.DataFrame:
    groups = selected_fit_groups(fit_rows, spec)
    return min(groups, key=lambda group: abs(float(group.iloc[0]["temperature_K"]) - temperature))


def g_peak_row(group: pd.DataFrame) -> pd.Series:
    match = group[group["peak_id"].astype(str).str.upper() == "G"]
    if match.empty:
        raise RuntimeError("Selected fit does not contain a G peak")
    return match.iloc[0]


def plot_g_peak_metrics(fit_rows: pd.DataFrame) -> Path:
    spec = FAMILIES[0]
    group = nearest_fit_group(fit_rows, spec, 100.0)
    g_row = g_peak_row(group)
    x_data, y_data = load_group_spectrum(group)
    x_min, x_max = 1548.0, 1648.0
    mask = (x_data >= x_min) & (x_data <= x_max)
    x = x_data[mask]
    y = y_data[mask]
    height = float(g_row["height"])
    position = float(g_row["position_cm-1"])
    width = float(g_row["width_fwhm_cm-1"])
    x_dense = np.linspace(x_min, x_max, 1200)
    g_dense = lorentzian(x_dense, height, position, width)
    half_height = height / 2.0
    left_half = position - width / 2.0
    right_half = position + width / 2.0

    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    style_axis(ax)
    ax.plot(x, y, color="#444444", linewidth=1.1, zorder=2)
    point_kwargs = marker_kwargs(spec, spec.color, size=6.0)
    ax.plot(x, y, linestyle="none", zorder=4, label=spec.label, **point_kwargs)
    ax.plot(x_dense, g_dense, color="#ef2b2d", linewidth=2.4, zorder=5, label="Lorentzian fit")
    ax.fill_between(
        [left_half, right_half],
        [half_height * 0.975, half_height * 0.975],
        [half_height * 1.025, half_height * 1.025],
        color="#1f4e79",
        alpha=0.16,
        zorder=5,
    )
    ax.annotate(
        "",
        xy=(right_half, half_height),
        xytext=(left_half, half_height),
        arrowprops=dict(arrowstyle="<->", color="#1f4e79", linewidth=3.3),
        zorder=7,
    )
    ax.plot([], [], color="#1f4e79", linewidth=3.3, label="FWHM")
    ax.vlines([left_half, right_half], 0.0, half_height, color="#1f4e79", linewidth=1.8, linestyle="--", zorder=5)
    ax.vlines(position, 0.0, height, color="#2e7d32", linewidth=1.8, linestyle="--", zorder=6, label="Height")
    ax.plot(position, 0.0, marker="x", linestyle="none", markersize=10, markeredgewidth=2.2, color="#d7191c", zorder=7, label="Peak position")
    ax.text(
        0.58,
        0.95,
        rf"Peak position = {position:.2f} cm$^{{-1}}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=14,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.94, pad=1.2),
        zorder=10,
    )
    ax.text(
        right_half + 4.5,
        half_height * 1.12,
        rf"FWHM = {width:.2f} cm$^{{-1}}$",
        ha="left",
        va="center",
        color="#1f4e79",
        fontsize=16,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.94, pad=1.3),
        zorder=10,
    )
    ax.set_xlim(x_min, x_max)
    lower = min(0.0, float(np.nanmin(y)))
    upper = max(float(np.nanmax(y)), height) * 1.16
    ax.set_ylim(lower - 0.02 * upper, upper)
    ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_xticks([1560, 1580, 1600, 1620, 1640])
    ax.legend(loc="upper right", frameon=False, handlelength=2.5)
    fig.tight_layout()
    return save_figure(fig, "Fig_S5_G_Peak_Lorentzian_Parameter_Extraction.png")


def normalized_peak_panel(
    group: pd.DataFrame,
    spec: FamilySpec,
    ax_main: plt.Axes,
    ax_residual: plt.Axes,
    panel_label: str,
) -> None:
    x_data, y_data = load_group_spectrum(group)
    x_min, x_max = 1550.0, 1650.0
    mask = (x_data >= x_min) & (x_data <= x_max)
    x = x_data[mask]
    y = y_data[mask]
    total = fit_sum_safe(x, group)
    g = g_peak_row(group)
    g_curve = lorentzian(x, float(g["height"]), float(g["position_cm-1"]), float(g["width_fwhm_cm-1"]))
    scale = max(float(np.nanmax(total)), 1e-12)
    y_norm = y / scale
    total_norm = total / scale
    g_norm = g_curve / scale
    residual = y_norm - total_norm
    ss_res = float(np.nansum(residual**2))
    ss_tot = float(np.nansum((y_norm - np.nanmean(y_norm)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.nanmean(residual**2)))
    temperature = float(group.iloc[0]["temperature_K"])

    style_axis(ax_main)
    style_axis(ax_residual)
    ax_main.plot(x, y_norm, color="#3f3f3f", linewidth=1.0, zorder=2)
    point_kwargs = marker_kwargs(spec, spec.color, size=5.0)
    legend_family_label = spec.label.replace(", ", ",\n", 1)
    ax_main.plot(x, y_norm, linestyle="none", zorder=4, label=legend_family_label, **point_kwargs)
    x_dense = np.linspace(x_min, x_max, 1000)
    total_dense = fit_sum_safe(x_dense, group) / scale
    g_dense = lorentzian(x_dense, float(g["height"]), float(g["position_cm-1"]), float(g["width_fwhm_cm-1"])) / scale
    ax_main.plot(x_dense, total_dense, color="#e52b2f", linewidth=2.0, zorder=5, label="Lorentzian fit")
    ax_main.plot(x_dense, g_dense, color="#10b9c5", linewidth=1.6, linestyle="--", zorder=6, label="G")
    ax_main.text(
        0.015,
        0.95,
        panel_label,
        transform=ax_main.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        zorder=10,
    )
    ax_main.text(
        0.975,
        0.95,
        f"{temperature:g} K\n"
        rf"$R^2$ = {r_squared:.5f}"
        f"\nRMSE = {rmse:.4f}",
        transform=ax_main.transAxes,
        ha="right",
        va="top",
        fontsize=12.5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=1.4),
        zorder=10,
    )
    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(-0.06, 1.08)
    ax_main.set_xticks([1560, 1580, 1600, 1620, 1640])
    ax_main.tick_params(labelbottom=False)

    ax_residual.axhline(0.0, color="#888888", linewidth=0.9, linestyle="--", zorder=1)
    ax_residual.plot(x, residual, color=spec.color, linewidth=1.1, marker=spec.marker, markersize=2.7, markerfacecolor="none" if spec.hollow else spec.color, markeredgecolor=spec.color, markeredgewidth=0.6, zorder=3)
    ax_residual.set_xlim(x_min, x_max)
    ax_residual.set_ylim(-0.12, 0.12)
    ax_residual.set_yticks([-0.1, 0.0, 0.1])
    ax_residual.set_xticks([1560, 1580, 1600, 1620, 1640])


def plot_four_panel_g_residuals(fit_rows: pd.DataFrame) -> Path:
    spec_3a = FAMILIES[0]
    spec_8a = FAMILIES[1]
    selections = (
        (spec_3a, 80.0, "a)"),
        (spec_8a, 80.0, "b)"),
        (spec_3a, 300.0, "c)"),
        (spec_8a, 300.0, "d)"),
    )
    fig = plt.figure(figsize=(12.2, 10.5))
    outer = GridSpec(
        2,
        2,
        figure=fig,
        left=0.09,
        right=0.985,
        bottom=0.075,
        top=0.985,
        hspace=0.16,
        wspace=0.18,
    )
    for index, (spec, temperature, panel_label) in enumerate(selections):
        inner = outer[index // 2, index % 2].subgridspec(2, 1, height_ratios=[3.2, 1.2], hspace=0.04)
        ax_main = fig.add_subplot(inner[0, 0])
        ax_residual = fig.add_subplot(inner[1, 0], sharex=ax_main)
        normalized_peak_panel(
            nearest_fit_group(fit_rows, spec, temperature),
            spec,
            ax_main,
            ax_residual,
            panel_label,
        )
        if index in (0, 1):
            ax_main.legend(
                loc="upper left",
                bbox_to_anchor=(0.025, 0.84),
                ncol=1,
                frameon=False,
                fontsize=14.5,
                handlelength=2.0,
                handletextpad=0.55,
                borderaxespad=0.0,
                labelspacing=0.22,
            )
        if index % 2 == 0:
            ax_main.set_ylabel("Normalized intensity")
            ax_residual.set_ylabel("Residual")
        else:
            ax_main.tick_params(labelleft=False)
            ax_residual.tick_params(labelleft=False)
        if index // 2 == 1:
            ax_residual.set_xlabel(r"Raman shift (cm$^{-1}$)")
        else:
            ax_residual.tick_params(labelbottom=False)
    return save_figure(fig, "Fig_S7_G_Fit_Residuals_Aligned_Au_80K_300K.png")


def copy_approved_figures() -> List[Path]:
    sources = (
        (
            AFTER_DIR
            / "FIGURE_3_PAPER"
            / "FIG3_AlignedAu8A_NestedSequences_Combination1_AbsFreq_AFTER.png",
            "Fig_S8_Aligned_Au_High_Coverage_Thermal_Cycle_Shifts.png",
        ),
        (
            AFTER_DIR
            / "WIDHT_FIGURES"
            / "WIDTH_after_all_families_UP1_3phonon_quadratic.png",
            "Fig_S9_Linewidth_Trends_3Phonon_All_Configurations.png",
        ),
    )
    copied: List[Path] = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for source, output_name in sources:
        if not source.exists():
            raise FileNotFoundError(source)
        destination = OUTPUT_DIR / output_name
        shutil.copy2(source, destination)
        copied.append(destination)
        print(f"Copied {source.name} -> {destination}", flush=True)
    return copied


def main() -> None:
    setup_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Final SI output folder: {OUTPUT_DIR}", flush=True)
    examples = load_example_selections()
    print("Loaded five exact representative selections from the ALS example manifests.", flush=True)
    print("Scanning read-only paired RAW/spike inputs...", flush=True)
    record_map = scan_original_records()
    plot_raw_vs_spike_removed(record_map, examples)
    plot_als_baseline_removal(examples)

    print("Loading final curated AFTER Lorentzian results...", flush=True)
    fit_rows = load_fit_rows()
    plot_global_lorentzian_fits(fit_rows, examples)
    plot_g_peak_metrics(fit_rows)
    plot_four_panel_g_residuals(fit_rows)
    copy_approved_figures()
    print("S5 is intentionally absent: the TEC figure is user supplied.", flush=True)
    print("Final SI figure package complete.", flush=True)


if __name__ == "__main__":
    main()
