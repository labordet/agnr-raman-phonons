#!/usr/bin/env python3
"""
Figure 2 paper waterfall from the final cleaned averaged spectra.

Reads the paper package written by Gamma_T_Comparison.py:
    GAMMA_T_COMPARISON/AFTER/AVERAGED_BASELINE_CORRECTED_TXT_FOR_CLUSTER

The input TXT files are the cleaned, spike-removed, baseline-corrected,
averaged spectra that passed the manual exam. Each TXT has column 1 = Raman
shift and column 2 = intensity. For Figure 2, sequence filters are applied
explicitly so the paper rows use the intended measurement branch.

Outputs paper-style broken-axis waterfall figures to:
    GAMMA_T_COMPARISON/AFTER/FIGURE_2_PAPER
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, StrMethodFormatter


APP_DIR = Path(__file__).resolve().parent
DEFAULT_AFTER_DIR = APP_DIR / "GAMMA_T_COMPARISON" / "AFTER"
DEFAULT_TXT_ROOT = DEFAULT_AFTER_DIR / "AVERAGED_BASELINE_CORRECTED_TXT_FOR_CLUSTER"
DEFAULT_OUT_DIR = DEFAULT_AFTER_DIR / "FIGURE_2_PAPER"
OUTPUT_STEM = "FIG2_waterfall_all"

FAMILY_ROWS = [
    ("Aligned_Au_3A", "Aligned Au 3A"),
    ("Aligned_Au_8A", "Aligned Au 8A"),
    ("MIRA_Au_unaligned_8A", "Unaligned Au 8A"),
    ("MIRA_RO_unaligned_8A", "Unaligned RO 8A"),
    ("Aligned_RO_8A", "Aligned RO 8A"),
]

SEQUENCE_FILTERS = {
    "Aligned_Au_3A": {"Spikes_Removed"},
    "Aligned_Au_8A": {"Spikes_Removed_UP_1"},
    "MIRA_Au_unaligned_8A": {"Spikes_Removed_UP_1"},
    "MIRA_RO_unaligned_8A": {"Spikes_Removed_UP_1"},
    "Aligned_RO_8A": {"Spikes_Removed_DOWN_1"},
}

ROMAN_NUMERALS = ("I", "II", "III", "IV", "V")

SAMPLE_COLORS = {
    "Aligned_Au_3A": "#feb715",
    "Aligned_Au_8A": "#f08228",
    "Aligned_RO_8A": "#00c8c8",
    "MIRA_Au_unaligned_8A": "#ae540b",
    "MIRA_RO_unaligned_8A": "#008686",
}

WINDOWS = [
    (270.0, 350.0),
    (1180.0, 1370.0),
    (1540.0, 1630.0),
]

XTICK_WINDOWS = [
    (3, 3),
    (7, 7),
    (4, 4),
]

STYLE = {
    "font_family": "Arial",
    "base": 24,
    "title": 24,
    "axis_label": 28,
    "tick_x": 24,
    "tick_y": 22,
    "legend": 16,
    "cbar_label": 28,
    "fig_width": 12.5,
    "row_height": 5.5,
    "dpi": 600,
    "wspace": 0.004,
    "hspace": 0.03,
    "facecolor": "white",
    "labelpad": 6,
    "save_pad_inches": 0.24,
    "left_xlabel_x": 0.58,
}

CBAR = {
    "label": "Temperature (K)",
    "tick_size": 22,
    "ticks": None,
}

TICKS = {
    "x_nbins": 7,
    "y_nbins": 6,
    "x_fmt": "{x:.0f}",
    "y_fmt": "{x:.2f}",
    "x_rotation": 30,
    "tick_length": 4,
    "tick_width": 0.8,
    "tick_direction": "out",
}

CMAP_NAME = "turbo"
TMIN_TARGET = 70.0
TMAX_TARGET = 300.0
CAP_MAX_K = 300.0
OFFSET = 0.05
LINE_WIDTH = 1.15
MARKER_SIZE = 7.0
MARKER_ALPHA = 0.95
POINT_STEP = 1
CUT_SIZE = 0.015
CBAR_REL_WIDTH = 0.03
SHOW_Y_TICK_LABELS = False

# The old Figure 2 paper-style code assumed normalized cleaned spectra. The
# second-pass TXT files are baseline-corrected intensities, so normalize each
# spectrum here to recover the same visual waterfall style.
NORMALIZATION = "each_spectrum"

YLIM_PAD = {"top_frac": 0.03, "bottom_frac": 0.02}
XPAD = {"left_frac": 0.0, "right_frac": 0.0}

TEMP_RE = re.compile(r"(\d+(?:\.\d+)?)\s*K", re.IGNORECASE)


def sample_marker_style(family: str) -> Dict[str, object]:
    is_unaligned = family in {"MIRA_Au_unaligned_8A", "MIRA_RO_unaligned_8A"}
    marker = "o" if family == "Aligned_Au_3A" else "D"
    return {
        "marker": marker,
        "hollow": is_unaligned,
        "edgecolor": SAMPLE_COLORS.get(family, "black"),
        "linewidth": 1.6 if is_unaligned else 0.0,
    }


def apply_fonts() -> None:
    want = STYLE["font_family"]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [want, "DejaVu Sans", "Liberation Sans", "Nimbus Sans L", "Arial"],
            "axes.titlesize": STYLE["title"],
            "axes.labelsize": STYLE["axis_label"],
            "axes.labelpad": STYLE["labelpad"],
            "xtick.labelsize": STYLE["tick_x"],
            "ytick.labelsize": STYLE["tick_y"],
            "legend.fontsize": STYLE["legend"],
            "figure.facecolor": STYLE["facecolor"],
            "savefig.facecolor": STYLE["facecolor"],
            "text.antialiased": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "rm",
        }
    )


def make_formatter(fmt: str):
    return StrMethodFormatter(fmt) if ("{" in fmt and "}" in fmt) else FormatStrFormatter(fmt)


def style_axis(ax, first: bool) -> None:
    ax.grid(False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=TICKS["x_nbins"]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=TICKS["y_nbins"]))
    ax.xaxis.set_major_formatter(make_formatter(TICKS["x_fmt"]))
    ax.yaxis.set_major_formatter(make_formatter(TICKS["y_fmt"]))
    ax.tick_params(
        which="both",
        direction=TICKS["tick_direction"],
        length=TICKS["tick_length"],
        width=TICKS["tick_width"],
        top=False,
        right=False,
    )
    for label in ax.get_xticklabels():
        label.set_rotation(TICKS["x_rotation"])
    if (not first) or (not SHOW_Y_TICK_LABELS):
        ax.tick_params(labelleft=False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def nice_xticks(lo: float, hi: float, n_target: int, n_max: int) -> Optional[np.ndarray]:
    if hi <= lo:
        return None
    candidates = [2, 5, 10, 20, 25, 40, 50, 100]
    best_ticks = None
    best_penalty = float("inf")
    for step in candidates:
        start = math.ceil(lo / step) * step
        end = math.floor(hi / step) * step
        if start > end:
            continue
        ticks = np.arange(start, end + 0.5 * step, step, dtype=float)
        n_ticks = len(ticks)
        if n_ticks < 2:
            continue
        penalty = abs(n_ticks - n_target)
        if n_ticks > n_max:
            penalty += 10 * (n_ticks - n_max)
        if penalty < best_penalty:
            best_penalty = penalty
            best_ticks = ticks
    return best_ticks


def resolve_cbar_ticks(norm, base_ticks=None) -> np.ndarray:
    vmin, vmax = norm.vmin, norm.vmax
    ticks = np.linspace(vmin, vmax, 6) if base_ticks is None else np.asarray(base_ticks, dtype=float)
    if not np.any(np.isclose(ticks, vmin)):
        ticks = np.r_[vmin, ticks]
    if not np.any(np.isclose(ticks, vmax)):
        ticks = np.r_[ticks, vmax]
    ticks = ticks[(ticks >= vmin - 1e-9) & (ticks <= vmax + 1e-9)]
    return np.unique(np.round(ticks, 10))


def read_txt_matrix(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", engine="python", header=None)
    if df.shape[1] < 2:
        raise ValueError(f"{path} does not have at least X and Y columns.")
    x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 2:
        raise ValueError(f"{path} has fewer than two finite XY points.")
    return x[finite], y[finite]


def interpolate_to_reference(x_source: np.ndarray, y_source: np.ndarray, x_reference: np.ndarray) -> np.ndarray:
    order = np.argsort(x_source)
    x_sorted = x_source[order]
    y_sorted = y_source[order]
    unique_x, unique_idx = np.unique(x_sorted, return_index=True)
    unique_y = y_sorted[unique_idx]
    y_interp = np.interp(x_reference, unique_x, unique_y)
    outside = (x_reference < unique_x[0]) | (x_reference > unique_x[-1])
    y_interp[outside] = np.nan
    return y_interp


def load_manifest(txt_root: Path) -> pd.DataFrame:
    manifest_path = txt_root / "second_pass_average_files.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing averaged-TXT manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    required = {"family", "sequence", "temperature_K", "relative_output_txt", "status"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"{manifest_path} is missing columns: {missing}")
    manifest = manifest[manifest["status"].astype(str).eq("WROTE_AVERAGED_TXT")].copy()
    if manifest.empty:
        raise ValueError(f"No WROTE_AVERAGED_TXT rows in {manifest_path}")
    manifest["temperature_K"] = pd.to_numeric(manifest["temperature_K"], errors="coerce")
    manifest = manifest[np.isfinite(manifest["temperature_K"])].copy()
    manifest = manifest[manifest["temperature_K"] <= CAP_MAX_K].copy()
    return manifest


def normalize_family_matrix(y_matrix: np.ndarray, segments: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    if NORMALIZATION == "none":
        return y_matrix, segments

    if NORMALIZATION == "each_spectrum":
        scale = np.nanmax(np.abs(y_matrix), axis=1)
        scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
        y_norm = y_matrix / scale[:, None]
    elif NORMALIZATION == "row_max":
        scale_value = float(np.nanmax(np.abs(y_matrix)))
        if not np.isfinite(scale_value) or scale_value <= 0:
            scale_value = 1.0
        y_norm = y_matrix / scale_value
    else:
        raise ValueError(f"Unknown NORMALIZATION mode: {NORMALIZATION}")

    normalized_segments = []
    for x_seg, y_seg in segments:
        if NORMALIZATION == "each_spectrum":
            normalized_segments.append((x_seg, y_seg / scale[:, None]))
        else:
            normalized_segments.append((x_seg, y_seg / scale_value))
    return y_norm, normalized_segments


def load_family_spectra(txt_root: Path, manifest: pd.DataFrame, family: str) -> Dict[str, object]:
    family_rows = manifest[manifest["family"].astype(str).eq(family)].copy()
    allowed_sequences = SEQUENCE_FILTERS.get(family)
    if allowed_sequences:
        family_rows = family_rows[family_rows["sequence"].astype(str).isin(allowed_sequences)].copy()
    family_rows = family_rows[
        (family_rows["temperature_K"] >= TMIN_TARGET)
        & (family_rows["temperature_K"] <= TMAX_TARGET)
    ].copy()
    if family_rows.empty:
        raise ValueError(f"No cleaned averaged TXT spectra for {family} in [{TMIN_TARGET}, {TMAX_TARGET}] K.")

    reference_x: Optional[np.ndarray] = None
    spectra_by_temp: Dict[float, List[np.ndarray]] = {}
    source_count_by_temp: Dict[float, int] = {}
    source_paths_by_temp: Dict[float, List[Path]] = {}

    for _idx, row in family_rows.sort_values(["temperature_K", "sequence", "relative_output_txt"]).iterrows():
        path = txt_root / str(row["relative_output_txt"])
        if not path.exists():
            # Windows paths in CSV may contain backslashes; Path handles them on Windows,
            # but this fallback keeps the script usable on POSIX too.
            path = txt_root / Path(str(row["relative_output_txt"].replace("\\", "/")))
        x, y = read_txt_matrix(path)
        if reference_x is None:
            reference_x = x.copy()
            y_ref = y.copy()
        elif len(x) == len(reference_x) and np.allclose(x, reference_x, rtol=0.0, atol=1e-8):
            y_ref = y.copy()
        else:
            y_ref = interpolate_to_reference(x, y, reference_x)
        temp = float(row["temperature_K"])
        spectra_by_temp.setdefault(temp, []).append(y_ref)
        source_count_by_temp[temp] = source_count_by_temp.get(temp, 0) + 1
        source_paths_by_temp.setdefault(temp, []).append(path.resolve())

    if reference_x is None:
        raise ValueError(f"No readable spectra for {family}.")

    temps = np.asarray(sorted(spectra_by_temp), dtype=float)
    y_rows = []
    for temp in temps:
        stack = np.vstack(spectra_by_temp[temp])
        y_rows.append(np.nanmedian(stack, axis=0))
    y_matrix = np.vstack(y_rows)

    segments: List[Tuple[np.ndarray, np.ndarray]] = []
    for lo, hi in WINDOWS:
        mask = (reference_x >= lo) & (reference_x < hi)
        if not np.any(mask):
            raise ValueError(f"{family}: no Raman-shift points in {lo:g}-{hi:g} cm^-1")
        segments.append((reference_x[mask], y_matrix[:, mask]))

    y_matrix, segments = normalize_family_matrix(y_matrix, segments)
    stack_indices, missing_slots, n_slots = build_stack_layout(temps, TMIN_TARGET, TMAX_TARGET)

    return {
        "family": family,
        "temps": temps,
        "segments": segments,
        "stack_indices": stack_indices,
        "missing_slots": missing_slots,
        "n_slots": n_slots,
        "source_count_by_temp": source_count_by_temp,
        "source_paths_by_temp": source_paths_by_temp,
    }


def build_stack_layout(
    temps: np.ndarray,
    tmin: float,
    tmax: float,
) -> Tuple[np.ndarray, List[Tuple[float, int]], int]:
    """Reserve absent display endpoints without inventing spectral rows."""
    stack_indices = np.arange(len(temps), dtype=int)
    missing_slots: List[Tuple[float, int]] = []

    if len(temps) and temps[0] > tmin:
        stack_indices += 1
        missing_slots.append((float(tmin), 0))

    n_slots = len(temps) + len(missing_slots)
    if len(temps) and temps[-1] < tmax:
        missing_slots.append((float(tmax), n_slots))
        n_slots += 1

    return stack_indices, missing_slots, n_slots


def row_y_limits(
    segments: Sequence[Tuple[np.ndarray, np.ndarray]],
    stack_indices: np.ndarray,
    missing_slots: Sequence[Tuple[float, int]],
) -> Tuple[float, float]:
    y_min = float("inf")
    y_max = -float("inf")
    offsets = np.asarray(stack_indices, dtype=float)[:, None] * OFFSET
    for _x_seg, y_seg in segments:
        y_plot = y_seg + offsets
        y_min = min(y_min, float(np.nanmin(y_plot)))
        y_max = max(y_max, float(np.nanmax(y_plot)))
    for _temp, slot_idx in missing_slots:
        guide_y = float(slot_idx) * OFFSET
        y_min = min(y_min, guide_y)
        y_max = max(y_max, guide_y)
    span = max(y_max - y_min, 1e-12)
    return y_min - YLIM_PAD["bottom_frac"] * span, y_max + YLIM_PAD["top_frac"] * span


def format_temperature(temperature: float) -> str:
    return f"{temperature:g}"


def print_temperature_audit(rows: Sequence[Dict[str, object]], txt_root: Path) -> None:
    print(f"Source manifest: {(txt_root / 'second_pass_average_files.csv').resolve()}")
    for roman, row in zip(ROMAN_NUMERALS, rows):
        measured = ", ".join(format_temperature(float(temp)) for temp in row["temps"])
        missing = ", ".join(format_temperature(float(temp)) for temp, _slot in row["missing_slots"])
        missing_text = f"{missing} K" if missing else "none"
        print(f"({roman}) {row['label']}: measured {measured} K; empty endpoint slots {missing_text}")


def plot_figure(txt_root: Path, out_dir: Path, formats: Iterable[str], show: bool = False) -> List[Path]:
    apply_fonts()
    manifest = load_manifest(txt_root)

    rows = []
    for family, label in FAMILY_ROWS:
        row = load_family_spectra(txt_root, manifest, family)
        row["label"] = label
        rows.append(row)
    print_temperature_audit(rows, txt_root)

    n_rows = len(rows)
    n_seg = len(WINDOWS)
    widths_ref = [row["segments"][idx][0].size for idx in range(n_seg)]

    vmin = TMIN_TARGET
    vmax = TMAX_TARGET
    cmap = colormaps[CMAP_NAME]
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar_ticks = resolve_cbar_ticks(norm, CBAR["ticks"])

    fig = plt.figure(
        figsize=(STYLE["fig_width"], STYLE["row_height"] * n_rows),
        dpi=STYLE["dpi"],
        facecolor=STYLE["facecolor"],
        constrained_layout=True,
    )
    total_w = float(sum(widths_ref))
    width_ratios = list(widths_ref) + [CBAR_REL_WIDTH * total_w]
    gs = fig.add_gridspec(
        nrows=n_rows,
        ncols=n_seg + 1,
        width_ratios=width_ratios,
        wspace=STYLE["wspace"],
        hspace=STYLE["hspace"],
    )

    for row_idx, row in enumerate(rows):
        temps = np.asarray(row["temps"], dtype=float)
        segments = row["segments"]
        stack_indices = np.asarray(row["stack_indices"], dtype=int)
        missing_slots = row["missing_slots"]
        colors = cmap(norm(temps))
        y_low, y_high = row_y_limits(segments, stack_indices, missing_slots)
        row_axes = []

        for seg_idx, (x_seg, y_seg) in enumerate(segments):
            ax = fig.add_subplot(gs[row_idx, seg_idx], sharey=row_axes[0] if row_axes else None)
            row_axes.append(ax)

            for temp, stack_idx, y_values, color in zip(temps, stack_indices, y_seg, colors):
                y_plot = y_values + float(stack_idx) * OFFSET
                ax.plot(x_seg, y_plot, color=color, lw=LINE_WIDTH, zorder=2)
                step_slice = slice(None, None, max(1, int(POINT_STEP)))
                ax.scatter(
                    x_seg[step_slice],
                    y_plot[step_slice],
                    s=MARKER_SIZE,
                    marker="o",
                    color=color,
                    edgecolors="none",
                    linewidths=0,
                    alpha=MARKER_ALPHA,
                    zorder=3,
                )

            width = float(x_seg[-1] - x_seg[0])
            ax.set_xlim(x_seg[0] - XPAD["left_frac"] * width, x_seg[-1] + XPAD["right_frac"] * width)
            ax.set_ylim(y_low, y_high)
            style_axis(ax, first=(seg_idx == 0))

            n_target, n_max = XTICK_WINDOWS[seg_idx]
            xticks = nice_xticks(float(x_seg[0]), float(x_seg[-1]), n_target=n_target, n_max=n_max)
            if xticks is not None:
                ax.set_xticks(xticks)

            if row_idx == n_rows - 1:
                ax.set_xlabel("Raman Shift (cm$^{-1}$)")
                if seg_idx == 0:
                    # The first broken-axis segment is narrow, so its centered
                    # label otherwise reaches the tight export boundary.
                    ax.xaxis.label.set_x(STYLE["left_xlabel_x"])
                ax.tick_params(axis="x", labelbottom=True, labelsize=STYLE["tick_x"])
            else:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)
            ax.tick_params(axis="y", labelsize=STYLE["tick_y"])

            d = CUT_SIZE
            if seg_idx < n_seg - 1:
                kwargs = dict(transform=ax.transAxes, color="k", clip_on=False, linewidth=1.0)
                ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
                ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
            if seg_idx > 0:
                kwargs = dict(transform=ax.transAxes, color="k", clip_on=False, linewidth=1.0)
                ax.plot((-d, +d), (-d, +d), **kwargs)
                ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)

        row_axes[0].set_ylabel("Intensity (a.u.)")

        cax = fig.add_subplot(gs[row_idx, -1])
        cbar = fig.colorbar(scalar_map, cax=cax, ticks=cbar_ticks)
        cbar.ax.set_ylim(vmin, vmax)
        cbar.ax.tick_params(labelsize=CBAR["tick_size"], width=0.8, length=4)
        cbar.set_label(CBAR["label"], size=STYLE["cbar_label"])

    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for fmt in formats:
        path = out_dir / f"{OUTPUT_STEM}.{fmt}"
        fig.savefig(
            path,
            dpi=STYLE["dpi"],
            bbox_inches="tight",
            pad_inches=STYLE["save_pad_inches"],
            facecolor=STYLE["facecolor"],
        )
        saved_paths.append(path)
        print(f"Saved {path.resolve()}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Figure 2 broken-axis waterfall from cleaned averaged TXT spectra.")
    parser.add_argument("--txt-root", type=Path, default=DEFAULT_TXT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_figure(args.txt_root, args.out_dir, args.formats, show=args.show)


if __name__ == "__main__":
    main()
