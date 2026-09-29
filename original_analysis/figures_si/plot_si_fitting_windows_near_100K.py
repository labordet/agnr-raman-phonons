#!/usr/bin/env python3
"""Plot lowest-temperature Raman spectra with the x windows used for fitting.

The script reads the final averaged, baseline-corrected TXT spectra and the
final paper-clean Lorentzian result tables. It never modifies those inputs.
The plotted windows are taken from the x_bound_min/x_bound_max values stored
with the final fits, including any adaptive expansions used during fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
AFTER_ROOT = ROOT / "GAMMA_T_COMPARISON" / "AFTER"
SPECTRA_ROOT = AFTER_ROOT / "AVERAGED_BASELINE_CORRECTED_TXT_FOR_CLUSTER"
RESULTS_ROOT = AFTER_ROOT / "FITTED_RESULTS_PAPER_CLEAN"
AFTER_OUTPUT = AFTER_ROOT / "SI_FITTING_WINDOWS_NEAR_100K"
FINAL_FIGURE_OUTPUT = ROOT / "SI FIGURES FINAL"
SI_COMPILE_OUTPUT = ROOT / "SI FINAL" / "SI_Figs"

FIGURE_NAME = "Fig_S3_Fitting_Windows_Near_100K_All_Configurations.png"
MANIFEST_NAME = "Fig_S3_Fitting_Windows_Near_100K_Sources_and_Bounds.csv"

# Match the readable printed typography used by Figures S1, S2, and S4.  S3's
# 13.2-inch canvas is reduced to 0.98\textwidth, so the source text must be
# large enough to remain approximately 8.5 pt after insertion in the SI.
SI_TICK_LABEL_SIZE = 17.5
SI_AXIS_LABEL_SIZE = 19.5
SI_LEGEND_SIZE = 17.0
SI_PANEL_TEXT_SIZE = 17.0
SI_SPINE_WIDTH = 1.25
SI_TICK_WIDTH = 1.25
SI_TICK_LENGTH = 6.0
SI_RAMAN_TICKS = [400, 800, 1200, 1600]


SAMPLE_COLORS = {
    "Aligned_Au_3A": "#feb715",
    "Aligned_Au_8A": "#f08228",
    "MIRA_Au_unaligned_8A": "#ae540b",
    "MIRA_RO_unaligned_8A": "#008686",
    "Aligned_RO_8A": "#00c8c8",
}

PEAK_COLORS = {
    "RBLM": "#2c7fb8",
    "CH_L1": "#8c6bb1",
    "CH_L2": "#8c6bb1",
    "CH_MID": "#8c6bb1",
    "D": "#b5bd00",
    "G": "#17becf",
}

PEAK_LABELS = {
    "RBLM": "RBLM",
    "CH_L1": "CH-related components",
    "CH_L2": "CH-related components",
    "CH_MID": "CH-related components",
    "D": "D",
    "G": "G",
}


@dataclass(frozen=True)
class Case:
    family: str
    sequence: str
    temperature: float
    label: str
    marker: str
    hollow: bool
    grid_position: tuple[int, int]


CASES = (
    Case(
        family="Aligned_Au_3A",
        sequence="Spikes_Removed",
        temperature=70.0,
        label="Aligned Au, low coverage",
        marker="o",
        hollow=False,
        grid_position=(0, 0),
    ),
    Case(
        family="Aligned_Au_8A",
        sequence="Spikes_Removed_DOWN_1",
        temperature=70.0,
        label="Aligned Au, high coverage",
        marker="D",
        hollow=False,
        grid_position=(1, 0),
    ),
    Case(
        family="MIRA_Au_unaligned_8A",
        sequence="Spikes_Removed_UP_1",
        temperature=75.8,
        label="Unaligned Au, high coverage",
        marker="D",
        hollow=True,
        grid_position=(2, 0),
    ),
    Case(
        family="MIRA_RO_unaligned_8A",
        sequence="Spikes_Removed_UP_1",
        temperature=75.0,
        label="Unaligned RO, high coverage",
        marker="D",
        hollow=True,
        grid_position=(0, 1),
    ),
    Case(
        family="Aligned_RO_8A",
        sequence="Spikes_Removed_DOWN_1",
        temperature=70.0,
        label="Aligned RO, high coverage",
        marker="D",
        hollow=False,
        grid_position=(1, 1),
    ),
)


def temperature_token(temperature: float) -> str:
    if float(temperature).is_integer():
        return f"{int(temperature)}K"
    return f"{temperature:g}".replace(".", "p") + "K"


def spectrum_path(case: Case) -> Path:
    path = (
        SPECTRA_ROOT
        / case.family
        / case.sequence
        / f"TEMP_{temperature_token(case.temperature)}_AVG.txt"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing averaged spectrum: {path}")
    return path


def result_path(case: Case) -> Path:
    path = RESULTS_ROOT / case.family / f"{case.family}_long_results.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing fitted-result table: {path}")
    return path


def read_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pd.read_csv(path, sep=None, engine="python")
    if table.shape[1] < 2:
        raise ValueError(f"Expected X and Y columns in {path}")
    x = pd.to_numeric(table.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(table.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 200.0) & (x <= 2000.0)
    if np.count_nonzero(valid) < 10:
        raise ValueError(f"Too few finite data points in 200-2000 cm-1 for {path}")
    x = x[valid]
    y = y[valid]
    order = np.argsort(x)
    return x[order], y[order]


def read_bounds(case: Case) -> pd.DataFrame:
    table = pd.read_csv(result_path(case))
    temperature = pd.to_numeric(table["temperature_K"], errors="coerce")
    selected = table.loc[
        (table["sequence"].astype(str) == case.sequence)
        & np.isclose(temperature, case.temperature, atol=1e-6)
    ].copy()
    if selected.empty:
        raise ValueError(
            f"No final fit rows for {case.family}, {case.sequence}, {case.temperature:g} K"
        )
    selected["x_bound_min"] = pd.to_numeric(selected["x_bound_min"], errors="coerce")
    selected["x_bound_max"] = pd.to_numeric(selected["x_bound_max"], errors="coerce")
    selected = selected.dropna(subset=["x_bound_min", "x_bound_max"])
    selected = selected.drop_duplicates(subset=["peak_id"], keep="first")
    order = {peak_id: index for index, peak_id in enumerate(PEAK_COLORS)}
    selected["_order"] = selected["peak_id"].map(order).fillna(999)
    return selected.sort_values("_order")


def normalize_for_display(y: np.ndarray) -> np.ndarray:
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return y.copy()
    scale = float(np.nanmax(finite))
    if not np.isfinite(scale) or abs(scale) < 1e-12:
        scale = float(np.nanmax(np.abs(finite)))
    if not np.isfinite(scale) or abs(scale) < 1e-12:
        scale = 1.0
    return y / scale


def style_axis(ax: plt.Axes, show_x_labels: bool) -> None:
    ax.set_xlim(200.0, 2000.0)
    # The last labeled tick stays inside the frame rather than on its corner.
    ax.set_xticks(SI_RAMAN_TICKS)
    ax.set_yticks([])
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        top=False,
        right=False,
        width=SI_TICK_WIDTH,
        length=SI_TICK_LENGTH,
        labelsize=SI_TICK_LABEL_SIZE,
    )
    ax.tick_params(labelbottom=show_x_labels)
    for spine in ax.spines.values():
        spine.set_linewidth(SI_SPINE_WIDTH)


def plot_case(ax: plt.Axes, case: Case, show_x_labels: bool) -> list[dict[str, object]]:
    path = spectrum_path(case)
    x, y = read_spectrum(path)
    y_display = normalize_for_display(y)
    bounds = read_bounds(case)

    for row in bounds.itertuples(index=False):
        peak_id = str(row.peak_id)
        color = PEAK_COLORS.get(peak_id, "#777777")
        low = float(min(row.x_bound_min, row.x_bound_max))
        high = float(max(row.x_bound_min, row.x_bound_max))
        ax.axvspan(low, high, color=color, alpha=0.16, zorder=0)
        ax.axvline(low, color=color, lw=0.9, ls="--", alpha=0.72, zorder=1)
        ax.axvline(high, color=color, lw=0.9, ls="--", alpha=0.72, zorder=1)

    ax.plot(x, y_display, color="#4b4b4b", lw=0.85, zorder=3)
    color = SAMPLE_COLORS[case.family]
    scatter_kwargs = {
        "s": 16.0 if case.hollow else 14.0,
        "marker": case.marker,
        "linewidths": 1.2 if case.hollow else 0.45,
        "zorder": 5,
    }
    if case.hollow:
        scatter_kwargs.update(facecolors="none", edgecolors=color)
    else:
        scatter_kwargs.update(facecolors=color, edgecolors=color)
    ax.scatter(x, y_display, **scatter_kwargs)

    y_min = float(np.nanmin(y_display))
    y_max = float(np.nanmax(y_display))
    span = max(y_max - y_min, 0.2)
    ax.set_ylim(y_min - 0.10 * span, y_max + 0.28 * span)
    ax.text(
        0.025,
        0.92,
        case.label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=SI_PANEL_TEXT_SIZE,
        fontweight="bold",
        color=color,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        zorder=8,
    )
    ax.text(
        0.975,
        0.92,
        f"T = {case.temperature:g} K",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=SI_PANEL_TEXT_SIZE,
        fontweight="bold",
        color="#202020",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        zorder=8,
    )
    style_axis(ax, show_x_labels=show_x_labels)
    if show_x_labels:
        ax.set_xlabel(r"Raman shift (cm$^{-1}$)", fontsize=SI_AXIS_LABEL_SIZE)

    manifest_rows: list[dict[str, object]] = []
    for row in bounds.itertuples(index=False):
        manifest_rows.append(
            {
                "sample_configuration": case.label,
                "internal_family_key": case.family,
                "thermal_sequence": case.sequence,
                "temperature_K": case.temperature,
                "spectrum_path": str(path),
                "fitted_results_path": str(result_path(case)),
                "peak_id": str(row.peak_id),
                "peak_label": str(row.peak_label),
                "x_bound_min_cm-1": float(row.x_bound_min),
                "x_bound_max_cm-1": float(row.x_bound_max),
            }
        )
    return manifest_rows


def add_window_key(ax: plt.Axes) -> None:
    ax.axis("off")
    legend_peak_ids = ("RBLM", "CH_L1", "D", "G")
    handles = [
        Patch(
            facecolor=PEAK_COLORS[peak_id],
            edgecolor=PEAK_COLORS[peak_id],
            alpha=0.20,
            linestyle="--",
            linewidth=1.2,
            label=PEAK_LABELS[peak_id],
        )
        for peak_id in legend_peak_ids
    ]
    ax.text(
        0.5,
        0.82,
        "Peak-position windows",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=18.0,
        fontweight="bold",
    )
    window_legend = ax.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.50),
        ncol=2,
        frameon=False,
        fontsize=SI_LEGEND_SIZE,
        handlelength=2.0,
        columnspacing=1.8,
        labelspacing=1.0,
    )
    ax.add_artist(window_legend)
    data_handle = Line2D(
        [],
        [],
        color="#4b4b4b",
        marker="o",
        markerfacecolor="#4b4b4b",
        markeredgecolor="#4b4b4b",
        markersize=4.5,
        lw=0.9,
        label="Measured spectrum",
    )
    ax.legend(
        handles=[data_handle],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.12),
        frameon=False,
        fontsize=SI_LEGEND_SIZE,
    )


def main() -> None:
    for directory in (AFTER_OUTPUT, FINAL_FIGURE_OUTPUT, SI_COMPILE_OUTPUT):
        directory.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 15,
            "axes.labelsize": SI_AXIS_LABEL_SIZE,
            "xtick.labelsize": SI_TICK_LABEL_SIZE,
            "ytick.labelsize": SI_TICK_LABEL_SIZE,
            "legend.fontsize": SI_LEGEND_SIZE,
            "axes.linewidth": SI_SPINE_WIDTH,
            "xtick.major.width": SI_TICK_WIDTH,
            "ytick.major.width": SI_TICK_WIDTH,
            "xtick.major.size": SI_TICK_LENGTH,
            "ytick.major.size": SI_TICK_LENGTH,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(13.2, 10.4),
        sharex=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.070,
        top=0.985,
        wspace=0.10,
        hspace=0.12,
    )

    manifest_rows: list[dict[str, object]] = []
    by_position = {case.grid_position: case for case in CASES}
    for row in range(3):
        for col in range(2):
            ax = axes[row, col]
            case = by_position.get((row, col))
            if case is None:
                add_window_key(ax)
                continue
            show_x_labels = (col == 0 and row == 2) or (col == 1 and row == 1)
            manifest_rows.extend(plot_case(ax, case, show_x_labels=show_x_labels))

    fig.supylabel(
        "Normalized intensity (a.u.)",
        x=0.018,
        fontsize=SI_AXIS_LABEL_SIZE,
    )

    output_paths = [
        AFTER_OUTPUT / FIGURE_NAME,
        FINAL_FIGURE_OUTPUT / FIGURE_NAME,
        SI_COMPILE_OUTPUT / FIGURE_NAME,
    ]
    for output_path in output_paths:
        fig.savefig(output_path, dpi=400, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(AFTER_OUTPUT / MANIFEST_NAME, index=False)

    print("Created fitting-window figure:")
    for output_path in output_paths:
        print(f"  {output_path}")
    print(f"Created source-and-bounds manifest: {AFTER_OUTPUT / MANIFEST_NAME}")


if __name__ == "__main__":
    main()
