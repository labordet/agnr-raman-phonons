# SPDX-License-Identifier: CC-BY-4.0
"""Make a PowerPoint-ready Figure 1c from the spectra used for fitting.

The original published Figure 1c display came from a separate reference chain.
This replacement opens the retained *temperature means used for the fits*, whose
raw/spike/selection/ALS/mean chains are recorded in the release. The selected
measured temperatures are I/II/V 100 K, III 95 K and IV 105 K, as the
manuscript caption states. Nothing is
interpolated or fitted by this plotter.

Run from anywhere with the plotting environment::

    python -B reproduce/plot_fig1c_fitting_means.py

All scientific inputs remain read-only. Output must be inside release outputs/
or _verification/; the default is outputs/fig1c_paper_panel/.
"""

from __future__ import annotations

from pathlib import Path
import os
import argparse
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(os.environ["RAMAN_DATA_ROOT"]).resolve()
PREPROCESSING = Path(__file__).resolve().parent / "stages/preprocessing/reproduce_preprocessing.py"
spec = importlib.util.spec_from_file_location("fig1c_verified_preprocessing", PREPROCESSING)
preprocessing = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = preprocessing
spec.loader.exec_module(preprocessing)

# Final paper order, top to bottom. III and IV are the two equally near 100 K
# choices for this revised panel, as stated in the manuscript caption.
PLAN = [
    ("I", "Aligned_Au_3A", "Spikes_Removed", 100.0,
     "Low Coverage Aligned\n9-AGNRs on Au", "#FEB816", "o", True, 1_000_000.0),
    ("II", "Aligned_Au_8A", "Spikes_Removed_UP_1", 100.0,
     "High Coverage Aligned\n9-AGNRs on Au", "#F08228", "D", True, 1_000_000.0),
    ("III", "MIRA_Au_unaligned_8A", "Spikes_Removed_UP_1", 95.0,
     "High Coverage Unaligned\n9-AGNRs on Au", "#AF550C", "D", False, 1_000_000.0),
    ("IV", "MIRA_RO_unaligned_8A", "Spikes_Removed_UP_1", 105.0,
     "High Coverage Unaligned\n9-AGNRs on Al$_2$O$_3$", "#008686", "D", False, 6_000.0),
    ("V", "Aligned_RO_8A", "Spikes_Removed_DOWN_1", 100.0,
     "High Coverage Aligned\n9-AGNRs on Al$_2$O$_3$", "#00C8C8", "D", True, 6_000.0),
]
# Paper panel style: edit these values to adjust the PowerPoint component.
# These offsets are intentionally nonuniform because IV has a strong D peak.
# They only change row position; every spectrum uses the same intensity scale.
OFFSETS = {"I": 5.1, "II": 4.1, "III": 3.05, "IV": 1.15, "V": 0.0}
FIGURE_SIZE_INCHES = (7.4, 9.4)
EXPORT_DPI = 400
FONT_FAMILY = "Arial"
AXIS_LABEL_FONT_SIZE_PT = 19
TICK_LABEL_FONT_SIZE_PT = 17
BOX_LINE_WIDTH_PT = 1.8
TICK_LINE_WIDTH_PT = 1.4
TICK_LENGTH_PT = 7
LINE_WIDTH_PT = 1.8
SCATTER_EVERY_NTH_POINT = 4
SCATTER_SIZE_PT2 = 1.8
X_LIMIT_CM1 = (150, 2000)
Y_LIMIT = (-0.20, 6.45)
X_TICKS_CM1 = (300, 600, 900, 1200, 1500, 1800)
# The axes rectangle has the ~0.72 width/height ratio of the original panel.
AXES_MARGINS = {"left": .100, "right": .917, "top": .980, "bottom": .090}
G_WINDOW = (1595.0, 1615.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_figure(output: Path) -> dict:
    output = preprocessing.prepare_output_directory(output)
    means = json.loads((ROOT / "metadata/temperature_mean_provenance.json").read_text(encoding="utf-8"))
    treatments = json.loads((ROOT / "metadata/preprocessing_provenance.json").read_text(encoding="utf-8"))
    checksums = {name: expected for expected, name in
                 (line.split("  ", 1) for line in
                  (ROOT / "checksums_sha256.txt").read_text(encoding="utf-8").splitlines())}

    inputs = []
    curves = []
    curve_frames = {}
    for label, family, sequence, temperature, _, _, _, _, expected_lambda in PLAN:
        selected_means = [row for row in means if row["family"] == family
                          and row["sequence"] == sequence and row["temperature_K"] == temperature]
        selected_treatments = [row for row in treatments if row["family"] == family
                               and row["sequence"] == sequence and row["temperature_K"] == temperature]
        if len(selected_means) != 1 or len(selected_treatments) != 1:
            raise ValueError(f"{label}: expected one fitting mean and one treatment at {temperature:g} K")
        mean, treatment = selected_means[0], selected_treatments[0]
        if (treatment["als_lambda"] != expected_lambda or treatment["als_p"] != .0055
                or treatment["als_iterations"] != 10):
            raise ValueError(f"{label}: main-series ALS settings differ from the paper")
        if label == "V" and sequence != "Spikes_Removed_DOWN_1":
            raise ValueError("V must use the author-confirmed Heating 1 fitting path")
        candidate_temperatures = [row["temperature_K"] for row in means
                                  if row["family"] == family and row["sequence"] == sequence]
        minimum_distance = min(abs(value - 100.0) for value in candidate_temperatures)
        if abs(temperature - 100.0) != minimum_distance:
            raise ValueError(f"{label}: selected temperature is not nearest to 100 K")

        mean_path = ROOT / mean["release_path"]
        raw_path = ROOT / treatment["raw_release"]
        for path in (mean_path, raw_path):
            relative = path.relative_to(ROOT).as_posix()
            if sha256(path) != checksums[relative]:
                raise ValueError(f"Protected scientific input differs from release checksum: {relative}")
        matrix = preprocessing.read_matrix(mean_path)
        if matrix.ndim != 2 or matrix.shape[1] != 2 or not np.isfinite(matrix).all():
            raise ValueError(f"{label}: expected finite two-column fitting mean")
        x, y = matrix[:, 0], matrix[:, 1]
        if not np.all(np.diff(x) > 0):
            raise ValueError(f"{label}: measured Raman grid must increase strictly")
        inside_g = (x >= G_WINDOW[0]) & (x <= G_WINDOW[1])
        if not inside_g.any():
            raise ValueError(f"{label}: G normalization window absent")
        denominator = float(y[inside_g].max())
        if denominator <= 0 or not np.isfinite(denominator):
            raise ValueError(f"{label}: invalid G normalization maximum")
        normalized = y / denominator
        frame = pd.DataFrame({
            "configuration": label, "measured_temperature_K": temperature,
            "raman_shift_cm-1": x, "normalized_intensity": normalized,
            "display_offset": OFFSETS[label],
            "stacked_intensity": normalized + OFFSETS[label],
        })
        curve_frames[label] = frame
        curves.append(frame)
        inputs.append({
            "configuration": label,
            "measured_temperature_K": temperature,
            "thermal_path": "Heating 1" if label == "V" else "main fitting path",
            "temperature_mean_file": mean["release_path"],
            "temperature_mean_sha256": sha256(mean_path),
            "raw_export": treatment["raw_release"],
            "raw_sha256": sha256(raw_path),
            "accepted_columns_in_mean": mean["n_used_columns"],
            "ALS_lambda": treatment["als_lambda"],
            "ALS_p": treatment["als_p"],
            "ALS_iterations": treatment["als_iterations"],
            "ALS_crop_min_cm-1": treatment["crop_min"],
            "ALS_crop_max_cm-1": treatment["crop_max"],
            "G_window_maximum": denominator,
            "measured_grid_points": len(x),
            "display_offset": OFFSETS[label],
        })

    combined = pd.concat(curves, ignore_index=True)
    if list(curve_frames) != ["I", "II", "III", "IV", "V"] or len(combined) != 3871:
        raise ValueError("Figure 1c order or point count is wrong")
    row_gaps = {}
    for upper, lower in zip(("I", "II", "III", "IV"), ("II", "III", "IV", "V")):
        upper_values = curve_frames[upper]
        lower_values = curve_frames[lower]
        upper_x = upper_values["raman_shift_cm-1"].to_numpy()
        upper_y = upper_values["stacked_intensity"].to_numpy()
        lower_x = lower_values["raman_shift_cm-1"].to_numpy()
        lower_y = lower_values["stacked_intensity"].to_numpy()
        visible = (upper_x >= X_LIMIT_CM1[0]) & (upper_x <= X_LIMIT_CM1[1])
        # Interpolation is only a crossing check between the different grids.
        # The displayed traces themselves use the original retained points.
        gap = float(np.min(upper_y[visible] - np.interp(
            upper_x[visible], lower_x, lower_y)))
        if gap < .12:
            raise ValueError(f"Displayed {lower}/{upper} traces overlap or crowd: {gap:.3g}")
        row_gaps[f"{lower}_to_{upper}"] = gap
    combined.to_csv(output / "Fig_1c_paper_fitting_means_curves.csv", index=False,
                    float_format="%.17g")
    pd.DataFrame(inputs).to_csv(output / "Fig_1c_paper_fitting_means_inputs.csv",
                                index=False, float_format="%.17g")

    # Artwork-only panel for insertion into the author's PowerPoint assembly.
    # The line joins every retained mean-spectrum grid point. Small filled
    # markers identify an evenly spaced subset of those real points, so dense
    # markers do not visually obscure the measured curves at paper size.
    plt.rcParams.update({"font.family": FONT_FAMILY, "svg.fonttype": "none"})
    fig_panel, ax_panel = plt.subplots(figsize=FIGURE_SIZE_INCHES, facecolor="white")
    ax_panel.set_facecolor("white")
    for label, _, _, _, _, color, _, _, _ in PLAN:
        curve = curve_frames[label]
        x = curve["raman_shift_cm-1"].to_numpy()
        stacked = curve["stacked_intensity"].to_numpy()
        ax_panel.plot(x, stacked, color=color, lw=LINE_WIDTH_PT, alpha=1.0,
                      zorder=2,
                      solid_joinstyle="round", solid_capstyle="round")
        marker_indices = np.arange(0, len(x), SCATTER_EVERY_NTH_POINT)
        ax_panel.scatter(x[marker_indices], stacked[marker_indices],
                         s=SCATTER_SIZE_PT2, marker="o", c=color, alpha=1.0,
                         edgecolors="none", zorder=3)
    ax_panel.set_xlim(*X_LIMIT_CM1)
    ax_panel.set_ylim(*Y_LIMIT)
    ax_panel.set_xticks(X_TICKS_CM1)
    ax_panel.set_yticks([])
    ax_panel.tick_params(axis="x", which="major", top=True, bottom=True,
                         direction="in", length=TICK_LENGTH_PT,
                         width=TICK_LINE_WIDTH_PT,
                         labelsize=TICK_LABEL_FONT_SIZE_PT, pad=7)
    ax_panel.tick_params(axis="y", left=False, right=False)
    ax_panel.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=AXIS_LABEL_FONT_SIZE_PT,
                        labelpad=10)
    ax_panel.set_ylabel("Intensity (a.u.)", fontsize=AXIS_LABEL_FONT_SIZE_PT,
                        labelpad=12)
    for spine in ax_panel.spines.values():
        spine.set_color("black")
        spine.set_linewidth(BOX_LINE_WIDTH_PT)
    fig_panel.subplots_adjust(**AXES_MARGINS)
    panel_png = output / "Fig_1c_paper_fitting_means_spectra_only_white.png"
    panel_svg = output / "Fig_1c_paper_fitting_means_spectra_only_white.svg"
    fig_panel.savefig(panel_png, dpi=EXPORT_DPI, facecolor="white")
    fig_panel.savefig(panel_svg, facecolor="white")
    transparent_png = output / "Fig_1c_paper_fitting_means_spectra_only_transparent.png"
    transparent_svg = output / "Fig_1c_paper_fitting_means_spectra_only_transparent.svg"
    fig_panel.savefig(transparent_png, dpi=EXPORT_DPI, transparent=True)
    fig_panel.savefig(transparent_svg, transparent=True)
    plt.close(fig_panel)

    report = {
        "purpose": "Replacement Figure 1c panel for the paper; user will assemble the full Figure 1 in PowerPoint.",
        "configuration_order_top_to_bottom": [row[0] for row in PLAN],
        "measured_temperatures_K": {row[0]: row[3] for row in PLAN},
        "selection_rule": "Nearest fitting means to 100 K; III and IV each have equally near 95/105 K choices. This revised panel uses III 95 K and IV 105 K, as the retained manuscript caption states.",
        "processing": "Retained fitting temperature means already contain the recorded spike edits, column selection, main ALS subtraction and averaging. This plot only G-normalizes and offsets them.",
        "G_normalization_window_cm-1": list(G_WINDOW),
        "plot_width_to_height_ratio": ((AXES_MARGINS["right"] - AXES_MARGINS["left"])
                                       * FIGURE_SIZE_INCHES[0]
                                       / ((AXES_MARGINS["top"] - AXES_MARGINS["bottom"])
                                          * FIGURE_SIZE_INCHES[1])),
        "line_points": "Every retained mean-spectrum grid point, with no smoothing or interpolation.",
        "scatter_points": f"Every {SCATTER_EVERY_NTH_POINT}th retained mean-spectrum grid point, shown as small filled circles to keep the publication panel readable.",
        "line_width_pt": LINE_WIDTH_PT,
        "font_family": FONT_FAMILY,
        "axis_label_font_size_pt": AXIS_LABEL_FONT_SIZE_PT,
        "tick_label_font_size_pt": TICK_LABEL_FONT_SIZE_PT,
        "box_line_width_pt": BOX_LINE_WIDTH_PT,
        "interpolation": False,
        "smoothing": False,
        "new_lorentzian_fits": False,
        "accepted_peak_or_temperature_model_changed": False,
        "original_published_figure_changed": False,
        "curve_rows": len(combined),
        "minimum_pointwise_vertical_gaps_between_adjacent_traces": row_gaps,
        "inputs": inputs,
        "exports": {"spectra_only_png": panel_png.name,
                    "spectra_only_svg": panel_svg.name,
                    "spectra_only_transparent_png": transparent_png.name,
                    "spectra_only_transparent_svg": transparent_svg.name,
                    "curves_csv": "Fig_1c_paper_fitting_means_curves.csv",
                    "inputs_csv": "Fig_1c_paper_fitting_means_inputs.csv"},
    }
    (output / "Fig_1c_paper_fitting_means_provenance.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/fig1c_paper_panel")
    args = parser.parse_args()
    report = make_figure(args.output)
    print(f"Figure 1c ready: {report['curve_rows']} measured-grid rows, "
          f"order {'/'.join(report['configuration_order_top_to_bottom'])}; "
          f"PNG/SVG/CSV in {args.output}")


if __name__ == "__main__":
    main()
