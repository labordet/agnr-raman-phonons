#!/usr/bin/env python3
"""Build traceable SI tables from the final curated AFTER analysis.

This script does not modify spectra or fitted-result files. It reads the same
curated temperature-means table and thermo-mechanical model implementation used
for the final Figure 3, then writes audit-friendly CSV tables for the SI.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
AFTER_DIR = APP_DIR / "GAMMA_T_COMPARISON" / "AFTER"
INPUT_CSV = (
    AFTER_DIR
    / "CURATED_TABLES_AND_MANIFESTS"
    / "Gamma_T_Comparison_curated_temperature_means.csv"
)
FIGURE3_SCRIPT = APP_DIR / "Figure_3_temperature_shift_from_after.py"
OUTPUT_DIR = APP_DIR / "SI FINAL" / "TABLE_SOURCE_AUDIT"

COMMON_T_LOW_K = 80.0
COMMON_T_HIGH_K = 290.0

SAMPLE_LABELS = {
    "Aligned_Au_3A": "Aligned Au, low coverage",
    "Aligned_Au_8A": "Aligned Au, high coverage",
    "MIRA_Au_unaligned_8A": "Unaligned Au, high coverage",
    "MIRA_RO_unaligned_8A": "Unaligned RO, high coverage",
    "Aligned_RO_8A": "Aligned RO, high coverage",
}

SEQUENCE_LABELS = {
    "Spikes_Removed": "single sweep",
    "Spikes_Removed_UP_1": "UP 1",
    "Spikes_Removed_DOWN_1": "DOWN 1",
    "Spikes_Removed_UP_2": "UP 2",
}


def load_figure3_module():
    spec = importlib.util.spec_from_file_location("figure3_after", FIGURE3_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {FIGURE3_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sequence_label(sequence: str) -> str:
    return SEQUENCE_LABELS.get(sequence, sequence)


def build_tables(
    input_csv: Path = INPUT_CSV,
    cte_dir: Path = APP_DIR.parent,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    figure3 = load_figure3_module()
    _, results, parameters = figure3.build_results(
        input_csv=input_csv,
        cte_dir=cte_dir,
    )

    parameter_table = parameters.copy()
    parameter_table["sample_family"] = parameter_table["family"].map(SAMPLE_LABELS)
    parameter_table["sweep"] = parameter_table["sequence"].map(sequence_label)
    parameter_table = parameter_table[
        [
            "peak_id",
            "family",
            "sample_family",
            "sequence",
            "sweep",
            "omega0_cm-1",
            "omega0_stderr",
            "omega0_n_points",
            "omega0_temperature_min_K",
            "omega0_temperature_max_K",
            "omega0_method",
            "A3_cm-1",
            "A3_stderr",
            "gamma_parallel",
            "gamma_stderr",
            "RMSE_cm-1",
            "R2",
            "fit_status",
            "source_csv",
        ]
    ]

    slope_rows: list[dict[str, object]] = []
    decomposition_rows: list[dict[str, object]] = []
    for peak_id in ("RBLM", "D", "G"):
        for family in figure3.SAMPLES_PLOT:
            result = results[peak_id][family]
            temperatures = np.asarray(result["T_exp"], dtype=float)
            measured = np.asarray(result["w_exp"], dtype=float)
            model_at_measured = result["omega0"] + np.interp(
                temperatures,
                np.asarray(result["T_grid"], dtype=float),
                np.asarray(result["delta_sum_raw"], dtype=float),
            )
            model_slope = float(np.polyfit(temperatures, model_at_measured, 1)[0])
            measured_slope = float(np.polyfit(temperatures, measured, 1)[0])
            relative_difference = (
                100.0 * (measured_slope - model_slope) / model_slope
                if model_slope != 0.0
                else np.nan
            )
            sequence = str(result["sequence"])
            slope_rows.append(
                {
                    "peak_id": peak_id,
                    "family": family,
                    "sample_family": SAMPLE_LABELS[family],
                    "sequence": sequence,
                    "sweep": sequence_label(sequence),
                    "temperature_min_K": float(np.min(temperatures)),
                    "temperature_max_K": float(np.max(temperatures)),
                    "model_effective_slope_cm-1_K-1": model_slope,
                    "measured_linear_slope_cm-1_K-1": measured_slope,
                    "relative_difference_percent": relative_difference,
                    "definition": (
                        "OLS slope of model values at measured temperatures; "
                        "OLS slope of curated measured positions"
                    ),
                    "source_csv": str(input_csv.resolve()),
                }
            )

            t_grid = np.asarray(result["T_grid"], dtype=float)
            te = np.asarray(result["delta_te"], dtype=float)
            anh = np.asarray(result["delta_anh_raw"], dtype=float)
            te_change = float(
                np.interp(COMMON_T_HIGH_K, t_grid, te)
                - np.interp(COMMON_T_LOW_K, t_grid, te)
            )
            anh_change = float(
                np.interp(COMMON_T_HIGH_K, t_grid, anh)
                - np.interp(COMMON_T_LOW_K, t_grid, anh)
            )
            total_change = te_change + anh_change
            te_share = 100.0 * te_change / total_change if total_change != 0.0 else np.nan
            anh_share = 100.0 * anh_change / total_change if total_change != 0.0 else np.nan
            decomposition_rows.append(
                {
                    "peak_id": peak_id,
                    "family": family,
                    "sample_family": SAMPLE_LABELS[family],
                    "sequence": sequence,
                    "sweep": sequence_label(sequence),
                    "temperature_low_K": COMMON_T_LOW_K,
                    "temperature_high_K": COMMON_T_HIGH_K,
                    "delta_omega_TE_cm-1": te_change,
                    "delta_omega_anh_cm-1": anh_change,
                    "delta_omega_total_cm-1": total_change,
                    "additivity_residual_cm-1": total_change - te_change - anh_change,
                    "TE_share_percent_signed": te_share,
                    "anh_share_percent_signed": anh_share,
                    "source_csv": str(input_csv.resolve()),
                }
            )

    return (
        parameter_table,
        pd.DataFrame(slope_rows),
        pd.DataFrame(decomposition_rows),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build traceable SI thermo-mechanical tables.")
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--cte-dir", type=Path, default=APP_DIR.parent)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameters, slopes, decomposition = build_tables(
        input_csv=args.input_csv,
        cte_dir=args.cte_dir,
    )
    outputs = {
        "SI_Table_S1_current_model_parameters.csv": parameters,
        "SI_Table_S2_current_temperature_coefficients.csv": slopes,
        "SI_Table_S3_current_additive_decomposition_80K_290K.csv": decomposition,
    }
    for name, frame in outputs.items():
        path = args.output_dir / name
        frame.to_csv(path, index=False)
        print(f"Wrote {path.resolve()} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
