#!/usr/bin/env python3
"""Compare legacy and corrected thermo-mechanical Raman fits.

The script compares outputs produced by Figure_3_temperature_shift_from_after.py
using the legacy and corrected thermal-expansion inputs. It never modifies either
set of fit results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FIT_STEMS = (
    "FIG3_fit_parameters_AFTER_AlignedAu8A_UP1",
    "FIG3_fit_parameters_AFTER_AlignedAu8A_DOWN1",
    "FIG3_fit_parameters_AFTER_AlignedAu8A_UP2",
    "FIG3_fit_parameters_AFTER_AlignedAu8A_NestedSequences",
)

PARAMETERS = (
    "omega0_cm-1",
    "omega0_stderr",
    "A3_cm-1",
    "A3_stderr",
    "gamma_parallel",
    "gamma_stderr",
    "RMSE_cm-1",
    "R2",
)

KEY_COLUMNS = ("peak_id", "family", "sequence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-dir", type=Path, required=True)
    parser.add_argument("--fixed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_pair(old_dir: Path, fixed_dir: Path, stem: str) -> pd.DataFrame:
    old_path = old_dir / f"{stem}.csv"
    fixed_path = fixed_dir / f"{stem}.csv"
    if not old_path.exists() or not fixed_path.exists():
        return pd.DataFrame()

    old = pd.read_csv(old_path)
    fixed = pd.read_csv(fixed_path)
    keep_old = [*KEY_COLUMNS, "family_label", *PARAMETERS]
    keep_fixed = [*KEY_COLUMNS, "family_label", *PARAMETERS]
    merged = old[keep_old].merge(
        fixed[keep_fixed],
        on=list(KEY_COLUMNS),
        how="outer",
        suffixes=("_old", "_fixed"),
        indicator=True,
    )
    merged.insert(0, "fit_set", stem.removeprefix("FIG3_fit_parameters_AFTER_"))
    for parameter in PARAMETERS:
        old_col = f"{parameter}_old"
        fixed_col = f"{parameter}_fixed"
        merged[f"{parameter}_delta"] = merged[fixed_col] - merged[old_col]
        denominator = merged[old_col].abs()
        merged[f"{parameter}_relative_change_percent"] = np.where(
            denominator > 0,
            100.0 * merged[f"{parameter}_delta"] / denominator,
            np.nan,
        )
    return merged


def build_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for parameter in PARAMETERS:
        delta = pd.to_numeric(comparison[f"{parameter}_delta"], errors="coerce")
        relative = pd.to_numeric(
            comparison[f"{parameter}_relative_change_percent"], errors="coerce"
        )
        finite_delta = delta[np.isfinite(delta)]
        finite_relative = relative[np.isfinite(relative)]
        rows.append(
            {
                "parameter": parameter,
                "n_compared": int(finite_delta.size),
                "max_abs_delta": float(finite_delta.abs().max())
                if not finite_delta.empty
                else np.nan,
                "median_abs_delta": float(finite_delta.abs().median())
                if not finite_delta.empty
                else np.nan,
                "max_abs_relative_change_percent": float(finite_relative.abs().max())
                if not finite_relative.empty
                else np.nan,
                "median_abs_relative_change_percent": float(
                    finite_relative.abs().median()
                )
                if not finite_relative.empty
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = [
        frame
        for stem in FIT_STEMS
        if not (frame := load_pair(args.old_dir, args.fixed_dir, stem)).empty
    ]
    if not frames:
        raise FileNotFoundError("No matching legacy/fixed fit-parameter CSV pairs found.")

    comparison = pd.concat(frames, ignore_index=True)
    summary = build_summary(comparison)
    comparison_path = args.output_dir / "FIT_PARAMETER_OLD_vs_FIXED.csv"
    summary_path = args.output_dir / "FIT_PARAMETER_CHANGE_SUMMARY.csv"
    comparison.to_csv(comparison_path, index=False)
    summary.to_csv(summary_path, index=False)

    omega_delta = pd.to_numeric(comparison["omega0_cm-1_delta"], errors="coerce")
    report = {
        "fit_pairs_compared": len(frames),
        "rows_compared": int(len(comparison)),
        "omega0_max_abs_delta_cm-1": float(omega_delta.abs().max()),
        "omega0_unchanged_within_1e-12": bool(
            np.allclose(omega_delta.fillna(0.0), 0.0, atol=1e-12, rtol=0.0)
        ),
        "legacy_directory": str(args.old_dir.resolve()),
        "corrected_directory": str(args.fixed_dir.resolve()),
        "comparison_csv": str(comparison_path.resolve()),
        "summary_csv": str(summary_path.resolve()),
    }
    report_path = args.output_dir / "FIT_PARAMETER_COMPARISON_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print("\nParameter-change summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
