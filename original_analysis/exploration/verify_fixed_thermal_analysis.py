#!/usr/bin/env python3
"""Validate the corrected TEC analysis package without modifying it."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image


def extract_table_rows(tex: str, label: str) -> list[list[str]]:
    label_token = rf"\\label\{{{re.escape(label)}\}}"
    label_match = re.search(label_token, tex)
    if not label_match:
        return []
    start = tex.rfind("\\begin{table}", 0, label_match.start())
    end = tex.find("\\end{table}", label_match.end())
    block = tex[start:end]
    rows: list[list[str]] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not re.match(r"^(RBLM|\$D\$|\$G\$)\s*&", stripped):
            continue
        stripped = re.sub(r"\\\\\s*$", "", stripped)
        rows.append([cell.strip() for cell in stripped.split("&")])
    return rows


def numeric_tail(row: list[str], count: int) -> list[float]:
    return [float(value) for value in row[-count:]]


def expected_table_values(table: pd.DataFrame, table_name: str) -> list[list[float]]:
    if table_name == "S1":
        columns = [
            "omega0_cm-1",
            "omega0_stderr",
            "A3_cm-1",
            "A3_stderr",
            "gamma_parallel",
            "gamma_stderr",
            "RMSE_cm-1",
            "R2",
        ]
        decimals = [2, 2, 3, 3, 3, 3, 3, 3]
    elif table_name == "S2":
        columns = [
            "model_effective_slope_cm-1_K-1",
            "measured_linear_slope_cm-1_K-1",
            "relative_difference_percent",
        ]
        decimals = [5, 5, 2]
    else:
        columns = [
            "delta_omega_TE_cm-1",
            "delta_omega_anh_cm-1",
            "delta_omega_total_cm-1",
            "TE_share_percent_signed",
            "anh_share_percent_signed",
        ]
        decimals = [3, 3, 3, 1, 1]
    return [
        [round(float(row[column]), decimals[index]) for index, column in enumerate(columns)]
        for _, row in table.iterrows()
    ]


def compare_printed_table(
    tex: str, label: str, csv_path: Path, table_name: str
) -> dict[str, object]:
    printed_rows = extract_table_rows(tex, label)
    csv = pd.read_csv(csv_path)
    expected = expected_table_values(csv, table_name)
    actual = [numeric_tail(row, len(expected[0])) for row in printed_rows]
    matches = len(actual) == len(expected) and all(
        len(actual_row) == len(expected_row)
        and all(abs(a - e) <= 5e-8 for a, e in zip(actual_row, expected_row))
        for actual_row, expected_row in zip(actual, expected)
    )
    return {
        "label": label,
        "csv": str(csv_path.resolve()),
        "printed_rows": len(actual),
        "csv_rows": len(expected),
        "values_match_display_precision": matches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    return parser.parse_args()


def image_info(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        return {
            "path": str(path.resolve()),
            "width_px": image.width,
            "height_px": image.height,
            "mode": image.mode,
        }


def main() -> None:
    args = parse_args()
    package = args.package_dir.resolve()
    si_dir = package / "SI_FINAL"
    tex_path = si_dir / "SI_FINAL_FIXED.tex"
    tex = tex_path.read_text(encoding="utf-8")

    included = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex)
    missing_includes = [
        name for name in included if not (si_dir / "SI_Figs" / name).exists()
    ]
    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
    uncommented = "\n".join(line.split("%", 1)[0] for line in tex.splitlines())
    brace_balance = len(re.findall(r"(?<!\\)\{", uncommented)) - len(
        re.findall(r"(?<!\\)\}", uncommented)
    )
    begin_envs = re.findall(r"\\begin\{([^}]+)\}", uncommented)
    end_envs = re.findall(r"\\end\{([^}]+)\}", uncommented)
    environment_count_mismatches = {
        env: {"begin": begin_envs.count(env), "end": end_envs.count(env)}
        for env in sorted(set(begin_envs) | set(end_envs))
        if begin_envs.count(env) != end_envs.count(env)
    }

    table_dir = package / "TABLES"
    table_rows = {
        path.name: int(len(pd.read_csv(path))) for path in sorted(table_dir.glob("*.csv"))
    }
    printed_table_checks = [
        compare_printed_table(
            tex,
            "tab:model_params",
            table_dir / "SI_Table_S1_current_model_parameters.csv",
            "S1",
        ),
        compare_printed_table(
            tex,
            "tab:slope_comparison",
            table_dir / "SI_Table_S2_current_temperature_coefficients.csv",
            "S2",
        ),
        compare_printed_table(
            tex,
            "tab:shift_decomposition",
            table_dir / "SI_Table_S3_current_additive_decomposition_80K_290K.csv",
            "S3",
        ),
    ]
    comparison_report = json.loads(
        (package / "AUDIT" / "FIT_PARAMETER_COMPARISON_REPORT.json").read_text(
            encoding="utf-8"
        )
    )

    key_images = [
        package / "MAIN_FIGURE_3" / "Figure_3_MAIN_FIXED.png",
        si_dir / "SI_Figs" / "Fig_S6_Thermal_Expansion_Coefficients.png",
        si_dir
        / "SI_Figs"
        / "Fig_S8_Aligned_Au_High_Coverage_Thermal_Cycle_Shifts.png",
    ]
    missing_images = [str(path) for path in key_images if not path.exists()]
    images = [image_info(path) for path in key_images if path.exists()]

    report = {
        "package_directory": str(package),
        "latex_source": str(tex_path),
        "latex_include_count": len(included),
        "missing_latex_figure_includes": missing_includes,
        "duplicate_latex_labels": duplicate_labels,
        "latex_brace_balance": brace_balance,
        "latex_environment_count_mismatches": environment_count_mismatches,
        "contains_unicode_em_dash": "\u2014" in tex,
        "table_row_counts": table_rows,
        "printed_table_checks": printed_table_checks,
        "key_images": images,
        "missing_key_images": missing_images,
        "omega0_unchanged_within_1e-12": comparison_report[
            "omega0_unchanged_within_1e-12"
        ],
        "checks_passed": bool(
            not missing_includes
            and not duplicate_labels
            and brace_balance == 0
            and not environment_count_mismatches
            and "\u2014" not in tex
            and not missing_images
            and all(count == 15 for count in table_rows.values())
            and all(
                check["values_match_display_precision"]
                for check in printed_table_checks
            )
            and comparison_report["omega0_unchanged_within_1e-12"]
        ),
    }
    output_path = package / "AUDIT" / "FIXED_PACKAGE_VALIDATION.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
