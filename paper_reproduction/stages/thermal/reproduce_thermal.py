#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-4.0
"""Reproduce adopted TEC curves, Figure S6, and SI Tables S1-S4."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(os.environ["RAMAN_DATA_ROOT"]).resolve()

def prepare_output_directory(path):
    output = Path(path).resolve()
    bases = [ROOT / "outputs", ROOT / "_verification"]
    if any(base.resolve() != base for base in bases):
        raise ValueError("Output roots must not be redirected by filesystem links.")
    if not any(output.is_relative_to(base) for base in bases):
        raise ValueError("Output must be inside release outputs/ or _verification/.")
    output.mkdir(parents=True, exist_ok=True)
    for directory, subdirectories, files in os.walk(output, followlinks=False):
        for name in subdirectories + files:
            item = Path(directory) / name
            if item.resolve() != item:
                raise ValueError("Output directory contains a redirected filesystem path.")
    return output



def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compare_table(expected, actual, tolerance=1e-8):
    import numpy as np
    import pandas as pd
    result = {"expected_rows": len(expected), "reproduced_rows": len(actual), "columns": {}}
    passed = len(expected) == len(actual)
    for col in expected.columns:
        if col == "source_csv":
            continue
        if col not in actual:
            result["columns"][col] = {"passed": False, "reason": "column missing"}
            passed = False
            continue
        if pd.api.types.is_numeric_dtype(expected[col]):
            a, b = expected[col].to_numpy(float), actual[col].to_numpy(float)
            equal = a.shape == b.shape and np.allclose(a, b, rtol=0, atol=tolerance, equal_nan=True)
            error = float(np.nanmax(np.abs(a - b))) if a.shape == b.shape else None
            result["columns"][col] = {"passed": bool(equal), "max_abs_difference": error}
        else:
            equal = expected[col].fillna("").astype(str).tolist() == actual[col].fillna("").astype(str).tolist()
            result["columns"][col] = {"passed": bool(equal)}
        passed = passed and equal
    result["passed"] = bool(passed)
    result["absolute_numeric_tolerance"] = tolerance
    return result


def run(output_dir, curves_only=False):
    output_dir = prepare_output_directory(output_dir)
    os.environ["MPLCONFIGDIR"] = str(output_dir / ".matplotlib")
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    import pandas as pd
    from PIL import Image

    ref = ROOT / "data/reference_data/thermal_expansion"
    original = Path(__file__).resolve().parents[3] / "original_analysis/thermal"
    tec = load_module("archive_tec", original / "build_fixed_tec_inputs.py")
    tec.GOLD_SOURCE = "Au_lattice_parameters.xlsx"
    tec.CNT_SOURCE = "CNT_axial_digitized.xlsx"
    grid = np.arange(0.0, 300.0 + 0.0001, 0.1)
    au, au_meta = tec.build_gold_curve(grid, ref)
    sapphire, sapphire_meta = tec.build_sapphire_curve(grid)
    cnt, cnt_meta = tec.build_cnt_curve(grid, ref)
    curves = {"Gold": au, "Sapphire": sapphire, "CNT": cnt}
    checks = {"TEC": {}}
    for name, alpha in curves.items():
        tec.validate_curve(name, grid, alpha)
        path = output_dir / f"{name}_CTE_alpha_SI.csv"
        tec.write_curve(path, grid, alpha)
        expected = pd.read_csv(ref / path.name)
        checks["TEC"][name] = compare_table(expected, pd.read_csv(path), tolerance=1e-18)
        # Temperature-grid rounding is tested separately from the TEC values.
        checks["TEC"][name]["columns"]["T (K)"]["passed"] = bool(np.allclose(expected.iloc[:, 0], grid, rtol=0, atol=5e-13))
        checks["TEC"][name]["passed"] = all(c["passed"] for c in checks["TEC"][name]["columns"].values())
    combined = pd.DataFrame({"T (K)": grid, "Au alpha (1/K)": au, "Al2O3 alpha (1/K)": sapphire, "axial CNT alpha (1/K)": cnt})
    combined.to_csv(output_dir / "thermal_expansion_curves.csv", index=False)
    figure = tec.plot_curves(output_dir, combined, ref, au_meta)
    target = output_dir / "Fig_S6_thermal_expansion.png"
    figure.replace(target)
    canonical = ROOT / "figures/supplementary/Fig_S6.png"
    if canonical.exists():
        a, b = Image.open(canonical).convert("RGB"), Image.open(target).convert("RGB")
        checks["Fig_S6"] = {"same_dimensions": a.size == b.size, "pixel_identical": a.size == b.size and a.tobytes() == b.tobytes(), "canonical_dimensions": list(a.size), "reproduced_dimensions": list(b.size), "canonical_renderer": Image.open(canonical).info.get("Software"), "reproduced_renderer": Image.open(target).info.get("Software"), "canonical_sha256": hashlib.sha256(canonical.read_bytes()).hexdigest(), "reproduced_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    else:
        checks["Fig_S6"] = {"canonical_comparison": "canonical image unavailable at expected release path"}

    if not curves_only:
        inputs = ROOT / "data/derived/peak_parameters.csv"
        model_path = Path(__file__).resolve().parents[3] / "original_analysis/figures_main/Figure_3_temperature_shift_from_after.py"
        tables = load_module("archive_tables", original / "build_final_si_tables.py")
        tables.FIGURE3_SCRIPT = model_path
        main_tables = tables.build_tables(input_csv=inputs, cte_dir=ref)
        names = ["Table_S1_model_parameters.csv", "Table_S2_temperature_slopes.csv", "Table_S3_shift_decomposition.csv"]
        checks["tables"] = {}
        for name, frame in zip(names, main_tables):
            frame["source_csv"] = "data/derived/peak_parameters.csv"
            frame.to_csv(output_dir / name, index=False)
            checks["tables"][name] = compare_table(pd.read_csv(ROOT / "tables/supplementary" / name), frame)
        model = load_module("archive_cycle_model", model_path)
        _, results, cycle = model.build_results(inputs, ref, sequence_filters=model.ALIGNED_AU_8A_NESTED_SEQUENCE_FILTERS, sample_order=model.ALIGNED_AU_8A_NESTED_SAMPLE_ORDER)
        data_slopes, model_slopes = [], []
        for _, row in cycle.iterrows():
            r = results[row.peak_id][row.family]
            times = np.asarray(r["T_exp"])
            data_slopes.append(float(np.polyfit(times, r["w_exp"], 1)[0]))
            modeled = r["omega0"] + np.interp(times, r["T_grid"], r["delta_sum_raw"])
            model_slopes.append(float(np.polyfit(times, modeled, 1)[0]))
        cycle["measured_linear_slope_cm-1_K-1"] = data_slopes
        cycle["model_effective_slope_cm-1_K-1"] = model_slopes
        cycle["source_csv"] = "data/derived/peak_parameters.csv"
        name = "Table_S4_thermal_paths.csv"
        cycle.to_csv(output_dir / name, index=False)
        checks["tables"][name] = compare_table(pd.read_csv(ROOT / "tables/supplementary" / name), cycle)
    checks["numeric_checks_passed"] = all(c["passed"] for c in checks["TEC"].values()) and all(c["passed"] for c in checks.get("tables", {}).values())
    checks["limitations"] = ["The author confirmed digitizing the CNT values from Jiang et al. Figure 5(a); the exact curve label and original digitization session are not recorded.", "Numerical fit comparisons allow small optimizer/version effects; accepted canonical parameters are never replaced."]
    checks["author_confirmed_label"] = "Configuration V source identifier Spikes_Removed_DOWN_1 denotes Heating 1, as confirmed by the author. Selected data and numerical contents are unchanged."
    (output_dir / "verification.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps({"numeric_checks_passed": checks["numeric_checks_passed"], "output": str(output_dir.relative_to(ROOT)), "Fig_S6": checks["Fig_S6"]}, indent=2))
    if not checks["numeric_checks_passed"]:
        raise SystemExit("Numerical verification failed; inspect verification.json")
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/thermal")
    parser.add_argument("--curves-only", action="store_true")
    args = parser.parse_args()
    run(args.output_dir, args.curves_only)
