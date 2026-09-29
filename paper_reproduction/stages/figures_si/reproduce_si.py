# SPDX-License-Identifier: CC-BY-4.0
"""Reproduce Figures S1-S5, S7, S9 and S10 from the released numerical inputs."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
RELEASE = Path(os.environ["RAMAN_DATA_ROOT"]).resolve()
ORIGINAL = HERE.parents[2] / "original_analysis/figures_si"
METADATA = RELEASE / "metadata"
sys.dont_write_bytecode = True


def load_original(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ORIGINAL / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_spectrum_helpers():
    """Load the original text readers and data classes without the graphical app."""
    path = ORIGINAL / "paper_spike_before_after_plotter.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    names = {
        "SpectrumColumnData", "SpectrumData", "_read_sample_lines", "_split_line",
        "_numeric_fraction", "_guess_separator", "_guess_header",
        "_coerce_numeric_columns", "_read_table", "_choose_xy_columns",
        "read_spectrum_file",
    }
    body = [node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    prelude = ast.parse(
        "from __future__ import annotations\n"
        "from pathlib import Path\nfrom typing import *\n"
        "from dataclasses import dataclass\nimport re\nimport numpy as np\nimport pandas as pd\n"
    ).body
    module = types.ModuleType("paper_spike_before_after_plotter")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(ast.Module(body=prelude + body, type_ignores=[]), str(path), "exec"),
         module.__dict__)
    return module


def prepare_output_directory(path):
    output = Path(path).resolve()
    bases = [RELEASE / "outputs", RELEASE / "_verification"]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RELEASE / "outputs/si")
    parser.add_argument("--refit-linewidths", action="store_true",
                        help="Also compare a fresh linewidth fit with the retained coefficients.")
    args = parser.parse_args()
    output = prepare_output_directory(args.output)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    import pandas as pd
    from PIL import Image

    selections = json.loads((METADATA / "figure_selections.json").read_text(encoding="utf-8"))
    expected = json.loads((METADATA / "figure_array_fingerprints.json").read_text(encoding="utf-8"))
    expected_by_key = {(x["figure"], x["label"]): x for x in expected}
    report = {"numerical_arrays": [], "linewidth_fits": {}, "figures": []}

    def record_array(figure, label, x, *ys):
        values = np.column_stack([x, *ys]).astype("<f8")
        sha = hashlib.sha256(values.tobytes()).hexdigest()
        ref = expected_by_key[(figure, label)]
        exact = sha == ref["sha256"] and list(values.shape) == ref["shape"]
        report["numerical_arrays"].append({
            "figure": figure, "label": label, "shape": list(values.shape),
            "sha256": sha, "exact_float64_match_to_original_analysis": exact,
        })
        if not exact:
            raise ValueError(f"Numerical array differs: {figure} {label}")
        columns = {
            "S1": ["raman_shift_cm-1", "raw_display_intensity", "spike_cleaned_display_intensity"],
            "S2": ["raman_shift_cm-1", "spike_cleaned_intensity", "baseline_corrected_intensity", "ALS_baseline"],
            "S4": ["raman_shift_cm-1", "baseline_corrected_intensity", "total_Lorentzian_fit"],
            "S5": ["raman_shift_cm-1", "baseline_corrected_intensity", "G_Lorentzian_component"],
            "S7": ["raman_shift_cm-1", "normalised_intensity", "normalised_total_fit", "normalised_residual"],
        }[figure]
        pd.DataFrame(values, columns=columns).to_csv(
            output / f"{figure}_{label}_numerical_data.csv", index=False, float_format="%.17g")

    helper = load_spectrum_helpers()
    si = load_original("si_original", "prepare_final_si_figures.py")
    si.ROOT = RELEASE / "data/processed/spike_cleaned"
    si.ALS_ROOT = RELEASE / "data/processed/baseline_corrected"
    si.FITTED_ROOT = RELEASE / "data/derived/fits"
    si.AVERAGED_TXT_ROOT = RELEASE / "data/processed/temperature_mean"
    si.SECOND_PASS_TXT_ROOT = si.AVERAGED_TXT_ROOT
    si.OUTPUT_DIR = output
    examples = {}
    records = {}
    for spec, saved in zip(si.FAMILIES, selections["selections"]):
        if spec.fit_family != saved["family"]:
            raise ValueError("Representative-spectrum configuration order differs.")
        s2 = saved["S2"]
        examples[spec.fit_family] = si.ExampleSelection(
            family=spec.fit_family, sequence=s2["sequence"],
            temperature=s2["temperature_K"], clean_name=Path(s2["clean"]).name,
            curve_index=s2["y_index_zero_based"], x_start=s2["x_start"],
            x_end=s2["x_end"], als_lambda=6000 if "RO" in spec.fit_family else 1000000,
            als_p=0.0055, als_iterations=10,
        )
        s1 = saved["S1"]
        records[spec.key] = [types.SimpleNamespace(
            raw_path=RELEASE / s1["raw"], clean_path=RELEASE / s1["clean"],
            clean_rel_dir=s1["sequence"], temperature_label=f'{s1["temperature_K"]:g}K',
        )]
        x, raw, clean, rec, index, score = si.clearest_spike_pair(
            records[spec.key], s2["x_start"], s2["x_end"])
        if index != s1["y_index_zero_based"] or len(x) != s1["points"]:
            raise ValueError(f"S1 spectrum selection differs: {spec.fit_family}")
        base, scale = si._display_scale(clean)
        record_array("S1", spec.fit_family, x, (raw-base)/scale, (clean-base)/scale)
        xx, yy, zz = si.selected_curve_pair(
            RELEASE/s2["clean"], RELEASE/s2["corrected"], s2["y_index_zero_based"],
            s2["x_start"], s2["x_end"])
        record_array("S2", spec.fit_family, xx, yy, zz, yy-zz)
    spikes = load_original("si_spikes_original", "spike_removal_figure.py")
    spikes.OUTPUT_DIR = output
    matplotlib.rcdefaults()
    spikes.setup_style()
    spikes.plot_raw_vs_spike_removed(records, examples)
    matplotlib.rcdefaults()
    si.setup_style()
    si.plot_als_baseline_removal(examples)
    fits = si.load_fit_rows()
    for spec in si.FAMILIES:
        group = si.example_fit_group(fits, examples[spec.fit_family])
        x, y = si.load_group_spectrum(group)
        record_array("S4", spec.fit_family, x, y, si.fit_sum_safe(x, group))
    group = si.nearest_fit_group(fits, si.FAMILIES[0], 100)
    g = si.g_peak_row(group)
    if not abs(float(g["position_cm-1"])-selections["S5"]["G_position"]) < 1e-10:
        raise ValueError("S5 accepted G-peak centre differs.")
    if not abs(float(g["width_fwhm_cm-1"])-selections["S5"]["G_FWHM"]) < 1e-10:
        raise ValueError("S5 accepted G-peak FWHM differs.")
    x, y = si.load_group_spectrum(group)
    mask = (x >= 1548) & (x <= 1648)
    record_array("S5", "Aligned_Au_3A", x[mask], y[mask], si.lorentzian(
        x[mask], float(g["height"]), float(g["position_cm-1"]), float(g["width_fwhm_cm-1"])))
    for spec, temp in ((si.FAMILIES[0],80), (si.FAMILIES[1],80),
                       (si.FAMILIES[0],300), (si.FAMILIES[1],300)):
        group = si.nearest_fit_group(fits, spec, temp)
        x, y = si.load_group_spectrum(group)
        mask = (x >= 1550) & (x <= 1650)
        x, y = x[mask], y[mask]
        total = si.fit_sum_safe(x, group)
        scale = max(float(np.nanmax(total)), 1e-12)
        record_array("S7", spec.fit_family+"_"+str(temp),
                     x, y/scale, total/scale, (y-total)/scale)
    si.plot_global_lorentzian_fits(fits, examples)
    si.plot_g_peak_metrics(fits)
    residuals = load_original("si_residual_original", "g_residual_figure.py")
    residuals.FITTED_ROOT = si.FITTED_ROOT
    residuals.AVERAGED_TXT_ROOT = si.AVERAGED_TXT_ROOT
    residuals.SECOND_PASS_TXT_ROOT = si.AVERAGED_TXT_ROOT
    residuals.OUTPUT_DIR = output
    matplotlib.rcdefaults()
    residuals.setup_style()
    residuals.plot_four_panel_g_residuals(fits)

    windows = load_original("si_windows_original", "plot_si_fitting_windows_near_100K.py")
    windows.SPECTRA_ROOT = RELEASE / "data/processed/temperature_mean"
    windows.RESULTS_ROOT = RELEASE / "data/derived/fits"
    windows.AFTER_OUTPUT = windows.FINAL_FIGURE_OUTPUT = windows.SI_COMPILE_OUTPUT = output
    matplotlib.rcdefaults()
    windows.main()
    bounds = pd.read_csv(output / windows.MANIFEST_NAME)
    bounds_reference = pd.read_csv(METADATA / "fitting_windows.csv")
    numeric_bounds = ["temperature_K", "x_bound_min_cm-1", "x_bound_max_cm-1"]
    ids = ["internal_family_key", "thermal_sequence", "peak_id"]
    if not bounds[ids].equals(bounds_reference[ids]):
        raise ValueError("S3 fitting-window configuration, path or peak identifiers differ.")
    if not np.array_equal(bounds[numeric_bounds].to_numpy(), bounds_reference[numeric_bounds].to_numpy()):
        raise ValueError("S3 accepted temperatures or centre-position bounds differ.")
    report["fitting_windows"] = {"rows": len(bounds), "identifiers_and_bounds_exact": True}

    width = load_original("si_width_original", "Figure_width_fwhm_after.py")
    width.INPUT_CSV = RELEASE / "data/derived/peak_parameters.csv"
    width.OUTPUT_DIR = output
    reference = pd.read_csv(RELEASE / "data/derived/linewidth_model_parameters.csv")
    matplotlib.rcdefaults()
    actual = reference.copy()
    if args.refit_linewidths:
        width.main()
        actual = pd.read_csv(output / "WIDTH_after_fit_parameters.csv")
        actual.to_csv(output / "linewidth_refitted_parameters.csv", index=False)
    keys = ["figure", "family", "sequence", "peak_id", "model", "n_points"]
    if not actual[keys].equals(reference[keys]):
        raise ValueError("Linewidth-fit identifiers or point counts differ.")
    numbers = ["omega0_position_cm-1", "gamma0", "c3"]
    difference = abs(actual[numbers].to_numpy() - reference[numbers].to_numpy())
    curve_difference = 0.0
    for (_, fitted), (_, retained) in zip(actual.iterrows(), reference.iterrows()):
        temp = np.linspace(70.0, 305.0, 1000)
        curve_difference = max(curve_difference, float(np.max(np.abs(
            width.gamma_model(temp, fitted.gamma0, fitted.c3, fitted['omega0_position_cm-1'])
            - width.gamma_model(temp, retained.gamma0, retained.c3, retained['omega0_position_cm-1'])))))
    tolerance = 1e-4
    matched = bool(np.all(difference <= tolerance) and curve_difference <= 1e-6)
    report["linewidth_fits"] = {
        "fresh_fit_performed": args.refit_linewidths,
        "rows": len(actual), "identifiers_and_point_counts_exact": True,
        "maximum_absolute_parameter_difference": float(np.max(difference)),
        "maximum_curve_difference_70_305K_cm_inverse": curve_difference,
        "comparison_tolerance_cm_inverse": tolerance, "numerically_equivalent": matched,
    }
    if not matched:
        raise ValueError("Linewidth fit parameters or model curves exceed the numerical tolerance.")
    def retained_linewidth_curve(x, y, omega0, sigma):
        x_fit, y_fit, sigma_fit = width.prepared_fit_data(x, y, sigma)
        candidates = reference.loc[np.isclose(reference['omega0_position_cm-1'],
                                             omega0, rtol=0, atol=1e-10)]
        if candidates.empty:
            raise ValueError("No retained linewidth coefficient row for this parent frequency.")
        coefficients = candidates[['gamma0', 'c3']].drop_duplicates()
        if len(coefficients) != 1 or not (candidates.n_points == len(x_fit)).all():
            raise ValueError("Ambiguous retained linewidth coefficients or point-count mismatch.")
        gamma0, c3 = coefficients.iloc[0].to_numpy(dtype=float)
        t = np.linspace(float(np.min(x_fit)), float(np.max(x_fit)), 300)
        return t, width.gamma_model(t, gamma0, c3, omega0), (gamma0, c3)
    width.fit_three_phonon_curve = retained_linewidth_curve
    width.main()
    retained_render = pd.read_csv(output / "WIDTH_after_fit_parameters.csv")
    if not retained_render[keys].equals(reference[keys]):
        raise ValueError("Rendered linewidth-series identifiers or point counts differ.")
    if not np.array_equal(retained_render[numbers].to_numpy(), reference[numbers].to_numpy()):
        raise ValueError("Rendered linewidth coefficients differ from the accepted values.")
    report['linewidth_fits']['figures_use_retained_coefficients'] = True
    for number in (1,2,3,4,5,7,9,10):
        paths = list(output.glob(f"Fig_S{number}_*.png"))
        if len(paths) != 1:
            raise ValueError(f"Expected one generated Fig S{number}; found {paths}")
        generated = paths[0]
        canonical = RELEASE / f"figures/supplementary/Fig_S{number}.png"
        a = np.asarray(Image.open(generated).convert("RGB"))
        b = np.asarray(Image.open(canonical).convert("RGB"))
        report["figures"].append({
            "figure": f"Fig_S{number}", "generated": str(generated.relative_to(RELEASE)),
            "RGB_pixel_equal": bool(a.shape == b.shape and np.array_equal(a,b)),
            "generated_shape": list(a.shape), "canonical_shape": list(b.shape),
            "binary_equal": generated.read_bytes() == canonical.read_bytes(),
        })
    (output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
