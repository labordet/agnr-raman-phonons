# SPDX-License-Identifier: CC-BY-4.0
"""Regenerate main-figure numerical components and SI thermal-path plots."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(os.environ["RAMAN_DATA_ROOT"]).resolve()
sys.dont_write_bytecode = True


def load_original(filename, module_name):
    path = Path(__file__).resolve().parents[3] / 'original_analysis/figures_main' / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked_output(path):
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


def compare_table(actual, expected_path, keys, columns, tolerance):
    import numpy as np
    import pandas as pd
    if not expected_path.exists():
        return {'verified': False, 'reason': f'Reference absent: {expected_path.relative_to(ROOT)}'}
    expected = pd.read_csv(expected_path)
    merged = actual.merge(expected, on=keys, suffixes=('_reproduced', '_accepted'), validate='one_to_one')
    differences = {}
    passes = len(actual) == len(expected) == len(merged)
    for col in columns:
        a = merged[col + '_reproduced'].to_numpy(float)
        b = merged[col + '_accepted'].to_numpy(float)
        diff = float(np.max(np.abs(a-b)))
        atol = tolerance.get(col, 1e-6)
        ok = bool(np.allclose(a, b, rtol=0, atol=atol, equal_nan=True))
        differences[col] = {'max_absolute_difference': diff, 'absolute_tolerance': atol, 'passed': ok}
        passes = passes and ok
    return {'verified': passes, 'actual_rows': len(actual), 'accepted_rows': len(expected), 'columns': differences}


def compare_exported_arrays(actual, expected_path, sort_columns):
    import numpy as np
    import pandas as pd
    expected = pd.read_csv(expected_path, float_precision='round_trip')
    if set(actual.columns) != set(expected.columns) or len(actual) != len(expected):
        return {'verified': False, 'reason': 'Released array shape or columns differ.'}
    actual = actual.sort_values(sort_columns).reset_index(drop=True)
    expected = expected.sort_values(sort_columns).reset_index(drop=True)
    checks = {}
    for col in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[col]):
            a, b = actual[col].to_numpy(float), expected[col].to_numpy(float)
            checks[col] = {'verified': bool(np.allclose(a, b, rtol=0, atol=1e-10, equal_nan=True)),
                           'max_absolute_difference': float(np.nanmax(np.abs(a-b)))}
        else:
            checks[col] = {'verified': actual[col].equals(expected[col])}
    return {'verified': all(c['verified'] for c in checks.values()), 'rows': len(actual), 'columns': checks}


def waterfall(out, report):
    import numpy as np
    import pandas as pd
    f2 = load_original('Figure_2_cleaned_spectra_waterfall.py', 'waterfall_original')
    inputs = ROOT / 'data/processed/temperature_mean'
    manifest = f2.load_manifest(inputs)
    selections = []
    interpolation = []
    hashes = {}
    array_comparisons = {}
    for family, label in f2.FAMILY_ROWS:
        records = manifest[
            manifest.family.eq(family) & manifest.sequence.isin(f2.SEQUENCE_FILTERS[family])
            & manifest.temperature_K.between(f2.TMIN_TARGET, f2.TMAX_TARGET)
        ].sort_values(['temperature_K', 'sequence', 'relative_output_txt'])
        reference_x = None
        for _, row in records.iterrows():
            path = inputs / Path(str(row.relative_output_txt).replace('\\', '/'))
            x, y = f2.read_txt_matrix(path)
            if reference_x is None:
                reference_x = x
            same_grid = len(x) == len(reference_x) and np.allclose(x, reference_x, rtol=0, atol=1e-8)
            if not same_grid:
                interpolation.append(str(path.relative_to(ROOT)))
            digest = hashlib.sha256(np.column_stack([x, y]).astype('<f8').tobytes()).hexdigest()
            hashes.setdefault(digest, []).append(str(path.relative_to(ROOT)))
            selections.append({
                'family': family, 'sequence': row.sequence, 'temperature_K': row.temperature_K,
                'input_file': path.relative_to(ROOT).as_posix(), 'spectral_points': len(x),
                'normalisation_max_abs_intensity': float(np.max(np.abs(y))),
                'Raman_grid_matches_family_reference': same_grid, 'array_sha256': digest,
            })
        row = f2.load_family_spectra(inputs, manifest, family)
        arrays = {'temperature_K': row['temps'], 'stack_index': row['stack_indices']}
        frames = []
        for i, (x, y) in enumerate(row['segments']):
            arrays[f'window_{i+1}_Raman_shift_cm-1'] = x
            arrays[f'window_{i+1}_normalised_intensity'] = y
            for j, temperature in enumerate(row['temps']):
                frames.append(pd.DataFrame({'temperature_K': temperature,
                    'stack_index': row['stack_indices'][j], 'window': i+1,
                    'Raman_shift_cm-1': x, 'normalised_intensity': y[j]}))
        array_comparisons[family] = compare_exported_arrays(
            pd.concat(frames, ignore_index=True),
            ROOT / f'data/derived/figure_data/Fig_2_{family}_spectral_windows.csv',
            ['temperature_K', 'window', 'Raman_shift_cm-1'])
        np.savez_compressed(out / f'Fig_2_{family}_plotted_arrays.npz', **arrays)
    selected = pd.DataFrame(selections)
    selected.to_csv(out / 'Fig_2_temperature_selection.csv', index=False)
    duplicate_temperature_rows = selected.duplicated(['family', 'temperature_K'], keep=False)
    report['Fig_2'] = {
        'selected_rows': len(selected),
        'by_family': selected.groupby('family').temperature_K.agg(['count','min','max']).to_dict('index'),
        'duplicate_temperature_rows': int(duplicate_temperature_rows.sum()),
        'grid_interpolation_inputs': interpolation,
        'identical_input_arrays': [v for v in hashes.values() if len(v)>1],
        'normalisation': 'maximum absolute intensity over each full input spectrum',
        'released_array_comparison': array_comparisons,
        'assembly_status': 'Numerical component regenerated; retained final insets, guides and composition are in figures/main/assembly/Fig_2_assembly.pptx.',
        'final_curve_equivalence': 'All 15 final panels agree at raster resolution with the traced generator output; see metadata/figure_2_artwork_comparison.json. This is not a binary-identity comparison.',
    }
    f2.OUTPUT_STEM = 'Fig_2_spectra_component'
    f2.plot_figure(inputs, out, ['pdf'], show=False)


def save_model_arrays(results, outpath):
    import pandas as pd
    frames=[]
    for peak, families in results.items():
        for family, d in families.items():
            frames.append(pd.DataFrame({
                'family': family, 'peak_id': peak, 'temperature_K': d['T_grid'],
                'omega_model_cm-1': d['omega0'] + d['delta_sum_raw'],
                'omega0_cm-1': d['omega0'], 'omega_at_zero_K_cm-1': d['omega0'] + d['A3'],
                'total_shift_from_zero_K_cm-1': d['delta_sum'],
                'thermoelastic_shift_cm-1': d['delta_te'],
                'anharmonic_shift_from_zero_K_cm-1': d['delta_anh'],
            }))
    pd.concat(frames, ignore_index=True).to_csv(outpath, index=False)


def replay_accepted_coefficients(module, results, accepted, tec):
    """Evaluate the original model with the retained publication coefficients."""
    import numpy as np
    from scipy.integrate import cumulative_trapezoid
    curves = {k:module.load_cte(tec,k) for k in ['gold','sapphire','cnt']}
    for peak, families in results.items():
        for family, d in families.items():
            r = accepted[accepted.family.eq(family) & accepted.peak_id.eq(peak)].iloc[0]
            t = d['T_grid']
            sub = curves[module.SAMPLE_CONFIG[family]['cte_kind']]
            cnt = curves['cnt']
            mismatch = cumulative_trapezoid(
                np.interp(t, sub['T (K)'], sub['alpha (1/K)'])
                - np.interp(t, cnt['T (K)'], cnt['alpha (1/K)']), t, initial=0.0,
            )
            w0, a3, gamma = float(r['omega0_cm-1']), float(r['A3_cm-1']), float(r.gamma_parallel)
            anh_raw = a3 * (1.0 + 2.0 * module.n_be(w0/2.0, t))
            te = w0 * (np.exp(-gamma*mismatch)-1.0)
            total_raw = anh_raw + te
            offset = float(total_raw[0])
            d.update(omega0=w0, omega0_stderr=float(r.omega0_stderr), A3=a3,
                     A3_err=float(r.A3_stderr), gamma=gamma, gamma_err=float(r.gamma_stderr),
                     delta_exp=d['w_exp']-w0-offset, delta_sum=total_raw-offset,
                     delta_te=te, delta_anh=anh_raw-offset, delta_sum_raw=total_raw,
                     delta_anh_raw=anh_raw, delta_offset0=offset,
                     rmse=float(r['RMSE_cm-1']), r2=float(r.R2))


def frequencies(out, report):
    import numpy as np
    import pandas as pd
    from PIL import Image
    f3 = load_original('Figure_3_temperature_shift_from_after.py', 'frequency_original')
    f3.INCLUDE_ERRORBARS = True
    f3.FIT_WITH_ERRORBARS = False
    f3.INCLUDE_MODEL_BAND = False
    f3.DISPLAY_FIGURES = False
    f3.apply_style()
    peak_file = ROOT / 'data/derived/peak_parameters.csv'
    tec = ROOT / 'data/reference_data/thermal_expansion'
    data, results, params = f3.build_results(peak_file, tec)
    params['source_csv'] = 'data/derived/peak_parameters.csv'
    data.to_csv(out / 'Fig_3_input_peak_positions.csv', index=False)
    params.to_csv(out / 'Fig_3_refit_validation_parameters.csv', index=False)
    columns = ['omega0_cm-1','omega0_stderr','omega0_n_points','A3_cm-1','A3_stderr','gamma_parallel','gamma_stderr','RMSE_cm-1','R2']
    report['Fig_3'] = {'refit_comparison':compare_table(
        params, ROOT / 'data/derived/main_frequency_model_parameters.csv',
        ['peak_id','family','sequence'], columns,
        {'omega0_cm-1':1e-10,'omega0_stderr':1e-10,'omega0_n_points':0,'A3_cm-1':1e-5,'A3_stderr':1e-5,'gamma_parallel':1e-5,'gamma_stderr':1e-5},
    )}
    accepted = pd.read_csv(ROOT / 'data/derived/main_frequency_model_parameters.csv')
    replay_accepted_coefficients(f3,results,accepted,tec)
    accepted.to_csv(out / 'Fig_3_model_parameters.csv', index=False)
    save_model_arrays(results, out / 'Fig_3_model_curves.csv')
    report['Fig_3']['released_model_array_comparison'] = compare_exported_arrays(
        pd.read_csv(out / 'Fig_3_model_curves.csv', float_precision='round_trip'),
        ROOT / 'data/derived/figure_data/Fig_3_model_curves.csv',
        ['family', 'peak_id', 'temperature_K'])
    report['Fig_3']['rendered_coefficients'] = 'Accepted publication coefficients replayed without alteration; refit comparison is separate and never replaces accepted values.'
    report['Fig_3']['accepted_coefficient_rows'] = len(accepted)
    report['Fig_3']['selected_peak_rows'] = len(data)
    report['Fig_3']['decomposition_axis'] = 'Absolute zero-K-referenced shift in cm^-1, as in final image; caption scaling description differs.'
    report['Fig_3']['assembly_status'] = 'Numerical component and legend regenerated; original displacement artwork and editable composition are retained in figures/main/assembly/Fig_3_assembly.pptx.'
    f3.make_combined_figure(results, f3.SAMPLES_PLOT, 'absolute', out / 'Fig_3_peak_positions_component.pdf', out / 'Fig_3_legend.pdf')
    f3.save_zero_temperature_omega0_comparison(results, out)
    reference = f3.omega0_comparison_table_from_results(results)
    report['Fig_4'] = compare_table(reference, ROOT / 'data/derived/reference_frequencies.csv', ['sample','peak_id'], ['omega0_cm-1','omega0_stderr','omega0_n_points'], {'omega0_cm-1':1e-10,'omega0_stderr':1e-10,'omega0_n_points':0})
    generated = out / 'FIG3_extrapolated_omega0_peak_positions_0K_white_background.png'
    canonical = ROOT / 'figures/main/Fig_4_reference_frequencies.png'
    with Image.open(generated) as a, Image.open(canonical) as b:
        report['Fig_4']['canonical_pixel_equality'] = a.size == b.size and np.array_equal(np.asarray(a.convert('RGBA')), np.asarray(b.convert('RGBA')))
        report['Fig_4']['generated_size'] = list(a.size)
        report['Fig_4']['canonical_size'] = list(b.size)
    generated.replace(out / 'Fig_4_reference_frequencies.png')
    (out / 'FIG3_extrapolated_omega0_peak_positions_0K.csv').replace(out / 'Fig_4_reference_frequencies.csv')
    for name in ['FIG3_extrapolated_omega0_peak_positions_0K.png','FIG3_extrapolated_omega0_peak_positions_0K_colored_bands.png']:
        (out / name).unlink()
    cycle_data, cycle_results, cycle_params = f3.build_results(peak_file, tec, sequence_filters=f3.ALIGNED_AU_8A_NESTED_SEQUENCE_FILTERS, sample_order=f3.ALIGNED_AU_8A_NESTED_SAMPLE_ORDER)
    cycle_params['source_csv'] = 'data/derived/peak_parameters.csv'
    cycle_data.to_csv(out / 'Fig_S8_input_peak_positions.csv', index=False)
    cycle_params.to_csv(out / 'Fig_S8_refit_validation_parameters.csv', index=False)
    cycle_accepted = pd.read_csv(ROOT / 'tables/supplementary/Table_S4_thermal_paths.csv')
    cycle_comparison = compare_table(cycle_params, ROOT / 'tables/supplementary/Table_S4_thermal_paths.csv', ['peak_id','family','sequence'],columns,{'omega0_cm-1':1e-10,'omega0_stderr':1e-10,'omega0_n_points':0,'A3_cm-1':1e-5,'A3_stderr':1e-5,'gamma_parallel':1e-5,'gamma_stderr':1e-5})
    replay_accepted_coefficients(f3,cycle_results,cycle_accepted,tec)
    cycle_accepted.to_csv(out / 'Fig_S8_model_parameters.csv', index=False)
    save_model_arrays(cycle_results, out / 'Fig_S8_model_curves.csv')
    cycle_array_comparison = compare_exported_arrays(
        pd.read_csv(out / 'Fig_S8_model_curves.csv', float_precision='round_trip'),
        ROOT / 'data/derived/figure_data/Fig_S8_model_curves.csv',
        ['family', 'peak_id', 'temperature_K'])
    f3.make_combined_figure(cycle_results, f3.ALIGNED_AU_8A_NESTED_SAMPLE_ORDER, 'absolute', out / 'Fig_S8_thermal_paths.pdf', out / 'Fig_S8_legend.pdf')
    report['Fig_S8'] = {'selected_peak_rows':len(cycle_data),'fit_parameter_rows':len(cycle_params),'refit_comparison':cycle_comparison,'rendered_coefficients':'Accepted Table S4 coefficients replayed without alteration.','verification':'Compare numerical parameters and slopes with Table S4 through the thermal-expansion verification script.','figure_assembly':'The plot PDF contains every visible element of final SI page 16, including component arrows. The separately emitted legend is an optional companion and is absent from the canonical figure; no manual assembly is required.'}
    report['Fig_S8']['released_model_array_comparison'] = cycle_array_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Separate output directory within release outputs/ or _verification/.')
    parser.add_argument('--only', choices=['all','waterfall','frequencies'], default='all')
    args = parser.parse_args()
    out = checked_output(args.output_dir)
    os.environ['MPLCONFIGDIR'] = str(out / '.matplotlib')
    os.environ['MPLBACKEND'] = 'Agg'
    report={}
    if args.only in ['all','waterfall']:
        waterfall(out,report)
    if args.only in ['all','frequencies']:
        frequencies(out,report)
    (out / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    failed = report.get('Fig_4', {}).get('verified') is False
    for figure in ('Fig_3', 'Fig_S8'):
        failed = failed or report.get(figure, {}).get('refit_comparison', {}).get('verified') is False
        failed = failed or report.get(figure, {}).get('released_model_array_comparison', {}).get('verified') is False
    failed = failed or any(not c['verified'] for c in report.get('Fig_2', {}).get('released_array_comparison', {}).values())
    if failed:
        raise SystemExit('Numerical comparison failed or a required comparison table is missing; see verification.json.')


if __name__ == '__main__':
    main()
