# SPDX-License-Identifier: CC-BY-4.0
"""Reproduce the five Figure 1c reference spectra from seven measured inputs.

The parameters and input selections are recorded in
metadata/figure_1_spectrum_provenance.json. Published arrays are read only;
recomputed spectra and numerical comparisons are written to a separate folder.
"""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import os

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

ROOT = Path(os.environ["RAMAN_DATA_ROOT"]).resolve()
TOLERANCE = 5e-10


def output_directory(root, path):
    root = Path(root).resolve()
    output = Path(path).resolve()
    bases = [root / "outputs", root / "_verification"]
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


def als_baseline(y, lam, p, niter, clipping=False):
    """Evaluate the recorded asymmetric least-squares baseline procedure."""
    length = len(y)
    difference = diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(length, length - 2))
    weights = np.ones(length)
    for _ in range(niter):
        baseline = spsolve(diags(weights, 0) + lam * difference @ difference.T, weights * y)
        if clipping:
            baseline = np.minimum(baseline, y)
        weights = p * (y > baseline) + (1 - p) * (y <= baseline)
    return baseline


def process_raw_input(root, raw_record, settings):
    """Select the recorded column, subtract its baseline and normalize it."""
    path = Path(root) / raw_record['release_path']
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != raw_record['original_sha256']:
        raise ValueError(f'Raw source checksum differs: {raw_record["release_path"]}')
    measured = np.loadtxt(path)
    lower, upper = settings['roi_cm1']
    selected = (measured[:, 0] >= lower) & (measured[:, 0] <= upper)
    x = measured[selected, 0]
    y = measured[selected, raw_record['intensity_column_1based'] - 1]
    baseline = als_baseline(y, settings['lambda_value'], settings['asymmetry_p'],
                            settings['iterations'], settings['clipping'])
    corrected = y - baseline
    peak_window = (x >= 1595) & (x <= 1615)
    scale = corrected[peak_window].max()
    normalized = corrected / scale
    if not np.isfinite(normalized).all():
        raise ValueError(f'Non-finite normalized values in {path.name}')
    return {'x': x, 'raw': y, 'baseline': baseline, 'corrected': corrected,
            'normalized': normalized, 'normalization_denominator': float(scale)}


def reproduce_reference(root=ROOT, out=None):
    """Write recomputed values and verify them against the retained workbook export."""
    root = Path(root).resolve()
    out = output_directory(root, out or root / 'outputs/reference_spectra')
    metadata = json.loads((root / 'metadata/figure_1_spectrum_provenance.json').read_text(encoding='utf-8'))
    settings = {item['configuration']: item for item in metadata['processing']}
    with (root / metadata['exported_data']['release_csv']).open(encoding='utf-8', newline='') as stream:
        retained = list(csv.DictReader(stream))
    processed = {}
    raw_checks = []
    for record in metadata['raw_sources']:
        configuration, temperature = record['configuration'], record['temperature_K']
        result = process_raw_input(root, record, settings[configuration])
        processed[(configuration, temperature)] = result
        name = f'{configuration}_{temperature:g}K_processing.csv'
        values = np.column_stack([result[key] for key in ('x', 'raw', 'baseline', 'corrected', 'normalized')])
        np.savetxt(out / name, values, delimiter=',', fmt='%.17g', comments='',
                   header='raman_shift_cm-1,raw_intensity,ALS_baseline,baseline_corrected_intensity,normalized_intensity')
        raw_checks.append({'configuration': configuration, 'temperature_K': temperature,
                           'source': record['release_path'], 'raw_checksum_verified': True,
                           'output': name, 'retained_points': len(result['x']),
                           'normalization_denominator': result['normalization_denominator']})

    comparisons, output_rows = [], []
    for configuration in ('I', 'II', 'III', 'IV', 'V'):
        inputs = [processed[(configuration, temperature)]
                  for temperature in settings[configuration]['measured_temperatures_K']]
        x = inputs[0]['x']
        if not all(np.array_equal(x, item['x']) for item in inputs):
            raise ValueError(f'Input grids differ for {configuration}; no resampling is allowed here.')
        normalized = np.nanmedian([item['normalized'] for item in inputs], axis=0)
        expected = np.asarray([[float(row['raman_shift_cm-1']), float(row['normalized_intensity'])]
                               for row in retained if row['configuration'] == configuration])
        if not np.array_equal(x, expected[:, 0]):
            raise ValueError(f'Recomputed Raman grid differs for {configuration}.')
        error = float(np.max(np.abs(normalized - expected[:, 1])))
        comparisons.append({'configuration': configuration, 'rows': len(x),
                            'measured_temperatures_K': settings[configuration]['measured_temperatures_K'],
                            'max_absolute_intensity_difference': error,
                            'tolerance': TOLERANCE, 'passed': error <= TOLERANCE})
        output_rows.extend(zip([configuration] * len(x), x, normalized))
    with (out / 'reference_spectra_recomputed.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['configuration', 'raman_shift_cm-1', 'normalized_intensity'])
        writer.writerows(output_rows)
    verification = {'raw_inputs': raw_checks, 'spectra': comparisons,
                    'all_numerical_checks_passed': all(item['passed'] for item in comparisons),
                    'scope': 'Raw-to-reference numerical processing. The original graph display and final composition are retained separately.',
                    'figure_assembly': 'figures/main/assembly/Fig_1_assembly.pptx',
                    'plot_component': 'figures/main/assembly/Fig_1c_reference_spectra.png'}
    (out / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')
    if not verification['all_numerical_checks_passed']:
        raise ValueError('Reference-spectrum verification failed; inspect verification.json.')
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/reference_spectra')
    arguments = parser.parse_args()
    result = reproduce_reference(ROOT, arguments.output)
    maximum = max(row['max_absolute_intensity_difference'] for row in result['spectra'])
    print(f'Five reference spectra verified; maximum intensity difference {maximum:.3g}.')


if __name__ == '__main__':
    main()
