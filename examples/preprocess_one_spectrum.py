"""Recalculate the ALS treatment of one archived fitting spectrum.

Example: python -m examples.preprocess_one_spectrum --data /path/to/ZENODO_RELEASE
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from raman_tools.baseline import subtract_als


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    args = parser.parse_args()
    root = args.data.resolve()
    records = json.loads((root / "metadata/preprocessing_provenance.json").read_text(encoding="utf-8"))
    record = next(r for r in records if r["family"] == "Aligned_Au_3A" and r["temperature_K"] == 100)
    selected = np.loadtxt(root / record["selected_spike_cleaned_release"], skiprows=1)
    expected = np.loadtxt(root / record["baseline_corrected_release"], skiprows=1)
    inside = (selected[:, 0] >= record["crop_min"]) & (selected[:, 0] <= record["crop_max"])
    y = selected[inside, 1]
    corrected = subtract_als(y, record["als_lambda"], record["als_p"], record["als_iterations"])
    error = float(np.max(np.abs(corrected - expected[:, 1])))
    print(f"Raw source: {record['raw_release']}")
    print(f"Selected input: {record['selected_spike_cleaned_release']}")
    print(f"ALS lambda={record['als_lambda']:g}, p={record['als_p']}, iterations={record['als_iterations']}")
    print(f"Maximum absolute difference from archived corrected column: {error:.3g}")


if __name__ == "__main__":
    main()
