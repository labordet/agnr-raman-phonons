"""Agreement tests using an optional unpacked Zenodo data archive."""

import json
import os
from argparse import Namespace
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from raman_tools.baseline import subtract_als
from raman_tools.selection import (
    REJECTED_MEAN_TEMPERATURES_K, acquisition_selection, accepted_mean_temperatures,
)


DATA = Path(os.environ["ZENODO_DATA_ROOT"]).resolve() if os.environ.get("ZENODO_DATA_ROOT") else None


@unittest.skipUnless(DATA and DATA.is_dir(), "Set ZENODO_DATA_ROOT to run archive agreement tests")
class DataAgreementTests(unittest.TestCase):
    def test_recorded_configuration_iii_selection(self):
        selected = acquisition_selection(DATA)
        self.assertEqual(len(selected), 28)
        self.assertEqual(int(selected.included_for_all_trend_peaks.sum()), 20)
        temperatures = accepted_mean_temperatures(DATA)
        self.assertEqual(len(temperatures), 16)
        self.assertTrue(set(REJECTED_MEAN_TEMPERATURES_K).isdisjoint(temperatures))

    def test_one_real_als_chain_matches_archived_values(self):
        records = json.loads((DATA / "metadata/preprocessing_provenance.json").read_text(encoding="utf-8"))
        rec = next(row for row in records if row["family"] == "Aligned_Au_3A"
                   and row["temperature_K"] == 100.)
        selected = np.loadtxt(DATA / rec["selected_spike_cleaned_release"], skiprows=1)
        expected = np.loadtxt(DATA / rec["baseline_corrected_release"], skiprows=1)
        mask = (selected[:, 0] >= rec["crop_min"]) & (selected[:, 0] <= rec["crop_max"])
        corrected = subtract_als(selected[mask, 1], rec["als_lambda"], rec["als_p"],
                                 rec["als_iterations"])
        np.testing.assert_allclose(corrected, expected[:, 1], rtol=0, atol=1e-7)

    def test_accepted_peak_modes_and_local_refits(self):
        peaks = pd.read_csv(DATA / "data/derived/peak_parameters.csv")
        self.assertEqual(len(peaks), 462)
        self.assertEqual(set(peaks.peak_id), {"RBLM", "D", "G"})
        refits = pd.read_csv(DATA / "metadata/original_records/unaligned_au_local_refit_replacements.csv")
        retained = refits.loc[refits.temperature_K.isin(accepted_mean_temperatures(DATA))]
        self.assertEqual(len(retained), 15)
        self.assertEqual(set(retained.peak_id), {"RBLM"})
        for _, row in retained.iterrows():
            matched = peaks.loc[(peaks.family == row["family"])
                                & (peaks.temperature_K == row["temperature_K"])
                                & (peaks.peak_id == "RBLM")]
            self.assertEqual(len(matched), 1)
            self.assertAlmostEqual(float(matched.iloc[0]["position_cm-1"]),
                                   float(row["position_cm-1"]), delta=0.02)

    def test_local_rblm_refit_keeps_other_components_fixed(self):
        """Recompute one retained local optimum without a new global campaign."""
        from original_analysis.fitting.Gamma_T_Comparison import trust_region_refit_single_peak

        rows = pd.read_csv(DATA / "data/derived/fits/MIRA_Au_unaligned_8A"
                           / "MIRA_Au_unaligned_8A_long_results.csv")
        target = rows.loc[(rows.temperature_K == 105.) & (rows.peak_id == "RBLM")].iloc[0]
        replacement, meta = trust_region_refit_single_peak(
            target, rows, DATA / "data/processed/temperature_mean",
            DATA / "metadata/original_records/fitting_bounds_primary.json",
            Namespace(second_pass_output_root=None), "RBLM",
            float(target.x_bound_min), float(target.x_bound_max),
            float(target.width_bound_min), float(target.width_bound_max),
            bootstrap_runs=10, random_seed=12345,
        )
        self.assertTrue(meta["success"])
        self.assertEqual(meta["bootstrap_success"], 10)
        self.assertEqual(len(replacement), 1)
        self.assertAlmostEqual(float(replacement.iloc[0]["position_cm-1"]),
                               float(target["position_cm-1"]), delta=1e-4)
        self.assertAlmostEqual(float(replacement.iloc[0]["width_fwhm_cm-1"]),
                               float(target["width_fwhm_cm-1"]), delta=1e-3)


if __name__ == "__main__":
    unittest.main()
