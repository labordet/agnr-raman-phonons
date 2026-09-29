"""Tests of reusable calculations and fitting conventions."""

import unittest

import numpy as np

from raman_tools.baseline import als_baseline, subtract_als
from raman_tools.fitting import PeakSpec, make_weight_vector, single_lorentzian
from raman_tools.quality_metrics import residual_r2_rmse
from thermal_models.klemens import gamma_model


class ScientificFunctionTests(unittest.TestCase):
    def test_als_recovers_smooth_offset_without_clipping(self):
        x = np.linspace(0, 1, 200)
        background = 3 + 0.3 * x
        y = background + single_lorentzian(x, 2, 0.5, 0.03)
        corrected = subtract_als(y, 1e5, 0.0055, 10)
        self.assertEqual(corrected.shape, y.shape)
        self.assertLess(abs(als_baseline(y, 1e5, 0.0055, 10)[0] - background[0]), 0.1)
        self.assertGreater(corrected.max(), 1.5)

    def test_fwhm_parameter_is_actual_half_maximum_width(self):
        x = np.array([300., 310., 320.])
        y = single_lorentzian(x, 8., 310., 20.)
        np.testing.assert_allclose(y, [4., 8., 4.], rtol=0, atol=1e-12)

    def test_sqrt_weights_correspond_to_objective_weight_six_to_twelve(self):
        x = np.linspace(200., 400., 201)
        y = single_lorentzian(x, 1., 310., 20.)
        peak = PeakSpec("RBLM", "RBLM", True, 310, 280, 340, 5, 50, 20)
        w = make_weight_vector(x, y, [peak], {
            "peak_window_weighting": True, "peak_window_weight": 6.,
            "peak_window_height_weighting": True, "peak_window_height_weight": 2.,
            "peak_window_height_floor_percentile": 10.,
        })
        self.assertEqual(w[0] ** 2, 1.)
        self.assertGreaterEqual(np.min(w[(x >= 280) & (x <= 340)] ** 2), 6. - 1e-12)
        self.assertAlmostEqual(np.max(w ** 2), 12.)

    def test_reported_quality_uses_unweighted_residuals(self):
        measured = np.array([1., 2., 3.])
        fitted = np.array([1., 2., 2.])
        r2, rmse = residual_r2_rmse(measured, fitted)
        self.assertAlmostEqual(r2, 0.5)
        self.assertAlmostEqual(rmse, 1 / np.sqrt(3))

    def test_klemens_linewidth_increases_with_temperature(self):
        temperatures = np.array([70., 80., 160., 300.])
        widths = gamma_model(temperatures, 10., 2., 1590.)
        self.assertTrue(np.isfinite(widths).all())
        self.assertTrue(np.all(np.diff(widths) >= 0))


if __name__ == "__main__":
    unittest.main()
