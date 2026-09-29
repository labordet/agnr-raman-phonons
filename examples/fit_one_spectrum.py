"""Fit a small synthetic Raman spectrum with the retained Lorentzian shape.

This demonstrates the optimizer interface; it is not a paper measurement.
Run: python -m examples.fit_one_spectrum
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from raman_tools.fitting import PeakSpec, make_weight_vector, multiple_lorentzian
from raman_tools.quality_metrics import residual_r2_rmse


def example_fit(seed: int = 7) -> tuple[np.ndarray, float, float]:
    rng = np.random.default_rng(seed)
    x = np.linspace(200, 1800, 801)
    true = np.array([0.35, 310.0, 20.0, 1.0, 1590.0, 28.0])
    y = multiple_lorentzian(x, *true) + rng.normal(0, 0.01, len(x))
    peaks = [
        PeakSpec("RBLM", "RBLM", True, 310, 280, 340, 5, 60, 20),
        PeakSpec("G", "G", True, 1590, 1540, 1640, 5, 60, 28),
    ]
    options = {"peak_window_weighting": True, "peak_window_weight": 6,
               "peak_window_height_weighting": True, "peak_window_height_weight": 2}
    weights = make_weight_vector(x, y, peaks, options)
    start = np.array([0.3, 307, 25, 0.9, 1586, 32])
    lower = [0, 280, 5, 0, 1540, 5]
    upper = [2, 340, 60, 2, 1640, 60]
    result = least_squares(lambda pars: (multiple_lorentzian(x, *pars) - y) * weights,
                           start, bounds=(lower, upper))
    r2, rmse = residual_r2_rmse(y, multiple_lorentzian(x, *result.x))
    return result.x, r2, rmse


if __name__ == "__main__":
    parameters, r2, rmse = example_fit()
    print("[height, position cm^-1, FWHM cm^-1] for RBLM then G:")
    print(np.array2string(parameters, precision=4))
    print(f"Unweighted R-squared={r2:.5f}; RMSE={rmse:.5f}")
