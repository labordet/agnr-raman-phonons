"""Public access to the retained full-window Raman fitting functions.

The historical fitter defines the Lorentzian shape, peak-window weighting,
bounded fit and residual bootstrap. Re-exporting those functions avoids a
second implementation with subtly different numerical behavior.
"""

from original_analysis.fitting.cluster_fitting_current_euler import (
    PeakSpec,
    single_lorentzian,
    multiple_lorentzian,
    make_weight_vector,
    fit_spectrum_column,
    bootstrap_parameter_std,
)

__all__ = [
    "PeakSpec", "single_lorentzian", "multiple_lorentzian",
    "make_weight_vector", "fit_spectrum_column", "bootstrap_parameter_std",
]
