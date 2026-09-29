"""Absolute comparison limits for quantities printed in the paper and SI.

Each nonlinear limit is at most half the final displayed unit. The limits
allow small SciPy optimizer differences while keeping measured quantities,
counts, temperature selections and source TEC values tightly constrained.
"""

TEC_TOLERANCES = {"T (K)": 5e-13, "alpha (1/K)": 1e-18}

MODEL_TOLERANCES = {
    "peak_target_cm-1": 0.0,
    "omega0_cm-1": 5e-3,  # 2 decimal places in the SI thermal-path table
    "omega0_stderr": 5e-4,
    "omega0_n_points": 0.0,
    "omega0_temperature_min_K": 0.0,
    "omega0_temperature_max_K": 0.0,
    "A3_cm-1": 5e-4,  # 3 decimal places
    "A3_stderr": 5e-4,
    "gamma_parallel": 5e-4,
    "gamma_stderr": 5e-4,
    "RMSE_cm-1": 5e-4,
    "R2": 5e-4,
    "temperature_min_K": 0.0,
    "temperature_max_K": 0.0,
    "temperature_low_K": 0.0,
    "temperature_high_K": 0.0,
    "measured_linear_slope_cm-1_K-1": 1e-8,  # direct linear regression
    "model_effective_slope_cm-1_K-1": 5e-6,  # 5 decimal places
    "relative_difference_percent": 0.05,  # 1 decimal place
    "delta_omega_TE_cm-1": 5e-4,  # 3 decimal places
    "delta_omega_anh_cm-1": 5e-4,
    "delta_omega_total_cm-1": 5e-4,
    "additivity_residual_cm-1": 1e-10,  # algebraic identity
    "TE_share_percent_signed": 0.05,  # 1 decimal place
    "anh_share_percent_signed": 0.05,
}
