"""Spectral diagnostics used for inspecting, not overriding, manual selection."""

from __future__ import annotations

import numpy as np


def residual_r2_rmse(measured: np.ndarray, fitted: np.ndarray) -> tuple[float, float]:
    """Unweighted full-window R² and RMSE used for reported fit quality."""
    y = np.asarray(measured, dtype=float)
    fit = np.asarray(fitted, dtype=float)
    if y.shape != fit.shape or y.ndim != 1 or not np.isfinite(y).all() or not np.isfinite(fit).all():
        raise ValueError("Require matching finite one-dimensional arrays")
    residual = y - fit
    sst = np.sum((y - np.mean(y)) ** 2)
    r2 = float(1 - np.sum(residual ** 2) / sst) if sst > 0 else float("nan")
    return r2, float(np.sqrt(np.mean(residual ** 2)))
