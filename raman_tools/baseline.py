"""Asymmetric least-squares (ALS) baseline subtraction.

The matrix orientation and update rule match the released paper processing.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def als_baseline(y: np.ndarray, lam: float, p: float, iterations: int = 10) -> np.ndarray:
    """Return the ALS baseline of one intensity vector.

    ``lam`` controls smoothness and ``p`` penalizes points above the baseline.
    The Raman-shift grid must already be cropped to the desired interval.
    """
    values = np.asarray(y, dtype=float)
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("y must be a finite one-dimensional spectrum")
    if lam <= 0 or not 0 < p < 1 or iterations < 1:
        raise ValueError("Require lam > 0, 0 < p < 1, iterations >= 1")
    n = len(values)
    difference = diags([np.ones(n), -2 * np.ones(n), np.ones(n)],
                       [0, -1, -2], shape=(n, n - 2), format="csc")
    penalty = (lam * difference.dot(difference.T)).tocsc()
    weights = np.ones(n)
    for _ in range(iterations):
        z = spsolve(diags(weights, 0, shape=(n, n), format="csc") + penalty,
                    weights * values)
        weights = p * (values > z) + (1 - p) * (values <= z)
    return np.asarray(z)


def subtract_als(y: np.ndarray, lam: float, p: float, iterations: int = 10) -> np.ndarray:
    """Subtract the ALS baseline without clipping negative residuals."""
    return np.asarray(y, dtype=float) - als_baseline(y, lam, p, iterations)
