"""Residual-bootstrap a synthetic Lorentzian demonstration fit.

The paper's full optimizer uses 100 successful refits and 3% start jitter.
This small example uses 20 local least-squares refits for quick exploration.
Run: python -m examples.bootstrap_one_fit
"""

import numpy as np
from scipy.optimize import least_squares

from raman_tools.fitting import single_lorentzian


def main() -> None:
    rng = np.random.default_rng(4)
    x = np.linspace(280, 340, 121)
    y = single_lorentzian(x, 0.8, 311.0, 18.0) + rng.normal(0, 0.02, len(x))
    bounds = ([0, 290, 5], [2, 330, 50])
    fit = least_squares(lambda p: single_lorentzian(x, *p) - y,
                        [0.7, 310, 20], bounds=bounds)
    model = single_lorentzian(x, *fit.x)
    residuals = y - model
    samples = []
    for _ in range(20):
        synthetic = model + rng.choice(residuals, size=len(residuals), replace=True)
        initial = np.clip(fit.x * (1 + rng.normal(0, 0.03, 3)),
                          np.array(bounds[0]) + 1e-8, np.array(bounds[1]) - 1e-8)
        result = least_squares(lambda p: single_lorentzian(x, *p) - synthetic,
                               initial, bounds=bounds)
        if result.success:
            samples.append(result.x)
    if len(samples) < 2:
        raise RuntimeError("Too few successful bootstrap refits")
    print("Best fit [height, position, FWHM]:", np.round(fit.x, 4))
    print("Bootstrap standard deviations:", np.round(np.std(samples, axis=0, ddof=1), 4))
    print("Successful refits:", len(samples))


if __name__ == "__main__":
    main()
