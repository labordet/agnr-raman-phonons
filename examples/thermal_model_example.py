"""Evaluate the retained three-phonon linewidth function.

Run: python -m examples.thermal_model_example
"""

import numpy as np

from thermal_models.klemens import gamma_model


if __name__ == "__main__":
    temperatures = np.array([70., 150., 300.])
    linewidths = gamma_model(temperatures, gamma0=10., c3=1., omega_cm1=1590.)
    for temperature, value in zip(temperatures, linewidths):
        print(f"{temperature:5.0f} K: {value:.5f} cm^-1")
