"""Model comparisons follow the precision reported in the SI tables."""

import unittest

import pandas as pd

from paper_reproduction.numerical_tolerances import MODEL_TOLERANCES, TEC_TOLERANCES
from paper_reproduction.stages.thermal.reproduce_thermal import compare_table


class ThermalToleranceTests(unittest.TestCase):
    def test_optimizer_difference_below_display_precision(self):
        original = pd.DataFrame({"A3_cm-1": [-2.0], "gamma_parallel": [1.0],
                                 "model_effective_slope_cm-1_K-1": [-0.01],
                                 "TE_share_percent_signed": [25.0]})
        other = original.copy()
        other.iloc[0] += [1.1e-5, 3.9e-5, 4.6e-8, 0.0075]
        self.assertTrue(compare_table(original, other, MODEL_TOLERANCES)["passed"])
        other.loc[0, "A3_cm-1"] += 1e-3
        self.assertFalse(compare_table(original, other, MODEL_TOLERANCES)["passed"])

    def test_tec_remains_strict(self):
        original = pd.DataFrame({"T (K)": [100.0], "alpha (1/K)": [1e-6]})
        other = original.copy()
        other.loc[0, "alpha (1/K)"] += 1e-14
        self.assertFalse(compare_table(original, other, TEC_TOLERANCES)["passed"])


if __name__ == "__main__":
    unittest.main()
