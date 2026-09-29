"""Access the retained Au, CNT and sapphire thermal-expansion continuations."""

from original_analysis.thermal.build_fixed_tec_inputs import (
    build_sapphire_curve,
    build_gold_curve,
    build_cnt_curve,
)

__all__ = ["build_sapphire_curve", "build_gold_curve", "build_cnt_curve"]
