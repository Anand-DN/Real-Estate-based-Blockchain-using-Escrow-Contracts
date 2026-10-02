"""MILLOW Experiment 2 - conformal prediction.

Phase 0 pre-registration (§0 of the protocol draft):
  cal_frac = 0.20
  methods  = A, B1, B3, C1, C2, C3   (B2 and C4 EXCLUDED)
  levels   = 0.80, 0.90, 0.95

This package must not modify Experiment 1. Everything under
scripts/valuation_v2_1, artifacts/valuation/v2_1, models/valuation/v2_1,
backend/, and .chain/ is read-only input.
"""

__all__ = ["conformal", "build_calibration_folds"]
