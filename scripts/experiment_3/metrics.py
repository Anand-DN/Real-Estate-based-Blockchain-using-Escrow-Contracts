"""Metric aggregation for Experiment 3 (spec Section 10).

Uses the frozen V2.1 ``compute_metrics`` contract. Folds are never pooled as
independent observations: reporting is per-fold values plus fold mean and
sample SD (ddof=1). Regime comparison rests on paired per-fold differences.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from scripts.valuation_v2_1 import protocol as v21
from scripts.experiment_3 import frozen

METRIC_KEYS = list(v21.METRIC_KEYS)


def compute_fold_metrics(price_true: np.ndarray, predictions_log: np.ndarray) -> Dict[str, float]:
    return v21.compute_metrics(price_true, predictions_log)


def aggregate(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Fold mean and sample SD (ddof=1) for every metric key."""
    out: Dict[str, float] = {}
    for key in METRIC_KEYS:
        values = np.asarray([r[key] for r in rows], dtype=float)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_sd"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return out


def paired_differences(
    a_rows: Sequence[Dict[str, float]], b_rows: Sequence[Dict[str, float]]
) -> Dict[str, List[float]]:
    """Per-fold paired differences a - b for the primary metrics."""
    primary = ["R2_log", "MAE_log", "MedAPE_percent"]
    return {
        key: [
            float(a[key]) - float(b[key])
            for a, b in zip(a_rows, b_rows)
        ]
        for key in primary
    }


def load_a0_catboost_per_fold() -> Dict[str, List[Dict[str, float]]]:
    """Read the frozen Experiment 1 CatBoost per-fold metrics (read-only)."""
    if not frozen.PER_FOLD_METRICS_CSV.exists():
        raise FileNotFoundError(frozen.PER_FOLD_METRICS_CSV)
    frame = pd.read_csv(frozen.PER_FOLD_METRICS_CSV)
    frame = frame[frame["model"] == "CatBoost"]
    out: Dict[str, List[Dict[str, float]]] = {}
    for regime in frozen.REGIMES:
        block = frame[frame["regime"] == regime].sort_values("fold")
        out[regime] = [
            {key: float(row[key]) for key in METRIC_KEYS}
            for _, row in block.iterrows()
        ]
    return out
