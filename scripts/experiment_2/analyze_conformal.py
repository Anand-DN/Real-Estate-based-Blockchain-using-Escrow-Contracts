"""Experiment 2 Phase 5: aggregation, subgroup analysis, decisions, figures.

PHASE 5 IS ANALYSIS ONLY.

This module is a pure consumer of the frozen Phase 4 interval artifacts. It
reads:

    artifacts/valuation/v2_2/conformal/conformal_scoring_manifest.json
    artifacts/valuation/v2_2/conformal/intervals/intervals_<regime>_fold<k>.csv
    artifacts/valuation/v2_2/conformal/scores/quantiles_<regime>_fold<k>.csv
    artifacts/valuation/v2_2/conformal/nested_models_manifest.json   (read only)

and computes coverage, coverage error, sharpness, the Gneiting-Raftery interval
score, paired random-vs-location-grouped deltas, city and price-band subgroup
coverage, the corrected descriptive F5 monotonicity characterisation, the
pre-registered F1-F7 decision criteria, and the twelve protocol figures.

It trains NOTHING, loads NO model, regenerates NO fold, and writes only under:

    artifacts/valuation/v2_2/conformal/analysis/

NO VALIDITY CLAIM IS MADE. The location_grouped coverage numbers are descriptive
measurements under distribution shift, never "coverage is X%" guarantees. No
composite score is computed, no method is ranked, and no method is selected.

METHOD SET AND LEVELS are inherited unchanged from the frozen Phase 1 module:
A, B1, B3, C1, C2, C3 at nominal 0.80 / 0.90 / 0.95.

INFINITE INTERVALS. A Monodrian stratum with no calibration support (or with
k > n) receives (-inf, +inf) with fallback_used=True. Such a row is genuinely
covered by its interval; it is counted as covered, and its count is always
reported separately (n_fallback, n_infinite). Width and interval-score metrics
are computed on bounded rows only and are explicitly scoped. Infinity is never
silently replaced with a finite number.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_2 import build_calibration_folds as B  # noqa: E402
from scripts.experiment_2 import conformal as C  # noqa: E402
from scripts.experiment_2 import run_conformal as R  # noqa: E402
from scripts.experiment_2 import train_nested_models as T  # noqa: E402
from scripts.valuation_v2_1 import protocol as P  # noqa: E402


# ============================================================
# CONTRACT
# ============================================================

ANALYSIS_DIRNAME = "analysis"
FIGURES_DIRNAME = "figures"

FOLD_METRICS_CSV = "fold_metrics.csv"
SUMMARY_METRICS_CSV = "summary_metrics.csv"
REGIME_SHIFT_CSV = "regime_shift_coverage.csv"
SUBGROUP_METRICS_CSV = "subgroup_metrics.csv"
DECISION_TABLE_CSV = "decision_table.csv"
ANALYSIS_MANIFEST_JSON = "analysis_manifest.json"
PROTECTED_BEFORE_NAME = "protected_state_before_phase5.json"
PROTECTED_AFTER_NAME = "protected_state_after_phase5.json"

#: The exact Experiment 2 protocol document this analysis is bound to.
PROTOCOL_DOC_PATH = (
    ROOT / "docs" / "research" / "EXPERIMENT_2_CONFORMAL_PROTOCOL_DRAFT.md"
)

RESULT_TABLES: Tuple[str, ...] = (
    FOLD_METRICS_CSV,
    SUMMARY_METRICS_CSV,
    REGIME_SHIFT_CSV,
    SUBGROUP_METRICS_CSV,
    DECISION_TABLE_CSV,
)

NOMINAL_LEVELS: Tuple[float, ...] = C.NOMINAL_LEVELS
METHODS: Tuple[str, ...] = C.METHODS
REGIMES: Tuple[str, ...] = B.REGIMES

#: F2/F4 absolute coverage-error decision threshold (protocol Section 16 F2).
COVERAGE_ERROR_THRESHOLD = 0.05

#: The approved F3 gate, inherited from the frozen Phase 1 module.
F3_GATE_LEVEL = C.F3_GATE_LEVEL
F3_GATE_MAX_MEDIAN_RELATIVE_WIDTH = C.F3_GATE_MAX_MEDIAN_RELATIVE_WIDTH

#: Pre-declared suppression rule (protocol Section 12.1).
MIN_SUBGROUP_N = C.MIN_SUBGROUP_N

#: Methods whose interval width is constant across the WHOLE test set by
#: construction. Resolution 8.2: C1/C2 are per-stratum constant, so they are
#: excluded from the global F0 homogeneity check and validated separately by
#: within-stratum invariance.
GLOBAL_HOMOGENEOUS_METHODS: Tuple[str, ...] = ("A", "B3")

#: Phase 4 persisted intervals with float_format="%.10g", so a mathematically
#: constant width is only recoverable to ~1e-9 relative after reload. F0 and
#: the C1/C2 within-stratum check use this serialization tolerance. It is an
#: implementation tolerance, not a statistical threshold (Resolution 8.2).
F0_RELATIVE_TOLERANCE = 1e-6

SUBGROUP_TYPES: Tuple[str, ...] = (
    "source_city",
    "true_price_band",
    "predicted_price_band",
)

#: Metric -> space, for the long-format summary table (mirrors Experiment 1).
SUMMARY_METRIC_SPACES: Dict[str, str] = {
    "n_evaluated": "count",
    "n_bounded": "count",
    "n_infinite": "count",
    "n_fallback": "count",
    "empirical_coverage": "probability",
    "coverage_error": "probability",
    "abs_coverage_error": "probability",
    "coverage_cp_lower": "probability",
    "coverage_cp_upper": "probability",
    "miscoverage_low": "probability",
    "miscoverage_high": "probability",
    "mean_width_log": "log1p(price)",
    "median_width_log": "log1p(price)",
    "sd_width_log": "log1p(price)",
    "min_width_log": "log1p(price)",
    "max_width_log": "log1p(price)",
    "mean_width_inr": "INR",
    "median_width_inr": "INR",
    "mean_relative_width_log": "ratio",
    "median_relative_width_log": "ratio",
    "mean_relative_width_inr": "ratio",
    "median_relative_width_inr": "ratio",
    "interval_score_log": "log1p(price)",
    "median_interval_score_log": "log1p(price)",
}

#: Cell metric keys carried into the fold table and the summary table.
CELL_METRIC_KEYS: Tuple[str, ...] = (
    "n_evaluated",
    "n_bounded",
    "n_infinite",
    "n_fallback",
    "empirical_coverage",
    "coverage_error",
    "abs_coverage_error",
    "coverage_cp_lower",
    "coverage_cp_upper",
    "miscoverage_low",
    "miscoverage_high",
    "mean_width_log",
    "median_width_log",
    "sd_width_log",
    "min_width_log",
    "max_width_log",
    "mean_width_inr",
    "median_width_inr",
    "mean_relative_width_log",
    "median_relative_width_log",
    "mean_relative_width_inr",
    "median_relative_width_inr",
    "interval_score_log",
    "median_interval_score_log",
)

FOLD_METRICS_COLUMNS: Tuple[str, ...] = (
    "regime",
    "fold",
    "method",
    "nominal",
    "n_evaluated",
    "n_bounded",
    "n_infinite",
    "n_fallback",
    "empirical_coverage",
    "coverage_error",
    "abs_coverage_error",
    "coverage_cp_lower",
    "coverage_cp_upper",
    "coverage_significant",
    "miscoverage_low",
    "miscoverage_high",
    "n_miscoverage_low",
    "n_miscoverage_high",
    "mean_width_log",
    "median_width_log",
    "sd_width_log",
    "min_width_log",
    "max_width_log",
    "mean_width_inr",
    "median_width_inr",
    "mean_relative_width_log",
    "median_relative_width_log",
    "mean_relative_width_inr",
    "median_relative_width_inr",
    "interval_score_log",
    "median_interval_score_log",
    "width_is_constant",
    "width_metrics_scope",
)

SUBGROUP_METRICS_COLUMNS: Tuple[str, ...] = (
    "regime",
    "fold",
    "method",
    "nominal",
    "subgroup_type",
    "subgroup",
    "n_evaluated",
    "n_covered",
    "empirical_coverage",
    "coverage_error",
    "abs_coverage_error",
    "coverage_cp_lower",
    "coverage_cp_upper",
    "coverage_significant",
    "coverage_reliable",
    "min_n_required",
)

REGIME_SHIFT_COLUMNS: Tuple[str, ...] = (
    "method",
    "nominal",
    "metric",
    "n_pairs",
    "mean_random",
    "mean_grouped",
    "mean_delta",
    "sd_delta_ddof1",
    "delta_fold1",
    "delta_fold2",
    "delta_fold3",
    "delta_fold4",
    "delta_fold5",
    "sign_convention",
)

DECISION_COLUMNS: Tuple[str, ...] = (
    "criterion",
    "scope",
    "metric",
    "value",
    "threshold",
    "outcome",
    "notes",
)

FIG_DPI = 200
FIGSIZE = (11.0, 6.4)

FIGURE_STEMS: Tuple[str, ...] = (
    "01_coverage_vs_nominal_random",
    "02_coverage_vs_nominal_grouped",
    "03_coverage_vs_nominal_both_regimes",
    "04_mean_width_vs_nominal",
    "05_regime_shift_coverage_delta",
    "06_interval_score_vs_nominal",
    "07_city_coverage",
    "08_price_band_coverage",
    "09_residual_vs_fitted",
    "10_qq_residuals",
    "11_scale_location",
    "12_calibration_transfer",
)

FIGURE_LABELS: Dict[str, str] = {
    "A": "A (split)",
    "B1": "B1 (normalized)",
    "B3": "B3 (control)",
    "C1": "C1 (city Mondrian)",
    "C2": "C2 (predicted band)",
    "C3": "C3 (grouped, secondary)",
}

METHOD_COLORS: Dict[str, str] = {
    "A": "#1f77b4",
    "B1": "#ff7f0e",
    "B3": "#2ca02c",
    "C1": "#d62728",
    "C2": "#9467bd",
    "C3": "#8c564b",
}

REGIME_COLORS: Dict[str, str] = {
    "random": "#1f77b4",
    "location_grouped": "#d62728",
}

REGIME_LABELS: Dict[str, str] = {
    "random": "random (positive control)",
    "location_grouped": "location_grouped (shift stress test)",
}


# ============================================================
# PATH HELPERS
# ============================================================


def analysis_dir(artifact_root: Path) -> Path:
    return artifact_root / ANALYSIS_DIRNAME


def figures_dir(analysis_root: Path) -> Path:
    return analysis_root / FIGURES_DIRNAME


def _rel(path: Path) -> str:
    return T.rel_to_root(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


# ============================================================
# EXTENDED PROTECTED-STATE SNAPSHOT (Phase 5)
# ============================================================

#: Trees whose bytes must not change during Phase 5. Phase 2/3/4 artifacts and
#: the Experiment 1 / V2.1 trees are all protected. The analysis output subtree
#: is excluded because Phase 5 writes there.
_PHASE5_PROTECTED_TREES: Tuple[str, ...] = (
    "models/valuation/v2_1",
    "artifacts/valuation/v2_1",
    "models/valuation/v2_2",
    "artifacts/valuation/v2_2/conformal",
)

_PHASE5_PROTECTED_FILES: Tuple[str, ...] = T.PROTECTED_FILES + (
    ".gitignore",
    ".gitattributes",
)


def _analysis_relative_prefixes() -> Tuple[str, ...]:
    base = _rel(T.ARTIFACT_ROOT / ANALYSIS_DIRNAME)
    return (base,)


def _iter_phase5_protected_files() -> List[Path]:
    skip = _analysis_relative_prefixes()
    files: List[Path] = []
    for rel in _PHASE5_PROTECTED_TREES:
        base = ROOT / rel
        if not base.exists():
            raise FileNotFoundError(f"protected tree missing: {rel}")
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            relpath = _rel(path)
            if any(relpath == s or relpath.startswith(s + "/") for s in skip):
                continue
            files.append(path)
    for rel in _PHASE5_PROTECTED_FILES:
        path = ROOT / rel
        if not path.exists():
            continue
        files.append(path)
    return files


def snapshot_protected_state_phase5() -> Dict[str, Any]:
    """Hash every protected file plus git HEAD / tags, as in Phases 3-4."""
    files: Dict[str, str] = {}
    for path in _iter_phase5_protected_files():
        rel = _rel(path)
        files[rel] = T.sha256_of(path)
    payload = json.dumps(files, sort_keys=True).encode("utf-8")
    tags = [t for t in T._git("tag", "--list").splitlines() if t]
    return {
        "git_head": T._git("rev-parse", "HEAD"),
        "git_tags": tags,
        "n_files": len(files),
        "files": files,
        "tree_digest": T.sha256_of_bytes(payload),
    }


# ============================================================
# INPUT LOADING
# ============================================================


def load_phase4_manifest(artifact_root: Path = T.ARTIFACT_ROOT) -> Dict[str, Any]:
    path = artifact_root / R.MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"Phase 4 manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("protected_state_clean") is not True:
        raise RuntimeError("Phase 4 manifest reports a protected-state violation")
    return manifest


def load_quantiles(artifact_root: Path, regime: str, fold_id: int) -> Dict[str, Any]:
    path = R.quantiles_json_path(artifact_root, regime, fold_id)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_intervals(artifact_root: Path, regime: str, fold_id: int) -> pd.DataFrame:
    path = R.intervals_csv_path(artifact_root, regime, fold_id)
    df = pd.read_csv(path, low_memory=False)
    if list(df.columns) != list(R.INTERVAL_COLUMNS):
        raise RuntimeError(f"interval column drift in {path}")
    return df


def cell_lookup(qdoc: Mapping[str, Any]) -> Dict[Tuple[str, float], Dict[str, Any]]:
    return {(c["method"], float(c["nominal"])): c for c in qdoc["cells"]}


# ============================================================
# CORE CELL METRICS (infinite-aware)
# ============================================================


def coverage_finite_aware(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal: float,
    *,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Exact coverage with infinite endpoints handled explicitly.

    An infinite endpoint is a legitimate endpoint (lower=-inf / upper=+inf), not
    a missing value. A row whose interval is (-inf, +inf) is covered. Counts of
    such rows are reported separately and never silently converted to finite.
    """
    yt = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    up = np.asarray(upper, dtype=float)
    if not (yt.shape == lo.shape == up.shape):
        raise ValueError("coverage inputs must share one shape")
    n = int(yt.size)
    if n == 0:
        raise ValueError("coverage requires at least one row")

    bounded = np.isfinite(lo) & np.isfinite(up)
    n_bounded = int(bounded.sum())
    n_infinite = int(n - n_bounded)

    inside = (yt >= lo) & (yt <= up)
    hits = int(inside.sum())
    below = int((yt < lo).sum())
    above = int((yt > up).sum())

    cp_low, cp_high = C._clopper_pearson(hits, n, confidence)
    empirical = hits / n
    error = empirical - float(nominal)
    return {
        "n_evaluated": n,
        "n_bounded": n_bounded,
        "n_infinite": n_infinite,
        "n_covered": hits,
        "empirical_coverage": float(empirical),
        "coverage_error": float(error),
        "abs_coverage_error": float(abs(error)),
        "coverage_cp_lower": float(cp_low),
        "coverage_cp_upper": float(cp_high),
        "coverage_significant": bool(cp_low > nominal or cp_high < nominal),
        "miscoverage_low": float(below / n),
        "miscoverage_high": float(above / n),
        "n_miscoverage_low": below,
        "n_miscoverage_high": above,
    }


def compute_cell_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal: float,
    method: str,
) -> Dict[str, Any]:
    """Coverage + sharpness for one (method, regime, fold, level) marginal cell.

    Sharpness (width, interval score, relative width) is computed on bounded
    rows only when any endpoint is infinite, and width_metrics_scope records
    that explicitly. Coverage uses all rows, with infinite intervals counted as
    covering (they do).
    """
    cov = coverage_finite_aware(y_true, lower, upper, nominal)
    bounded = np.isfinite(lower) & np.isfinite(upper)
    n_bounded = int(bounded.sum())

    if n_bounded >= 1:
        sh = C.sharpness_metrics(
            y_true[bounded],
            y_pred[bounded],
            lower[bounded],
            upper[bounded],
            1.0 - float(nominal),
            method=method,
        )
        scope = "all_rows" if cov["n_infinite"] == 0 else "bounded_rows_only"
    else:
        sh = {
            "mean_width_log": float("nan"),
            "median_width_log": float("nan"),
            "sd_width_log": float("nan"),
            "min_width_log": float("nan"),
            "max_width_log": float("nan"),
            "mean_width_inr": float("nan"),
            "median_width_inr": float("nan"),
            "mean_relative_width_log": float("nan"),
            "median_relative_width_log": float("nan"),
            "mean_relative_width_inr": float("nan"),
            "median_relative_width_inr": float("nan"),
            "interval_score_log": float("nan"),
            "median_interval_score_log": float("nan"),
        }
        scope = "bounded_rows_only"

    return {
        **cov,
        "mean_width_log": sh["mean_width_log"],
        "median_width_log": sh["median_width_log"],
        "sd_width_log": sh["sd_width_log"],
        "min_width_log": sh["min_width_log"],
        "max_width_log": sh["max_width_log"],
        "mean_width_inr": sh["mean_width_inr"],
        "median_width_inr": sh["median_width_inr"],
        "mean_relative_width_log": sh["mean_relative_width_log"],
        "median_relative_width_log": sh["median_relative_width_log"],
        "mean_relative_width_inr": sh["mean_relative_width_inr"],
        "median_relative_width_inr": sh["median_relative_width_inr"],
        "interval_score_log": sh["interval_score_log"],
        "median_interval_score_log": sh["median_interval_score_log"],
        "width_is_constant": bool(method in C.HOMOGENEOUS_METHODS),
        "width_metrics_scope": scope,
    }


# ============================================================
# PER-FOLD PASS
# ============================================================


def _stable_subgroup_types(labels: np.ndarray) -> List[Any]:
    return C._unique_stable(labels)


def build_fold_rows(
    df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """One marginal row per (method, nominal) for a single fold's interval table."""
    rows: List[Dict[str, Any]] = []
    for method in METHODS:
        for nominal in NOMINAL_LEVELS:
            mask = (df["method"] == method) & (
                np.isclose(df["nominal"], float(nominal))
            )
            sub = df.loc[mask]
            if sub.empty:
                continue
            m = compute_cell_metrics(
                sub["actual_log"].to_numpy(dtype=float),
                sub["point_pred_log"].to_numpy(dtype=float),
                sub["lower_log"].to_numpy(dtype=float),
                sub["upper_log"].to_numpy(dtype=float),
                float(nominal),
                method,
            )
            m["n_fallback"] = int(sub["fallback_used"].sum())
            m["method"] = method
            m["nominal"] = float(nominal)
            rows.append(m)
    return rows


def build_subgroup_rows(
    *,
    regime: str,
    fold_id: int,
    df: pd.DataFrame,
    cutpoints: Sequence[float],
) -> List[Dict[str, Any]]:
    """Subgroup coverage per (method, nominal, subgroup_type, subgroup)."""
    rows: List[Dict[str, Any]] = []
    true_band_all = C.price_band_labels(df["actual_price"].to_numpy(dtype=float))
    for method in METHODS:
        for nominal in NOMINAL_LEVELS:
            mask = (df["method"] == method) & (
                np.isclose(df["nominal"], float(nominal))
            )
            sub = df.loc[mask]
            if sub.empty:
                continue
            y = sub["actual_log"].to_numpy(dtype=float)
            lo = sub["lower_log"].to_numpy(dtype=float)
            up = sub["upper_log"].to_numpy(dtype=float)

            band_pred, _ = C.predicted_tertile_labels(
                sub["point_pred_log"].to_numpy(dtype=float), cutpoints=cutpoints
            )
            true_band = true_band_all[sub.index.to_numpy()]

            labelled = {
                "source_city": sub["source_city"].to_numpy(dtype=object),
                "true_price_band": np.asarray(true_band, dtype=object),
                "predicted_price_band": np.asarray(band_pred, dtype=object),
            }
            for subgroup_type in SUBGROUP_TYPES:
                labels = labelled[subgroup_type]
                for key in _stable_subgroup_types(labels):
                    cell = labels == key
                    cov = coverage_finite_aware(
                        y[cell], lo[cell], up[cell], float(nominal)
                    )
                    n = cov["n_evaluated"]
                    rows.append(
                        {
                            "regime": regime,
                            "fold": int(fold_id),
                            "method": method,
                            "nominal": float(nominal),
                            "subgroup_type": subgroup_type,
                            "subgroup": str(key),
                            "n_evaluated": n,
                            "n_covered": cov["n_covered"],
                            "empirical_coverage": cov["empirical_coverage"],
                            "coverage_error": cov["coverage_error"],
                            "abs_coverage_error": cov["abs_coverage_error"],
                            "coverage_cp_lower": cov["coverage_cp_lower"],
                            "coverage_cp_upper": cov["coverage_cp_upper"],
                            "coverage_significant": cov["coverage_significant"],
                            "coverage_reliable": bool(n >= MIN_SUBGROUP_N),
                            "min_n_required": int(MIN_SUBGROUP_N),
                        }
                    )
    return rows


def within_stratum_invariance(
    df: pd.DataFrame,
    *,
    tolerance: float = F0_RELATIVE_TOLERANCE,
) -> Dict[str, Any]:
    """F0 (Resolution 8.2): C1/C2 log width must be constant within each stratum.

    C1 is constant-width within ``source_city`` strata and C2 within predicted
    price-band strata, so the pooled SD is expected to be nonzero and is
    descriptive only. This checks the structural invariance that *is* required,
    using the same relative serialization tolerance as the A/B3 global check.
    """
    worst_relative_sd = 0.0
    n_strata = 0
    n_violations = 0
    for method in ("C1", "C2"):
        for nominal in NOMINAL_LEVELS:
            sub = df[
                (df["method"] == method)
                & np.isclose(df["nominal"], float(nominal))
            ]
            for _, grp in sub.groupby("stratum", sort=False, dropna=False):
                width = (
                    grp["upper_log"].to_numpy(dtype=float)
                    - grp["lower_log"].to_numpy(dtype=float)
                )
                width = width[np.isfinite(width)]
                if width.size < 2:
                    continue
                denom = max(float(np.mean(np.abs(width))), 1e-12)
                relative_sd = float(np.std(width, ddof=1)) / denom
                worst_relative_sd = max(worst_relative_sd, relative_sd)
                n_strata += 1
                if relative_sd > tolerance:
                    n_violations += 1
    return {
        "max_relative_sd": worst_relative_sd,
        "n_strata": n_strata,
        "n_violations": n_violations,
        "tolerance": tolerance,
    }


def calibration_coverage_from_quantiles(
    *,
    qdoc: Mapping[str, Any],
    df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Empirical coverage of s <= qhat on C_k, reconstructed from quantile metadata.

    For pooled methods (A, B1, B3, C3) the exact order-statistic quantile makes
    the calibration coverage exactly k/n. For C1/C2 it is the test-size-weighted
    mean of the per-stratum k/n, with fallback strata (unbounded) counted as
    covering. This is derived from frozen Phase 4 metadata only; no model is
    loaded and no calibration score is recomputed.
    """
    del df  # per-stratum test counts are carried inside the quantile metadata
    lookup = cell_lookup(qdoc)
    rows: List[Dict[str, Any]] = []
    for method in METHODS:
        for nominal in NOMINAL_LEVELS:
            cell = lookup[(method, float(nominal))]
            if method in ("C1", "C2"):
                strata = cell["strata"]
                weights = np.array(
                    [float(s["n_test_rows"]) for s in strata], dtype=float
                )
                if weights.sum() <= 0:
                    cal_cov = float("nan")
                else:
                    per = np.array(
                        [
                            1.0
                            if s.get("fallback_used") or s.get("qhat") is None
                            else (float(s["k"]) / float(s["n_calibration"])
                                  if s["n_calibration"] > 0 else 1.0)
                            for s in strata
                        ],
                        dtype=float,
                    )
                    cal_cov = float(np.average(per, weights=weights))
            else:
                n = float(cell["n"])
                cal_cov = 1.0 if cell.get("unbounded") else float(cell["k"]) / n
            rows.append(
                {
                    "method": method,
                    "nominal": float(nominal),
                    "calibration_coverage": cal_cov,
                    "partition": cell.get("partition", "pooled"),
                    "n_calibration": int(cell.get("n", 0)),
                }
            )
    return rows


# ============================================================
# SUMMARY / REGIME SHIFT
# ============================================================


def summary_rows(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Long-format fold mean / sample SD, mirroring Experiment 1 summary_metrics."""
    records: List[Dict[str, Any]] = []
    for method in METHODS:
        for regime in REGIMES:
            block = fold_metrics[
                (fold_metrics["method"] == method)
                & (fold_metrics["regime"] == regime)
            ]
            for metric in CELL_METRIC_KEYS:
                values = block[metric].to_numpy(dtype=float)
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    mean = std = vmin = vmax = float("nan")
                else:
                    mean = float(np.mean(finite))
                    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
                    vmin = float(np.min(finite))
                    vmax = float(np.max(finite))
                records.append(
                    {
                        "method": method,
                        "regime": regime,
                        "metric": metric,
                        "space": SUMMARY_METRIC_SPACES.get(metric, ""),
                        "mean": mean,
                        "std": std,
                        "min": vmin,
                        "max": vmax,
                        "n_folds": int(block["fold"].nunique()),
                        "std_ddof": 1,
                    }
                )
    return pd.DataFrame(records, columns=[
        "method", "regime", "metric", "space",
        "mean", "std", "min", "max", "n_folds", "std_ddof",
    ])


def regime_shift_rows(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Paired per-fold deltas, declared sign convention grouped - random."""
    rows: List[Dict[str, Any]] = []
    for method in METHODS:
        for nominal in NOMINAL_LEVELS:
            rnd = fold_metrics[
                (fold_metrics["method"] == method)
                & (fold_metrics["regime"] == "random")
                & np.isclose(fold_metrics["nominal"], float(nominal))
            ].sort_values("fold")
            grp = fold_metrics[
                (fold_metrics["method"] == method)
                & (fold_metrics["regime"] == "location_grouped")
                & np.isclose(fold_metrics["nominal"], float(nominal))
            ].sort_values("fold")
            for metric in (
                "empirical_coverage",
                "coverage_error",
                "mean_width_log",
                "interval_score_log",
            ):
                r = rnd[metric].to_numpy(dtype=float)
                g = grp[metric].to_numpy(dtype=float)
                if r.size != g.size or r.size == 0:
                    continue
                delta = g - r
                row: Dict[str, Any] = {
                    "method": method,
                    "nominal": float(nominal),
                    "metric": metric,
                    "n_pairs": int(delta.size),
                    "mean_random": float(np.mean(r)),
                    "mean_grouped": float(np.mean(g)),
                    "mean_delta": float(np.mean(delta)),
                    "sd_delta_ddof1": float(np.std(delta, ddof=1))
                    if delta.size > 1 else 0.0,
                    "sign_convention": "delta = grouped - random; negative = worse under shift",
                }
                for i in range(5):
                    row[f"delta_fold{i + 1}"] = (
                        float(delta[i]) if i < delta.size else float("nan")
                    )
                rows.append(row)
    return pd.DataFrame(rows, columns=list(REGIME_SHIFT_COLUMNS))


def monotonicity_by_cell(fold_metrics: pd.DataFrame) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    out: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for method in METHODS:
        for regime in REGIMES:
            for fold in sorted(fold_metrics["fold"].unique()):
                block = fold_metrics[
                    (fold_metrics["method"] == method)
                    & (fold_metrics["regime"] == regime)
                    & (fold_metrics["fold"] == fold)
                ].sort_values("nominal")
                seq = C.coverage_sequence_by_level(
                    block["empirical_coverage"].tolist(), NOMINAL_LEVELS
                )
                out[(method, regime, int(fold))] = seq
    return out


# ============================================================
# FIGURES
# ============================================================


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": FIG_DPI,
            "savefig.dpi": FIG_DPI,
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.30,
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(
        pdf,
        format="pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None},
    )
    plt.close(fig)
    return [png, pdf]


def _fold_mean(fold_metrics: pd.DataFrame, method: str, regime: str,
               metric: str, nominal: float) -> float:
    block = fold_metrics[
        (fold_metrics["method"] == method)
        & (fold_metrics["regime"] == regime)
        & np.isclose(fold_metrics["nominal"], float(nominal))
    ]
    if block.empty:
        return float("nan")
    return float(np.nanmean(block[metric].to_numpy(dtype=float)))


def figure_coverage_vs_nominal(
    fold_metrics: pd.DataFrame, out_dir: Path, regimes: Sequence[str], stem: str,
    title: str,
) -> List[Path]:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.array([0.80, 0.90, 0.95])
    ax.plot(x, x, color="#444444", linestyle=":", linewidth=1.6,
            label="perfect calibration")
    for method in METHODS:
        for regime in regimes:
            y = [_fold_mean(fold_metrics, method, regime,
                            "empirical_coverage", lev) for lev in x]
            style = "-" if len(regimes) == 1 else "--"
            label = FIGURE_LABELS[method] if len(regimes) == 1 else (
                f"{FIGURE_LABELS[method]} ({regime})"
            )
            ax.plot(x, y, marker="o", linestyle=style,
                    color=METHOD_COLORS[method], label=label)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Mean empirical coverage across folds (n=5)")
    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["0.80", "0.90", "0.95"])
    ax.legend(loc="lower right", ncol=2)
    return save_figure(fig, out_dir, stem)


def figure_mean_width_vs_nominal(
    fold_metrics: pd.DataFrame, out_dir: Path
) -> List[Path]:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.array([0.80, 0.90, 0.95])
    for method in METHODS:
        for regime in REGIMES:
            y = [_fold_mean(fold_metrics, method, regime, "mean_width_log", lev)
                 for lev in x]
            ax.plot(x, y, marker="o", linestyle="-" if regime == "random" else "--",
                    color=METHOD_COLORS[method],
                    label=f"{FIGURE_LABELS[method]} ({regime})")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Mean interval width (log1p price)")
    ax.set_title("Interval Width vs Nominal Coverage", pad=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["0.80", "0.90", "0.95"])
    ax.legend(loc="upper left", ncol=2)
    return save_figure(fig, out_dir, "04_mean_width_vs_nominal")


def figure_regime_shift_delta(
    fold_metrics: pd.DataFrame, out_dir: Path
) -> List[Path]:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    levels = np.array([0.80, 0.90, 0.95])
    positions = np.arange(len(METHODS), dtype=float)
    width = 0.25
    for li, nominal in enumerate(levels):
        deltas = []
        for method in METHODS:
            d = (_fold_mean(fold_metrics, method, "location_grouped",
                            "empirical_coverage", nominal)
                 - _fold_mean(fold_metrics, method, "random",
                              "empirical_coverage", nominal))
            deltas.append(d)
        ax.bar(positions + (li - 1) * width, deltas, width=width * 0.9,
               label=f"nominal {nominal:.2f}")
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels([FIGURE_LABELS[m] for m in METHODS], rotation=15, ha="right")
    ax.set_ylabel("Δ coverage = grouped − random")
    ax.set_title("Paired Regime-Shift Coverage Difference (mean across folds)",
                 pad=12, fontweight="bold")
    ax.legend(loc="lower left")
    return save_figure(fig, out_dir, "05_regime_shift_coverage_delta")


def figure_interval_score(
    fold_metrics: pd.DataFrame, out_dir: Path
) -> List[Path]:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.array([0.80, 0.90, 0.95])
    for method in METHODS:
        for regime in REGIMES:
            y = [_fold_mean(fold_metrics, method, regime, "interval_score_log", lev)
                 for lev in x]
            ax.plot(x, y, marker="o", linestyle="-" if regime == "random" else "--",
                    color=METHOD_COLORS[method],
                    label=f"{FIGURE_LABELS[method]} ({regime})")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Mean interval score (log1p price)")
    ax.set_title("Interval Score vs Nominal Coverage", pad=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["0.80", "0.90", "0.95"])
    ax.legend(loc="upper left", ncol=2)
    return save_figure(fig, out_dir, "06_interval_score_vs_nominal")


def figure_city_coverage(subgroup: pd.DataFrame, out_dir: Path) -> List[Path]:
    cities = sorted(
        subgroup.loc[subgroup["subgroup_type"] == "source_city", "subgroup"].unique()
    )
    fig, axes = plt.subplots(
        len(NOMINAL_LEVELS), 1, figsize=(12.0, 12.0), sharex=True
    )
    positions = np.arange(len(cities), dtype=float)
    width = 0.14
    for ax, nominal in zip(axes, NOMINAL_LEVELS):
        rows = subgroup[
            (subgroup["subgroup_type"] == "source_city")
            & np.isclose(subgroup["nominal"], float(nominal))
        ]
        for mi, method in enumerate(METHODS):
            means, errs = [], []
            for city in cities:
                cell = rows[(rows["method"] == method) & (rows["subgroup"] == city)]
                if cell.empty:
                    means.append(np.nan)
                    errs.append(0.0)
                    continue
                vals = cell["empirical_coverage"].to_numpy(dtype=float)
                means.append(float(np.mean(vals)))
                errs.append(float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0)
            ax.bar(positions + (mi - 2.5) * width, means, width=width * 0.9,
                   yerr=errs, capsize=2, color=METHOD_COLORS[method],
                   label=FIGURE_LABELS[method])
        ax.axhline(float(nominal), color="#444444", linestyle=":", linewidth=1.2)
        ax.set_ylabel(f"coverage @ {nominal:.2f}")
        ax.set_title(f"City-Level Coverage at Nominal {nominal:.2f}", fontsize=12)
    axes[0].legend(loc="upper center", ncol=3)
    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels(cities, rotation=20, ha="right")
    fig.suptitle("City-Level Coverage (mean ± SD across folds)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_figure(fig, out_dir, "07_city_coverage")


def figure_price_band_coverage(subgroup: pd.DataFrame, out_dir: Path) -> List[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.6))
    for ax, stype, title in (
        (axes[0], "true_price_band", "True-Price Bands (diagnostic; label-dependent)"),
        (axes[1], "predicted_price_band", "Predicted-Price Bands (assignable; C2)"),
    ):
        groups = sorted(subgroup.loc[subgroup["subgroup_type"] == stype, "subgroup"].unique())
        positions = np.arange(len(groups), dtype=float)
        width = 0.14
        rows = subgroup[subgroup["subgroup_type"] == stype]
        for mi, method in enumerate(METHODS):
            means = []
            for band in groups:
                cell = rows[(rows["method"] == method) & (rows["subgroup"] == band)]
                means.append(
                    float(np.mean(cell["empirical_coverage"].to_numpy(dtype=float)))
                    if not cell.empty else np.nan
                )
            ax.bar(positions + (mi - 2.5) * width, means, width=width * 0.9,
                   color=METHOD_COLORS[method], label=FIGURE_LABELS[method])
        ax.set_xticks(positions)
        ax.set_xticklabels(groups, rotation=20, ha="right")
        ax.set_ylabel("Mean empirical coverage across folds")
        ax.set_title(title, fontsize=12)
        for lev in NOMINAL_LEVELS:
            ax.axhline(float(lev), color="#999999", linestyle=":", linewidth=0.8)
    axes[0].legend(loc="lower right", ncol=2)
    fig.suptitle("Price-Band Coverage Profile (all nominal levels overlaid as dotted lines)",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save_figure(fig, out_dir, "08_price_band_coverage")


def figure_residual_vs_fitted(residuals: pd.DataFrame, out_dir: Path) -> List[Path]:
    regimes = list(REGIMES)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.4), sharey=True)
    for ax, regime in zip(axes, regimes):
        block = residuals[residuals["regime"] == regime]
        for fold in sorted(block["fold"].unique()):
            f = block[block["fold"] == fold]
            ax.scatter(f["fitted"], f["residual"], s=4, alpha=0.20,
                       color=matplotlib.colormaps["viridis"]((fold - 1) / 5.0),
                       label=f"fold {fold}")
        ax.axhline(0.0, color="#444444", linewidth=1.0)
        ax.set_xlabel("Fitted log1p(price)")
        ax.set_title(f"{REGIME_LABELS[regime]}", fontsize=12)
    axes[0].set_ylabel("Residual (actual − fitted, log space)")
    axes[0].legend(loc="upper left", markerscale=3)
    fig.suptitle("Residual vs Fitted (Method A, test rows)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, out_dir, "09_residual_vs_fitted")


def figure_qq(residuals: pd.DataFrame, out_dir: Path) -> List[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.4))
    for ax, regime in zip(axes, REGIMES):
        values = residuals.loc[residuals["regime"] == regime, "residual"].to_numpy(dtype=float)
        stats.probplot(values, dist="norm", plot=ax)
        ax.set_title(f"{REGIME_LABELS[regime]}", fontsize=12)
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Ordered residuals")
    fig.suptitle("QQ Plot of Residuals vs Normal Reference (Method A)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, out_dir, "10_qq_residuals")


def figure_scale_location(
    residuals: pd.DataFrame,
    out_dir: Path,
    b1_qhat: Mapping[str, float],
) -> List[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.4), sharey=True)
    for ax, regime in zip(axes, REGIMES):
        block = residuals[residuals["regime"] == regime]
        ax.scatter(block["fitted"], np.abs(block["residual"]), s=4, alpha=0.18,
                   color=REGIME_COLORS[regime])
        qhat = float(b1_qhat.get(regime, float("nan")))
        xs = np.linspace(block["fitted"].min(), block["fitted"].max(), 50)
        ax.plot(xs, qhat * xs, color="#111111", linewidth=1.4,
                label="B1 half-width q̂·ŷ (q̂ = mean B1 quantile @0.90)")
        ax.set_xlabel("Fitted log1p(price)")
        ax.set_title(f"{REGIME_LABELS[regime]}", fontsize=12)
    axes[0].set_ylabel("|Residual| (log space)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Scale–Location: |Residual| vs Fitted (Method A) with B1 scale overlaid",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, out_dir, "11_scale_location")


def figure_calibration_transfer(
    calibrated: pd.DataFrame, fold_metrics: pd.DataFrame, out_dir: Path
) -> List[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.4))
    for ax, regime in zip(axes, REGIMES):
        ax.plot([0, 1], [0, 1], color="#444444", linestyle=":", linewidth=1.4)
        block = calibrated[calibrated["regime"] == regime]
        for method in METHODS:
            cal_x, test_y = [], []
            for nominal in NOMINAL_LEVELS:
                cal = block[(block["method"] == method)
                            & np.isclose(block["nominal"], float(nominal))]
                if cal.empty:
                    continue
                cal_x.append(float(np.nanmean(cal["calibration_coverage"].to_numpy(dtype=float))))
                test_y.append(_fold_mean(fold_metrics, method, regime,
                                         "empirical_coverage", nominal))
            ax.scatter(cal_x, test_y, s=45, color=METHOD_COLORS[method],
                       label=FIGURE_LABELS[method])
            ax.plot(cal_x, test_y, color=METHOD_COLORS[method], alpha=0.4, linewidth=1.0)
        ax.set_xlabel("Mean calibration coverage on C_k (s ≤ q̂)")
        ax.set_ylabel("Mean test coverage on test_k")
        ax.set_title(f"{REGIME_LABELS[regime]}", fontsize=12)
        ax.set_xlim(0.75, 1.0)
        ax.set_ylim(0.75, 1.0)
    axes[0].legend(loc="lower right")
    fig.suptitle("Calibration Transfer: C_k vs test_k coverage", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, out_dir, "12_calibration_transfer")


def build_figures(
    *,
    out_dir: Path,
    fold_metrics: pd.DataFrame,
    subgroup: pd.DataFrame,
    residuals: pd.DataFrame,
    calibrated: pd.DataFrame,
    b1_qhat: Mapping[str, float],
) -> List[Path]:
    apply_style()
    produced: List[Path] = []
    produced += figure_coverage_vs_nominal(
        fold_metrics, out_dir, ("random",), "01_coverage_vs_nominal_random",
        "Coverage vs Nominal — random regime",
    )
    produced += figure_coverage_vs_nominal(
        fold_metrics, out_dir, ("location_grouped",),
        "02_coverage_vs_nominal_grouped",
        "Coverage vs Nominal — location_grouped regime",
    )
    produced += figure_coverage_vs_nominal(
        fold_metrics, out_dir, REGIMES, "03_coverage_vs_nominal_both_regimes",
        "Coverage vs Nominal — both regimes overlaid",
    )
    produced += figure_mean_width_vs_nominal(fold_metrics, out_dir)
    produced += figure_regime_shift_delta(fold_metrics, out_dir)
    produced += figure_interval_score(fold_metrics, out_dir)
    produced += figure_city_coverage(subgroup, out_dir)
    produced += figure_price_band_coverage(subgroup, out_dir)
    produced += figure_residual_vs_fitted(residuals, out_dir)
    produced += figure_qq(residuals, out_dir)
    produced += figure_scale_location(residuals, out_dir, b1_qhat)
    produced += figure_calibration_transfer(calibrated, fold_metrics, out_dir)
    return produced


# ============================================================
# DECISION TABLE (F0–F7)
# ============================================================


def _decision(
    rows: List[Dict[str, Any]], criterion: str, scope: str, metric: str,
    value: Any, threshold: Any, outcome: str, notes: str,
) -> None:
    rows.append(
        {
            "criterion": criterion,
            "scope": scope,
            "metric": metric,
            "value": value,
            "threshold": threshold,
            "outcome": outcome,
            "notes": notes,
        }
    )


def _fmt_levels(values: Mapping[float, float]) -> str:
    return "; ".join(f"{k:.2f}:{v:.6g}" for k, v in sorted(values.items()))


def build_decision_rows(
    *,
    fold_metrics: pd.DataFrame,
    subgroup: pd.DataFrame,
    monotonicity: Mapping[Tuple[str, str, int], Mapping[str, Any]],
    b3_max_interval_difference: float,
    within_stratum: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    # ---- F0: implementation validity gate (Resolution 8.2) ---------------
    # Global constant-width check applies ONLY to A and B3, with a relative
    # serialization tolerance because Phase 4 used float_format="%.10g".
    glob = fold_metrics[
        fold_metrics["method"].isin(list(GLOBAL_HOMOGENEOUS_METHODS))
    ]
    if len(glob):
        relative_sd = (
            glob["sd_width_log"].abs()
            / glob["mean_width_log"].abs().clip(lower=1e-12)
        )
        max_relative_sd = float(relative_sd.max())
    else:
        max_relative_sd = 0.0
    glob_pass = max_relative_sd <= F0_RELATIVE_TOLERANCE

    b3_pass = b3_max_interval_difference <= C.B3_MATCH_TOLERANCE
    ws_pass = int(within_stratum.get("n_violations", 0)) == 0
    f0_pass = glob_pass and b3_pass and ws_pass

    _decision(
        rows, "F0", "global|A_B3_constant_width",
        "max_relative_sd_width_log_A_B3", max_relative_sd,
        F0_RELATIVE_TOLERANCE, "pass" if glob_pass else "fail",
        "Global constant width is required only for A and B3. Tolerance is a "
        "serialization artifact of %.10g persistence, not a scientific "
        "threshold (Resolution 8.2).",
    )
    _decision(
        rows, "F0", "global|B3_reproduces_A",
        "max_abs_A_vs_B3_interval_difference", b3_max_interval_difference,
        C.B3_MATCH_TOLERANCE, "pass" if b3_pass else "fail",
        "B3 (sigma=1) must reproduce Method A to floating-point tolerance.",
    )
    _decision(
        rows, "F0", "C1_C2|within_stratum_width_invariance",
        "max_within_stratum_relative_sd_width",
        float(within_stratum.get("max_relative_sd", float("nan"))),
        F0_RELATIVE_TOLERANCE, "pass" if ws_pass else "fail",
        f"strata checked={within_stratum.get('n_strata', 0)}, "
        f"violations={within_stratum.get('n_violations', 0)}. C1/C2 are "
        "per-stratum constant; pooled SD is descriptive only (Resolution 8.2).",
    )

    # ---- F1: negative control on the random regime -----------------------
    for method in METHODS:
        per_level: Dict[float, float] = {}
        significant = 0
        for nominal in NOMINAL_LEVELS:
            block = fold_metrics[
                (fold_metrics["method"] == method)
                & (fold_metrics["regime"] == "random")
                & np.isclose(fold_metrics["nominal"], float(nominal))
            ]
            err = float(np.nanmean(block["abs_coverage_error"].to_numpy(dtype=float)))
            per_level[float(nominal)] = err
            if bool(block["coverage_significant"].any()):
                significant += 1
        _decision(
            rows, "F1", f"{method}|random", "abs_coverage_error_by_level",
            _fmt_levels(per_level), "all levels within Clopper-Pearson 95%",
            "pass" if significant == 0 else "fail",
            "Negative control: if random-regime coverage deviates from nominal, "
            "split conformal is not working even where assumptions hold. "
            "coverage_significant is anti-conservative (correlated rows, §15.4).",
        )

    # ---- F2: headline falsifier (Method A, location_grouped) -------------
    a_errors: Dict[float, float] = {}
    a_not_rejected = False
    for nominal in NOMINAL_LEVELS:
        block = fold_metrics[
            (fold_metrics["method"] == "A")
            & (fold_metrics["regime"] == "location_grouped")
            & np.isclose(fold_metrics["nominal"], float(nominal))
        ]
        err = abs(float(np.nanmean(block["empirical_coverage"].to_numpy(dtype=float)))
                  - float(nominal))
        a_errors[float(nominal)] = err
        if err <= COVERAGE_ERROR_THRESHOLD and not bool(block["coverage_significant"].any()):
            a_not_rejected = True
    all_over = all(v > COVERAGE_ERROR_THRESHOLD for v in a_errors.values())
    if all_over:
        f2 = "rejected"
    elif a_not_rejected:
        f2 = "not_rejected"
    else:
        f2 = "inconclusive"
    _decision(
        rows, "F2", "A|location_grouped", "abs_coverage_error_by_level",
        _fmt_levels(a_errors), f"all three > {COVERAGE_ERROR_THRESHOLD} -> rejected",
        f2,
        "Rejected only if abs_coverage_error > 0.05 at ALL three levels. "
        "Location-grouped coverage is descriptive; no distribution-free "
        "guarantee is claimed there.",
    )

    # ---- F3: sharpness / usability gate at 0.90 --------------------------
    for method in METHODS:
        for regime in REGIMES:
            per_fold = fold_metrics[
                (fold_metrics["method"] == method)
                & (fold_metrics["regime"] == regime)
                & np.isclose(fold_metrics["nominal"], F3_GATE_LEVEL)
            ].sort_values("fold")
            med = per_fold["median_relative_width_inr"].to_numpy(dtype=float)
            mean_med = float(np.nanmean(med))
            n_pass = int(np.sum(med <= F3_GATE_MAX_MEDIAN_RELATIVE_WIDTH))
            usable = mean_med <= F3_GATE_MAX_MEDIAN_RELATIVE_WIDTH
            _decision(
                rows, "F3", f"{method}|{regime}",
                "mean_median_relative_width_inr@0.90", mean_med,
                F3_GATE_MAX_MEDIAN_RELATIVE_WIDTH,
                "usable" if usable else "not_usable",
                f"per-fold passes at 0.90: {n_pass}/{len(med)}. "
                "Experiment-specific operational threshold, not an industry "
                "standard. Calibration and usability are separate axes.",
            )

    # ---- F4: overcoverage falsifier at 0.90 ------------------------------
    for method in METHODS:
        for regime in REGIMES:
            block = fold_metrics[
                (fold_metrics["method"] == method)
                & (fold_metrics["regime"] == regime)
                & np.isclose(fold_metrics["nominal"], 0.90)
            ]
            err = float(np.nanmean(block["coverage_error"].to_numpy(dtype=float)))
            width = float(np.nanmean(block["median_relative_width_inr"].to_numpy(dtype=float)))
            triggered = (err >= COVERAGE_ERROR_THRESHOLD) and (
                width > F3_GATE_MAX_MEDIAN_RELATIVE_WIDTH
            )
            _decision(
                rows, "F4", f"{method}|{regime}", "coverage_error@0.90",
                err, f">={COVERAGE_ERROR_THRESHOLD} with width>{F3_GATE_MAX_MEDIAN_RELATIVE_WIDTH}",
                "conservative_and_useless" if triggered else "not_triggered",
                f"mean median_relative_width_inr={width:.4g}. Conservative and "
                "useless is its own outcome, distinct from F2.",
            )

    # ---- F5: descriptive monotonicity ------------------------------------
    for method in METHODS:
        for regime in REGIMES:
            directions = [
                monotonicity[(method, regime, fold)]["direction"]
                for fold in sorted(fold_metrics["fold"].unique())
            ]
            counts = {
                d: int(sum(1 for x in directions if x == d))
                for d in ("increasing", "decreasing", "non_monotone", "undefined")
            }
            _decision(
                rows, "F5", f"{method}|{regime}", "monotone_direction_counts",
                json.dumps(counts, sort_keys=True), "descriptive only",
                "descriptive",
                "Corrected F5: no Spearman sign is a validity requirement; "
                "non-monotonicity is not a falsifier.",
            )

    # ---- F6: uniformity falsifier ----------------------------------------
    for method in METHODS:
        for regime in REGIMES:
            n_sig = 0
            examples: List[str] = []
            for nominal in NOMINAL_LEVELS:
                pooled = fold_metrics[
                    (fold_metrics["method"] == method)
                    & (fold_metrics["regime"] == regime)
                    & np.isclose(fold_metrics["nominal"], float(nominal))
                ]
                pooled_calibrated = not bool(pooled["coverage_significant"].any())
                if not pooled_calibrated:
                    continue
                cells = subgroup[
                    (subgroup["method"] == method)
                    & (subgroup["regime"] == regime)
                    & np.isclose(subgroup["nominal"], float(nominal))
                    & (subgroup["coverage_reliable"])
                    & (subgroup["coverage_significant"])
                ]
                n_sig += int(len(cells))
                if len(examples) < 3 and not cells.empty:
                    for _, row in cells.head(3).iterrows():
                        examples.append(
                            f"{row['subgroup_type']}:{row['subgroup']}@{nominal:.2f}"
                        )
            _decision(
                rows, "F6", f"{method}|{regime}",
                "significant_reliable_subgroup_cells_at_calibrated_levels",
                n_sig, 0, "uniform" if n_sig == 0 else "nonuniform",
                "Marginal coverage may hide conditional miscalibration. "
                f"examples={examples}. Suppression rule n<{MIN_SUBGROUP_N} applied.",
            )

    # ---- F7: C1/C2 improvement under shift -------------------------------
    def mean_abs(method: str) -> float:
        vals = []
        for nominal in NOMINAL_LEVELS:
            block = fold_metrics[
                (fold_metrics["method"] == method)
                & (fold_metrics["regime"] == "location_grouped")
                & np.isclose(fold_metrics["nominal"], float(nominal))
            ]
            vals.append(float(np.nanmean(block["abs_coverage_error"].to_numpy(dtype=float))))
        return float(np.mean(vals))

    a_m = mean_abs("A")
    c1_m = mean_abs("C1")
    c2_m = mean_abs("C2")
    improved = min(c1_m, c2_m) < a_m
    _decision(
        rows, "F7", "location_grouped",
        "mean_abs_coverage_error A/C1/C2",
        json.dumps({"A": a_m, "C1": c1_m, "C2": c2_m}),
        "min(C1,C2) < A",
        "improved" if improved else "not_improved",
        "If neither Mondrian variant improves shift coverage, no further method "
        "may be invented within Experiment 2.",
    )
    return rows


# ============================================================
# DIGEST
# ============================================================


def deterministic_digest(paths: Sequence[Path], base: Path) -> str:
    """SHA-256 over sorted '<relative path>:<file sha>' lines."""
    lines = []
    for p in sorted(paths, key=lambda q: q.relative_to(base).as_posix()):
        rel = p.relative_to(base).as_posix()
        lines.append(f"{rel}:{T.sha256_of(p)}")
    return T.sha256_of_bytes(("\n".join(lines) + "\n").encode("utf-8"))


# ============================================================
# RUN
# ============================================================


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    T.assert_safe_write_target(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        index=False,
        na_rep="",
        float_format="%.10g",
        lineterminator="\n",
        columns=list(df.columns),
    )
    return path


def run_phase5(
    *,
    artifact_root: Path = T.ARTIFACT_ROOT,
    output_root: Path | None = None,
) -> Dict[str, Any]:
    """Run the full Phase 5 analysis and write the analysis artifacts."""
    input_root = artifact_root
    out_root = Path(output_root) if output_root is not None else analysis_dir(artifact_root)
    fig_root = figures_dir(out_root)

    T.assert_safe_write_target(out_root / "probe.csv")
    T.assert_safe_write_target(fig_root / "probe.png")

    phase4 = load_phase4_manifest(input_root)

    before = snapshot_protected_state_phase5()
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / PROTECTED_BEFORE_NAME, "w", encoding="utf-8") as fh:
        json.dump(before, fh, indent=2)
        fh.write("\n")

    fold_rows: List[Dict[str, Any]] = []
    subgroup_rows: List[Dict[str, Any]] = []
    calibrated_rows: List[Dict[str, Any]] = []
    resid_frames: List[pd.DataFrame] = []
    b3_max_interval_difference = 0.0
    b1_qhat_acc: Dict[str, List[float]] = {r: [] for r in REGIMES}
    ws_max_relative_sd = 0.0
    ws_n_strata = 0
    ws_n_violations = 0

    for regime in REGIMES:
        for summary in phase4["folds"]:
            if summary["regime"] != regime:
                continue
            fold_id = int(summary["fold"])
            df = load_intervals(input_root, regime, fold_id)
            qdoc = load_quantiles(input_root, regime, fold_id)
            cutpoints = qdoc["c2_cutpoints_from_T_prime"]

            rows = build_fold_rows(df)
            for r in rows:
                r["regime"] = regime
                r["fold"] = fold_id
            fold_rows.extend(rows)

            subgroup_rows.extend(
                build_subgroup_rows(
                    regime=regime, fold_id=fold_id, df=df, cutpoints=cutpoints
                )
            )

            ws = within_stratum_invariance(df)
            ws_max_relative_sd = max(
                ws_max_relative_sd, float(ws["max_relative_sd"])
            )
            ws_n_strata += int(ws["n_strata"])
            ws_n_violations += int(ws["n_violations"])

            for c in calibration_coverage_from_quantiles(qdoc=qdoc, df=df):
                calibrated_rows.append(
                    {"regime": regime, "fold": fold_id, **c}
                )

            lookup = cell_lookup(qdoc)
            for nominal in NOMINAL_LEVELS:
                b1 = lookup[("B1", float(nominal))]
                if not b1.get("unbounded") and b1.get("quantile") is not None:
                    b1_qhat_acc[regime].append(float(b1["quantile"]))

            a = df[(df["method"] == "A") & np.isclose(df["nominal"], 0.80)]
            b3 = df[(df["method"] == "B3") & np.isclose(df["nominal"], 0.80)]
            a_lo = a["lower_log"].to_numpy(dtype=float)
            a_up = a["upper_log"].to_numpy(dtype=float)
            b3_lo = b3["lower_log"].to_numpy(dtype=float)
            b3_up = b3["upper_log"].to_numpy(dtype=float)
            if a_lo.size and a_lo.size == b3_lo.size:
                b3_max_interval_difference = max(
                    b3_max_interval_difference,
                    float(np.max(np.abs(a_lo - b3_lo))),
                    float(np.max(np.abs(a_up - b3_up))),
                )

            sub_a = df[(df["method"] == "A") & np.isclose(df["nominal"], 0.80)]
            resid_frames.append(
                sub_a[["point_pred_log", "actual_log"]].assign(
                    regime=regime, fold=fold_id
                )
            )
            print(
                f"  [{regime:<17} fold {fold_id}] "
                f"cells={len(rows)} subgroup_rows={len(subgroup_rows)}"
            )

    fold_metrics = pd.DataFrame(fold_rows, columns=list(FOLD_METRICS_COLUMNS))
    fold_metrics = fold_metrics.sort_values(
        ["regime", "method", "nominal", "fold"], kind="mergesort"
    ).reset_index(drop=True)

    summary = summary_rows(fold_metrics)
    shift = regime_shift_rows(fold_metrics)
    subgroup = pd.DataFrame(subgroup_rows, columns=list(SUBGROUP_METRICS_COLUMNS))
    subgroup = subgroup.sort_values(
        ["regime", "method", "nominal", "subgroup_type", "subgroup", "fold"],
        kind="mergesort",
    ).reset_index(drop=True)

    calibrated = pd.DataFrame(calibrated_rows)
    calibrated = calibrated.sort_values(
        ["regime", "method", "nominal", "fold"], kind="mergesort"
    ).reset_index(drop=True)
    residuals = pd.concat(resid_frames, ignore_index=True)
    residuals["fitted"] = residuals["point_pred_log"].astype(float)
    residuals["residual"] = (
        residuals["actual_log"].astype(float) - residuals["fitted"]
    )
    residuals = residuals[["regime", "fold", "fitted", "residual"]]

    b1_qhat = {
        regime: float(np.mean(values)) if values else float("nan")
        for regime, values in b1_qhat_acc.items()
    }

    monotonicity = monotonicity_by_cell(fold_metrics)
    within_stratum = {
        "max_relative_sd": ws_max_relative_sd,
        "n_strata": ws_n_strata,
        "n_violations": ws_n_violations,
        "tolerance": F0_RELATIVE_TOLERANCE,
    }
    decisions = pd.DataFrame(
        build_decision_rows(
            fold_metrics=fold_metrics,
            subgroup=subgroup,
            monotonicity=monotonicity,
            b3_max_interval_difference=b3_max_interval_difference,
            within_stratum=within_stratum,
        ),
        columns=list(DECISION_COLUMNS),
    )

    tables = {
        FOLD_METRICS_CSV: fold_metrics,
        SUMMARY_METRICS_CSV: summary,
        REGIME_SHIFT_CSV: shift,
        SUBGROUP_METRICS_CSV: subgroup,
        DECISION_TABLE_CSV: decisions,
    }
    table_paths = [
        _write_csv(table_df, out_root / name)
        for name, table_df in tables.items()
    ]

    figure_paths = build_figures(
        out_dir=fig_root,
        fold_metrics=fold_metrics,
        subgroup=subgroup,
        residuals=residuals,
        calibrated=calibrated,
        b1_qhat=b1_qhat,
    )
    if len(figure_paths) != 2 * len(FIGURE_STEMS):
        raise RuntimeError(
            f"expected {2 * len(FIGURE_STEMS)} figure files, got {len(figure_paths)}"
        )

    after = snapshot_protected_state_phase5()
    with open(out_root / PROTECTED_AFTER_NAME, "w", encoding="utf-8") as fh:
        json.dump(after, fh, indent=2)
        fh.write("\n")
    diff = T.compare_protected_states(before, after)

    digest = deterministic_digest(table_paths + figure_paths, out_root)

    inputs = phase4["folds"]
    manifest: Dict[str, Any] = {
        "purpose": (
            "Experiment 2 Phase 5 conformal coverage, sharpness, subgroup and "
            "decision analysis."
        ),
        "analysis_only": True,
        "models_trained": 0,
        "models_loaded_for_prediction": 0,
        "folds_regenerated": False,
        "protocol_modified": False,
        "protocol": {
            "path": _rel(PROTOCOL_DOC_PATH),
            "sha256": T.sha256_of(PROTOCOL_DOC_PATH),
            "note": (
                "SHA-256 of the exact Experiment 2 protocol document used for "
                "this experiment (Phase 0 decisions, Resolutions 8.1 and 8.2, "
                "F1-F7 criteria, and the Section 15.10 audit-limitation note)."
            ),
        },
        "dataset_modified": False,
        "chain_touched": False,
        "live_tests_run": False,
        "deterministic": True,
        "determinism_note": (
            "Fixed figure geometry and DPI, no random jitter, no bootstrap, no "
            "seed dependence. PDF CreationDate suppressed. Re-running "
            "reproduces byte-identical tables and figures."
        ),
        "composite_score_computed": False,
        "models_ranked": False,
        "methods": list(METHODS),
        "nominal_levels": [float(x) for x in NOMINAL_LEVELS],
        "regimes": list(REGIMES),
        "min_subgroup_n": int(MIN_SUBGROUP_N),
        "coverage_error_threshold": COVERAGE_ERROR_THRESHOLD,
        "f3_gate": {
            "level": F3_GATE_LEVEL,
            "max_median_relative_width": F3_GATE_MAX_MEDIAN_RELATIVE_WIDTH,
            "provenance": "Experiment-specific operational threshold (§0.4); "
                          "NOT a universal real-estate industry standard",
        },
        "infinite_interval_policy": (
            "(-inf, +inf) rows are counted as covered and reported via "
            "n_fallback/n_infinite; width and interval-score metrics are scoped "
            "to bounded rows. Infinity is never replaced with a finite value."
        ),
        "environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "executable": "python",
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "scipy": __import__("scipy").__version__,
            "plot_backend": matplotlib.get_backend(),
        },
        "figure_dpi": FIG_DPI,
        "figure_size_inches": list(FIGSIZE),
        "phase4_inputs": {
            "manifest": _rel(input_root / R.MANIFEST_NAME),
            "manifest_sha256": T.sha256_of(input_root / R.MANIFEST_NAME),
            "n_fold_tables": len(inputs),
            "deterministic_content_digest": phase4.get("deterministic_content_digest"),
        },
        "row_counts": {
            "fold_metrics": int(len(fold_metrics)),
            "summary_metrics": int(len(summary)),
            "regime_shift_coverage": int(len(shift)),
            "subgroup_metrics": int(len(subgroup)),
            "decision_table": int(len(decisions)),
            "n_figures": len(FIGURE_STEMS),
            "n_figure_files": len(figure_paths),
        },
        "decision_outcomes": {
            f"{row.criterion}|{row.scope}": row.outcome
            for row in decisions.itertuples(index=False)
        } if not decisions.empty else {},
        "protected_state": diff,
        "protected_state_clean": bool(diff["clean"]),
        "deterministic_content_digest": digest,
        "outputs": {
            _rel(p): T.sha256_of(p)
            for p in (table_paths + figure_paths
                      + [out_root / PROTECTED_BEFORE_NAME,
                         out_root / PROTECTED_AFTER_NAME])
        },
        "figure_directory": _rel(fig_root),
    }

    manifest_path = out_root / ANALYSIS_MANIFEST_JSON
    T.assert_safe_write_target(manifest_path)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False, default=_json_safe)
        fh.write("\n")

    if not diff["clean"]:
        raise RuntimeError(
            "Protected state changed during Phase 5 analysis: "
            f"{json.dumps(diff, indent=2)}"
        )
    return manifest


# ============================================================
# VERIFICATION
# ============================================================


def verify_phase5(
    *,
    artifact_root: Path = T.ARTIFACT_ROOT,
    output_root: Path | None = None,
) -> Dict[str, Any]:
    """Reload and independently verify the Phase 5 analysis artifacts."""
    out_root = Path(output_root) if output_root is not None else analysis_dir(artifact_root)
    manifest_path = out_root / ANALYSIS_MANIFEST_JSON
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    if manifest.get("protected_state_clean") is not True:
        raise RuntimeError("Phase 5 manifest reports a protected-state violation")

    table_paths = [out_root / name for name in RESULT_TABLES]
    fig_paths = []
    for stem in FIGURE_STEMS:
        fig_paths.append(figures_dir(out_root) / f"{stem}.png")
        fig_paths.append(figures_dir(out_root) / f"{stem}.pdf")

    for p in table_paths + fig_paths:
        if not p.exists():
            raise RuntimeError(f"missing Phase 5 output: {p}")

    expected = deterministic_digest(table_paths + fig_paths, out_root)
    if expected != manifest["deterministic_content_digest"]:
        raise RuntimeError("Phase 5 deterministic content digest drift")

    for rel, sha in manifest["outputs"].items():
        p = ROOT / rel
        if not p.exists():
            raise RuntimeError(f"recorded Phase 5 output missing: {rel}")
        if T.sha256_of(p) != sha:
            raise RuntimeError(f"Phase 5 output hash drift: {rel}")

    counts = {
        "fold_metrics": int(len(pd.read_csv(out_root / FOLD_METRICS_CSV))),
        "summary_metrics": int(len(pd.read_csv(out_root / SUMMARY_METRICS_CSV))),
        "regime_shift_coverage": int(len(pd.read_csv(out_root / REGIME_SHIFT_CSV))),
        "subgroup_metrics": int(len(pd.read_csv(out_root / SUBGROUP_METRICS_CSV))),
        "decision_table": int(len(pd.read_csv(out_root / DECISION_TABLE_CSV))),
    }
    for key, observed in counts.items():
        if observed != manifest["row_counts"][key]:
            raise RuntimeError(f"{key}: row count drift ({observed})")

    return {
        "verified": True,
        "n_figures": len(FIGURE_STEMS),
        "n_figure_files": len(fig_paths),
        "row_counts": counts,
        "deterministic_content_digest": manifest["deterministic_content_digest"],
        "protected_state_clean": manifest["protected_state_clean"],
    }


# ============================================================
# CLI
# ============================================================


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Experiment 2 Phase 5 analyzer")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    out_root = Path(args.out_dir).resolve() if args.out_dir else None

    if args.verify_only:
        report = verify_phase5(output_root=out_root)
        print(json.dumps(report, indent=2))
        return 0

    manifest = run_phase5(output_root=out_root)
    report = verify_phase5(output_root=out_root)
    payload = {
        "status": "Phase 5 analysis complete",
        "row_counts": manifest["row_counts"],
        "deterministic_content_digest": manifest["deterministic_content_digest"],
        "protected_state_clean": manifest["protected_state_clean"],
        "verification": report,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
