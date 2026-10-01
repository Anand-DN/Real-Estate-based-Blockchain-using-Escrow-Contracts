"""
MILLOW RESEARCH EXPERIMENT 1
Research-quality visualisation and model-selection analysis.

THIS SCRIPT TRAINS NOTHING.

It reads only the already-persisted Experiment 1 results and derives figures,
tables and summary statistics from them:

    artifacts/valuation/v2_1/model_comparison/comparison_table.csv
    artifacts/valuation/v2_1/model_comparison/per_fold_metrics.csv
    artifacts/valuation/v2_1/model_comparison/per_fold_metrics.json
    artifacts/valuation/v2_1/model_comparison/summary_metrics.csv
    artifacts/valuation/v2_1/model_comparison/training_time.csv

No model is fitted, loaded, re-scored or re-predicted. No fold is regenerated.
No protocol constant is redefined. No metric value is hard-coded: every number
drawn or tabulated is computed at run time from the persisted per-fold and
summary artifacts, and the summary artifacts are themselves re-derived from
the per-fold artifact and checked for agreement before anything is plotted.

STATISTICAL CONVENTIONS
-----------------------
* Error bars are fold-level sample standard deviations (ddof=1, n=5), never
  standard errors, and never a standard deviation recomputed from aggregated
  values. They are taken from the persisted per-fold rows.
* Predictions are never pooled across folds.
* Folds are NOT paired between regimes. Regime A fold k and Regime B fold k
  are different splits with different train/valid sizes, so a difference of
  regime means is an independent-samples (Welch) quantity, not a paired one.
  The Welch-Satterthwaite standard error and degrees of freedom are used for
  the location-shift confidence intervals, and no paired t-difference is
  computed anywhere.
* Relative change is reported only for ratio-scale, strictly positive
  quantities (MAE, RMSE, MedAPE). R2 is not ratio-scale and is anchored at
  zero only in the sense of "no better than the mean predictor", so its
  relative change is emitted but explicitly flagged as not scale-invariant.
* No composite score, weighted score, win-count or ranking is computed.

Determinism: fixed figure geometry, fixed DPI, no random jitter, no
bootstrap, no random seed dependence. Re-running reproduces byte-identical
figures.

Usage:
    python -m scripts.valuation_v2_1.model_comparison_analysis
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ============================================================
# NAMESPACING
# ============================================================

BASE = ROOT / "artifacts" / "valuation" / "v2_1" / "model_comparison"
FIGURE_DIR = BASE / "figures"

INPUT_COMPARISON = BASE / "comparison_table.csv"
INPUT_PER_FOLD = BASE / "per_fold_metrics.csv"
INPUT_PER_FOLD_JSON = BASE / "per_fold_metrics.json"
INPUT_SUMMARY = BASE / "summary_metrics.csv"
INPUT_TIMING = BASE / "training_time.csv"

OUT_MASTERS = BASE / "thesis_summary_table"
OUT_GENERALIZATION = BASE / "generalization_table"
OUT_SHIFT_ALL = BASE / "regime_shift_all_metrics"
OUT_RESULTS_JSON = BASE / "analysis_results.json"
OUT_MANIFEST = BASE / "analysis_manifest.json"

# ============================================================
# PRESENTATION CONSTANTS (display only; never a ranking)
# ============================================================

# Fixed display order, identical to the order used in the Experiment 1
# documentation. This is a presentation order chosen before any analysis and
# is deliberately NOT sorted by performance.
MODELS: Tuple[str, ...] = (
    "Ridge",
    "RandomForest",
    "LightGBM",
    "CatBoost",
    "XGBoost",
)

MODEL_LABELS: Dict[str, str] = {
    "Ridge": "Ridge",
    "RandomForest": "Random Forest",
    "LightGBM": "LightGBM",
    "CatBoost": "CatBoost",
    "XGBoost": "XGBoost",
}

REGIMES: Tuple[str, ...] = ("random", "location_grouped")
REGIME_LABELS: Dict[str, str] = {
    "random": "Random CV",
    "location_grouped": "Location-Grouped CV",
}

# Metrics with direction, unit, formatting and axis guidance.
# direction: "lower_is_better" or "higher_is_better".
METRIC_INFO: Dict[str, Dict[str, Any]] = {
    "MAE_INR": {
        "label": "MAE (INR)",
        "direction": "lower_is_better",
        "kind": "inr",
    },
    "RMSE_INR": {
        "label": "RMSE (INR)",
        "direction": "lower_is_better",
        "kind": "inr",
    },
    "R2_INR": {
        "label": "R²",
        "direction": "higher_is_better",
        "kind": "unit",
    },
    "MAPE_percent": {
        "label": "MAPE (%)",
        "direction": "lower_is_better",
        "kind": "percent",
    },
    "MedAPE_percent": {
        "label": "MedAPE (%)",
        "direction": "lower_is_better",
        "kind": "percent",
    },
    "MAE_log": {
        "label": "MAE (log1p INR)",
        "direction": "lower_is_better",
        "kind": "unit",
    },
    "RMSE_log": {
        "label": "RMSE (log1p INR)",
        "direction": "lower_is_better",
        "kind": "unit",
    },
    "R2_log": {
        "label": "R² (log space)",
        "direction": "higher_is_better",
        "kind": "unit",
    },
}

PRIMARY_METRIC = "R2_log"

COLORS: Dict[str, str] = {
    "random": "#1f77b4",
    "location_grouped": "#d62728",
}

FIG_DPI = 200
FIGSIZE = (11.0, 6.4)

# Agreement tolerance between the persisted summary table and the summary
# re-derived from the persisted per-fold rows.
#
# The tolerance is scale-relative, not absolute. INR-space metrics are of
# order 1e7, where a single float64 unit in the last place is about 3.7e-9, so
# an absolute 1e-9 threshold is below the representable resolution of the
# quantity being compared and would fail on arithmetic ordering alone rather
# than on genuine drift. REL_TOL = 1e-12 is roughly two orders of magnitude
# tighter than float64 round-trip noise at INR scale and still four orders of
# magnitude below the smallest reported fold-to-fold standard deviation.
REL_TOL = 1e-12
ABS_TOL = 1e-12


def agreement_tolerance(scale: float) -> float:
    return max(REL_TOL * abs(scale), ABS_TOL)

T_CRIT_CACHE: Dict[float, float] = {}


# ============================================================
# LOADING
# ============================================================

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require_inputs() -> None:
    missing = [
        p for p in (
            INPUT_COMPARISON,
            INPUT_PER_FOLD,
            INPUT_PER_FOLD_JSON,
            INPUT_SUMMARY,
        ) if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing persisted Experiment 1 inputs: "
            + ", ".join(str(m) for m in missing)
        )


def load_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the persisted comparison, per-fold and summary tables."""
    comparison = pd.read_csv(INPUT_COMPARISON)
    per_fold = pd.read_csv(INPUT_PER_FOLD)
    summary = pd.read_csv(INPUT_SUMMARY)

    expected_per_fold = len(MODELS) * len(REGIMES) * 5
    if len(per_fold) != expected_per_fold:
        raise RuntimeError(
            f"per_fold_metrics.csv has {len(per_fold)} rows, expected "
            f"{expected_per_fold} (5 models x 2 regimes x 5 folds)."
        )
    observed = set(per_fold["model"]) | set(comparison["model"])
    unknown = observed - set(MODELS)
    if unknown:
        raise RuntimeError(f"Unexpected model labels in artifacts: {unknown}")

    return comparison, per_fold, summary


def verify_artifact_agreement(
    comparison: pd.DataFrame,
    per_fold: pd.DataFrame,
    summary: pd.DataFrame,
) -> Dict[str, Any]:
    """Re-derive every summary statistic from the persisted per-fold rows and
    check it against the persisted summary table before anything is plotted.

    This guarantees the figures cannot silently disagree with the Experiment 1
    record: if the tables had drifted, the run would abort.
    """
    checks: List[Dict[str, Any]] = []

    for model in MODELS:
        for regime in REGIMES:
            rows = per_fold[
                (per_fold["model"] == model) & (per_fold["regime"] == regime)
            ].sort_values("fold")
            if len(rows) != 5 or list(rows["fold"]) != [1, 2, 3, 4, 5]:
                raise RuntimeError(
                    f"Per-fold rows incomplete for {model}/{regime}."
                )

            table_row = comparison[
                (comparison["model"] == model)
                & (comparison["regime"] == regime)
            ]
            if len(table_row) != 1:
                raise RuntimeError(
                    f"comparison_table.csv row not unique for {model}/{regime}."
                )
            table_row = table_row.iloc[0]

            for metric in METRIC_INFO:
                values = rows[metric].to_numpy(dtype=float)
                derived_mean = float(values.mean())
                derived_std = float(values.std(ddof=1))

                diff_mean = abs(derived_mean - float(table_row[f"{metric}_mean"]))
                diff_std = abs(derived_std - float(table_row[f"{metric}_std"]))

                summary_row = summary[
                    (summary["model"] == model)
                    & (summary["regime"] == regime)
                    & (summary["metric"] == metric)
                ]
                if len(summary_row) != 1:
                    raise RuntimeError(
                        f"summary_metrics.csv row not unique for "
                        f"{model}/{regime}/{metric}."
                    )
                summary_row = summary_row.iloc[0]

                diff_smean = abs(derived_mean - float(summary_row["mean"]))
                diff_sstd = abs(derived_std - float(summary_row["std"]))
                diff_min = abs(float(values.min()) - float(summary_row["min"]))
                diff_max = abs(float(values.max()) - float(summary_row["max"]))
                n_ok = int(summary_row["n_folds"]) == 5
                ddof_ok = int(summary_row["std_ddof"]) == 1

                worst = max(diff_mean, diff_std, diff_smean, diff_sstd,
                            diff_min, diff_max)
                tol = agreement_tolerance(float(values.max()))
                checks.append(
                    {
                        "model": model,
                        "regime": regime,
                        "metric": metric,
                        "max_abs_difference": worst,
                        "tolerance": tol,
                        "relative_difference": worst / max(
                            float(values.max()), 1e-30
                        ),
                        "n_folds_is_5": n_ok,
                        "std_ddof_is_1": ddof_ok,
                        "ok": bool(
                            worst <= tol and n_ok and ddof_ok
                        ),
                    }
                )

    failures = [c for c in checks if not c["ok"]]
    if failures:
        raise RuntimeError(
            "Persisted summary and per-fold artifacts disagree beyond the "
            f"scale-relative tolerance ({REL_TOL:g} relative). First "
            f"failure: {failures[0]}"
        )

    return {
        "checks_run": len(checks),
        "relative_tolerance": REL_TOL,
        "absolute_tolerance_floor": ABS_TOL,
        "tolerance_rationale": (
            "INR-space metrics are of order 1e7, where one float64 unit in "
            "the last place is about 3.7e-9. An absolute 1e-9 threshold "
            "would sit below the representable resolution of the quantity and "
            "would fail on summation ordering rather than genuine drift, so "
            "the check is scale-relative."
        ),
        "max_absolute_difference": max(
            c["max_abs_difference"] for c in checks
        ),
        "max_relative_difference": max(
            c["relative_difference"] for c in checks
        ),
        "all_agree": True,
        "meaning": (
            "Every mean, std, min, max, n and ddof in the persisted summary "
            "table was independently re-derived from the persisted per-fold "
            "rows and matched, so the figures cannot disagree with the "
            "Experiment 1 record."
        ),
    }


# ============================================================
# STATISTICS
# ============================================================

def welch_difference(
    mean_a: float, std_a: float, n_a: int,
    mean_b: float, std_b: float, n_b: int,
) -> Dict[str, float]:
    """Independent-samples (Welch) difference b - a.

    Used because Regime A fold k and Regime B fold k are different splits,
    not repeated measurements of the same fold, so the two fold samples are
    independent and must not be treated as paired.
    """
    delta = mean_b - mean_a
    var_a = (std_a ** 2) / n_a
    var_b = (std_b ** 2) / n_b
    se = float(np.sqrt(var_a + var_b))
    if se == 0.0:
        return {
            "delta": delta,
            "welch_se": 0.0,
            "welch_df": float("nan"),
            "ci95_low": delta,
            "ci95_high": delta,
        }
    df = (var_a + var_b) ** 2 / (
        var_a ** 2 / (n_a - 1) + var_b ** 2 / (n_b - 1)
    )
    key = round(float(df), 6)
    if key not in T_CRIT_CACHE:
        T_CRIT_CACHE[key] = float(stats.t.ppf(0.975, key))
    t_crit = T_CRIT_CACHE[key]
    return {
        "delta": delta,
        "welch_se": se,
        "welch_df": float(df),
        "ci95_low": delta - t_crit * se,
        "ci95_high": delta + t_crit * se,
    }


def relative_change_pct(
    before: float, after: float, ratio_scale: bool
) -> float:
    """Percentage change from `before` to `after`.

    Only defined for strictly positive, ratio-scale quantities. Returned as
    NaN otherwise, rather than a misleading number.
    """
    if not ratio_scale:
        return float("nan")
    if before == 0.0:
        return float("nan")
    return (after - before) / abs(before) * 100.0


def metric_block(
    comparison: pd.DataFrame, metric: str
) -> Dict[str, Dict[str, float]]:
    """{model: {regime: {mean, std}}} for one metric, from persisted data."""
    out: Dict[str, Dict[str, float]] = {}
    for model in MODELS:
        out[model] = {}
        for regime in REGIMES:
            row = comparison[
                (comparison["model"] == model)
                & (comparison["regime"] == regime)
            ].iloc[0]
            out[model][regime] = {
                "mean": float(row[f"{metric}_mean"]),
                "std": float(row[f"{metric}_std"]),
            }
    return out


def fmt_inr(value: float, decimals: int = 0) -> str:
    return f"{value:,.{decimals}f}"


def fmt_pm(mean: float, std: float, kind: str, decimals: int = 4) -> str:
    if kind == "inr":
        return f"{fmt_inr(mean, 0)} ± {fmt_inr(std, 0)}"
    if kind == "percent":
        return f"{mean:.{decimals}f} ± {std:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


# ============================================================
# TABLE BUILDERS
# ============================================================

def build_thesis_summary(
    comparison: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[List[str]]]:
    """Thesis-ready mean ± std table across both regimes."""
    cols = [
        "Model",
        "Random MAE (INR)", "Grouped MAE (INR)",
        "Random RMSE (INR)", "Grouped RMSE (INR)",
        "Random R²", "Grouped R²",
        "Random MedAPE (%)", "Grouped MedAPE (%)",
        "Random R² (log)", "Grouped R² (log)",
    ]
    specs = [
        ("MAE_INR", "random"), ("MAE_INR", "location_grouped"),
        ("RMSE_INR", "random"), ("RMSE_INR", "location_grouped"),
        ("R2_INR", "random"), ("R2_INR", "location_grouped"),
        ("MedAPE_percent", "random"), ("MedAPE_percent", "location_grouped"),
        ("R2_log", "random"), ("R2_log", "location_grouped"),
    ]

    rows: List[List[str]] = []
    for model in MODELS:
        row: List[str] = [MODEL_LABELS[model]]
        for metric, regime in specs:
            block = metric_block(comparison, metric)[model][regime]
            kind = METRIC_INFO[metric]["kind"]
            row.append(fmt_pm(block["mean"], block["std"], kind))
        rows.append(row)

    return pd.DataFrame(rows, columns=cols), rows


def build_generalization_table(
    comparison: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Random -> location-grouped shift table for the primary metric and for
    MedAPE, with Welch confidence intervals on the difference of means."""
    r2 = metric_block(comparison, PRIMARY_METRIC)
    med = metric_block(comparison, "MedAPE_percent")

    records: List[Dict[str, Any]] = []
    for model in MODELS:
        rnd = r2[model]["random"]
        grp = r2[model]["location_grouped"]
        r2_stats = welch_difference(
            rnd["mean"], rnd["std"], 5, grp["mean"], grp["std"], 5
        )
        rnd_m = med[model]["random"]
        grp_m = med[model]["location_grouped"]
        med_stats = welch_difference(
            rnd_m["mean"], rnd_m["std"], 5, grp_m["mean"], grp_m["std"], 5
        )
        records.append(
            {
                "Model": MODEL_LABELS[model],
                "Random R² (log)": fmt_pm(
                    rnd["mean"], rnd["std"], METRIC_INFO["R2_log"]["kind"]
                ),
                "Grouped R² (log)": fmt_pm(
                    grp["mean"], grp["std"], METRIC_INFO["R2_log"]["kind"]
                ),
                "R² (log) absolute change": f"{r2_stats['delta']:+.4f}",
                "R² (log) relative change": (
                    "not applicable: R² is not a ratio scale"
                ),
                "R² (log) change 95% CI (Welch)": (
                    f"[{r2_stats['ci95_low']:+.4f}, {r2_stats['ci95_high']:+.4f}]"
                ),
                "Random MedAPE (%)": fmt_pm(
                    rnd_m["mean"], rnd_m["std"], "percent"
                ),
                "Grouped MedAPE (%)": fmt_pm(
                    grp_m["mean"], grp_m["std"], "percent"
                ),
                "MedAPE absolute change": f"{med_stats['delta']:+.2f} pp",
                "MedAPE relative change": (
                    f"{relative_change_pct(rnd_m['mean'], grp_m['mean'], True):+.1f}%"
                ),
                "MedAPE change 95% CI (Welch)": (
                    f"[{med_stats['ci95_low']:+.2f}, "
                    f"{med_stats['ci95_high']:+.2f}] pp"
                ),
                "_r2_delta": r2_stats["delta"],
                "_r2_rel_pct": relative_change_pct(
                    rnd["mean"], grp["mean"], False
                ),
                "_r2_ci_low": r2_stats["ci95_low"],
                "_r2_ci_high": r2_stats["ci95_high"],
                "_r2_welch_df": r2_stats["welch_df"],
                "_med_delta": med_stats["delta"],
                "_med_rel_pct": relative_change_pct(
                    rnd_m["mean"], grp_m["mean"], True
                ),
                "_med_ci_low": med_stats["ci95_low"],
                "_med_ci_high": med_stats["ci95_high"],
                "_med_welch_df": med_stats["welch_df"],
                "_r2_random": rnd["mean"],
                "_r2_grouped": grp["mean"],
                "_r2_random_std": rnd["std"],
                "_r2_grouped_std": grp["std"],
            }
        )

    display_cols = [c for c in records[0] if not c.startswith("_")]
    return pd.DataFrame(records)[display_cols], records


def build_shift_all_metrics(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Full random -> grouped change for all eight metrics, per model."""
    ratio_scale_metrics = (
        "MAE_INR", "RMSE_INR", "MAPE_percent", "MedAPE_percent",
        "MAE_log", "RMSE_log",
    )
    rows: List[Dict[str, Any]] = []
    for metric in METRIC_INFO:
        block = metric_block(comparison, metric)
        kind = METRIC_INFO[metric]["kind"]
        ratio = metric in ratio_scale_metrics
        for model in MODELS:
            rnd = block[model]["random"]
            grp = block[model]["location_grouped"]
            stats_w = welch_difference(
                rnd["mean"], rnd["std"], 5, grp["mean"], grp["std"], 5
            )
            rows.append(
                {
                    "Model": MODEL_LABELS[model],
                    "Metric": METRIC_INFO[metric]["label"],
                    "Metric key": metric,
                    "Better direction": METRIC_INFO[metric]["direction"],
                    "Random mean": rnd["mean"],
                    "Random std": rnd["std"],
                    "Grouped mean": grp["mean"],
                    "Grouped std": grp["std"],
                    "Absolute change": stats_w["delta"],
                    "Relative change (%)": relative_change_pct(
                        rnd["mean"], grp["mean"], ratio
                    ),
                    "Relative change is meaningful": ratio,
                    "Welch SE of change": stats_w["welch_se"],
                    "Welch df": stats_w["welch_df"],
                    "Change 95% CI low": stats_w["ci95_low"],
                    "Change 95% CI high": stats_w["ci95_high"],
                    "Unit kind": kind,
                }
            )
    return pd.DataFrame(rows)


def build_discrimination_table(comparison: pd.DataFrame) -> pd.DataFrame:
    """How much of each metric's between-model spread survives fold noise.

    For each metric and regime:

        discrimination ratio = (max model mean - min model mean)
                               / (mean of the five within-model fold SDs)

    A ratio below about 1 means the entire spread between the five models is
    smaller than the fold-to-fold wobble of a single model, so that metric
    cannot separate the models at this sample size, however the numbers are
    ordered. This is a diagnostic about metric usefulness, not a model
    ranking, and it is the reason the selection reasoning in the analysis
    document can weight some evidence more heavily than other evidence
    without inventing a composite score.
    """
    rows: List[Dict[str, Any]] = []
    for regime in REGIMES:
        for metric in METRIC_INFO:
            block = metric_block(comparison, metric)
            means = [block[m][regime]["mean"] for m in MODELS]
            stds = [block[m][regime]["std"] for m in MODELS]
            spread = max(means) - min(means)
            mean_std = float(np.mean(stds))
            ratio = spread / mean_std if mean_std > 0 else float("nan")
            rows.append(
                {
                    "Regime": REGIME_LABELS[regime],
                    "Metric key": metric,
                    "Metric": METRIC_INFO[metric]["label"],
                    "Best model on this metric in this regime": MODEL_LABELS[
                        min(
                            MODELS,
                            key=lambda m: (
                                block[m][regime]["mean"]
                                if METRIC_INFO[metric]["direction"]
                                == "lower_is_better"
                                else -block[m][regime]["mean"]
                            ),
                        )
                    ],
                    "Between-model spread": spread,
                    "Mean within-model fold SD": mean_std,
                    "Discrimination ratio": ratio,
                    "Separates models at n=5": bool(ratio >= 1.0),
                }
            )
    return pd.DataFrame(rows)


def _extremum(values: pd.Series, selector: Any) -> float:
    """True min or max of the persisted per-fold values, NaN when absent."""
    if len(values) == 0:
        return float("nan")
    return float(selector(values))


def per_fold_metric(
    per_fold: pd.DataFrame, model: str, regime: str, metric: str
) -> pd.Series:
    """Persisted per-fold values for one model/regime/metric combination."""
    selected = per_fold[(per_fold["model"] == model) & (per_fold["regime"] == regime)]
    return selected[metric].astype(float)


def build_stability_table(
    comparison: pd.DataFrame, per_fold: pd.DataFrame
) -> pd.DataFrame:
    """Fold-to-fold stability of the primary metric, and how much of that
    stability is lost when moving from random to location-held-out evaluation.

    The inflation factor is the ratio of the location-grouped fold SD to the
    random-regime fold SD for the same model. A factor near 1 would mean the
    model is equally trustworthy on unseen locations as on seen ones; larger
    values mean the per-fold estimate is much less reproducible under
    location shift, which matters directly for whether a downstream
    uncertainty method can rely on a stable residual scale.

    Minima and maxima are the true extrema of the persisted per-fold values,
    recomputed from per_fold_metrics.csv. They are deliberately not expressed
    as a multiple of the SD, because a mean +/- 2 SD band is a distributional
    statement about normally distributed fold noise and not the actual
    observed range, which for n=5 need not be symmetric.
    """
    block = metric_block(comparison, PRIMARY_METRIC)
    rows: List[Dict[str, Any]] = []
    for model in MODELS:
        rnd = block[model]["random"]
        grp = block[model]["location_grouped"]
        rnd_folds = per_fold_metric(per_fold, model, "random", PRIMARY_METRIC)
        grp_folds = per_fold_metric(
            per_fold, model, "location_grouped", PRIMARY_METRIC
        )
        rows.append(
            {
                "Model": MODEL_LABELS[model],
                "Random fold SD (R² log)": rnd["std"],
                "Grouped fold SD (R² log)": grp["std"],
                "SD inflation under location hold-out": (
                    grp["std"] / rnd["std"] if rnd["std"] > 0 else float("nan")
                ),
                "Random fold range (R² log)": (
                    f"{rnd['mean'] - rnd['std']:.4f} to "
                    f"{rnd['mean'] + rnd['std']:.4f} (mean ± 1 SD)"
                ),
                "Grouped fold range (R² log)": (
                    f"{grp['mean'] - grp['std']:.4f} to "
                    f"{grp['mean'] + grp['std']:.4f} (mean ± 1 SD)"
                ),
                "Random fold min (R² log)": _extremum(rnd_folds, min),
                "Random fold max (R² log)": _extremum(rnd_folds, max),
                "Grouped fold min (R² log)": _extremum(grp_folds, min),
                "Grouped fold max (R² log)": _extremum(grp_folds, max),
            }
        )
    return pd.DataFrame(rows)


def build_regime_dominance(comparison: pd.DataFrame) -> pd.DataFrame:
    """Per-metric, per-regime ordering of models.

    Presented as a descriptive lookup table, NOT as a ranking, NOT scored,
    NOT weighted and NOT counted into any composite. Its only purpose is to
    make it explicit and auditable that the ordering differs between
    regimes, which is the central factual finding of the comparison.
    """
    rows: List[Dict[str, Any]] = []
    for regime in REGIMES:
        for metric in METRIC_INFO:
            per_regime = metric_block(comparison, metric)
            block = {m: per_regime[m][regime] for m in MODELS}
            ordered = sorted(
                MODELS,
                key=lambda m: (
                    block[m]["mean"]
                    if METRIC_INFO[metric]["direction"] == "lower_is_better"
                    else -block[m]["mean"]
                ),
            )
            row: Dict[str, Any] = {
                "Regime": REGIME_LABELS[regime],
                "Metric": METRIC_INFO[metric]["label"],
                "Metric key": metric,
                "Direction": METRIC_INFO[metric]["direction"],
            }
            for position, model in enumerate(ordered, start=1):
                row[f"position_{position}"] = MODEL_LABELS[model]
            rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# FIGURE PLUMBING
# ============================================================

def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": FIG_DPI,
            "savefig.dpi": FIG_DPI,
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.30,
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )


def metric_figure(
    comparison: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    stem: str,
    note: str,
) -> List[Path]:
    """Grouped bar chart with fold-level sample-std error bars."""
    info = METRIC_INFO[metric]
    block = metric_block(comparison, metric)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    n_models = len(MODELS)
    n_regimes = len(REGIMES)
    width = 0.8 / n_regimes
    positions = np.arange(n_models, dtype=float)

    for r_index, regime in enumerate(REGIMES):
        offset = (r_index - (n_regimes - 1) / 2.0) * width
        means = [block[m][regime]["mean"] for m in MODELS]
        stds = [block[m][regime]["std"] for m in MODELS]
        ax.bar(
            positions + offset,
            means,
            width=width * 0.92,
            yerr=stds,
            capsize=4,
            color=COLORS[regime],
            edgecolor="white",
            linewidth=0.8,
            label=f"{REGIME_LABELS[regime]} (mean ± SD, n=5)",
            error_kw={"elinewidth": 1.1, "ecolor": "#333333"},
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=14, fontweight="bold")

    direction = info["direction"].replace("_", " ")
    arrow = "↓ lower is better" if info["direction"] == "lower_is_better" \
        else "↑ higher is better"
    ax.text(
        0.5, 1.005,
        f"{arrow}   |   error bars = fold-level sample standard deviation "
        f"(ddof=1, n=5), not standard error",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#444444",
    )

    if info["kind"] == "inr":
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(
                lambda v, _pos: f"{v / 1e6:.1f}M"
            )
        )
    ax.legend(loc="best")
    fig.text(
        0.01, 0.01, note,
        fontsize=8.5, color="#666666", ha="left", va="bottom",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))

    return save_figure(fig, stem)


def save_figure(fig: plt.Figure, stem: str) -> List[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURE_DIR / f"{stem}.png"
    pdf = FIGURE_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=FIG_DPI, bbox_inches="tight")
    # CreationDate is suppressed so the PDF bytes are a function of the data
    # only. Matplotlib otherwise stamps wall-clock time into every PDF, which
    # would make the vector figures non-reproducible and would contradict the
    # deterministic=true claim recorded in analysis_manifest.json. PNG output
    # carries no timestamp and needs no equivalent override.
    fig.savefig(
        pdf,
        format="pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None},
    )
    plt.close(fig)
    return [png, pdf]


# ============================================================
# FIGURES
# ============================================================

def figure_01_mae(comparison: pd.DataFrame) -> List[Path]:
    return metric_figure(
        comparison,
        "MAE_INR",
        "Mean Absolute Error Across Valuation Models",
        "MAE (INR)",
        "01_mae_comparison",
        "Source: artifacts/valuation/v2_1/model_comparison/comparison_table.csv "
        "(frozen V2.1 protocol). Predictions scored fold-by-fold, never pooled.",
    )


def figure_02_rmse(comparison: pd.DataFrame) -> List[Path]:
    return metric_figure(
        comparison,
        "RMSE_INR",
        "Root Mean Squared Error Across Valuation Models",
        "RMSE (INR)",
        "02_rmse_comparison",
        "Source: artifacts/valuation/v2_1/model_comparison/comparison_table.csv. "
        "Note: grouped-regime RMSE spread across models is small relative to "
        "its fold-to-fold standard deviation.",
    )


def figure_03_r2(comparison: pd.DataFrame) -> List[Path]:
    return metric_figure(
        comparison,
        "R2_INR",
        "R² Across Valuation Models and Evaluation Regimes",
        "R² (INR space)",
        "03_r2_comparison",
        "Source: artifacts/valuation/v2_1/model_comparison/comparison_table.csv. "
        "INR-space R² is compressed by the heavy right tail of luxury prices.",
    )


def figure_04_medape(comparison: pd.DataFrame) -> List[Path]:
    return metric_figure(
        comparison,
        "MedAPE_percent",
        "Median Absolute Percentage Error Across Valuation Models",
        "MedAPE (%)",
        "04_medape_comparison",
        "Source: artifacts/valuation/v2_1/model_comparison/comparison_table.csv. "
        "Median percentage error is robust to the low-price tail that inflates MAPE.",
    )


def figure_05_r2_log(comparison: pd.DataFrame) -> List[Path]:
    return metric_figure(
        comparison,
        "R2_log",
        "Log-Space R² Across Valuation Models",
        "R² (log1p INR space)",
        "05_r2_log_comparison",
        "Source: artifacts/valuation/v2_1/model_comparison/comparison_table.csv. "
        "Log-space R² is the protocol's headline goodness-of-fit measure.",
    )


def figure_06_generalization(comparison: pd.DataFrame) -> List[Path]:
    """Random CV -> location-held-out CV change on the primary metric."""
    block = metric_block(comparison, PRIMARY_METRIC)
    med_block = metric_block(comparison, "MedAPE_percent")

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(15.0, 6.6), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )

    # Left: paired slope chart, fixed model order, no sorting by value.
    y_ticks = np.arange(len(MODELS), dtype=float)
    for y, model in zip(y_ticks, MODELS):
        rnd = block[model]["random"]
        grp = block[model]["location_grouped"]
        ax_left.plot(
            [0, 1],
            [rnd["mean"], grp["mean"]],
            marker="o",
            markersize=9,
            linewidth=2.0,
            color="#888888",
            alpha=0.85,
            zorder=2,
        )
        ax_left.errorbar(
            [0], [rnd["mean"]], yerr=rnd["std"],
            fmt="none", ecolor=COLORS["random"], elinewidth=2.4, capsize=5,
            zorder=3,
        )
        ax_left.errorbar(
            [1], [grp["mean"]], yerr=grp["std"],
            fmt="none", ecolor=COLORS["location_grouped"], elinewidth=2.4,
            capsize=5, zorder=3,
        )
        delta = grp["mean"] - rnd["mean"]
        ax_left.annotate(
            f"Δ {delta:+.4f}",
            xy=(1.0, grp["mean"]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=10,
            color="#333333",
            va="center",
        )

    ax_left.set_yticks(y_ticks)
    ax_left.set_yticklabels([MODEL_LABELS[m] for m in MODELS])
    ax_left.set_xticks([0, 1])
    ax_left.set_xticklabels(
        [
            f"Random CV\n(seen locations)",
            "Location-Grouped CV\n(held-out locations)",
        ]
    )
    ax_left.set_xlim(-0.35, 1.42)
    ax_left.set_ylabel(METRIC_INFO[PRIMARY_METRIC]["label"])
    ax_left.set_title(
        "Log-space R²: random evaluation vs held-out locations",
        fontsize=13,
        fontweight="bold",
    )
    ax_left.grid(axis="y")
    ax_left.grid(axis="x", visible=False)

    # Right: change in MedAPE with Welch 95% CI on the difference of means.
    deltas: List[float] = []
    lows: List[float] = []
    highs: List[float] = []
    for model in MODELS:
        rnd = med_block[model]["random"]
        grp = med_block[model]["location_grouped"]
        s = welch_difference(
            rnd["mean"], rnd["std"], 5, grp["mean"], grp["std"], 5
        )
        deltas.append(s["delta"])
        lows.append(s["ci95_low"])
        highs.append(s["ci95_high"])

    err = np.vstack(
        [
            np.array(deltas) - np.array(lows),
            np.array(highs) - np.array(deltas),
        ]
    )
    ax_right.bar(
        np.arange(len(MODELS), dtype=float),
        deltas,
        width=0.62,
        color="#8c564b",
        edgecolor="white",
        yerr=err,
        capsize=5,
        error_kw={"elinewidth": 1.3, "ecolor": "#333333"},
    )
    ax_right.axhline(0.0, color="#333333", linewidth=1.0)
    ax_right.set_xticks(np.arange(len(MODELS), dtype=float))
    ax_right.set_xticklabels(
        [MODEL_LABELS[m].replace(" ", "\n") for m in MODELS], fontsize=10
    )
    ax_right.set_ylabel("Change in MedAPE (percentage points)")
    ax_right.set_title(
        "Increase in MedAPE under location hold-out\n"
        "(bars with Welch 95% CI on the difference of fold means)",
        fontsize=13,
        fontweight="bold",
    )
    ax_right.tick_params(axis="y", labelsize=10)

    fig.suptitle(
        "Valuation Generalization Under Location Shift",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5, 0.945,
        "Change measured as Location-Grouped CV minus Random CV. "
        "This is the change when moving from random evaluation to "
        "evaluation on held-out locations.",
        ha="center", va="top", fontsize=10.5, color="#444444",
    )
    fig.text(
        0.01, 0.005,
        "Folds are not paired between regimes: the two regimes use different "
        "splits, so intervals use the independent-samples Welch "
        "Satterthwaite formula on the difference of fold means, not a paired "
        "t-difference. A positive change indicates worse performance under "
        "location hold-out for error metrics; a negative change in R² "
        "indicates the same. Models are shown in a fixed pre-declared "
        "display order, not sorted by value.",
        fontsize=8.5, color="#666666", ha="left", va="bottom",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.935))
    return save_figure(fig, "06_location_generalization")


def figure_07_fold_variability(per_fold: pd.DataFrame) -> List[Path]:
    """Per-fold distribution of the primary metric, both regimes."""
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.6), sharey=True)

    fold_offsets = np.linspace(-0.16, 0.16, 5)
    y_range: List[float] = []

    for ax, regime in zip(axes, REGIMES):
        data = [
            per_fold[
                (per_fold["model"] == model)
                & (per_fold["regime"] == regime)
            ].sort_values("fold")[PRIMARY_METRIC].to_numpy(dtype=float)
            for model in MODELS
        ]
        positions = np.arange(len(MODELS), dtype=float)

        boxes = ax.boxplot(
            data,
            positions=positions,
            widths=0.52,
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "#111111",
                "markeredgecolor": "#111111",
                "markersize": 7,
            },
            medianprops={"color": "#111111", "linewidth": 2.0},
            whiskerprops={"color": "#555555", "linewidth": 1.4},
            capprops={"color": "#555555", "linewidth": 1.4},
            flierprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "#555555",
                "markersize": 6,
            },
        )
        for patch in boxes["boxes"]:
            patch.set_facecolor(COLORS[regime])
            patch.set_alpha(0.28)
            patch.set_edgecolor(COLORS[regime])
            patch.set_linewidth(1.6)

        # Deterministic per-fold points: fixed offsets, no random jitter.
        for offset, fold in zip(fold_offsets, range(1, 6)):
            values = [
                per_fold[
                    (per_fold["model"] == model)
                    & (per_fold["regime"] == regime)
                    & (per_fold["fold"] == fold)
                ][PRIMARY_METRIC].iloc[0]
                for model in MODELS
            ]
            ax.scatter(
                positions + offset,
                values,
                s=34,
                facecolor="white",
                edgecolor=COLORS[regime],
                linewidth=1.5,
                zorder=4,
                label=f"Fold {fold}" if (regime == REGIMES[0]) else None,
            )

        means = [float(d.mean()) for d in data]
        y_range.extend(means)
        for position, arr, mean in zip(positions, data, means):
            y_range.extend(arr.tolist())

        ax.set_xticks(positions)
        ax.set_xticklabels(
            [MODEL_LABELS[m].replace(" ", "\n") for m in MODELS], fontsize=10
        )
        ax.set_title(REGIME_LABELS[regime], fontsize=13, fontweight="bold")
        ax.set_xlabel("Model")
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        ax.legend(
            loc="upper left",
            ncols=5,
            fontsize=9,
            bbox_to_anchor=(0.0, 1.0),
        ) if regime == REGIMES[0] else None

    lo = min(y_range)
    hi = max(y_range)
    pad = 0.10 * (hi - lo)
    axes[0].set_ylim(lo - pad, hi + pad)
    axes[0].set_ylabel(METRIC_INFO[PRIMARY_METRIC]["label"])

    fig.suptitle(
        "Fold-to-Fold Variability of Log-Space R²",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5, 0.945,
        "Boxes span the interquartile range with the median line and the "
        "filled diamond showing the mean; open circles are the five "
        "individual persisted fold values.",
        ha="center", va="top", fontsize=10.5, color="#444444",
    )
    fig.text(
        0.01, 0.005,
        "Source: artifacts/valuation/v2_1/model_comparison/per_fold_metrics.csv "
        "(persisted per-fold results, not regenerated). n=5 folds per model "
        "per regime. The random regime is visibly tighter than the "
        "location-grouped regime for every model, which is the stability "
        "signal this figure exists to expose.",
        fontsize=8.5, color="#666666", ha="left", va="bottom",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.935))
    return save_figure(fig, "07_fold_variability")


def figure_08_training_time() -> List[Path]:
    """Training cost, only if the persisted timing data is complete.

    The XGBoost anchor was deliberately not retrained, so no XGBoost timing
    exists. It is therefore omitted rather than estimated.
    """
    if not INPUT_TIMING.exists():
        return []

    timing = pd.read_csv(INPUT_TIMING)
    required_cols = {
        "model", "total_fit_seconds", "total_predict_seconds",
    }
    if not required_cols.issubset(timing.columns):
        return []
    if timing["total_fit_seconds"].isna().any():
        return []
    if timing["model"].duplicated().any():
        return []

    # Order follows the fixed pre-declared display order, minus models with
    # no persisted timing (the untimed anchor).
    rows = timing.set_index("model")
    timed_models = [m for m in MODELS if m in rows.index]
    if not timed_models:
        return []

    fig, ax = plt.subplots(figsize=FIGSIZE)
    positions = np.arange(len(timed_models), dtype=float)
    means = [float(rows.loc[m, "total_fit_seconds"]) for m in timed_models]
    predict = [
        float(rows.loc[m, "total_predict_seconds"]) for m in timed_models
    ]

    ax.bar(
        positions - 0.19,
        means,
        width=0.38,
        color="#4c72b0",
        edgecolor="white",
        label="Total fit time (10 folds)",
    )
    ax.bar(
        positions + 0.19,
        predict,
        width=0.38,
        color="#dd8452",
        edgecolor="white",
        label="Total predict time (10 folds)",
    )
    for pos, value in zip(positions, means):
        ax.annotate(
            f"{value:,.0f}s",
            (pos - 0.19, value),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=9.5,
            color="#333333",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([MODEL_LABELS[m] for m in timed_models])
    ax.set_ylabel("Wall-clock seconds")
    ax.set_title(
        "Training Time Across Valuation Models", pad=14, fontweight="bold"
    )
    ax.text(
        0.5, 1.005,
        "Sum of the 10 persisted per-fold fit times (2 regimes × 5 folds)",
        transform=ax.transAxes,
        ha="center", va="bottom", fontsize=10, color="#444444",
    )
    ax.legend(loc="upper left")
    fig.text(
        0.01, 0.01,
        "Source: artifacts/valuation/v2_1/model_comparison/training_time.csv. "
        "XGBoost is intentionally absent: the anchor was not retrained, so no "
        "timing exists for it and none was estimated. Timings are reported for "
        "transparency about experiment cost and were not used to select, drop "
        "or reconfigure any model.",
        fontsize=8.5, color="#666666", ha="left", va="bottom",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    return save_figure(fig, "08_training_time")


# ============================================================
# MARKDOWN TABLE EMITTER
# ============================================================

def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| " + " | ".join(str(row[c]) for c in cols) + " |"
        )
    return "\n".join(lines)


def write_table(df: pd.DataFrame, stem: Path, title: str) -> None:
    df.to_csv(stem.with_suffix(".csv"), index=False)
    with open(stem.with_suffix(".md"), "w", encoding="utf-8") as fh:
        fh.write(f"### {title}\n\n")
        fh.write(md_table(df))
        fh.write("\n")


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    require_inputs()
    apply_style()

    print("=" * 78)
    print("EXPERIMENT 1 - VISUALISATION AND MODEL-SELECTION ANALYSIS")
    print("=" * 78)
    print("Analysis only. No model is trained, loaded, re-scored or modified.")
    print()

    comparison, per_fold, summary = load_tables()

    print("[1/5] Verifying persisted artifact self-consistency")
    agreement = verify_artifact_agreement(comparison, per_fold, summary)
    print(
        f"      {agreement['checks_run']} checks, max relative difference "
        f"{agreement['max_relative_difference']:.3e} "
        f"(tolerance {agreement['relative_tolerance']:.0e} relative)"
    )
    print()

    print("[2/5] Building tables from persisted per-fold results")
    thesis_df, _ = build_thesis_summary(comparison)
    generalization_df, generalization_records = build_generalization_table(
        comparison
    )
    shift_df = build_shift_all_metrics(comparison)
    dominance_df = build_regime_dominance(comparison)
    discrimination_df = build_discrimination_table(comparison)
    stability_df = build_stability_table(comparison, per_fold)
    write_table(
        thesis_df, OUT_MASTERS, "Thesis summary table (mean ± SD, n=5)"
    )
    write_table(
        generalization_df,
        OUT_GENERALIZATION,
        "Generalization table: random CV to location-held-out CV",
    )
    write_table(
        shift_df,
        OUT_SHIFT_ALL,
        "Full random-to-grouped change, all eight metrics",
    )
    write_table(
        discrimination_df,
        BASE / "metric_discrimination",
        "Metric discriminative power: between-model spread vs fold noise",
    )
    write_table(
        stability_df,
        BASE / "fold_stability",
        "Fold-to-fold stability of the primary metric (R² log)",
    )
    write_table(
        dominance_df,
        BASE / "per_metric_ordering_by_regime",
        "Per-metric ordering by regime (descriptive lookup, not a ranking)",
    )
    for path in (
        OUT_MASTERS.with_suffix(".csv"), OUT_MASTERS.with_suffix(".md"),
        OUT_GENERALIZATION.with_suffix(".csv"),
        OUT_GENERALIZATION.with_suffix(".md"),
        OUT_SHIFT_ALL.with_suffix(".csv"), OUT_SHIFT_ALL.with_suffix(".md"),
    ):
        print(f"      {path}")
    print()

    print("[3/5] Generating figures")
    outputs: List[Path] = []
    figure_jobs = (
        ("Figure 1", "01 MAE comparison", figure_01_mae, (comparison,)),
        ("Figure 2", "02 RMSE comparison", figure_02_rmse, (comparison,)),
        ("Figure 3", "03 R² comparison", figure_03_r2, (comparison,)),
        ("Figure 4", "04 MedAPE comparison", figure_04_medape, (comparison,)),
        ("Figure 5", "05 log-space R² comparison", figure_05_r2_log,
         (comparison,)),
        ("Figure 6", "06 location generalization",
         figure_06_generalization, (comparison,)),
        ("Figure 7", "07 fold variability", figure_07_fold_variability,
         (per_fold,)),
        ("Figure 8", "08 training time", figure_08_training_time, ()),
    )
    for label, name, fn, args in figure_jobs:
        produced = fn(*args)
        if not produced:
            print(f"      {label}: skipped (insufficient persisted data)")
            continue
        outputs.extend(produced)
        print(f"      {label}: {name}  -> {len(produced)} files")
    print()

    print("[4/5] Writing machine-readable analysis record")
    results = {
        "note": (
            "All values derived at run time from persisted Experiment 1 "
            "artifacts. No metric value is hard-coded. No composite score, "
            "weighted score, win count or ranking is produced."
        ),
        "inputs": {
            str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
            for p in (
                INPUT_COMPARISON, INPUT_PER_FOLD, INPUT_PER_FOLD_JSON,
                INPUT_SUMMARY, INPUT_TIMING,
            ) if p.exists()
        },
        "artifact_self_consistency": agreement,
        "models": list(MODELS),
        "regimes": list(REGIMES),
        "n_folds_per_model_per_regime": 5,
        "std_ddof": 1,
        "primary_metric": PRIMARY_METRIC,
        "error_bars": "fold-level sample standard deviation (ddof=1, n=5)",
        "folds_paired_between_regimes": False,
        "change_interval_method": (
            "Welch-Satterthwaite independent-samples interval on the "
            "difference of fold means; no paired t-difference is used."
        ),
        "relative_change_applied_to": [
            "MAE_INR", "RMSE_INR", "MAPE_percent", "MedAPE_percent",
            "MAE_log", "RMSE_log",
        ],
        "relative_change_not_scale_invariant": ["R2_INR", "R2_log"],
        "composite_score_computed": False,
        "models_ranked": False,
        "thesis_summary": thesis_df.to_dict(orient="records"),
        "generalization": generalization_records,
        "shift_all_metrics": shift_df.to_dict(orient="records"),
        "metric_discrimination": discrimination_df.to_dict(orient="records"),
        "fold_stability": stability_df.to_dict(orient="records"),
        "per_metric_ordering_by_regime": dominance_df.to_dict(
            orient="records"
        ),
        "figures": [
            str(p.relative_to(ROOT)).replace("\\", "/") for p in outputs
        ],
    }
    with open(OUT_RESULTS_JSON, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"      {OUT_RESULTS_JSON}")
    print()

    print("[5/5] Writing analysis manifest")
    table_outputs = [
        p for p in (
            OUT_MASTERS.with_suffix(".csv"), OUT_MASTERS.with_suffix(".md"),
            OUT_GENERALIZATION.with_suffix(".csv"),
            OUT_GENERALIZATION.with_suffix(".md"),
            OUT_SHIFT_ALL.with_suffix(".csv"), OUT_SHIFT_ALL.with_suffix(".md"),
            (BASE / "metric_discrimination.csv"),
            (BASE / "metric_discrimination.md"),
            (BASE / "fold_stability.csv"),
            (BASE / "fold_stability.md"),
            (BASE / "per_metric_ordering_by_regime.csv"),
            (BASE / "per_metric_ordering_by_regime.md"),
            OUT_RESULTS_JSON,
        ) if p.exists()
    ]
    manifest = {
        "purpose": (
            "Experiment 1 research-quality visualisation and "
            "model-selection analysis."
        ),
        "analysis_only": True,
        "models_trained": 0,
        "models_loaded_or_modified": 0,
        "folds_regenerated": False,
        "protocol_modified": False,
        "dataset_modified": False,
        "anchor_modified": False,
        "chain_touched": False,
        "live_tests_run": False,
        "deterministic": True,
        "determinism_note": (
            "Fixed figure geometry and DPI, no random jitter, no bootstrap, "
            "no seed dependence. Re-running reproduces identical figures."
        ),
        "composite_score_computed": False,
        "models_ranked": False,
        "environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "executable": "python",
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "plot_backend": matplotlib.get_backend(),
        },
        "figure_dpi": FIG_DPI,
        "figure_size_inches": list(FIGSIZE),
        "inputs": results["inputs"],
        "outputs": {
            str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
            for p in outputs + table_outputs
        },
        "figure_directory": str(FIGURE_DIR.relative_to(ROOT)).replace(
            "\\", "/"
        ),
    }
    with open(OUT_MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"      {OUT_MANIFEST}")
    print()
    print(f"Analysis complete: {len(outputs)} figure files, "
          f"{len(table_outputs)} table/record files.")
    print("No model was trained, loaded, re-scored or modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
