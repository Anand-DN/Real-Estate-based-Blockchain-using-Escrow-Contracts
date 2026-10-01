"""
MILLOW RESEARCH EXPERIMENT 1
Five-model baseline comparison on the FROZEN V2.1 protocol.

XGBoost is NOT retrained. Its per-fold results are READ from the existing V2.1
anchor artifacts and joined into the comparison tables.

This script trains only: Ridge, Random Forest, LightGBM, CatBoost.

MODEL-FAIRNESS RULE enforced by this file
----------------------------------------
Every configuration below is PRE-DECLARED in this file, before any model is
trained, and is frozen for the whole run. There is no search of any kind:
no grid search, no random search, no optuna, no early stopping, and no
configuration is selected after observing results. The four new
configurations are deliberately matched to the already-frozen XGBoost anchor
on the three structural axes that govern capacity, so the comparison is not
confounded by unequal budget:

                       Ridge  RF     LGBM   CatBoost  XGBoost(anchor)
    boosting rounds      n/a    n/a    700     700       700
    learning rate        n/a    n/a    0.04    0.04      0.04
    max depth            n/a    7      7       7         7

FOLDS
-----
Fold assignments are LOADED from the persisted, committed contract at
artifacts/valuation/v2_1/cv_folds_v2_1.json. generate_folds, KFold and
GroupKFold are never called. Every model therefore sees byte-identical
train/validation indices, which is what makes the comparison valid.

METRICS
-------
All eight protocol metrics are computed per fold. Predictions are scored
fold-by-fold and are NEVER pooled into a single headline score. Summary
statistics are fold-level mean, sample std (ddof=1), min, max and n.

SAFETY
------
Writes only under models/valuation/v2_1/model_comparison/ and
artifacts/valuation/v2_1/model_comparison/. Historical V2 artifacts and the
XGBoost anchor are never written. No blockchain interaction whatsoever.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.valuation_v2_1 import protocol as P  # noqa: E402


# ============================================================
# NAMESPACING
# ============================================================

COMPARISON_DIR = ROOT / "artifacts" / "valuation" / "v2_1" / "model_comparison"
MODEL_DIR = ROOT / "models" / "valuation" / "v2_1" / "model_comparison"

CONFIG_JSON = COMPARISON_DIR / "model_configurations.json"
PER_FOLD_CSV = COMPARISON_DIR / "per_fold_metrics.csv"
SUMMARY_CSV = COMPARISON_DIR / "summary_metrics.csv"
PER_FOLD_JSON = COMPARISON_DIR / "per_fold_metrics.json"
ENVIRONMENT_JSON = COMPARISON_DIR / "environment.json"
MANIFEST_JSON = COMPARISON_DIR / "experiment_manifest.json"
COMPARISON_TABLE_CSV = COMPARISON_DIR / "comparison_table.csv"
TIMING_CSV = COMPARISON_DIR / "training_time.csv"

# XGBoost anchor: READ ONLY, never retrained.
ANCHOR_METRICS_JSON = (
    ROOT / "artifacts" / "valuation" / "v2_1" / "xgb_anchor_metrics_v2_1.json"
)
ANCHOR_FOLD_DIR = ROOT / "models" / "valuation" / "v2_1" / "xgb_anchor_folds"

REGIMES: Tuple[str, ...] = ("random", "location_grouped")


def assert_safe_write_targets() -> None:
    """Refuse to run if any output path could collide with protected files."""
    forbidden = (
        ROOT / "models" / "valuation" / "valuation_v2",
        ROOT / "artifacts" / "valuation" / "valuation_v2",
        ROOT / "models" / "valuation" / "v2_1" / "xgb_anchor_folds",
        ROOT / "artifacts" / "valuation" / "v2_1" / "cv_folds_v2_1.json",
        ROOT / "artifacts" / "valuation" / "v2_1" / "xgb_anchor_metrics_v2_1.json",
    )
    for path in (
        COMPARISON_DIR, MODEL_DIR, CONFIG_JSON, PER_FOLD_CSV, SUMMARY_CSV,
        PER_FOLD_JSON, ENVIRONMENT_JSON, MANIFEST_JSON,
        COMPARISON_TABLE_CSV, TIMING_CSV,
    ):
        resolved = str(path.resolve())
        for bad in forbidden:
            if resolved.startswith(str(bad)):
                raise RuntimeError(
                    f"Refusing to write {resolved}: collides with {bad}."
                )


# ============================================================
# PRE-DECLARED FIXED CONFIGURATIONS
# Frozen here before any training. Not changeable after results are seen.
# ============================================================

# Ridge: alpha=1.0 is scikit-learn's documented default. Fixed here, not
# selected by search. No early stopping, no solver search.
RIDGE_CONFIG: Dict[str, Any] = {
    "estimator": "sklearn.linear_model.Ridge",
    "params": {
        "alpha": 1.0,
        "random_state": 42,
        "fit_intercept": True,
    },
    "requires_dense_output": False,
    "rationale": (
        "alpha=1.0 is the scikit-learn documented default, fixed a priori "
        "rather than searched. Ridge is a closed-form convex model with no "
        "boosting rounds, no learning rate and no depth, so the three "
        "capacity axes matched against the boosting models are not "
        "applicable to it."
    ),
}

# RandomForest: n_estimators and max_depth matched to the frozen XGBoost
# anchor (700 rounds-equivalent trees, depth 7) so no model gets a larger
# budget than another. leaf/ node sizes matched to XGBoost's
# min_child_weight=5 by using min_samples_leaf=5.
RF_CONFIG: Dict[str, Any] = {
    "estimator": "sklearn.ensemble.RandomForestRegressor",
    "params": {
        "n_estimators": 700,
        "max_depth": 7,
        "min_samples_leaf": 5,
        "min_samples_split": 10,
        "random_state": 42,
        "n_jobs": -1,
    },
    "requires_dense_output": False,
    "rationale": (
        "n_estimators=700 and max_depth=7 are set equal to the frozen XGBoost "
        "anchor's boosting-round and depth budget. min_samples_leaf=5 and "
        "min_samples_split=10 are fixed a priori to approximate XGBoost's "
        "min_child_weight=5 leaf-size control. No search was performed."
    ),
}

# LightGBM: boosting rounds, learning rate and depth matched to the anchor.
# num_leaves is set to 2**depth - 1 = 127, the maximum expressible at depth 7,
# so the depth cap binds exactly as it does for XGBoost.
LGBM_CONFIG: Dict[str, Any] = {
    "estimator": "lightgbm.LGBMRegressor",
    "params": {
        "n_estimators": 700,
        "learning_rate": 0.04,
        "num_leaves": 127,
        "max_depth": 7,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },
    "requires_dense_output": False,
    "rationale": (
        "n_estimators=700, learning_rate=0.04 and max_depth=7 equal the frozen "
        "XGBoost anchor exactly. num_leaves=127 = 2**7-1 so the depth-7 cap "
        "binds identically to XGBoost rather than being separately tunable. "
        "Subsample and colsample values match the anchor. No early stopping: "
        "verbose=-1 and no early_stopping argument. No search."
    ),
}

# CatBoost: iterations, learning rate and depth matched to the anchor.
# CatBoost does not accept scipy sparse input, so it is the one model that
# needs dense_output=True. That changes no fitted statistic.
CATBOOST_CONFIG: Dict[str, Any] = {
    "estimator": "catboost.CatBoostRegressor",
    "params": {
        "iterations": 700,
        "learning_rate": 0.04,
        "depth": 7,
        "l2_leaf_reg": 2.0,
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
    },
    "requires_dense_output": True,
    "rationale": (
        "iterations=700, learning_rate=0.04 and depth=7 equal the frozen XGBoost "
        "anchor exactly. l2_leaf_reg=2.0 matches the anchor's reg_lambda=2.0. "
        "verbose=False. No early_stopping_rounds. No search. "
        "requires_dense_output=True because CatBoost rejects scipy sparse "
        "matrices; this is an input-format accommodation that changes no "
        "fitted statistic, not a model-specific preprocessing advantage."
    ),
}

ALL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Ridge": RIDGE_CONFIG,
    "RandomForest": RF_CONFIG,
    "LightGBM": LGBM_CONFIG,
    "CatBoost": CATBOOST_CONFIG,
}


def build_estimator(name: str) -> Any:
    """Instantiate one pre-declared estimator."""
    if name == "Ridge":
        return Ridge(**RIDGE_CONFIG["params"])
    if name == "RandomForest":
        return RandomForestRegressor(**RF_CONFIG["params"])
    if name == "LightGBM":
        return LGBMRegressor(**LGBM_CONFIG["params"])
    if name == "CatBoost":
        return CatBoostRegressor(**CATBOOST_CONFIG["params"])
    raise ValueError(f"Unknown model: {name!r}")


# ============================================================
# ENVIRONMENT RECORD (machine-independent)
# ============================================================

def environment_record() -> Dict[str, Any]:
    """Record exact installed versions. No machine-specific paths.

    Delegates to the protocol's canonical recorder so the comparison
    environment can never drift from artifacts/valuation/v2_1/environment_v2_1.json.
    """
    record = P.environment_record()
    record["install_note"] = (
        "lightgbm 4.7.0 and catboost 1.2.10 were installed for this "
        "experiment because both were absent at V2.1 protocol creation. "
        "No pre-existing package was upgraded or downgraded: python, "
        "pandas, numpy, scikit-learn, xgboost and joblib are at the exact "
        "versions recorded for the XGBoost anchor. catboost pulled in "
        "plotly and graphviz as transitive dependencies."
    )
    return record


# ============================================================
# TRAINING AND SCORING
# ============================================================

def run_model(
    name: str,
    df: pd.DataFrame,
    folds: P.FoldSet,
    regime: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Train one model on each persisted fold. Never regenerates folds."""
    records: List[Dict[str, Any]] = []
    timing: Dict[str, float] = {}

    dense = ALL_CONFIGS[name]["requires_dense_output"]
    estimator_params = ALL_CONFIGS[name]["params"]

    x_all = df[P.INPUT_FEATURES]
    y_all = P.log_target(df[P.TARGET_COLUMN])
    price_all = df[P.TARGET_COLUMN].to_numpy(dtype=float)

    model_subdir = MODEL_DIR / name
    model_subdir.mkdir(parents=True, exist_ok=True)

    for fold_index, (tr, va) in enumerate(zip(folds.train, folds.valid)):
        tr_idx = np.asarray(tr, dtype=np.int64)
        va_idx = np.asarray(va, dtype=np.int64)

        pipe = P.make_pipeline(
            build_estimator(name),
            dense_output=dense,
        )

        started = time.perf_counter()
        pipe.fit(x_all.iloc[tr_idx], y_all[tr_idx])
        fit_seconds = time.perf_counter() - started

        pred_started = time.perf_counter()
        y_pred_log = np.asarray(pipe.predict(x_all.iloc[va_idx]), dtype=float)
        predict_seconds = time.perf_counter() - pred_started

        metrics = P.compute_metrics(price_all[va_idx], y_pred_log)

        records.append(
            {
                "model": name,
                "regime": regime,
                "fold": fold_index + 1,
                "train_rows": int(len(tr_idx)),
                "valid_rows": int(len(va_idx)),
                "fit_seconds": round(fit_seconds, 3),
                "predict_seconds": round(predict_seconds, 3),
                **metrics,
            }
        )

        timing[f"fold{fold_index + 1}_fit_seconds"] = round(fit_seconds, 3)
        timing["total_fit_seconds"] = round(timing.get("total_fit_seconds", 0.0) + fit_seconds, 3)
        timing["total_predict_seconds"] = round(
            timing.get("total_predict_seconds", 0.0) + predict_seconds, 3
        )

        joblib.dump(pipe, model_subdir / f"{regime}_fold{fold_index + 1}.joblib")

        print(
            f"    fold {fold_index + 1}: fit={fit_seconds:7.1f}s  "
            f"MAE_INR={metrics['MAE_INR']:>12,.0f}  "
            f"R2_log={metrics['R2_log']:>7.4f}  "
            f"MedAPE={metrics['MedAPE_percent']:>5.2f}%"
        )

    return records, timing


def summarise(
    records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Fold-level mean, std (ddof=1), min, max and n. Never pooled."""
    out: Dict[str, Dict[str, float]] = {}
    for key in P.METRIC_KEYS:
        values = np.array([r[key] for r in records], dtype=float)
        out[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
            "n_folds": int(len(values)),
            "std_ddof": 1,
        }
    return out


# ============================================================
# XGBOOST ANCHOR: READ, NEVER RETRAINED
# ============================================================

def load_xgboost_anchor() -> List[Dict[str, Any]]:
    """Read the existing V2.1 XGBoost anchor per-fold results."""
    if not ANCHOR_METRICS_JSON.exists():
        raise FileNotFoundError(ANCHOR_METRICS_JSON)

    with open(ANCHOR_METRICS_JSON, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if payload.get("model") != "XGBRegressor":
        raise RuntimeError(
            f"Unexpected anchor model: {payload.get('model')!r}"
        )
    if payload.get("feature_count") != P.FINAL_FEATURE_COUNT:
        raise RuntimeError(
            "Anchor feature_count does not match the current protocol: "
            f"{payload.get('feature_count')} != {P.FINAL_FEATURE_COUNT}"
        )

    records = []
    for row in payload["per_fold"]:
        if row["model_path"] and not (
            ROOT / row["model_path"]
        ).exists():
            raise FileNotFoundError(
                f"Anchor fold model missing: {row['model_path']}"
            )
        records.append(
            {
                "model": "XGBoost",
                "regime": row["regime"],
                "fold": row["fold"],
                "train_rows": row["train_rows"],
                "valid_rows": row["valid_rows"],
                "fit_seconds": None,
                "predict_seconds": None,
                "source": "V2.1 anchor (not retrained)",
                **{k: row[k] for k in P.METRIC_KEYS},
            }
        )
    return records


# ============================================================
# WRITING
# ============================================================

def write_all(
    per_fold: List[Dict[str, Any]],
    summaries: Dict[str, Dict[str, Dict[str, float]]],
    timings: Dict[str, Dict[str, float]],
    duplicate_report: P.DuplicateReport,
    invalid_removed: int,
) -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(per_fold).to_csv(PER_FOLD_CSV, index=False)

    summary_rows = []
    for model, per_regime in summaries.items():
        for regime, per_metric in per_regime.items():
            for key in P.METRIC_KEYS:
                s = per_metric[key]
                summary_rows.append(
                    {
                        "model": model,
                        "regime": regime,
                        "metric": key,
                        "space": P.METRIC_SPACE[key],
                        "mean": s["mean"],
                        "std": s["std"],
                        "min": s["min"],
                        "max": s["max"],
                        "n_folds": s["n_folds"],
                        "std_ddof": 1,
                    }
                )
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)

    # Wide comparison table: one row per model per regime, primary metrics.
    comparison_rows = []
    for model in ("Ridge", "RandomForest", "LightGBM", "CatBoost", "XGBoost"):
        for regime in REGIMES:
            per_metric = summaries[model][regime]
            row = {"model": model, "regime": regime}
            for key in P.METRIC_KEYS:
                row[f"{key}_mean"] = per_metric[key]["mean"]
                row[f"{key}_std"] = per_metric[key]["std"]
            comparison_rows.append(row)
    pd.DataFrame(comparison_rows).to_csv(COMPARISON_TABLE_CSV, index=False)

    pd.DataFrame(
        [{"model": m, **t} for m, t in timings.items()]
    ).to_csv(TIMING_CSV, index=False)

    with open(PER_FOLD_JSON, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "note": (
                    "Per-fold metrics for all five models. XGBoost rows are "
                    "read from the existing V2.1 anchor and were NOT "
                    "retrained."
                ),
                "per_fold": per_fold,
                "summary": summaries,
            },
            fh,
            indent=2,
        )

    with open(CONFIG_JSON, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "declaration": (
                    "Configurations are pre-declared and frozen. No grid "
                    "search, no random search, no early stopping, no "
                    "post-hoc selection."
                ),
                "xgboost_anchor_reference": {
                    "note": (
                        "Not retrained. Configuration recorded for "
                        "comparability only; copied from the anchor run "
                        "record, not re-derived here."
                    ),
                    "params": json.load(
                        open(ANCHOR_METRICS_JSON, encoding="utf-8")
                    )["xgboost_params"],
                },
                "trained_models": ALL_CONFIGS,
                "fairness_axes": {
                    "boosting_rounds": {
                        "LightGBM": 700, "CatBoost": 700, "XGBoost": 700,
                        "Ridge": "n/a", "RandomForest": "n/a (700 trees)",
                    },
                    "learning_rate": {
                        "LightGBM": 0.04, "CatBoost": 0.04, "XGBoost": 0.04,
                        "Ridge": "n/a", "RandomForest": "n/a",
                    },
                    "max_depth": {
                        "LightGBM": 7, "CatBoost": 7, "XGBoost": 7,
                        "Ridge": "n/a", "RandomForest": 7,
                    },
                },
            },
            fh,
            indent=2,
        )

    with open(ENVIRONMENT_JSON, "w", encoding="utf-8") as fh:
        json.dump(environment_record(), fh, indent=2)

    with open(MANIFEST_JSON, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "experiment": (
                    "MILLOW Experiment 1 - five-model baseline comparison"
                ),
                "protocol": "V2.1 (leakage-controlled research protocol)",
                "historical_reference": (
                    "Historical V2 reference - pre-leakage-control baseline"
                ),
                "non_equivalence_warning": (
                    "V2.1 and historical V2 are NOT directly comparable. "
                    "Frequency leakage removed, 737 duplicates removed, "
                    "feature set changed, evaluation changed from a single "
                    "split to 5-fold CV. No better-or-worse claim is made."
                ),
                "models_trained_this_run": [
                    "Ridge", "RandomForest", "LightGBM", "CatBoost",
                ],
                "xgboost": (
                    "NOT retrained. Results read from the existing V2.1 "
                    "anchor at artifacts/valuation/v2_1/xgb_anchor_metrics_v2_1.json."
                ),
                "tuning_performed": False,
                "hyperparameter_search": False,
                "early_stopping_used": False,
                "post_hoc_configuration_changes": False,
                "predictions_pooled_across_folds": False,
                "winner_chosen": False,
                "models_ranked": False,
                "primary_metrics_for_interpretation": [
                    "R2_log", "MAE_log", "MedAPE_percent",
                ],
                "metric_keys": list(P.METRIC_KEYS),
                "feature_count": P.FINAL_FEATURE_COUNT,
                "random_state": P.RANDOM_STATE,
                "folds_loaded_from": str(
                    P.FOLDS_PATH.relative_to(ROOT)
                ).replace("\\", "/"),
                "folds_regenerated": False,
                "dataset": {
                    "path": str(
                        P.DATA_PATH.relative_to(ROOT)
                    ).replace("\\", "/"),
                    **duplicate_report.as_dict(),
                    "invalid_rows_removed": invalid_removed,
                },
                "training_time_seconds": timings,
                "artifacts": {
                    "configurations": str(
                        CONFIG_JSON.relative_to(ROOT)
                    ).replace("\\", "/"),
                    "per_fold_csv": str(
                        PER_FOLD_CSV.relative_to(ROOT)
                    ).replace("\\", "/"),
                    "per_fold_json": str(
                        PER_FOLD_JSON.relative_to(ROOT)
                    ).replace("\\", "/"),
                    "summary_csv": str(
                        SUMMARY_CSV.relative_to(ROOT)
                    ).replace("\\", "/"),
                    "comparison_table_csv": str(
                        COMPARISON_TABLE_CSV.relative_to(ROOT)
                    ).replace("\\", "/"),
                    "environment_json": str(
                        ENVIRONMENT_JSON.relative_to(ROOT)
                    ).replace("\\", "/"),
                    "training_time_csv": str(
                        TIMING_CSV.relative_to(ROOT)
                    ).replace("\\", "/"),
                    "artifact_verification_json": str(
                        (COMPARISON_DIR / "artifact_verification.json").relative_to(ROOT)
                    ).replace("\\", "/"),
                    "model_dir": str(
                        MODEL_DIR.relative_to(ROOT)
                    ).replace("\\", "/"),
                },
                "protected_and_untouched": [
                    "artifacts/valuation/valuation_v2_*",
                    "models/valuation/valuation_v2_*",
                    "artifacts/valuation/v2_1/cv_folds_v2_1.json",
                    "artifacts/valuation/v2_1/xgb_anchor_metrics_v2_1.json",
                    "models/valuation/v2_1/xgb_anchor_folds/",
                    ".chain/state.json",
                ],
            },
            fh,
            indent=2,
        )


# ============================================================
# PRINTING
# ============================================================

def print_configs() -> None:
    print("=" * 74)
    print("PRE-DECLARED FIXED CONFIGURATIONS (frozen before any training)")
    print("=" * 74)
    print()
    print("No grid search. No random search. No early stopping.")
    print("No configuration is changed after results are observed.")
    print()

    for name in ("Ridge", "RandomForest", "LightGBM", "CatBoost"):
        cfg = ALL_CONFIGS[name]
        print(f"  {name}  ({cfg['estimator']})")
        for key, value in cfg["params"].items():
            print(f"      {key:<20} = {value!r}")
        print(f"      dense_output        = {cfg['requires_dense_output']}")
        print()

    print("  XGBoost (ANCHOR - NOT RETRAINED, read from existing artifacts)")
    with open(ANCHOR_METRICS_JSON, encoding="utf-8") as fh:
        anchor = json.load(fh)["xgboost_params"]
    for key, value in anchor.items():
        print(f"      {key:<20} = {value!r}")
    print()

    print("  Matched capacity axes (boosting models vs the anchor):")
    print("      rounds      LightGBM 700   CatBoost 700   XGBoost 700")
    print("      learning_rate      0.04           0.04          0.04")
    print("      max_depth                 7              7             7")
    print()


def print_regime(
    regime: str,
    per_fold_by_model: Dict[str, List[Dict[str, Any]]],
    summaries: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    order = ("Ridge", "RandomForest", "LightGBM", "CatBoost", "XGBoost")

    print(f"  {regime.upper()} - PER FOLD (all five models)")
    print(
        f"    {'model':<14}{'fold':>5}{'MAE_INR':>14}{'R2_INR':>8}"
        f"{'MAPE%':>7}{'MedAPE%':>8}{'MAE_log':>9}{'R2_log':>8}"
    )
    for model in order:
        for r in per_fold_by_model[model]:
            if r["regime"] != regime:
                continue
            print(
                f"    {model:<14}{r['fold']:>5}"
                f"{r['MAE_INR']:>14,.0f}{r['R2_INR']:>8.4f}"
                f"{r['MAPE_percent']:>7.2f}{r['MedAPE_percent']:>8.2f}"
                f"{r['MAE_log']:>9.4f}{r['R2_log']:>8.4f}"
            )
    print()

    print(f"  {regime.upper()} - MEAN +/- STD (ddof=1, n=5, not pooled)")
    header = (
        f"    {'model':<14}{'MAE_INR':>22}{'R2_INR':>16}"
        f"{'MedAPE%':>16}{'MAE_log':>16}{'R2_log':>16}"
    )
    print(header)
    for model in order:
        s = summaries[model][regime]
        print(
            f"    {model:<14}"
            f"{s['MAE_INR']['mean']:>13,.0f}+/-{s['MAE_INR']['std']:>7,.0f}"
            f"{s['R2_INR']['mean']:>9.4f}+/-{s['R2_INR']['std']:>6.4f}"
            f"{s['MedAPE_percent']['mean']:>9.2f}+/-{s['MedAPE_percent']['std']:>5.2f}"
            f"{s['MAE_log']['mean']:>9.4f}+/-{s['MAE_log']['std']:>6.4f}"
            f"{s['R2_log']['mean']:>9.4f}+/-{s['R2_log']['std']:>6.4f}"
        )
    print()


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    assert_safe_write_targets()

    print()
    print_configs()

    print("=" * 74)
    print("1. DATASET")
    print("=" * 74)
    raw = P.load_raw()
    df, duplicate_report, invalid_removed = P.build_dataset(raw)
    print(f"   {duplicate_report}")
    print(f"   invalid rows removed: {invalid_removed}")
    print(f"   feature count       : {P.FINAL_FEATURE_COUNT}")
    print()

    print("=" * 74)
    print("2. PERSISTED FOLDS (loaded only, never regenerated)")
    print("=" * 74)
    folds_by_regime = P.load_folds()
    for regime in REGIMES:
        folds = folds_by_regime[regime]
        report = P.validate_folds(folds, df)
        if not report["valid"]:
            raise RuntimeError(f"Fold validation failed: {report['problems']}")
        print(
            f"   {regime:<17} digest={folds.digest()[:16]}  valid={report['valid']}  "
            f"train={[len(t) for t in folds.train]}"
        )
    print()

    print("=" * 74)
    print("3. XGBOOST ANCHOR (read, NOT retrained)")
    print("=" * 74)
    anchor_records = load_xgboost_anchor()
    anchor_by_regime: Dict[str, List[Dict[str, Any]]] = {}
    for r in anchor_records:
        anchor_by_regime.setdefault(r["regime"], []).append(r)
    for regime in REGIMES:
        rows = anchor_by_regime[regime]
        print(
            f"   {regime:<17} {len(rows)} folds read, "
            f"R2_log={[round(r['R2_log'], 4) for r in rows]}"
        )
    print()

    per_fold_by_model: Dict[str, List[Dict[str, Any]]] = {
        "XGBoost": anchor_records
    }
    summaries: Dict[str, Dict[str, Dict[str, float]]] = {
        "XGBoost": {
            regime: summarise(anchor_by_regime[regime]) for regime in REGIMES
        }
    }
    timings: Dict[str, Dict[str, float]] = {}

    for name in ("Ridge", "RandomForest", "LightGBM", "CatBoost"):
        print("=" * 74)
        print(f"4. TRAINING {name}")
        print("=" * 74)
        model_records: List[Dict[str, Any]] = []
        model_timing: Dict[str, float] = {}

        for regime in REGIMES:
            print(f"   regime: {regime}")
            records, timing = run_model(name, df, folds_by_regime[regime], regime)
            model_records.extend(records)
            for key, value in timing.items():
                if key == "total_fit_seconds" or key == "total_predict_seconds":
                    model_timing[key] = round(
                        model_timing.get(key, 0.0) + value, 3
                    )
                else:
                    model_timing[f"{regime}_{key}"] = value
        model_timing["overall_fit_seconds"] = round(
            model_timing["total_fit_seconds"], 3
        )
        print(
            f"   total fit time: {model_timing['total_fit_seconds']:.1f}s"
            f"  predict: {model_timing['total_predict_seconds']:.1f}s"
        )
        print()

        per_fold_by_model[name] = model_records
        timings[name] = model_timing
        summaries[name] = {
            regime: summarise(
                [r for r in model_records if r["regime"] == regime]
            )
            for regime in REGIMES
        }

    print("=" * 74)
    print("5. RESULTS")
    print("=" * 74)
    print()
    for regime in REGIMES:
        print_regime(regime, per_fold_by_model, summaries)

    print("=" * 74)
    print("6. WRITING ARTIFACTS (new namespace only)")
    print("=" * 74)
    all_per_fold: List[Dict[str, Any]] = []
    for model in ("Ridge", "RandomForest", "LightGBM", "CatBoost", "XGBoost"):
        all_per_fold.extend(per_fold_by_model[model])

    write_all(
        all_per_fold,
        summaries,
        timings,
        duplicate_report=duplicate_report,
        invalid_removed=invalid_removed,
    )
    for path in (
        CONFIG_JSON, PER_FOLD_CSV, PER_FOLD_JSON, SUMMARY_CSV,
        COMPARISON_TABLE_CSV, ENVIRONMENT_JSON, TIMING_CSV, MANIFEST_JSON,
    ):
        print(f"   {path}")
    print(f"   {MODEL_DIR}")
    print()

    print("=" * 74)
    print("FIVE-MODEL COMPARISON COMPLETE - 40 MODELS TRAINED")
    print("=" * 74)
    print()
    print("4 models x 2 regimes x 5 folds = 40 trained.")
    print("XGBoost was NOT retrained; its 10 rows are read from the anchor.")
    print()
    print("No winner is chosen and no ranking is produced. No configuration")
    print("was changed after results were observed. Interpretation is a")
    print("separate step, per the protocol.")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
