"""
MILLOW RESEARCH EXPERIMENT 1
V2.1 XGBoost ANCHOR experiment.

This is NOT a tuning exercise and NOT the five-model comparison. It runs one
model, XGBoost, with the FROZEN V2.1 protocol and the UNCHANGED historical V2
hyperparameter configuration, across the persisted V2.1 folds. Its purpose is
to establish a leakage-controlled anchor row that the later five-model
comparison can be read against.

Frozen protocol (scripts/valuation_v2_1/protocol.py), all unchanged:
    target                     log1p(price)
    features                   48, incl. location_frequency + log_location_frequency
    frequency features         train-only, fitted inside the Pipeline
    unseen frequency fallback  0.0
    amenity value 9            its own one-hot level, never reinterpreted
    duplicate policy           5-column exact key, keep first
    preprocessing              StandardScaler + OneHotEncoder(handle_unknown="ignore")
    folds                      persisted V2.1 assignments, loaded not regenerated
    grouped regime             source_city + "__" + location
    random seed                42

NOT done here, deliberately:
    - no early stopping
    - no hyperparameter tuning or search
    - no early-stopping validation slice
    - no pooling of predictions across folds for the headline result
    - no write outside models/valuation/v2_1/ and artifacts/valuation/v2_1/

Historical V2 outputs are never read or written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.valuation_v2_1 import protocol as P  # noqa: E402


# ============================================================
# NAMESPACING
# ============================================================

# V2.1 outputs live in their own namespace. Historical V2 files at
# models/valuation/valuation_v2_*.joblib and
# artifacts/valuation/valuation_v2_* are never touched.
MODEL_DIR = ROOT / "models" / "valuation" / "v2_1"
ARTIFACT_DIR = ROOT / "artifacts" / "valuation" / "v2_1"

ANCHOR_TAG = "v2_1_xgboost_anchor"

METRICS_CSV = ARTIFACT_DIR / "xgb_anchor_metrics_v2_1.csv"
METRICS_JSON = ARTIFACT_DIR / "xgb_anchor_metrics_v2_1.json"
PER_FOLD_CSV = ARTIFACT_DIR / "xgb_anchor_per_fold_v2_1.csv"
RUN_RECORD = ARTIFACT_DIR / "xgb_anchor_run_v2_1.json"
FOLD_MODELS_DIR = MODEL_DIR / "xgb_anchor_folds"

# Guard: refuse to run if any historical V2 path is ever a write target.
FORBIDDEN_WRITE_PREFIXES = (
    ROOT / "models" / "valuation" / "valuation_v2",
    ROOT / "artifacts" / "valuation" / "valuation_v2",
)


# ============================================================
# FROZEN XGBOOST CONFIGURATION
# ============================================================

# Identical to the historical V2 XGBoost configuration. eval_metric is
# included because V2 set it; with no early stopping it only affects logging.
XGBOOST_PARAMS: Dict[str, Any] = {
    "objective": "reg:squarederror",
    "n_estimators": 700,
    "max_depth": 7,
    "learning_rate": 0.04,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "eval_metric": "rmse",
}


def assert_safe_write_targets() -> None:
    """Fail loudly if a V2.1 path could overwrite a historical V2 file."""
    targets = [
        METRICS_CSV,
        METRICS_JSON,
        PER_FOLD_CSV,
        RUN_RECORD,
        FOLD_MODELS_DIR,
    ]
    for path in targets:
        resolved = path.resolve()
        for forbidden in FORBIDDEN_WRITE_PREFIXES:
            if str(resolved).startswith(str(forbidden)):
                raise RuntimeError(
                    f"Refusing to write {resolved}: it collides with a "
                    f"historical V2 path ({forbidden})."
                )


# ============================================================
# FOLD EXECUTION
# ============================================================

def run_regime(
    df: pd.DataFrame,
    regime: str,
    folds: P.FoldSet,
) -> List[Dict[str, Any]]:
    """Train and score one XGBoost model per persisted fold.

    Returns one record per fold. Predictions are scored fold-by-fold and are
    never pooled into a single headline number.
    """
    records: List[Dict[str, Any]] = []

    x_all = df[P.INPUT_FEATURES]
    y_all = P.log_target(df[P.TARGET_COLUMN])

    for fold_index, (tr, va) in enumerate(zip(folds.train, folds.valid)):
        tr_idx = np.asarray(tr, dtype=np.int64)
        va_idx = np.asarray(va, dtype=np.int64)

        x_tr = x_all.iloc[tr_idx]
        x_va = x_all.iloc[va_idx]
        # log_target returns a numpy array, so index positionally.
        y_tr = y_all[tr_idx]
        y_va_true_price = df[P.TARGET_COLUMN].to_numpy(dtype=float)[va_idx]

        pipe = P.make_pipeline(XGBRegressor(**XGBOOST_PARAMS))
        pipe.fit(x_tr, y_tr)

        # Pipeline.predict returns log1p(price) space. Scoring owns inversion.
        y_va_pred_log = np.asarray(pipe.predict(x_va), dtype=float)
        metrics = P.compute_metrics(y_va_true_price, y_va_pred_log)

        record: Dict[str, Any] = {
            "regime": regime,
            "fold": fold_index + 1,
            "train_rows": int(len(tr_idx)),
            "valid_rows": int(len(va_idx)),
            **metrics,
        }
        records.append(record)

        # Fold models are persisted so the run is auditable, under the V2.1
        # namespace only.
        FOLD_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = FOLD_MODELS_DIR / f"{regime}_fold{fold_index + 1}.joblib"
        import joblib
        joblib.dump(pipe, model_path)
        record["model_path"] = str(
            model_path.relative_to(ROOT)
        ).replace("\\", "/")

        print(
            f"    fold {fold_index + 1}: "
            f"train={len(tr_idx):,} valid={len(va_idx):,}  "
            f"MAE_INR={metrics['MAE_INR']:,.0f}  "
            f"R2_log={metrics['R2_log']:.4f}  "
            f"MedAPE={metrics['MedAPE_percent']:.2f}%"
        )

    return records


def summarise(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Fold-level mean and sample standard deviation for every metric.

    Standard deviation is ddof=1, i.e. the sample standard deviation across
    the five folds, and is reported as such.
    """
    out: Dict[str, Dict[str, float]] = {}
    for key in P.METRIC_KEYS:
        values = np.array([r[key] for r in records], dtype=float)
        out[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return out


# ============================================================
# REPORTING
# ============================================================

def write_outputs(
    all_records: List[Dict[str, Any]],
    summaries: Dict[str, Dict[str, Dict[str, float]]],
    dataset_rows: int,
    duplicate_report: P.DuplicateReport,
    invalid_removed: int,
) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    per_fold = pd.DataFrame(all_records)
    per_fold.to_csv(PER_FOLD_CSV, index=False)

    # Summary table: one row per regime per metric, with fold-level mean and
    # sample standard deviation. Kept separate from the per-fold table so the
    # two are not confused.
    summary_rows = []
    for regime, per_metric in summaries.items():
        for key in P.METRIC_KEYS:
            s = per_metric[key]
            summary_rows.append(
                {
                    "regime": regime,
                    "metric": key,
                    "space": P.METRIC_SPACE[key],
                    "mean": s["mean"],
                    "std": s["std"],
                    "min": s["min"],
                    "max": s["max"],
                    "n_folds": len(all_records) // len(summaries),
                    "std_ddof": 1,
                }
            )
    pd.DataFrame(summary_rows).to_csv(METRICS_CSV, index=False)

    payload = {
        "experiment": "MILLOW Experiment 1 - V2.1 XGBoost anchor",
        "purpose": (
            "Single-model anchor on the frozen V2.1 protocol. NOT a tuning "
            "run and NOT the five-model comparison."
        ),
        "protocol": "V2.1 (leakage-controlled research protocol)",
        "historical_reference": (
            "Historical V2 reference - pre-leakage-control baseline"
        ),
        "non_equivalence_warning": (
            "V2.1 and historical V2 are NOT directly comparable. V2.1 removed "
            "frequency leakage, removed 737 exact duplicates, changed the "
            "feature set (dropped amenity_known_count, amenity_unknown_count "
            "and city_frequency; added amenities_fully_specified; added "
            "StandardScaler), and replaced a single train/test draw with 5-fold "
            "CV. No better-or-worse claim is made."
        ),
        "model": "XGBRegressor",
        "xgboost_params": dict(XGBOOST_PARAMS),
        "tuning_performed": False,
        "early_stopping_used": False,
        "hyperparameter_search": False,
        "predictions_pooled_across_folds": False,
        "feature_count": P.FINAL_FEATURE_COUNT,
        "features": list(P.FINAL_FEATURES),
        "pipeline_input_columns": list(P.INPUT_FEATURES),
        "dataset": {
            "path": str(P.DATA_PATH.relative_to(ROOT)).replace("\\", "/"),
            **duplicate_report.as_dict(),
            "invalid_rows_removed": invalid_removed,
        },
        "target": {
            "column": P.TARGET_COLUMN,
            "training_space": P.TRAINING_SPACE,
            "prediction_space": P.PREDICTION_SPACE,
        },
        "random_state": P.RANDOM_STATE,
        "folds": {
            regime: {
                "splitter": folds.splitter,
                "group_column": folds.group_column,
                "digest": folds.digest(),
                "fold_train_rows": [len(t) for t in folds.train],
                "fold_valid_rows": [len(v) for v in folds.valid],
            }
            for regime, folds in load_all_folds().items()
        },
        "metrics": {
            "keys": list(P.METRIC_KEYS),
            "space": dict(P.METRIC_SPACE),
            "description": dict(P.METRIC_DESCRIPTION),
        },
        "per_fold": all_records,
        "summary": summaries,
        "dataset_rows": dataset_rows,
    }

    with open(METRICS_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    with open(RUN_RECORD, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "experiment": payload["experiment"],
                "protocol": payload["protocol"],
                "model": payload["model"],
                "xgboost_params": payload["xgboost_params"],
                "tuning_performed": payload["tuning_performed"],
                "early_stopping_used": payload["early_stopping_used"],
                "predictions_pooled_across_folds": payload[
                    "predictions_pooled_across_folds"
                ],
                "feature_count": P.FINAL_FEATURE_COUNT,
                "folds": payload["folds"],
                "summary": summaries,
                "artifacts": {
                    "per_fold_csv": str(
                        PER_FOLD_CSV.relative_to(ROOT)
                    ).replace("\\", "/"),
                    "metrics_csv": str(
                        METRICS_CSV.relative_to(ROOT)
                    ).replace("\\", "/"),
                    "metrics_json": str(
                        METRICS_JSON.relative_to(ROOT)
                    ).replace("\\", "/"),
                    "fold_models_dir": str(
                        FOLD_MODELS_DIR.relative_to(ROOT)
                    ).replace("\\", "/"),
                },
            },
            fh,
            indent=2,
        )


def load_all_folds() -> Dict[str, P.FoldSet]:
    return P.load_folds()


def print_regime_summary(
    regime: str,
    records: List[Dict[str, Any]],
    summary: Dict[str, Dict[str, float]],
) -> None:
    print()
    print(f"  {regime.upper()} - per fold")
    header = (
        f"    {'fold':>4}  {'MAE_INR':>15}  {'RMSE_INR':>15}  "
        f"{'R2_INR':>7}  {'MAPE%':>6}  {'MedAPE%':>7}  "
        f"{'MAE_log':>8}  {'RMSE_log':>9}  {'R2_log':>7}"
    )
    print(header)
    for r in records:
        print(
            f"    {r['fold']:>4}  {r['MAE_INR']:>15,.0f}  "
            f"{r['RMSE_INR']:>15,.0f}  {r['R2_INR']:>7.4f}  "
            f"{r['MAPE_percent']:>6.2f}  {r['MedAPE_percent']:>7.2f}  "
            f"{r['MAE_log']:>8.4f}  {r['RMSE_log']:>9.4f}  "
            f"{r['R2_log']:>7.4f}"
        )

    print()
    print(f"  {regime.upper()} - mean +/- std over {len(records)} folds")
    for key in P.METRIC_KEYS:
        s = summary[key]
        print(
            f"    {key:<15} {s['mean']:>18,.4f} +/- {s['std']:>14,.4f}"
            f"   [min {s['min']:,.4f}  max {s['max']:,.4f}]"
        )
    print()
    print("    Predictions are scored per fold and NOT pooled.")
    print()


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    assert_safe_write_targets()

    print("=" * 70)
    print("MILLOW RESEARCH EXPERIMENT 1 - V2.1 XGBOOST ANCHOR")
    print("=" * 70)
    print()
    print("Frozen V2.1 protocol. No tuning. No early stopping.")
    print("Historical V2 reference is NOT touched and NOT overwritten.")
    print()

    print("1. DATASET")
    raw = P.load_raw()
    df, duplicate_report, invalid_removed = P.build_dataset(raw)
    print(f"   {duplicate_report}")
    print(f"   invalid rows removed by validity filter: {invalid_removed}")
    print()

    print("2. PROTOCOL")
    print(f"   final feature count   : {P.FINAL_FEATURE_COUNT}")
    print(f"   pipeline input columns: {len(P.INPUT_FEATURES)}")
    print(f"   numeric (scaled)      : {len(P.NUMERIC_FEATURES)}")
    print(f"   one-hot               : {len(P.ONEHOT_FEATURES)}")
    print(f"   train-only frequency  : {P.FREQUENCY_FEATURES}")
    print(f"   unseen fallback       : {P.FALLBACK_FREQUENCY}")
    print(f"   target space          : {P.TRAINING_SPACE}")
    print(f"   random seed           : {P.RANDOM_STATE}")
    print()

    print("3. PERSISTED FOLDS (loaded, never regenerated)")
    folds_by_regime = load_all_folds()

    validation = P.validate_folds(folds_by_regime["random"], df)
    if not validation["valid"]:
        raise RuntimeError(f"Fold validation failed: {validation['problems']}")

    for regime, folds in folds_by_regime.items():
        print(
            f"   {regime:<17} digest={folds.digest()[:16]}  "
            f"train={[len(t) for t in folds.train]}  "
            f"valid={[len(v) for v in folds.valid]}"
        )
    print()

    print("4. XGBOOST CONFIGURATION (identical to historical V2)")
    for key, value in XGBOOST_PARAMS.items():
        print(f"   {key:<20} = {value}")
    print()

    all_records: List[Dict[str, Any]] = []
    summaries: Dict[str, Dict[str, Dict[str, float]]] = {}

    for regime in ("random", "location_grouped"):
        print(f"5. RUNNING REGIME: {regime}")
        records = run_regime(df, regime, folds_by_regime[regime])
        summary = summarise(records)
        all_records.extend(records)
        summaries[regime] = summary
        print_regime_summary(regime, records, summary)

    print("6. WRITING V2.1 ARTIFACTS (new namespace only)")
    write_outputs(
        all_records,
        summaries,
        dataset_rows=len(df),
        duplicate_report=duplicate_report,
        invalid_removed=invalid_removed,
    )
    for path in (PER_FOLD_CSV, METRICS_CSV, METRICS_JSON, RUN_RECORD):
        print(f"   {path}")
    print(f"   {FOLD_MODELS_DIR}")
    print()

    print("=" * 70)
    print("V2.1 XGBOOST ANCHOR COMPLETE - 10 MODELS TRAINED")
    print("=" * 70)
    print()
    print("These results are NOT comparable to historical V2 as a")
    print("better-or-worse claim. The protocols differ in leakage control,")
    print("duplicate handling, feature set and evaluation design.")
    print()
    print("Next step, not performed here: the five-model comparison")
    print("(Ridge, RandomForest, LightGBM, CatBoost, XGBoost).")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
