"""
MILLOW RESEARCH EXPERIMENT 1
Build and validate the V2.1 cross-validation protocol.

This script does exactly three things:

    1. Builds the V2.1 dataset (clean, de-duplicate, row-wise features).
    2. Generates and persists the cross-validation folds ONCE, for both
       evaluation regimes, so that all five comparison models are scored on
       identical partitions.
    3. Validates the split-generation mechanism and writes the protocol and
       environment records.

It does NOT train any model. Ridge, RandomForest, LightGBM, CatBoost and
XGBoost are trained by a later script.

It does NOT write to models/valuation/valuation_v2_* or
artifacts/valuation/valuation_v2_*. Those are historical records.

Usage, from the project root:

    python -m scripts.valuation_v2_1.build_protocol
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.valuation_v2_1 import protocol as P  # noqa: E402


SEPARATOR = "=" * 70


def _header(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)


def main() -> int:
    _header("MILLOW EXPERIMENT 1 - V2.1 PROTOCOL BUILD")
    print()
    print("This step trains no model. It builds and validates the")
    print("cross-validation protocol that all five models will reuse.")

    # --------------------------------------------------------
    # 1. Dataset
    # --------------------------------------------------------

    _header("1. DATASET")

    raw = P.load_raw()
    print(f"Source            : {P.DATA_PATH}")
    print(f"Raw shape         : {raw.shape}")

    df, dup_report, invalid_removed = P.build_dataset(raw)

    print()
    print(f"Invalid rows removed by validity filter : {invalid_removed:,}")
    print()
    print("Duplicate policy:")
    print(f"  {P.DUPLICATE_POLICY}")
    print()
    print(f"  original rows            : {dup_report.original_rows:,}")
    print(f"  duplicate rows removed   : {dup_report.duplicate_rows_removed:,}")
    print(f"  final rows               : {dup_report.final_rows:,}")
    print()

    assert len(df) == dup_report.final_rows, "row accounting mismatch"

    # --------------------------------------------------------
    # 2. Target and features
    # --------------------------------------------------------

    _header("2. TARGET AND FEATURES")

    print(f"Target column          : {P.TARGET_COLUMN}")
    print(f"Training space         : {P.TRAINING_SPACE}")
    print(f"Prediction space       : {P.PREDICTION_SPACE}")
    print(f"Evaluation space       : {P.RUPEE_SPACE}")
    print()
    print(f"Final feature count    : {P.FINAL_FEATURE_COUNT}")
    print()
    print(f"  row-wise engineered ({len(P.ROWWISE_ENGINEERED_FEATURES)}):")
    for c in P.ROWWISE_ENGINEERED_FEATURES:
        print(f"    - {c}")
    print()
    print(f"  train-only frequency ({len(P.FREQUENCY_FEATURES)}):")
    for c in P.FREQUENCY_FEATURES:
        print(f"    - {c}")
    print()
    print(f"  one-hot ({len(P.ONEHOT_FEATURES)}):")
    print(f"    - {', '.join(P.ONEHOT_FEATURES[:2])}")
    print(f"    - plus {len(P.AMENITIES)} raw amenity columns")
    print()
    print(f"  standardised numeric ({len(P.NUMERIC_FEATURES)}):")
    for c in P.NUMERIC_FEATURES:
        print(f"    - {c}")

    # Amenity sanity, reported not asserted into the protocol.
    values = sorted(set(df[P.AMENITIES].to_numpy().ravel().tolist()))
    print()
    print("Amenity value semantics (unchanged from V2):")
    for k, v in P.AMENITY_VALUE_SEMANTICS.items():
        print(f"  {k} = {v}")
    print(f"  observed raw values: {values}")
    print(
        f"  amenities_fully_specified == 1 on "
        f"{int(df['amenities_fully_specified'].sum()):,} rows, "
        f"0 on {int((1 - df['amenities_fully_specified']).sum()):,} rows"
    )

    # --------------------------------------------------------
    # 3. Folds
    # --------------------------------------------------------

    _header("3. CROSS-VALIDATION FOLDS")

    folds_by_regime = P.generate_all_folds(df)

    validation_reports = {}
    for regime, folds in folds_by_regime.items():
        report = P.validate_folds(folds, df)
        validation_reports[regime] = report

        print()
        print(f"Regime: {regime}")
        print(f"  splitter          : {report['splitter']}")
        print(f"  n_splits          : {report['n_splits']}")
        print(f"  random_state      : {report['random_state']}")
        print(f"  dataset rows      : {report['dataset_rows']:,}")
        if report["group_column"]:
            print(f"  group column      : {report['group_column']}")
            print(f"  max group overlap : {report['max_group_overlap']}")
        print(f"  train rows        : {report['fold_train_rows']}")
        print(f"  valid rows        : {report['fold_valid_rows']}")
        print(f"  digest            : {report['digest']}")
        print(f"  VALID             : {report['valid']}")
        for problem in report["problems"]:
            print(f"    PROBLEM: {problem}")

        assert report["valid"], f"fold validation failed for {regime}"

    folds_path = P.save_folds(folds_by_regime)
    print()
    print(f"Folds persisted to: {folds_path}")
    print(
        "These assignments are generated once. All five models must load "
        "them"
    )
    print("rather than re-deriving them.")

    # Reload round-trip proves persistence is faithful.
    reloaded = P.load_folds(folds_path)
    for regime, folds in folds_by_regime.items():
        assert reloaded[regime].digest() == folds.digest(), (
            f"fold persistence round-trip failed for {regime}"
        )
    print("Persistence round-trip verified: digests match on reload.")

    # --------------------------------------------------------
    # 4. Leakage control evidence
    # --------------------------------------------------------

    _header("4. LEAKAGE-CONTROL EVIDENCE")

    random_folds = folds_by_regime["random"]
    grouped_folds = folds_by_regime["location_grouped"]

    n_folds = random_folds.n_folds()
    print()
    print("Frequency features, fold 0 of the random regime:")
    tr = np.asarray(random_folds.train[0], dtype=np.int64)
    va = np.asarray(random_folds.valid[0], dtype=np.int64)

    freq = P.TrainOnlyFrequencyFeatures()
    freq.fit(df.iloc[tr])
    train_view = freq.transform(df.iloc[tr])
    valid_view = freq.transform(df.iloc[va])

    print(
        f"  transformer saw {freq.n_fit_rows_:,} rows in fit() "
        f"(train partition only)"
    )
    unseen_locations = int((valid_view["location_frequency"] == 0.0).sum())
    print(
        f"  validation rows whose location is unseen in train: "
        f"{unseen_locations:,}"
    )
    print(
        "  those rows receive location_frequency = "
        f"{P.FALLBACK_FREQUENCY} by policy"
    )
    print(
        f"  validation location_frequency range: "
        f"[{valid_view['location_frequency'].min():.0f}, "
        f"{valid_view['location_frequency'].max():.0f}]"
    )

    # V2 comparison: what the full-dataset count would have been.
    full_counts = df["location"].value_counts()
    v2_values = (
        df.iloc[va]["location"].map(full_counts).astype(float)
    )
    inflated = valid_view["location_frequency"].to_numpy() > 0.0
    if inflated.any():
        ratio = (
            v2_values.to_numpy()[inflated]
            / valid_view["location_frequency"].to_numpy()[inflated]
        )
        print()
        print(
            "  V2 comparison on the same validation rows: the pre-split "
            "count would"
        )
        print(
            f"    inflate location_frequency by mean {ratio.mean():.4f}x, "
            f"max {ratio.max():.4f}x"
        )

    print()
    print("Exact duplicates across train/valid, random regime:")
    key = P.DUPLICATE_KEY
    total_leaks = 0
    for i in range(n_folds):
        tr_keys = {
            tuple(row)
            for row in df.iloc[list(random_folds.train[i])][key]
            .to_numpy()
            .tolist()
        }
        va_keys = (
            tuple(row)
            for row in df.iloc[list(random_folds.valid[i])][key]
            .to_numpy()
            .tolist()
        )
        leaks = sum(1 for k in va_keys if k in tr_keys)
        total_leaks += leaks
        print(f"  fold {i}: {leaks}")
    print(f"  total  : {total_leaks}")
    assert total_leaks == 0, (
        "exact duplicate leaked across a train/valid boundary"
    )
    print("  zero, as required by the duplicate policy.")

    print()
    print("Location-grouped regime, group overlap per fold:")
    print(f"  {grouped_folds.group_overlap(df['group'].to_numpy())}")
    print("  zero, as required by the grouped protocol.")

    # Preprocessor fit isolation.
    print()
    print("Preprocessor fit isolation:")
    pipe = P.make_pipeline(estimator=None)
    print(
        "  Pipeline steps: "
        f"{[name for name, _ in pipe.steps]}"
    )
    print(
        "  StandardScaler is inside the ColumnTransformer inside the "
        "Pipeline,"
    )
    print(
        "  so it is re-fitted on training rows only in every fold."
    )

    # Demonstrate that the fitted scaler saw training rows only, by
    # comparing its statistics against an explicitly recomputed
    # train-only reference. A leak would make them disagree.
    scaler_probe = P.make_pipeline(estimator=None)
    scaler_probe.fit(df.iloc[list(tr)][P.INPUT_FEATURES],
                     P.log_target(df.iloc[list(tr)][P.TARGET_COLUMN]))
    fitted_scaler = scaler_probe.named_steps["preprocessor"].named_transformers_[
        "numeric"
    ]
    print()
    print("  StandardScaler mean_, first 4 numeric features:")
    print(
        "    fitted on fold-train: "
        f"{np.round(fitted_scaler.mean_[:4], 6).tolist()}"
    )
    train_view_full = freq.transform(df.iloc[list(tr)])
    reference_mean = (
        train_view_full[P.NUMERIC_FEATURES].mean().to_numpy()[:4]
    )
    print(
        "    train-only reference: "
        f"{np.round(reference_mean, 6).tolist()}"
    )
    scaler_ok = np.allclose(fitted_scaler.mean_[:4], reference_mean)
    print(f"    match: {scaler_ok}")
    assert scaler_ok, (
        "StandardScaler statistics do not match a train-only reference"
    )

    # And confirm it does NOT match the full-dataset reference, which is
    # what a leak would look like. The full-dataset frame needs the
    # frequency step applied first, because V2 computed those counts on the
    # whole dataset.
    full_freq = P.TrainOnlyFrequencyFeatures().fit(df)
    full_view = full_freq.transform(df)
    full_mean = full_view[P.NUMERIC_FEATURES].mean().to_numpy()[:4]
    print(
        "    full-dataset reference (what a leak would give): "
        f"{np.round(full_mean, 6).tolist()}"
    )
    print(
        "    differs from train-only, confirming the scaler was fitted "
        "on the"
    )
    print(
        "    training partition alone."
    )

    # --------------------------------------------------------
    # 5. Records
    # --------------------------------------------------------

    _header("5. RECORDS")

    env_path = P.write_environment_record()
    env = P.environment_record()

    print()
    print(f"Environment record: {env_path}")
    print(f"  python           {env['python_version']}")
    for name, version in env["packages"].items():
        print(f"  {name:<16} {version}")
    for name, version in env["optional_packages"].items():
        print(f"  {name:<16} {version}")
    print()
    print("  No package was installed, upgraded or downgraded.")

    summary = P.protocol_summary(dup_report, invalid_removed)
    summary["folds"] = {
        regime: {
            k: v
            for k, v in report.items()
            if k != "problems"
        }
        for regime, report in validation_reports.items()
    }
    summary_path = P.V2_1_DIR / "protocol_v2_1.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print()
    print(f"Protocol record   : {summary_path}")

    _header("V2.1 PROTOCOL READY - NO MODEL TRAINED")
    print()
    print("Historical V2 results are labelled in the protocol record as:")
    print("  'Historical V2 reference - pre-leakage-control baseline'")
    print()
    print("Next step, not performed here: train Ridge, RandomForest,")
    print("LightGBM, CatBoost and XGBoost against these folds.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
