"""
MILLOW RESEARCH EXPERIMENT 1
V2.1 leakage-controlled baseline protocol.

This module is the single source of truth for the V2.1 research baseline.
It contains ONLY protocol machinery:

    - dataset loading and cleaning
    - exact-duplicate control
    - row-wise feature engineering
    - TRAIN-ONLY frequency feature transformer
    - leakage-safe preprocessing pipeline
    - reproducible cross-validation fold generation
    - metric computation

It deliberately contains NO model definitions and NO training loop.
The five comparison models (Ridge, RandomForest, LightGBM, CatBoost,
XGBoost) are trained by a later script against this protocol.

Historical V2 (scripts/train_valuation_v2.py) is retained unmodified as the
pre-leakage-control reference record and must not be edited or re-run
against this protocol.
"""

from .protocol import (  # noqa: F401
    AMENITIES,
    AMENITY_VALUE_SEMANTICS,
    BASIC_FEATURES,
    CATEGORICAL_FEATURES,
    DATA_PATH,
    DUPLICATE_KEY,
    DUPLICATE_POLICY,
    ENVIRONMENT_PATH,
    FALLBACK_FREQUENCY,
    FINAL_FEATURE_COUNT,
    FINAL_FEATURES,
    FOLDS_PATH,
    FREQUENCY_FEATURES,
    INPUT_FEATURES,
    METRIC_DESCRIPTION,
    METRIC_KEYS,
    METRIC_SPACE,
    N_SPLITS,
    NUMERIC_FEATURES,
    ONEHOT_FEATURES,
    RANDOM_STATE,
    REDUNDANT_FEATURES_REMOVED,
    ROWWISE_ENGINEERED_FEATURES,
    TARGET_COLUMN,
    TEST_SIZE,
    UNSEEN_FREQUENCY_POLICY,
    V2_1_DIR,
    DuplicateReport,
    FoldSet,
    TrainOnlyFrequencyFeatures,
    add_group_labels,
    build_dataset,
    compute_metrics,
    create_preprocessor,
    environment_record,
    frequency_feature_names,
    full_feature_list,
    generate_all_folds,
    generate_folds,
    get_folds,
    load_folds,
    load_raw,
    log_target,
    make_pipeline,
    protocol_summary,
    remove_duplicates,
    save_folds,
    to_rupees,
    validate_folds,
    write_environment_record,
)

__all__ = [
    "AMENITIES",
    "AMENITY_VALUE_SEMANTICS",
    "BASIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "DATA_PATH",
    "DUPLICATE_KEY",
    "DUPLICATE_POLICY",
    "DuplicateReport",
    "ENVIRONMENT_PATH",
    "FALLBACK_FREQUENCY",
    "FINAL_FEATURE_COUNT",
    "FINAL_FEATURES",
    "FOLDS_PATH",
    "FREQUENCY_FEATURES",
    "FoldSet",
    "INPUT_FEATURES",
    "METRIC_DESCRIPTION",
    "METRIC_KEYS",
    "METRIC_SPACE",
    "N_SPLITS",
    "NUMERIC_FEATURES",
    "ONEHOT_FEATURES",
    "RANDOM_STATE",
    "REDUNDANT_FEATURES_REMOVED",
    "ROWWISE_ENGINEERED_FEATURES",
    "TARGET_COLUMN",
    "TEST_SIZE",
    "TrainOnlyFrequencyFeatures",
    "UNSEEN_FREQUENCY_POLICY",
    "V2_1_DIR",
    "add_group_labels",
    "build_dataset",
    "compute_metrics",
    "create_preprocessor",
    "environment_record",
    "frequency_feature_names",
    "full_feature_list",
    "generate_all_folds",
    "generate_folds",
    "get_folds",
    "load_folds",
    "load_raw",
    "log_target",
    "make_pipeline",
    "protocol_summary",
    "remove_duplicates",
    "save_folds",
    "to_rupees",
    "validate_folds",
    "write_environment_record",
]
