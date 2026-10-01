"""
MILLOW RESEARCH EXPERIMENT 1
V2.1 leakage-controlled baseline protocol.

Single source of truth for the V2.1 research baseline. Contains protocol
machinery only: dataset loading, duplicate control, row-wise feature
engineering, TRAIN-ONLY frequency features, leakage-safe preprocessing,
reproducible CV folds, and metrics.

Contains NO model definitions and NO training loop. The five comparison
models (Ridge, RandomForest, LightGBM, CatBoost, XGBoost) are trained by a
separate script against this protocol.

V2 CHANGES ADDRESSED HERE (see docs/research/EXPERIMENT_1_V2_1_BASELINE.md):

  1. Frequency leakage. V2 computed location_frequency / city_frequency /
     log_location_frequency over all 29,135 rows before splitting. Here
     location_frequency and log_location_frequency are fitted inside an
     sklearn transformer on training rows only, so no validation or test row
     can contribute to its own frequency feature, and the guarantee holds
     identically inside every CV fold. city_frequency is dropped entirely as
     redundant with the one-hot source_city feature.

  2. Duplicate control. V2 kept exact duplicate property records, so an
     identical record could appear in both train and test. Here exact
     duplicates are removed before any split.

  3. Amenity degeneracy. V2 kept amenity_known_count and
     amenity_unknown_count, which satisfy
     known + unknown == 35 and therefore correlate at exactly -1.0. Here a
     single explicit binary amenities_fully_specified replaces them. It is a
     data-recording / provenance indicator, NOT an ordinary property amenity.

  4. Fair Ridge preprocessing. V2 had no scaler. Here numeric features are
     standardised by a StandardScaler fitted inside the Pipeline, so the
     scaler is re-fitted independently in every fold.

  5. Cross-validation. V2 used a single train/test draw per regime. Here
     K-fold fold assignments are generated once, persisted, and reused by
     every model.

  6. Metrics. V2 reported MAE, RMSE, R2 and MAPE on the rupee scale only.
     Here MedAPE and log-space metrics are added.

Historical V2 outputs under models/valuation/valuation_v2_* and
artifacts/valuation/valuation_v2_* are never written by this module.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "MREID_property.csv"

# V2.1 outputs are namespaced under their own directory so that historical
# V2 models and artifacts are never read or written by this protocol.
V2_1_DIR = ROOT / "artifacts" / "valuation" / "v2_1"

FOLDS_PATH = V2_1_DIR / "cv_folds_v2_1.json"
ENVIRONMENT_PATH = V2_1_DIR / "environment_v2_1.json"


# ============================================================
# PROTOCOL CONSTANTS
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5

# Retained from V2 so that V2 and V2.1 stay directly comparable.
TEST_SIZE = 0.20

AMENITIES: List[str] = [
    "maintenancestaff",
    "gymnasium",
    "swimmingpool",
    "landscapedgardens",
    "joggingtrack",
    "rainwaterharvesting",
    "indoorgames",
    "shoppingmall",
    "intercom",
    "sportsfacility",
    "atm",
    "clubhouse",
    "school",
    "24x7security",
    "powerbackup",
    "carparking",
    "staffquarter",
    "cafeteria",
    "multipurposeroom",
    "hospital",
    "washingmachine",
    "gasconnection",
    "ac",
    "wifi",
    "children_splayarea",
    "liftavailable",
    "bed",
    "vaastucompliant",
    "microwave",
    "golfcourse",
    "tv",
    "diningtable",
    "sofa",
    "wardrobe",
    "refrigerator",
]

# Value semantics of the 35 raw amenity columns. 9 is never reinterpreted.
AMENITY_NO = 0
AMENITY_YES = 1
AMENITY_UNKNOWN = 9

AMENITY_KNOWN_VALUES = (AMENITY_NO, AMENITY_YES)
AMENITY_VALUE_SEMANTICS = {
    "0": "No",
    "1": "Yes",
    "9": "Not specified / unknown (kept as its own categorical level)",
}

# ------------------------------------------------------------
# DUPLICATE POLICY (V2.1 change 2)
# ------------------------------------------------------------

# Exact-duplicate rule. Two rows are exact duplicates if and only if every
# column below is equal. mreid_id is deliberately EXCLUDED, because mreid_id
# is a unique row identifier and can never match across rows; deduping on it
# would be a no-op. source_file is excluded because the same listing
# signature can legitimately arrive from more than one scrape run.
#
# Consequence: only rows that are identical on all five fields are removed.
# Distinct properties that merely share a location, an area or a bedroom
# count are always retained.
DUPLICATE_KEY: List[str] = [
    "source_city",
    "location",
    "area",
    "no_of_bedrooms",
    "price",
]

DUPLICATE_POLICY = (
    "Drop rows that are exact duplicates on "
    f"{DUPLICATE_KEY}, keeping the first occurrence. "
    "mreid_id and source_file are excluded from the key. "
    "No fuzzy, near-duplicate or within-location de-duplication is performed."
)


# ------------------------------------------------------------
# FEATURE DEFINITION (V2.1 change 3)
# ------------------------------------------------------------

BASIC_FEATURES: List[str] = [
    "area",
    "no_of_bedrooms",
    "resale",
]

# Row-wise engineered features. Each depends only on the row's own raw
# columns, so computing them before the split cannot leak anything.
ROWWISE_ENGINEERED_FEATURES: List[str] = [
    "log_area",
    "log_bedrooms",
    "area_per_bedroom",
    "area_bedroom_interaction",
    "amenity_yes_count",
    "amenities_fully_specified",
]

# Aggregate engineered features. These depend on the training distribution
# and therefore MUST be fitted inside the Pipeline on training rows only.
# Frozen decision: city_frequency is DROPPED. It is a deterministic 1:1
# restatement of source_city, which is already a one-hot categorical feature,
# so it added zero information. See REDUNDANT_FEATURES_REMOVED.
FREQUENCY_FEATURES: List[str] = [
    "location_frequency",
    "log_location_frequency",
]

REDUNDANT_FEATURES_REMOVED: Dict[str, str] = {
    "city_frequency": (
        "Dropped. Deterministic 1:1 restatement of source_city, which is "
        "already represented as a one-hot categorical feature. Retained "
        "location_frequency and log_location_frequency only."
    ),
    "amenity_known_count": (
        "Dropped. Satisfied amenity_known_count + amenity_unknown_count == 35 "
        "identically and correlated with amenity_unknown_count at exactly "
        "-1.0, so the pair carried a single degree of freedom."
    ),
    "amenity_unknown_count": (
        "Dropped. Same collinearity as amenity_known_count. Replaced by the "
        "single binary amenities_fully_specified."
    ),
}

ENGINEERED_FEATURES: List[str] = (
    ROWWISE_ENGINEERED_FEATURES + FREQUENCY_FEATURES
)

CATEGORICAL_FEATURES: List[str] = [
    "source_city",
    "location",
]

# One-hot encoded columns: the two free-text keys plus all 35 raw amenities.
# The amenities are encoded categorically because each takes three discrete
# values (0, 1, 9) and 9 is a distinct "not specified" state, not an
# ordered magnitude. This matches V2 and is kept unchanged.
ONEHOT_FEATURES: List[str] = CATEGORICAL_FEATURES + AMENITIES

FINAL_FEATURES: List[str] = (
    BASIC_FEATURES
    + ROWWISE_ENGINEERED_FEATURES
    + FREQUENCY_FEATURES
    + CATEGORICAL_FEATURES
    + AMENITIES
)

FINAL_FEATURE_COUNT = len(FINAL_FEATURES)

# Columns the model receives, i.e. the columns the Pipeline is fitted with.
# The two frequency features are absent here on purpose: they are created
# by TrainOnlyFrequencyFeatures from training rows, so asking a caller for
# them would invite pre-computing them on the full dataset, which is the
# exact V2 defect this protocol removes.
INPUT_FEATURES: List[str] = [
    c for c in FINAL_FEATURES if c not in FREQUENCY_FEATURES
]

NUMERIC_FEATURES: List[str] = [
    c for c in FINAL_FEATURES if c not in ONEHOT_FEATURES
]

# ------------------------------------------------------------
# FREQUENCY POLICY (V2.1 change 1)
# ------------------------------------------------------------

FALLBACK_FREQUENCY = 0.0

UNSEEN_FREQUENCY_POLICY = (
    "A location absent from the training partition receives a frequency of "
    "exactly 0.0, meaning zero training observations, and "
    f"log1p(0.0) = {np.log1p(0.0):.1f}. No smoothing constant is added and no "
    "global or full-dataset count is consulted. 0.0 is preferred over 1.0 so "
    "that an unseen locality is never confused with a singleton locality "
    "that genuinely appears once in training."
)

# ------------------------------------------------------------
# METRIC CONTRACT (V2.1 change 6)
# ------------------------------------------------------------

TARGET_COLUMN = "price"

# Space bookkeeping, made explicit so no model can be scored in the wrong
# space by accident.
TRAINING_SPACE = "log1p(price)"
PREDICTION_SPACE = "log1p(price)  (models predict in log space)"
RUPEE_SPACE = "INR"

METRIC_KEYS: Tuple[str, ...] = (
    "MAE_INR",
    "RMSE_INR",
    "R2_INR",
    "MAPE_percent",
    "MedAPE_percent",
    "MAE_log",
    "RMSE_log",
    "R2_log",
)

METRIC_SPACE = {
    "MAE_INR": RUPEE_SPACE,
    "RMSE_INR": RUPEE_SPACE,
    "R2_INR": RUPEE_SPACE,
    "MAPE_percent": RUPEE_SPACE,
    "MedAPE_percent": RUPEE_SPACE,
    "MAE_log": "log1p(price)",
    "RMSE_log": "log1p(price)",
    "R2_log": "log1p(price)",
}

METRIC_DESCRIPTION = {
    "MAE_INR": "mean_absolute_error on INR after expm1 inversion",
    "RMSE_INR": "sqrt(mean_squared_error) on INR after expm1 inversion",
    "R2_INR": "r2_score on INR after expm1 inversion",
    "MAPE_percent": "mean absolute percentage error on INR, x100",
    "MedAPE_percent": "median absolute percentage error on INR, x100",
    "MAE_log": "mean_absolute_error in log1p(price) space, no inversion",
    "RMSE_log": "sqrt(mean_squared_error) in log1p(price) space, no inversion",
    "R2_log": "r2_score in log1p(price) space, no inversion",
}


# ============================================================
# REPORT CONTAINERS
# ============================================================

@dataclass(frozen=True)
class DuplicateReport:
    """Row accounting for the exact-duplicate policy."""

    original_rows: int
    duplicate_rows_removed: int
    final_rows: int
    key: Tuple[str, ...] = tuple(DUPLICATE_KEY)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "policy": DUPLICATE_POLICY,
            "key": list(self.key),
            "original_rows": self.original_rows,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "final_rows": self.final_rows,
        }

    def __str__(self) -> str:
        return (
            f"original_rows={self.original_rows:,}  "
            f"duplicate_rows_removed={self.duplicate_rows_removed:,}  "
            f"final_rows={self.final_rows:,}"
        )


@dataclass(frozen=True)
class FoldSet:
    """Immutable fold assignment for one evaluation regime."""

    regime: str
    splitter: str
    group_column: str | None
    n_splits: int
    random_state: int
    train: Tuple[Tuple[int, ...], ...]
    valid: Tuple[Tuple[int, ...], ...]
    dataset_rows: int

    def n_folds(self) -> int:
        return len(self.train)

    def group_overlap(self, groups: Sequence[str]) -> List[int]:
        """Per-fold count of groups present in both train and valid."""
        g = np.asarray(groups)
        out: List[int] = []
        for tr, va in zip(self.train, self.valid):
            tr_set = set(g[list(tr)].tolist())
            va_set = set(g[list(va)].tolist())
            out.append(len(tr_set & va_set))
        return out

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(self.regime.encode("utf-8"))
        for tr, va in zip(self.train, self.valid):
            h.update(np.asarray(tr, dtype=np.int64).tobytes())
            h.update(np.asarray(va, dtype=np.int64).tobytes())
        return h.hexdigest()


# ============================================================
# LOADING AND CLEANING
# ============================================================

def load_raw() -> pd.DataFrame:
    """Read the processed property dataset, unmodified on disk."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)
    return pd.read_csv(DATA_PATH)


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def _clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Type coercion, validity filter and text normalisation.

    Identical to V2 clean_data() so that V2 and V2.1 share one cleaning
    definition and differ only in the audited areas.
    """
    _require_columns(
        df,
        ["price", "area", "location", "no_of_bedrooms", "resale", "source_city"]
        + AMENITIES,
    )

    out = df.copy()

    for c in ["price", "area", "no_of_bedrooms", "resale"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    before = len(out)
    out = out[
        (out["price"] > 0)
        & (out["area"] > 0)
        & (out["no_of_bedrooms"] > 0)
    ].copy()
    invalid_removed = before - len(out)

    out["location"] = out["location"].astype(str).str.strip().str.lower()
    out["source_city"] = out["source_city"].astype(str).str.strip().str.lower()
    # Grouped-split key. Built here so that every downstream consumer,
    # including the location-grouped fold generator, uses one definition.
    out["group"] = add_group_labels(out)

    for c in AMENITIES:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out, invalid_removed


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, DuplicateReport]:
    """Apply the exact-duplicate policy. See DUPLICATE_POLICY."""
    _require_columns(df, DUPLICATE_KEY)

    original_rows = len(df)
    mask = df.duplicated(subset=DUPLICATE_KEY, keep="first")
    deduped = df[~mask].copy()

    report = DuplicateReport(
        original_rows=original_rows,
        duplicate_rows_removed=int(mask.sum()),
        final_rows=len(deduped),
    )
    return deduped, report


def _add_rowwise_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-local feature engineering.

    Every feature here is a pure function of the row's own raw columns, so
    running it before the split cannot leak information across partitions.
    The aggregate frequency features are deliberately NOT created here; they
    are produced by TrainOnlyFrequencyFeatures inside the Pipeline.
    """
    out = df.copy()

    out["log_area"] = np.log1p(out["area"])
    out["log_bedrooms"] = np.log1p(out["no_of_bedrooms"])
    out["area_per_bedroom"] = out["area"] / out["no_of_bedrooms"].clip(lower=1)
    out["area_bedroom_interaction"] = out["area"] * out["no_of_bedrooms"]

    amenity_data = out[AMENITIES]

    out["amenity_yes_count"] = (amenity_data == AMENITY_YES).sum(axis=1)

    # V2.1 change 3: replaces the perfectly collinear
    # amenity_known_count / amenity_unknown_count pair.
    # 1 when every one of the 35 values is a definite 0 or 1,
    # 0 when at least one value is 9 (not specified / unknown).
    out["amenities_fully_specified"] = (
        amenity_data.isin(AMENITY_KNOWN_VALUES).all(axis=1)
    ).astype(int)

    return out


def add_group_labels(df: pd.DataFrame) -> pd.Series:
    """source_city + '__' + location, the location-grouped split key."""
    return df["source_city"].astype(str) + "__" + df["location"].astype(str)


def build_dataset(
    raw: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, DuplicateReport, int]:
    """Load, clean, de-duplicate and add row-wise features.

    Returns (dataset, duplicate_report, invalid_rows_removed).

    The returned frame contains NO frequency feature columns. Those are added
    by TrainOnlyFrequencyFeatures, fitted on training rows only.
    """
    frame = load_raw() if raw is None else raw
    cleaned, invalid_removed = _clean(frame)
    deduped, report = remove_duplicates(cleaned)
    deduped = _add_rowwise_features(deduped)
    return deduped.reset_index(drop=True), report, invalid_removed


def frequency_feature_names() -> List[str]:
    return list(FREQUENCY_FEATURES)


def full_feature_list() -> List[str]:
    """The columns a caller passes to Pipeline.fit / predict.

    This is FINAL_FEATURES minus the two frequency features, which the
    Pipeline derives itself from training rows only.
    """
    return list(INPUT_FEATURES)


# ============================================================
# TRAIN-ONLY FREQUENCY FEATURES (V2.1 change 1)
# ============================================================

class TrainOnlyFrequencyFeatures(BaseEstimator, TransformerMixin):
    """Fit location counts on training rows only.

    This transformer is the structural fix for the V2 leakage defect. It is
    placed as the FIRST step of the Pipeline, so sklearn calls fit() with the
    training rows of the current fold and transform() separately for
    training and for validation. Because fit() never sees validation or test
    rows, no held-out row can contribute to its own frequency value.

    Frozen decision: city_frequency is NOT computed. It is a deterministic 1:1
    restatement of source_city, which is already a one-hot feature, so it
    carried no information. Only location_frequency and its log are produced.

    Unseen locations receive FALLBACK_FREQUENCY (0.0). See
    UNSEEN_FREQUENCY_POLICY.
    """

    def __init__(
        self,
        location_col: str = "location",
        fallback: float = FALLBACK_FREQUENCY,
    ) -> None:
        self.location_col = location_col
        self.fallback = fallback

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TrainOnlyFrequencyFeatures":
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "TrainOnlyFrequencyFeatures requires a pandas DataFrame so "
                "that location is addressable by name."
            )
        self.location_counts_ = X[self.location_col].value_counts()
        self.n_fit_rows_ = int(len(X))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "location_counts_"):
            raise RuntimeError(
                "TrainOnlyFrequencyFeatures must be fitted before transform."
            )

        out = X.copy()

        location_frequency = (
            out[self.location_col]
            .map(self.location_counts_)
            .fillna(self.fallback)
            .astype(float)
        )

        out["location_frequency"] = location_frequency
        out["log_location_frequency"] = np.log1p(location_frequency)

        return out

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return list(FREQUENCY_FEATURES)


# ============================================================
# PREPROCESSING (V2.1 change 4)
# ============================================================

def create_preprocessor(dense_output: bool = False) -> ColumnTransformer:
    """Leakage-safe preprocessor.

    Numeric branch : StandardScaler, fitted on training rows only.
    Categorical    : OneHotEncoder(handle_unknown="ignore").

    V2 used "passthrough" for numerics. That is harmless for XGBoost, which
    is invariant to feature scale, but it makes Ridge unevaluable because
    area_bedroom_interaction reaches 1.0e5 while amenity counts are 0..35.
    Standardising here means Ridge, RandomForest, LightGBM, CatBoost and
    XGBoost all receive the identical numeric representation.

    Tree models are not required to be scale-sensitive, so standardising
    them costs nothing and removes any model-specific preprocessing
    decision from the comparison.

    dense_output=True is provided for CatBoost, whose estimator does not
    accept scipy sparse matrices. It is a documented per-model input-format
    accommodation only; it changes no fitted statistic.
    """
    numeric = list(NUMERIC_FEATURES)
    categorical = list(ONEHOT_FEATURES)

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=not dense_output,
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def make_pipeline(
    estimator: Any,
    dense_output: bool = False,
) -> Pipeline:
    """Assemble the standard V2.1 pipeline around any estimator.

    Step 1 TrainOnlyFrequencyFeatures : train-only aggregate features.
    Step 2 preprocessor                : StandardScaler + OneHotEncoder.
    Step 3 estimator                   : supplied by the caller.

    Identical construction for all five comparison models.
    """
    return Pipeline(
        steps=[
            (
                "frequency",
                TrainOnlyFrequencyFeatures(),
            ),
            (
                "preprocessor",
                create_preprocessor(dense_output=dense_output),
            ),
            (
                "model",
                estimator,
            ),
        ]
    )


# ============================================================
# CROSS-VALIDATION FOLDS (V2.1 change 5)
# ============================================================

def generate_folds(
    df: pd.DataFrame,
    regime: str,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> FoldSet:
    """Generate fold indices for one evaluation regime.

    regime "random"            : KFold, shuffled, random_state=42.
                                 Reproducible standard CV.
    regime "location_grouped"  : GroupKFold on source_city + "__" + location,
                                 shuffled, random_state=42. No group can
                                 appear in both train and validation.

    GroupKFold is used rather than GroupShuffleSplit because K-fold reuses
    every row for validation exactly once, which makes fold-to-fold variance
    comparable across models. Groups are held out whole.
    """
    if regime == "random":
        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        group_column = None
        name = "KFold(shuffle=True)"
        train, valid = [], []
        for tr, va in splitter.split(df):
            train.append(tuple(int(i) for i in tr))
            valid.append(tuple(int(i) for i in va))

    elif regime == "location_grouped":
        splitter = GroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        group_column = "group"
        name = "GroupKFold(shuffle=True)"
        groups = df["group"].to_numpy()
        train, valid = [], []
        for tr, va in splitter.split(df, groups=groups):
            train.append(tuple(int(i) for i in tr))
            valid.append(tuple(int(i) for i in va))

    else:
        raise ValueError(
            f"Unknown regime: {regime!r}. "
            "Expected 'random' or 'location_grouped'."
        )

    return FoldSet(
        regime=regime,
        splitter=name,
        group_column=group_column,
        n_splits=n_splits,
        random_state=random_state,
        train=tuple(train),
        valid=tuple(valid),
        dataset_rows=int(len(df)),
    )


def generate_all_folds(
    df: pd.DataFrame,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> Dict[str, FoldSet]:
    """Generate both regimes. These are generated ONCE and reused by all
    five models so that every model is scored on identical partitions."""
    return {
        regime: generate_folds(
            df,
            regime=regime,
            n_splits=n_splits,
            random_state=random_state,
        )
        for regime in ("random", "location_grouped")
    }


def validate_folds(
    folds: FoldSet,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """Assert the structural guarantees the protocol depends on."""
    n = len(df)
    problems: List[str] = []

    for i, (tr, va) in enumerate(zip(folds.train, folds.valid)):
        tr_arr = np.asarray(tr, dtype=np.int64)
        va_arr = np.asarray(va, dtype=np.int64)

        if len(set(tr_arr.tolist()) & set(va_arr.tolist())):
            problems.append(f"fold {i}: train/valid row overlap")

        union = set(tr_arr.tolist()) | set(va_arr.tolist())
        if len(union) != n:
            problems.append(
                f"fold {i}: coverage {len(union)} != dataset rows {n}"
            )

    overlap = None
    if folds.group_column is not None:
        overlap = folds.group_overlap(df[folds.group_column].to_numpy())
        if any(overlap):
            problems.append(
                f"group overlap across folds: {overlap}"
            )

    return {
        "regime": folds.regime,
        "splitter": folds.splitter,
        "n_splits": folds.n_folds(),
        "random_state": folds.random_state,
        "dataset_rows": n,
        "fold_train_rows": [len(tr) for tr in folds.train],
        "fold_valid_rows": [len(va) for va in folds.valid],
        "group_column": folds.group_column,
        "max_group_overlap": None if overlap is None else max(overlap),
        "digest": folds.digest(),
        "problems": problems,
        "valid": not problems,
    }


def save_folds(
    folds_by_regime: Dict[str, FoldSet],
    path: Path = FOLDS_PATH,
) -> Path:
    """Persist fold assignments so they are generated exactly once."""
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "protocol": "V2.1",
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS,
        "dataset": {
            "path": str(DATA_PATH.relative_to(ROOT)),
            "rows": next(iter(folds_by_regime.values())).dataset_rows,
            "duplicate_policy": DUPLICATE_POLICY,
            "duplicate_key": list(DUPLICATE_KEY),
        },
        "feature_count": FINAL_FEATURE_COUNT,
        "features": list(FINAL_FEATURES),
        "pipeline_input_columns": list(INPUT_FEATURES),
        "pipeline_input_note": (
            "The two frequency features are absent because the Pipeline "
            "derives them from training rows only."
        ),
        "metric_keys": list(METRIC_KEYS),
        "regimes": {},
    }

    for regime, folds in folds_by_regime.items():
        payload["regimes"][regime] = {
            "splitter": folds.splitter,
            "group_column": folds.group_column,
            "n_splits": folds.n_splits,
            "random_state": folds.random_state,
            "digest": folds.digest(),
            "train": [list(tr) for tr in folds.train],
            "valid": [list(va) for va in folds.valid],
        }

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    return path


def load_folds(path: Path = FOLDS_PATH) -> Dict[str, FoldSet]:
    """Load previously persisted folds. Regenerating instead of reloading
    would risk a silent protocol drift."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Generate folds with "
            "python -m scripts.valuation_v2_1.build_protocol"
        )

    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    out: Dict[str, FoldSet] = {}
    for regime, block in payload["regimes"].items():
        out[regime] = FoldSet(
            regime=regime,
            splitter=block["splitter"],
            group_column=block["group_column"],
            n_splits=block["n_splits"],
            random_state=block["random_state"],
            train=tuple(tuple(x) for x in block["train"]),
            valid=tuple(tuple(x) for x in block["valid"]),
            dataset_rows=payload["dataset"]["rows"],
        )
    return out


def get_folds(
    regime: str,
    df: pd.DataFrame | None = None,
    path: Path = FOLDS_PATH,
) -> FoldSet:
    """Load persisted folds for one regime, generating them once if absent."""
    if path.exists():
        return load_folds(path)[regime]
    if df is None:
        raise FileNotFoundError(path)
    return generate_folds(df, regime=regime)


# ============================================================
# METRICS (V2.1 change 6)
# ============================================================

def log_target(price: Sequence[float] | np.ndarray) -> np.ndarray:
    """Training space: log1p(price)."""
    return np.log1p(np.asarray(price, dtype=float))


def to_rupees(log_prediction: Sequence[float] | np.ndarray) -> np.ndarray:
    """Prediction-space conversion: log1p(price) -> INR, floored at zero."""
    return np.maximum(np.expm1(np.asarray(log_prediction, dtype=float)), 0.0)


def compute_metrics(
    price_true: Sequence[float] | np.ndarray,
    log_prediction: Sequence[float] | np.ndarray,
) -> Dict[str, float]:
    """Compute the full V2.1 metric contract.

    price_true       : true INR price.
    log_prediction   : model output, which must be in log1p(price) space.

    Space separation, made explicit:

        INR metrics   : expm1(log_prediction), floored at 0, compared to
                        price_true. This is where the business cares.
        Log metrics   : log_prediction compared to log1p(price_true) with no
                        inversion. This is the space the model optimises.

    R2_log is the headline goodness-of-fit measure, because the target is
    heavy-tailed (max/min price ratio 427) and INR-space R2 is dominated by
    a handful of luxury rows.

    MAPE and MedAPE are implemented here rather than imported, because
    scikit-learn 1.9.1 exposes mean_absolute_percentage_error but not a
    median counterpart. Writing both explicitly keeps the two percentage
    metrics on one denominator convention and keeps the protocol runnable
    across sklearn versions.

    Both percentage metrics divide by max(|y_true|, EPSILON) exactly as
    sklearn does. EPSILON is machine epsilon for float64. Since clean_data
    enforces price > 0, the smallest denominator on this dataset is
    2e6 and epsilon never activates, so neither metric is distorted by the
    guard. The guard exists only so that a future unfiltered dataset cannot
    silently produce inf or NaN.
    """
    y_inr = np.asarray(price_true, dtype=float)
    y_log = np.log1p(y_inr)
    p_log = np.asarray(log_prediction, dtype=float)
    p_inr = np.maximum(np.expm1(p_log), 0.0)

    denominator = np.maximum(
        np.abs(y_inr), np.finfo(np.float64).eps
    )
    abs_pct_error = np.abs(p_inr - y_inr) / denominator

    return {
        "MAE_INR": float(mean_absolute_error(y_inr, p_inr)),
        "RMSE_INR": float(np.sqrt(mean_squared_error(y_inr, p_inr))),
        "R2_INR": float(r2_score(y_inr, p_inr)),
        "MAPE_percent": float(np.mean(abs_pct_error) * 100.0),
        "MedAPE_percent": float(np.median(abs_pct_error) * 100.0),
        "MAE_log": float(mean_absolute_error(y_log, p_log)),
        "RMSE_log": float(np.sqrt(mean_squared_error(y_log, p_log))),
        "R2_log": float(r2_score(y_log, p_log)),
    }


# ============================================================
# ENVIRONMENT RECORD (V2.1 change 7)
# ============================================================

# Machine-independent interpreter label recorded in the environment record.
# sys.executable is an absolute path such as
#   C:\Users\<user>\AppData\Local\Programs\Python\<ver>\python.exe
# which discloses the local username and host layout and is useless for
# reproducibility. python_version() already pins the interpreter version.
ENVIRONMENT_EXECUTABLE_LABEL = "python"


def environment_record() -> Dict[str, Any]:
    """Record exact installed versions. Records only; changes nothing."""
    return {
        "recorded_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": ENVIRONMENT_EXECUTABLE_LABEL,
        "packages": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
        "optional_packages": _optional_versions(),
        "note": (
            "python, pandas, numpy, scikit-learn, xgboost and joblib are at "
            "exactly the versions recorded for the XGBoost anchor run: none "
            "was installed, upgraded or downgraded at any point. lightgbm and "
            "catboost were absent when this record was first generated and "
            "were installed afterwards, unmodified from the versions shown, "
            "solely to run the Experiment 1 five-model baseline comparison. "
            "catboost pulled in plotly and graphviz as transitive "
            "dependencies; no research-relevant package was affected."
        ),
    }


def _optional_versions() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in ("lightgbm", "catboost"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "unknown")
        except Exception:
            out[name] = "not installed"
    return out


def write_environment_record(path: Path = ENVIRONMENT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = environment_record()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    return path


# ============================================================
# PROTOCOL SUMMARY
# ============================================================

def protocol_summary(
    duplicate_report: DuplicateReport,
    invalid_rows_removed: int,
) -> Dict[str, Any]:
    """Machine-readable description of the frozen V2.1 protocol."""
    return {
        "protocol": "V2.1",
        "protocol_name": "MILLOW Experiment 1 research baseline",
        "historical_reference": (
            "Historical V2 reference - pre-leakage-control baseline"
        ),
        "dataset": {
            "path": str(DATA_PATH.relative_to(ROOT)),
            "invalid_rows_removed": invalid_rows_removed,
            **duplicate_report.as_dict(),
        },
        "target": {
            "column": TARGET_COLUMN,
            "training_space": TRAINING_SPACE,
            "prediction_space": PREDICTION_SPACE,
            "evaluation_space": RUPEE_SPACE,
        },
        "features": {
            "count": FINAL_FEATURE_COUNT,
            "list": list(FINAL_FEATURES),
            "pipeline_input_columns": list(INPUT_FEATURES),
            "numeric_scaled": list(NUMERIC_FEATURES),
            "one_hot": list(ONEHOT_FEATURES),
            "rowwise_engineered": list(ROWWISE_ENGINEERED_FEATURES),
            "train_only_frequency": list(FREQUENCY_FEATURES),
            "removed_as_redundant": dict(REDUNDANT_FEATURES_REMOVED),
        },
        "amenity_semantics": dict(AMENITY_VALUE_SEMANTICS),
        "amenity_note": (
            "Raw 35 amenity columns are one-hot encoded as three discrete "
            "levels (0, 1, 9). 9 is never reinterpreted as 0 or 1. "
            "amenities_fully_specified replaces the perfectly collinear "
            "amenity_known_count / amenity_unknown_count pair of V2. "
            "amenities_fully_specified is a DATA-RECORDING / PROVENANCE "
            "indicator (did the source record every amenity field), NOT an "
            "ordinary property amenity, and must be interpreted as such."
        ),
        "frequency_policy": UNSEEN_FREQUENCY_POLICY,
        "cv": {
            "n_splits": N_SPLITS,
            "random_state": RANDOM_STATE,
            "regimes": ["random", "location_grouped"],
            "grouped_group_column": "source_city + '__' + location",
        },
        "metrics": {
            "keys": list(METRIC_KEYS),
            "space": dict(METRIC_SPACE),
            "description": dict(METRIC_DESCRIPTION),
        },
        "excluded_from_features": {
            "mreid_id": "unique row identifier, carries no signal",
            "source_file": "scrape provenance, near-duplicate of source_city",
            "derived_price_per_sqft": (
                "equals price / area; including it would be target leakage"
            ),
        },
    }
