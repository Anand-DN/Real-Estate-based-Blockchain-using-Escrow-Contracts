"""
MILLOW - AI Property Valuation v2

v2 improvements over v1:
- Feature engineering
- XGBoost regression
- Same random and location-grouped evaluation
- Same target: log1p(price)
- No target leakage

Important:
0 = No
1 = Yes
9 = Not specified / unknown
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    train_test_split,
    GroupShuffleSplit,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "MREID_property.csv"
)

MODEL_DIR = ROOT / "models" / "valuation"
ARTIFACT_DIR = ROOT / "artifacts" / "valuation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# AMENITIES
# ============================================================

AMENITIES = [
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


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("MILLOW - PROPERTY VALUATION v2")
    print("=" * 70)

    print("\nLoading:")
    print(DATA_PATH)

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    print(f"\nShape: {df.shape}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(df):

    required = [
        "price",
        "area",
        "location",
        "no_of_bedrooms",
        "resale",
        "source_city",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    numeric = [
        "price",
        "area",
        "no_of_bedrooms",
        "resale",
    ]

    for c in numeric:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    before = len(df)

    df = df[
        (df["price"] > 0)
        & (df["area"] > 0)
        & (df["no_of_bedrooms"] > 0)
    ].copy()

    print(
        f"\nInvalid rows removed: "
        f"{before - len(df)}"
    )

    # Normalize text.
    df["location"] = (
        df["location"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["source_city"] = (
        df["source_city"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Preserve 0 / 1 / 9.
    for c in AMENITIES:

        if c in df.columns:

            df[c] = pd.to_numeric(
                df[c],
                errors="coerce",
            )

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # Basic nonlinear property features
    # --------------------------------------------------------

    df["log_area"] = np.log1p(
        df["area"]
    )

    df["log_bedrooms"] = np.log1p(
        df["no_of_bedrooms"]
    )

    df["area_per_bedroom"] = (
        df["area"]
        / df["no_of_bedrooms"].clip(lower=1)
    )

    df["area_bedroom_interaction"] = (
        df["area"]
        * df["no_of_bedrooms"]
    )

    # --------------------------------------------------------
    # Amenity information
    # --------------------------------------------------------

    existing_amenities = [
        c for c in AMENITIES
        if c in df.columns
    ]

    if existing_amenities:

        amenity_data = df[
            existing_amenities
        ]

        # Number explicitly marked Yes.
        df["amenity_yes_count"] = (
            amenity_data == 1
        ).sum(axis=1)

        # Number explicitly known (0 or 1).
        df["amenity_known_count"] = (
            amenity_data.isin([0, 1])
        ).sum(axis=1)

        # Number unknown / not specified.
        df["amenity_unknown_count"] = (
            amenity_data == 9
        ).sum(axis=1)

    # --------------------------------------------------------
    # Location frequency
    # --------------------------------------------------------

    location_counts = (
        df["location"]
        .value_counts()
    )

    df["location_frequency"] = (
        df["location"]
        .map(location_counts)
        .astype(float)
    )

    city_counts = (
        df["source_city"]
        .value_counts()
    )

    df["city_frequency"] = (
        df["source_city"]
        .map(city_counts)
        .astype(float)
    )

    # Log-transform frequency.
    df["log_location_frequency"] = np.log1p(
        df["location_frequency"]
    )

    # --------------------------------------------------------
    # Price-per-area MUST NOT be created.
    #
    # Price / Area would use the target and create leakage.
    # --------------------------------------------------------

    print("\nEngineered features added:")
    engineered = [
        "log_area",
        "log_bedrooms",
        "area_per_bedroom",
        "area_bedroom_interaction",
        "amenity_yes_count",
        "amenity_known_count",
        "amenity_unknown_count",
        "location_frequency",
        "city_frequency",
        "log_location_frequency",
    ]

    for c in engineered:
        if c in df.columns:
            print(f"  {c}")

    return df


# ============================================================
# FEATURE SET
# ============================================================

def get_features(df):

    basic = [
        "area",
        "no_of_bedrooms",
        "resale",
    ]

    engineered = [
        "log_area",
        "log_bedrooms",
        "area_per_bedroom",
        "area_bedroom_interaction",
        "amenity_yes_count",
        "amenity_known_count",
        "amenity_unknown_count",
        "location_frequency",
        "city_frequency",
        "log_location_frequency",
    ]

    categorical = [
        "source_city",
        "location",
    ]

    amenity_features = [
        c for c in AMENITIES
        if c in df.columns
    ]

    features = (
        basic
        + engineered
        + categorical
        + amenity_features
    )

    return [
        c for c in features
        if c in df.columns
    ]


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor(features):

    categorical = [
        c for c in features
        if c in [
            "source_city",
            "location",
        ]
        or c in AMENITIES
    ]

    numeric = [
        c for c in features
        if c not in categorical
    ]

    transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )

    return transformer


# ============================================================
# MODEL
# ============================================================

def create_model(features):

    preprocessor = create_preprocessor(
        features
    )

    model = XGBRegressor(
        n_estimators=700,
        max_depth=7,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


# ============================================================
# METRICS
# ============================================================

def evaluate(y_true_log, y_pred_log):

    y_true = np.expm1(
        y_true_log
    )

    y_pred = np.expm1(
        y_pred_log
    )

    y_pred = np.maximum(
        y_pred,
        0,
    )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    nonzero = y_true != 0

    mape = (
        np.mean(
            np.abs(
                (
                    y_true[nonzero]
                    - y_pred[nonzero]
                )
                / y_true[nonzero]
            )
        )
        * 100
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE_percent": float(mape),
    }


# ============================================================
# TRAIN
# ============================================================

def train_model(
    df,
    features,
    split_name,
    train_idx,
    test_idx,
):

    print("\n" + "-" * 70)
    print(
        f"TRAINING XGBOOST - {split_name}"
    )
    print("-" * 70)

    train = df.iloc[train_idx]
    test = df.iloc[test_idx]

    X_train = train[features]
    X_test = test[features]

    y_train = np.log1p(
        train["price"].values
    )

    y_test = np.log1p(
        test["price"].values
    )

    print(
        f"Train rows: {len(train):,}"
    )

    print(
        f"Test rows:  {len(test):,}"
    )

    model = create_model(
        features
    )

    print("\nTraining XGBoost...")

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    metrics = evaluate(
        y_test,
        predictions,
    )

    print("\nRESULTS")

    print(
        f"MAE:  ₹{metrics['MAE']:,.2f}"
    )

    print(
        f"RMSE: ₹{metrics['RMSE']:,.2f}"
    )

    print(
        f"R²:   {metrics['R2']:.4f}"
    )

    print(
        f"MAPE: {metrics['MAPE_percent']:.2f}%"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / f"valuation_v2_xgboost_{split_name}.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"\nModel saved:"
        f"\n{model_path}"
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    try:

        preprocessor = (
            model.named_steps[
                "preprocessor"
            ]
        )

        xgb = (
            model.named_steps[
                "model"
            ]
        )

        names = (
            preprocessor
            .get_feature_names_out()
        )

        importance = (
            xgb.feature_importances_
        )

        importance_df = pd.DataFrame(
            {
                "feature": names,
                "importance": importance,
            }
        ).sort_values(
            "importance",
            ascending=False,
        )

        importance_path = (
            ARTIFACT_DIR
            / f"valuation_v2_feature_importance_{split_name}.csv"
        )

        importance_df.to_csv(
            importance_path,
            index=False,
        )

        print(
            f"\nTop 20 features:"
        )

        print(
            importance_df
            .head(20)
            .to_string(index=False)
        )

    except Exception as e:

        print(
            "\nFeature importance error:"
        )

        print(e)

    return {
        "version": "v2",
        "model": "XGBoost",
        "split": split_name,
        "features": len(features),
        "train_rows": len(train),
        "test_rows": len(test),
        **metrics,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    df = clean_data(
        df
    )

    df = engineer_features(
        df
    )

    features = get_features(
        df
    )

    print("\n" + "=" * 70)
    print("FINAL FEATURE SET")
    print("=" * 70)

    print(
        f"Number of features: "
        f"{len(features)}"
    )

    print(
        ", ".join(features)
    )

    # --------------------------------------------------------
    # Random split
    # --------------------------------------------------------

    indices = np.arange(
        len(df)
    )

    train_random, test_random = (
        train_test_split(
            indices,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
    )

    # --------------------------------------------------------
    # Location-grouped split
    # --------------------------------------------------------

    groups = (
        df["source_city"].astype(str)
        + "__"
        + df["location"].astype(str)
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_grouped, test_grouped = next(
        splitter.split(
            df,
            groups=groups,
        )
    )

    print("\n" + "=" * 70)
    print("SPLITS")
    print("=" * 70)

    print(
        f"Random:"
        f" train={len(train_random):,}"
        f" test={len(test_random):,}"
    )

    print(
        f"Location grouped:"
        f" train={len(train_grouped):,}"
        f" test={len(test_grouped):,}"
    )

    # --------------------------------------------------------
    # Train both
    # --------------------------------------------------------

    results = []

    results.append(
        train_model(
            df,
            features,
            "random",
            train_random,
            test_random,
        )
    )

    results.append(
        train_model(
            df,
            features,
            "location_grouped",
            train_grouped,
            test_grouped,
        )
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_path = (
        ARTIFACT_DIR
        / "valuation_v2_metrics.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    json_path = (
        ARTIFACT_DIR
        / "valuation_v2_metrics.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("V2 FINAL RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nMetrics:"
        f"\n{results_path}"
    )

    print(
        f"\nJSON:"
        f"\n{json_path}"
    )

    print("\n" + "=" * 70)
    print("MILLOW VALUATION v2 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()