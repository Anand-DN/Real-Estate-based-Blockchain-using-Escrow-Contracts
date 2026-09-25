"""
MILLOW - AI Property Valuation v3

Main improvement:
Leakage-safe location/market statistics.

Location statistics are calculated ONLY from the training set.

Target:
    log1p(price)

Never use:
    derived_price_per_sqft
    price_per_sqft
    target-derived features
"""

from pathlib import Path
import json
import sys
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass


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
# LOAD + CLEAN
# ============================================================

def load_data():

    print("=" * 70)
    print("MILLOW - PROPERTY VALUATION v3")
    print("=" * 70)

    print("\nLoading:")
    print(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    print(
        f"\nDataset shape: {df.shape}"
    )

    return df


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
        f"Invalid rows removed: "
        f"{before - len(df)}"
    )

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

    for c in AMENITIES:

        if c in df.columns:

            df[c] = pd.to_numeric(
                df[c],
                errors="coerce",
            )

    return df


# ============================================================
# BASIC FEATURE ENGINEERING
# ============================================================

def engineer_basic_features(df):

    df = df.copy()

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

    existing = [
        c for c in AMENITIES
        if c in df.columns
    ]

    if existing:

        amenity_data = df[existing]

        df["amenity_yes_count"] = (
            amenity_data == 1
        ).sum(axis=1)

        df["amenity_known_count"] = (
            amenity_data.isin([0, 1])
        ).sum(axis=1)

        df["amenity_unknown_count"] = (
            amenity_data == 9
        ).sum(axis=1)

    return df


# ============================================================
# LEAKAGE-SAFE MARKET FEATURES
# ============================================================

def add_market_features(
    train_df,
    test_df,
):

    train = train_df.copy()
    test = test_df.copy()

    # --------------------------------------------------------
    # Create training-only price per sqft.
    #
    # This is NOT the target-derived column from the dataset.
    # It is calculated ONLY from training observations and then
    # aggregated by location/city.
    # --------------------------------------------------------

    train["_train_ppsf"] = (
        train["price"]
        / train["area"]
    )

    # --------------------------------------------------------
    # Global training statistics
    # --------------------------------------------------------

    global_median_price = (
        train["price"]
        .median()
    )

    global_median_ppsf = (
        train["_train_ppsf"]
        .median()
    )

    # --------------------------------------------------------
    # City statistics
    # --------------------------------------------------------

    city_price = (
        train.groupby("source_city")[
            "price"
        ]
        .median()
    )

    city_ppsf = (
        train.groupby("source_city")[
            "_train_ppsf"
        ]
        .median()
    )

    city_count = (
        train.groupby("source_city")
        .size()
    )

    # --------------------------------------------------------
    # Location statistics
    # --------------------------------------------------------

    location_price = (
        train.groupby("location")[
            "price"
        ]
        .median()
    )

    location_ppsf = (
        train.groupby("location")[
            "_train_ppsf"
        ]
        .median()
    )

    location_count = (
        train.groupby("location")
        .size()
    )

    # --------------------------------------------------------
    # Map statistics to both train and test.
    # --------------------------------------------------------

    for frame in [train, test]:

        frame["city_median_price"] = (
            frame["source_city"]
            .map(city_price)
            .fillna(global_median_price)
        )

        frame["city_median_ppsf"] = (
            frame["source_city"]
            .map(city_ppsf)
            .fillna(global_median_ppsf)
        )

        frame["city_sample_count"] = (
            frame["source_city"]
            .map(city_count)
            .fillna(0)
        )

        frame["location_median_price"] = (
            frame["location"]
            .map(location_price)
            .fillna(
                frame["city_median_price"]
            )
        )

        frame["location_median_ppsf"] = (
            frame["location"]
            .map(location_ppsf)
            .fillna(
                frame["city_median_ppsf"]
            )
        )

        frame["location_sample_count"] = (
            frame["location"]
            .map(location_count)
            .fillna(0)
        )

        frame["log_location_sample_count"] = (
            np.log1p(
                frame["location_sample_count"]
            )
        )

    train.drop(
        columns=["_train_ppsf"],
        errors="ignore",
        inplace=True,
    )

    test.drop(
        columns=["_train_ppsf"],
        errors="ignore",
        inplace=True,
    )

    return train, test


# ============================================================
# FEATURES
# ============================================================

def get_features(df):

    basic = [
        "area",
        "no_of_bedrooms",
        "resale",
        "log_area",
        "log_bedrooms",
        "area_per_bedroom",
        "area_bedroom_interaction",
        "amenity_yes_count",
        "amenity_known_count",
        "amenity_unknown_count",
    ]

    market = [
        "city_median_price",
        "city_median_ppsf",
        "city_sample_count",
        "location_median_price",
        "location_median_ppsf",
        "location_sample_count",
        "log_location_sample_count",
    ]

    categorical = [
        "source_city",
        "location",
    ]

    amenities = [
        c for c in AMENITIES
        if c in df.columns
    ]

    features = (
        basic
        + market
        + categorical
        + amenities
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

    return ColumnTransformer(
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


# ============================================================
# MODEL
# ============================================================

def create_model(features):

    preprocessor = create_preprocessor(
        features
    )

    xgb = XGBRegressor(
        n_estimators=800,
        max_depth=7,
        learning_rate=0.035,
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

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                xgb,
            ),
        ]
    )


# ============================================================
# METRICS
# ============================================================

def evaluate(
    y_true_log,
    y_pred_log,
):

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
# TRAIN ONE SPLIT
# ============================================================

def train_split(
    df,
    train_idx,
    test_idx,
    split_name,
):

    print("\n" + "-" * 70)
    print(
        f"V3 TRAINING: {split_name}"
    )
    print("-" * 70)

    train_raw = df.iloc[
        train_idx
    ].copy()

    test_raw = df.iloc[
        test_idx
    ].copy()

    # --------------------------------------------------------
    # BASIC FEATURES BEFORE MARKET FEATURES
    # --------------------------------------------------------

    train_raw = engineer_basic_features(
        train_raw
    )

    test_raw = engineer_basic_features(
        test_raw
    )

    # --------------------------------------------------------
    # CRITICAL:
    # Calculate market statistics using TRAIN ONLY.
    # --------------------------------------------------------

    train, test = add_market_features(
        train_raw,
        test_raw,
    )

    features = get_features(
        train
    )

    print(
        f"Train rows: {len(train):,}"
    )

    print(
        f"Test rows: {len(test):,}"
    )

    print(
        f"Features: {len(features)}"
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    X_train = train[features]
    X_test = test[features]

    y_train = np.log1p(
        train["price"].values
    )

    y_test = np.log1p(
        test["price"].values
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(
        features
    )

    print(
        "\nTraining XGBoost..."
    )

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
        / f"valuation_v3_xgboost_{split_name}.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"\nSaved model:"
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
            / (
                f"valuation_v3_feature_importance_"
                f"{split_name}.csv"
            )
        )

        importance_df.to_csv(
            importance_path,
            index=False,
        )

        print(
            "\nTop 20 features:"
        )

        print(
            importance_df
            .head(20)
            .to_string(index=False)
        )

    except Exception as exc:

        print(
            "\nFeature importance error:"
        )

        print(exc)

    return {
        "version": "v3",
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

    indices = np.arange(
        len(df)
    )

    # --------------------------------------------------------
    # RANDOM SPLIT
    # --------------------------------------------------------

    train_random, test_random = (
        train_test_split(
            indices,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
    )

    # --------------------------------------------------------
    # LOCATION-GROUPED SPLIT
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
    print("DATA SPLITS")
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
    # TRAIN
    # --------------------------------------------------------

    results = []

    results.append(
        train_split(
            df,
            train_random,
            test_random,
            "random",
        )
    )

    results.append(
        train_split(
            df,
            train_grouped,
            test_grouped,
            "location_grouped",
        )
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    csv_path = (
        ARTIFACT_DIR
        / "valuation_v3_metrics.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False,
    )

    json_path = (
        ARTIFACT_DIR
        / "valuation_v3_metrics.json"
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
    print("V3 FINAL RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nMetrics saved:"
        f"\n{csv_path}"
    )

    print(
        f"\nJSON saved:"
        f"\n{json_path}"
    )

    print("\n" + "=" * 70)
    print("MILLOW VALUATION v3 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()