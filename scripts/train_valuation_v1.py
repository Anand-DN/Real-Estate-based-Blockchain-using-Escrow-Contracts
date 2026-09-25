"""
MILLOW - AI Property Valuation v1

Purpose:
    Train and evaluate property price prediction models using the
    processed MREID property dataset.

Experiments:
    A. Property-only
    B. Property + City
    C. Property + City + Location + Amenities

Evaluation:
    1. Random property split
    2. Location-grouped split

Target:
    log1p(price)

Important:
    Amenity value 9 is preserved as a separate categorical state.
    We do NOT assume 9 means 0.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "MREID_property.csv"

MODEL_DIR = PROJECT_ROOT / "models" / "valuation"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "valuation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# AMENITY COLUMNS
# ============================================================

AMENITY_COLUMNS = [
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
# DATA LOADING
# ============================================================

def load_data():
    print("=" * 70)
    print("MILLOW - PROPERTY VALUATION v1")
    print("=" * 70)

    print(f"\nLoading dataset:")
    print(DATA_PATH)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"\nDataset shape: {df.shape}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return df


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_data(df):
    print("\n" + "=" * 70)
    print("DATA VALIDATION")
    print("=" * 70)

    required_columns = [
        "price",
        "area",
        "location",
        "no_of_bedrooms",
        "resale",
        "source_city",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Numeric validation
    numeric_columns = [
        "price",
        "area",
        "no_of_bedrooms",
        "resale",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    before = len(df)

    # Remove technically invalid records only.
    # We are NOT removing statistical outliers.
    df = df[
        (df["price"] > 0)
        & (df["area"] > 0)
        & (df["no_of_bedrooms"] > 0)
    ].copy()

    removed = before - len(df)

    print(f"\nInvalid rows removed: {removed}")
    print(f"Remaining rows: {len(df):,}")

    # Normalize location strings.
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

    # Preserve amenity 0 / 1 / 9 as categorical values.
    for col in AMENITY_COLUMNS:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(
                    df[col],
                    errors="coerce"
                )
                .astype("Int64")
                .astype(str)
            )

    print("\nAmenity encoding:")
    print("0 = No")
    print("1 = Yes")
    print("9 = Not specified / unknown state preserved")

    return df


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

def get_experiments(df):

    property_features = [
        "area",
        "no_of_bedrooms",
    ]

    city_features = [
        "source_city",
    ]

    location_features = [
        "location",
        "resale",
    ]

    available_amenities = [
        col
        for col in AMENITY_COLUMNS
        if col in df.columns
    ]

    experiments = {
        "A_property_only": property_features,

        "B_property_city": (
            property_features
            + city_features
        ),

        "C_full_property_intelligence": (
            property_features
            + city_features
            + location_features
            + available_amenities
        ),
    }

    return experiments


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor(feature_columns):

    categorical_columns = [
        col
        for col in feature_columns
        if col in [
            "source_city",
            "location",
        ]
        or col in AMENITY_COLUMNS
    ]

    numeric_columns = [
        col
        for col in feature_columns
        if col not in categorical_columns
    ]

    transformers = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                StandardScaler(),
                numeric_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor


# ============================================================
# MODELS
# ============================================================

def create_models(feature_columns):

    preprocessor_ridge = create_preprocessor(
        feature_columns
    )

    preprocessor_rf = create_preprocessor(
        feature_columns
    )

    ridge = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor_ridge,
            ),
            (
                "model",
                Ridge(
                    alpha=10.0
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor_rf,
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=None,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return {
        "Ridge": ridge,
        "RandomForest": random_forest,
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true_log,
    y_pred_log,
):

    # Convert back to original price scale.
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    y_pred = np.maximum(
        y_pred,
        0
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

    # Avoid division by zero.
    non_zero = y_true != 0

    mape = (
        np.mean(
            np.abs(
                (
                    y_true[non_zero]
                    - y_pred[non_zero]
                )
                / y_true[non_zero]
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
# TRAIN / EVALUATE
# ============================================================

def train_experiment(
    df,
    experiment_name,
    feature_columns,
    split_name,
    train_indices,
    test_indices,
):

    print("\n" + "-" * 70)
    print(
        f"EXPERIMENT: {experiment_name}"
    )
    print(
        f"FEATURES: {len(feature_columns)}"
    )
    print(
        f"SPLIT: {split_name}"
    )
    print("-" * 70)

    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]

    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]

    # Log transform target.
    y_train = np.log1p(
        train_df["price"].values
    )

    y_test = np.log1p(
        test_df["price"].values
    )

    print(
        f"Train rows: {len(train_df):,}"
    )

    print(
        f"Test rows:  {len(test_df):,}"
    )

    models = create_models(
        feature_columns
    )

    results = []

    for model_name, model in models.items():

        print(
            f"\nTraining {model_name}..."
        )

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

        metrics = calculate_metrics(
            y_test,
            predictions,
        )

        result = {
            "experiment": experiment_name,
            "split": split_name,
            "model": model_name,
            "n_features": len(feature_columns),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            **metrics,
        }

        results.append(result)

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

        # Save model.
        model_filename = (
            f"{experiment_name}"
            f"__{split_name}"
            f"__{model_name}.joblib"
        )

        model_path = (
            MODEL_DIR
            / model_filename
        )

        joblib.dump(
            model,
            model_path,
        )

        print(
            f"Saved: {model_path}"
        )

        # Save feature importance for Random Forest.
        if model_name == "RandomForest":

            try:
                preprocessor = (
                    model.named_steps[
                        "preprocessor"
                    ]
                )

                rf_model = (
                    model.named_steps[
                        "model"
                    ]
                )

                feature_names = (
                    preprocessor
                    .get_feature_names_out()
                )

                importances = (
                    rf_model
                    .feature_importances_
                )

                importance_df = pd.DataFrame(
                    {
                        "feature": feature_names,
                        "importance": importances,
                    }
                ).sort_values(
                    "importance",
                    ascending=False,
                )

                importance_path = (
                    ARTIFACT_DIR
                    / (
                        f"{experiment_name}"
                        f"__{split_name}"
                        f"__feature_importance.csv"
                    )
                )

                importance_df.to_csv(
                    importance_path,
                    index=False,
                )

                print(
                    f"Feature importance saved: "
                    f"{importance_path}"
                )

                print(
                    "\nTop 15 features:"
                )

                print(
                    importance_df.head(15)
                    .to_string(index=False)
                )

            except Exception as exc:
                print(
                    "Could not save feature importance:"
                )
                print(exc)

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    df = validate_data(df)

    experiments = get_experiments(
        df
    )

    print("\n" + "=" * 70)
    print("EXPERIMENTS")
    print("=" * 70)

    for name, features in experiments.items():

        print(
            f"\n{name}:"
        )

        print(
            f"  Number of features: "
            f"{len(features)}"
        )

        print(
            "  "
            + ", ".join(features[:15])
        )

        if len(features) > 15:
            print(
                f"  ... +{len(features) - 15} more"
            )

    # ========================================================
    # RANDOM SPLIT
    # ========================================================

    all_indices = np.arange(
        len(df)
    )

    train_indices, test_indices = (
        train_test_split(
            all_indices,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
    )

    # ========================================================
    # LOCATION-GROUPED SPLIT
    # ========================================================

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

    grouped_train_indices, grouped_test_indices = next(
        splitter.split(
            df,
            groups=groups,
        )
    )

    print("\n" + "=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)

    print(
        f"\nRandom split:"
        f"\n  Train: {len(train_indices):,}"
        f"\n  Test:  {len(test_indices):,}"
    )

    print(
        f"\nLocation-grouped split:"
        f"\n  Train: {len(grouped_train_indices):,}"
        f"\n  Test:  {len(grouped_test_indices):,}"
    )

    # ========================================================
    # RUN EXPERIMENTS
    # ========================================================

    all_results = []

    for experiment_name, feature_columns in experiments.items():

        # Random split
        results = train_experiment(
            df=df,
            experiment_name=experiment_name,
            feature_columns=feature_columns,
            split_name="random",
            train_indices=train_indices,
            test_indices=test_indices,
        )

        all_results.extend(
            results
        )

        # Location-grouped split
        results = train_experiment(
            df=df,
            experiment_name=experiment_name,
            feature_columns=feature_columns,
            split_name="location_grouped",
            train_indices=grouped_train_indices,
            test_indices=grouped_test_indices,
        )

        all_results.extend(
            results
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_path = (
        ARTIFACT_DIR
        / "valuation_metrics_v1.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    # JSON copy
    json_path = (
        ARTIFACT_DIR
        / "valuation_metrics_v1.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            all_results,
            f,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        results_df[
            [
                "experiment",
                "split",
                "model",
                "MAE",
                "RMSE",
                "R2",
                "MAPE_percent",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nMetrics saved to:"
        f"\n{results_path}"
    )

    print(
        f"\nJSON saved to:"
        f"\n{json_path}"
    )

    print("\n" + "=" * 70)
    print("MILLOW VALUATION v1 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()