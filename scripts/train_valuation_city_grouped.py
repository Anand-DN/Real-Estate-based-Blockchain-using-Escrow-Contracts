import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

DATA_PATH = "data/processed/MREID_property.csv"

OUTPUT_DIR = "artifacts/valuation/city_grouped"
MODEL_DIR = "models/valuation/city_grouped"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    mape = np.mean(
        np.abs((y_true - y_pred) / y_true)
    ) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
    }


def add_features(df):

    df = df.copy()

    df["log_area"] = np.log1p(df["area"])
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

    excluded = {
        "mreid_id",
        "price",
        "area",
        "location",
        "source_city",
        "source_file",
        "derived_price_per_sqft",
        "no_of_bedrooms",
    }

    candidate_amenities = [
        c for c in df.columns
        if c not in excluded
    ]

    amenity_cols = []

    for c in candidate_amenities:
        values = set(
            df[c].dropna().unique()
        )

        if values.issubset({0, 1, 9}):
            amenity_cols.append(c)

    df["amenity_yes_count"] = (
        df[amenity_cols] == 1
    ).sum(axis=1)

    df["amenity_known_count"] = (
        df[amenity_cols].isin([0, 1])
    ).sum(axis=1)

    df["amenity_unknown_count"] = (
        df[amenity_cols] == 9
    ).sum(axis=1)

    # Frequency is calculated from the city dataset.
    # It is NOT a price statistic.
    location_counts = (
        df["location"].value_counts()
    )

    df["location_frequency"] = (
        df["location"].map(location_counts)
    )

    df["log_location_frequency"] = np.log1p(
        df["location_frequency"]
    )

    # Keep 0 / 1 / 9 as categorical states.
    for c in amenity_cols:
        df[c] = df[c].astype(str)

    return df


def train_city_grouped(city_df, city):

    # Groups are locations.
    groups = city_df["location"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(
            city_df,
            groups=groups
        )
    )

    train = city_df.iloc[train_idx].copy()
    test = city_df.iloc[test_idx].copy()

    train_locations = set(
        train["location"]
    )

    test_locations = set(
        test["location"]
    )

    overlap = (
        train_locations
        & test_locations
    )

    print(
        f"  Train locations: {len(train_locations)}"
    )

    print(
        f"  Test locations:  {len(test_locations)}"
    )

    print(
        f"  Location overlap: {len(overlap)}"
    )

    feature_exclude = [
        "price",
        "mreid_id",
        "source_file",
        "derived_price_per_sqft",
    ]

    features = [
        c for c in train.columns
        if c not in feature_exclude
    ]

    X_train = train[features]
    X_test = test[features]

    y_train = np.log1p(
        train["price"]
    )

    y_test = test["price"].values

    categorical_cols = [
        c for c in features
        if not pd.api.types.is_numeric_dtype(
            X_train[c]
        )
    ]

    numeric_cols = [
        c for c in features
        if c not in categorical_cols
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_cols,
            ),
            (
                "num",
                "passthrough",
                numeric_cols,
            ),
        ]
    )

    X_train_enc = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_test_enc = (
        preprocessor.transform(
            X_test
        )
    )

    model = XGBRegressor(
        n_estimators=800,
        max_depth=7,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train_enc,
        y_train
    )

    pred_log = model.predict(
        X_test_enc
    )

    pred = np.expm1(
        pred_log
    )

    pred = np.maximum(
        pred,
        0
    )

    metrics = evaluate(
        y_test,
        pred
    )

    metrics.update({
        "city": city,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_locations": len(train_locations),
        "test_locations": len(test_locations),
        "location_overlap": len(overlap),
    })

    return (
        model,
        preprocessor,
        metrics,
        y_test,
        pred,
    )


def main():

    print("Loading dataset...")

    df = pd.read_csv(
        DATA_PATH
    )

    required = [
        "price",
        "area",
        "location",
        "source_city",
        "no_of_bedrooms",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # Remove only technically invalid records.
    df = df[
        (df["price"] > 0)
        & (df["area"] > 0)
        & (df["no_of_bedrooms"] > 0)
    ].copy()

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
    )

    df = add_features(df)

    cities = sorted(
        df["source_city"].unique()
    )

    print("\nCities:")
    print(cities)

    results = []

    all_y = []
    all_pred = []

    for city in cities:

        city_df = df[
            df["source_city"] == city
        ].copy()

        print(
            f"\n================================"
        )

        print(
            f"Training {city} "
            f"({len(city_df)} rows)"
        )

        print(
            "================================"
        )

        (
            model,
            preprocessor,
            metrics,
            y_test,
            pred,
        ) = train_city_grouped(
            city_df,
            city
        )

        print(
            f"  MAE  = ₹{metrics['MAE']:,.0f}"
        )

        print(
            f"  RMSE = ₹{metrics['RMSE']:,.0f}"
        )

        print(
            f"  R²   = {metrics['R2']:.4f}"
        )

        print(
            f"  MAPE = {metrics['MAPE']:.2f}%"
        )

        results.append(
            metrics
        )

        all_y.extend(
            y_test
        )

        all_pred.extend(
            pred
        )

        # Save model
        model.save_model(
            os.path.join(
                MODEL_DIR,
                f"{city.lower()}_xgb.json"
            )
        )

        joblib.dump(
            preprocessor,
            os.path.join(
                MODEL_DIR,
                f"{city.lower()}_preprocessor.joblib"
            )
        )

    # Overall evaluation
    overall = evaluate(
        np.array(all_y),
        np.array(all_pred)
    )

    overall.update({
        "city": "ALL_CITIES",
        "train_rows": "-",
        "test_rows": len(all_y),
        "train_locations": "-",
        "test_locations": "-",
        "location_overlap": "-",
    })

    results.append(
        overall
    )

    result_df = pd.DataFrame(
        results
    )

    print(
        "\n\n=============================================="
    )

    print(
        "CITY-SPECIFIC LOCATION-GROUPED RESULTS"
    )

    print(
        "==============================================\n"
    )

    columns = [
        "city",
        "train_rows",
        "test_rows",
        "train_locations",
        "test_locations",
        "location_overlap",
        "MAE",
        "RMSE",
        "R2",
        "MAPE",
    ]

    print(
        result_df[
            columns
        ].to_string(
            index=False
        )
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "city_grouped_metrics.csv"
    )

    result_df.to_csv(
        output_path,
        index=False
    )

    print(
        "\nSaved:",
        output_path
    )


if __name__ == "__main__":
    main()