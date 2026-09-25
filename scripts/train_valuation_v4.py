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

OUTPUT_DIR = "artifacts/valuation/v4"
MODEL_DIR = "models/valuation/v4"

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


def add_basic_features(df):
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

    amenity_cols = []

    for c in df.columns:
        if c in excluded:
            continue

        values = set(df[c].dropna().unique())

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

    for c in amenity_cols:
        df[c] = df[c].astype(str)

    return df, amenity_cols


def add_train_only_market_features(train, test):
    """
    Calculate every market/frequency statistic using TRAIN ONLY.
    """

    train = train.copy()
    test = test.copy()

    train_ppsf = (
        train["price"] / train["area"]
    )

    global_median_ppsf = train_ppsf.median()
    global_median_price = train["price"].median()

    # -----------------------------
    # City statistics
    # -----------------------------

    city_stats = (
        pd.DataFrame({
            "price": train["price"],
            "ppsf": train_ppsf,
            "city": train["source_city"]
        })
        .groupby("city")
        .agg(
            city_median_price=("price", "median"),
            city_median_ppsf=("ppsf", "median"),
            city_sample_count=("price", "size"),
        )
    )

    # -----------------------------
    # Location statistics
    # -----------------------------

    location_stats = (
        pd.DataFrame({
            "price": train["price"],
            "ppsf": train_ppsf,
            "location": train["location"]
        })
        .groupby("location")
        .agg(
            location_median_price=("price", "median"),
            location_median_ppsf=("ppsf", "median"),
            location_sample_count=("price", "size"),
        )
    )

    def apply_stats(df):

        df = df.copy()

        # City statistics
        df = df.join(
            city_stats,
            on="source_city"
        )

        # Location statistics
        df = df.join(
            location_stats,
            on="location"
        )

        # City fallback for unseen city/location statistics
        df["city_median_price"] = (
            df["city_median_price"]
            .fillna(global_median_price)
        )

        df["city_median_ppsf"] = (
            df["city_median_ppsf"]
            .fillna(global_median_ppsf)
        )

        df["city_sample_count"] = (
            df["city_sample_count"]
            .fillna(0)
        )

        # Hierarchical location fallback
        valid_location = (
            df["location_sample_count"]
            .fillna(0) >= 5
        )

        df["location_median_price"] = np.where(
            valid_location,
            df["location_median_price"],
            df["city_median_price"]
        )

        df["location_median_ppsf"] = np.where(
            valid_location,
            df["location_median_ppsf"],
            df["city_median_ppsf"]
        )

        df["location_sample_count"] = (
            df["location_sample_count"]
            .fillna(0)
        )

        # Frequency features
        df["log_location_sample_count"] = np.log1p(
            df["location_sample_count"]
        )

        df["log_city_sample_count"] = np.log1p(
            df["city_sample_count"]
        )

        # Market-prior estimate
        df["location_market_value"] = (
            df["location_median_ppsf"]
            * df["area"]
        )

        df["log_location_market_value"] = np.log1p(
            df["location_market_value"]
        )

        return df

    train = apply_stats(train)
    test = apply_stats(test)

    return train, test


def train_one_city(city_df, city):

    city_df = city_df.copy()

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

    train_locations = set(train["location"])
    test_locations = set(test["location"])

    overlap = train_locations & test_locations

    # Basic feature engineering first
    train, amenity_cols = add_basic_features(train)
    test, _ = add_basic_features(test)

    # STRICT train-only market statistics
    train, test = add_train_only_market_features(
        train,
        test
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

    y_train = np.log1p(train["price"])
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
                categorical_cols
            ),
            (
                "num",
                "passthrough",
                numeric_cols
            )
        ]
    )

    X_train_enc = preprocessor.fit_transform(
        X_train
    )

    X_test_enc = preprocessor.transform(
        X_test
    )

    model = XGBRegressor(
        n_estimators=900,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_enc,
        y_train
    )

    pred_log = model.predict(
        X_test_enc
    )

    pred = np.expm1(pred_log)
    pred = np.maximum(pred, 0)

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
        pred
    )


def main():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

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
            f"\n======================================"
        )

        print(
            f"V4 training: {city} "
            f"({len(city_df)} rows)"
        )

        print(
            f"======================================"
        )

        (
            model,
            preprocessor,
            metrics,
            y_test,
            pred
        ) = train_one_city(
            city_df,
            city
        )

        print(
            f"Train rows:      {metrics['train_rows']}"
        )

        print(
            f"Test rows:       {metrics['test_rows']}"
        )

        print(
            f"Train locations: {metrics['train_locations']}"
        )

        print(
            f"Test locations:  {metrics['test_locations']}"
        )

        print(
            f"Overlap:         {metrics['location_overlap']}"
        )

        print(
            f"MAE:             ₹{metrics['MAE']:,.0f}"
        )

        print(
            f"RMSE:            ₹{metrics['RMSE']:,.0f}"
        )

        print(
            f"R²:              {metrics['R2']:.4f}"
        )

        print(
            f"MAPE:            {metrics['MAPE']:.2f}%"
        )

        results.append(metrics)

        all_y.extend(y_test)
        all_pred.extend(pred)

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
        "location_overlap": "-"
    })

    results.append(overall)

    result_df = pd.DataFrame(results)

    print(
        "\n\n=============================================="
    )

    print(
        "MILLOW VALUATION V4 RESULTS"
    )

    print(
        "==============================================\n"
    )

    print(
        result_df.to_string(
            index=False
        )
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "v4_city_grouped_metrics.csv"
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