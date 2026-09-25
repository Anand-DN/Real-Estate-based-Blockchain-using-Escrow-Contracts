import os
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/processed/MREID_property.csv"

OUTPUT_DIR = "artifacts/valuation/v5_1_fixed"
MODEL_DIR = "models/valuation/v5_1_fixed"

RANDOM_STATE = 42
TEST_SIZE = 0.20

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["area"] = pd.to_numeric(df["area"], errors="coerce")
df["no_of_bedrooms"] = pd.to_numeric(
    df["no_of_bedrooms"],
    errors="coerce"
)

df = df[
    (df["price"] > 0)
    & (df["area"] > 0)
    & (df["no_of_bedrooms"] > 0)
].copy()

df["source_city"] = (
    df["source_city"]
    .astype(str)
    .str.strip()
)

df["location"] = (
    df["location"]
    .astype(str)
    .str.strip()
)

cities = sorted(df["source_city"].unique())

print("\nCities:")
print(cities)


# ============================================================
# AMENITIES
# ============================================================

exclude_cols = {
    "mreid_id",
    "price",
    "area",
    "no_of_bedrooms",
    "derived_price_per_sqft",
    "source_file",
    "source_city",
    "location",
}

amenity_cols = [
    c for c in df.columns
    if c not in exclude_cols
]

amenity_cols = [
    c for c in amenity_cols
    if pd.api.types.is_numeric_dtype(df[c])
]


# ============================================================
# BASIC FEATURES
# ============================================================

def create_basic_features(data):

    x = data.copy()

    x["log_area"] = np.log1p(x["area"])

    x["log_bedrooms"] = np.log1p(
        x["no_of_bedrooms"]
    )

    x["area_per_bedroom"] = (
        x["area"] /
        x["no_of_bedrooms"]
    )

    x["area_bedroom_interaction"] = (
        x["area"] *
        x["no_of_bedrooms"]
    )

    amenity_values = x[amenity_cols].apply(
        pd.to_numeric,
        errors="coerce"
    )

    x["amenity_yes_count"] = (
        amenity_values == 1
    ).sum(axis=1)

    x["amenity_known_count"] = (
        amenity_values.isin([0, 1])
    ).sum(axis=1)

    x["amenity_unknown_count"] = (
        amenity_values == 9
    ).sum(axis=1)

    x = x.drop(
        columns=amenity_cols
    )

    return x


# ============================================================
# TRAIN-ONLY MARKET FEATURES
# ============================================================

def add_market_features(
    train_x,
    test_x,
    train_raw
):

    train_x = train_x.copy()
    test_x = test_x.copy()

    # --------------------------------------------------------
    # Training-only global statistics
    # --------------------------------------------------------

    train_ppsf = (
        train_raw["price"] /
        train_raw["area"]
    )

    global_median_ppsf = (
        train_ppsf.median()
    )

    global_median_price = (
        train_raw["price"].median()
    )

    # --------------------------------------------------------
    # City statistics
    # --------------------------------------------------------

    city_stats = (
        train_raw
        .assign(
            ppsf=train_raw["price"]
            / train_raw["area"]
        )
        .groupby("source_city")
        .agg(
            city_median_price=(
                "price",
                "median"
            ),
            city_median_ppsf=(
                "ppsf",
                "median"
            ),
            city_sample_count=(
                "price",
                "size"
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Location statistics
    # --------------------------------------------------------

    location_stats = (
        train_raw
        .assign(
            ppsf=train_raw["price"]
            / train_raw["area"]
        )
        .groupby(
            ["source_city", "location"]
        )
        .agg(
            location_median_price=(
                "price",
                "median"
            ),
            location_median_ppsf=(
                "ppsf",
                "median"
            ),
            location_sample_count=(
                "price",
                "size"
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Merge city statistics
    # --------------------------------------------------------

    train_x = train_x.merge(
        city_stats,
        on="source_city",
        how="left"
    )

    test_x = test_x.merge(
        city_stats,
        on="source_city",
        how="left"
    )

    # --------------------------------------------------------
    # Merge location statistics
    # --------------------------------------------------------

    train_x = train_x.merge(
        location_stats,
        on=["source_city", "location"],
        how="left"
    )

    test_x = test_x.merge(
        location_stats,
        on=["source_city", "location"],
        how="left"
    )

    # --------------------------------------------------------
    # Hierarchical fallback
    #
    # If location has fewer than 5 observations,
    # use city-level market statistics.
    # --------------------------------------------------------

    reliable_location = (
        train_x["location_sample_count"]
        >= 5
    )

    for frame in [train_x, test_x]:

        reliable = (
            frame["location_sample_count"]
            >= 5
        )

        frame["location_median_price"] = (
            frame["location_median_price"]
            .where(
                reliable,
                frame["city_median_price"]
            )
        )

        frame["location_median_ppsf"] = (
            frame["location_median_ppsf"]
            .where(
                reliable,
                frame["city_median_ppsf"]
            )
        )

        frame["location_sample_count"] = (
            frame["location_sample_count"]
            .where(
                reliable,
                frame["city_sample_count"]
            )
        )

    # --------------------------------------------------------
    # Fill unseen city/location values
    # --------------------------------------------------------

    for frame in [train_x, test_x]:

        frame["city_median_price"] = (
            frame["city_median_price"]
            .fillna(global_median_price)
        )

        frame["city_median_ppsf"] = (
            frame["city_median_ppsf"]
            .fillna(global_median_ppsf)
        )

        frame["city_sample_count"] = (
            frame["city_sample_count"]
            .fillna(0)
        )

        frame["location_median_price"] = (
            frame["location_median_price"]
            .fillna(
                frame["city_median_price"]
            )
        )

        frame["location_median_ppsf"] = (
            frame["location_median_ppsf"]
            .fillna(
                frame["city_median_ppsf"]
            )
        )

        frame["location_sample_count"] = (
            frame["location_sample_count"]
            .fillna(
                frame["city_sample_count"]
            )
        )

        frame["log_city_sample_count"] = (
            np.log1p(
                frame["city_sample_count"]
            )
        )

        frame["log_location_sample_count"] = (
            np.log1p(
                frame["location_sample_count"]
            )
        )

        frame["location_market_value"] = (
            frame["location_median_ppsf"]
            * frame["area"]
        )

        frame["log_location_market_value"] = (
            np.log1p(
                frame["location_market_value"]
            )
        )

    return train_x, test_x


# ============================================================
# MODEL
# ============================================================

def build_model():

    return XGBRegressor(
        n_estimators=900,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=2,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    mask = y_true > 0

    mape = np.mean(
        np.abs(
            (
                y_true[mask]
                - y_pred[mask]
            )
            / y_true[mask]
        )
    ) * 100

    return mae, rmse, r2, mape


# ============================================================
# RESULT STORAGE
# ============================================================

all_results = []


# ============================================================
# CITY LOOP
# ============================================================

for city in cities:

    city_df = df[
        df["source_city"] == city
    ].copy()

    print("\n==========================================")
    print(
        f"FIXED-SPLIT VALIDATION: {city} "
        f"({len(city_df)} rows)"
    )
    print("==========================================")

    # --------------------------------------------------------
    # ONE FIXED SPLIT
    # --------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(
        splitter.split(
            city_df,
            groups=city_df["location"]
        )
    )

    train_raw = city_df.iloc[
        train_idx
    ].copy()

    test_raw = city_df.iloc[
        test_idx
    ].copy()

    train_locations = set(
        train_raw["location"]
    )

    test_locations = set(
        test_raw["location"]
    )

    overlap = (
        train_locations
        & test_locations
    )

    print(
        f"Train rows:      {len(train_raw)}"
    )

    print(
        f"Test rows:       {len(test_raw)}"
    )

    print(
        f"Train locations: {len(train_locations)}"
    )

    print(
        f"Test locations:  {len(test_locations)}"
    )

    print(
        f"Overlap:         {len(overlap)}"
    )

    # ========================================================
    # BASIC FEATURES
    # ========================================================

    train_basic = create_basic_features(
        train_raw
    )

    test_basic = create_basic_features(
        test_raw
    )

    # ========================================================
    # MARKET FEATURES
    # ========================================================

    train_market, test_market = (
        add_market_features(
            train_basic,
            test_basic,
            train_raw
        )
    )

    # ========================================================
    # COMMON FEATURES
    # ========================================================

    common_categorical = [
        "location"
    ]

    common_numerical = [
        "log_area",
        "log_bedrooms",
        "area_per_bedroom",
        "area_bedroom_interaction",
        "amenity_yes_count",
        "amenity_known_count",
        "amenity_unknown_count",
    ]

    # ========================================================
    # MODEL A — V3 RAW PRICE
    # ========================================================

    print("\n--- Model A: Raw-price XGBoost ---")

    pre_a = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                ),
                common_categorical,
            ),
            (
                "num",
                "passthrough",
                common_numerical,
            ),
        ]
    )

    Xa_train = pre_a.fit_transform(
        train_basic
    )

    Xa_test = pre_a.transform(
        test_basic
    )

    model_a = build_model()

    model_a.fit(
        Xa_train,
        train_raw["price"].values
    )

    pred_a = model_a.predict(
        Xa_test
    )

    mae_a, rmse_a, r2_a, mape_a = (
        calculate_metrics(
            test_raw["price"].values,
            pred_a
        )
    )

    print(
        f"MAE: ₹{mae_a:,.0f} | "
        f"RMSE: ₹{rmse_a:,.0f} | "
        f"R²: {r2_a:.4f} | "
        f"MAPE: {mape_a:.2f}%"
    )

    # ========================================================
    # MODEL B — V4 HIERARCHICAL RAW PRICE
    # ========================================================

    print(
        "\n--- Model B: Hierarchical raw-price XGBoost ---"
    )

    market_numerical = [
        "log_area",
        "log_bedrooms",
        "area_per_bedroom",
        "area_bedroom_interaction",
        "amenity_yes_count",
        "amenity_known_count",
        "amenity_unknown_count",
        "city_median_price",
        "city_median_ppsf",
        "city_sample_count",
        "location_median_price",
        "location_median_ppsf",
        "location_sample_count",
        "log_city_sample_count",
        "log_location_sample_count",
        "location_market_value",
        "log_location_market_value",
    ]

    pre_b = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                ),
                common_categorical,
            ),
            (
                "num",
                "passthrough",
                market_numerical,
            ),
        ]
    )

    Xb_train = pre_b.fit_transform(
        train_market
    )

    Xb_test = pre_b.transform(
        test_market
    )

    model_b = build_model()

    model_b.fit(
        Xb_train,
        train_raw["price"].values
    )

    pred_b = model_b.predict(
        Xb_test
    )

    mae_b, rmse_b, r2_b, mape_b = (
        calculate_metrics(
            test_raw["price"].values,
            pred_b
        )
    )

    print(
        f"MAE: ₹{mae_b:,.0f} | "
        f"RMSE: ₹{rmse_b:,.0f} | "
        f"R²: {r2_b:.4f} | "
        f"MAPE: {mape_b:.2f}%"
    )

    # ========================================================
    # MODEL C — V5 LOG PRICE
    # ========================================================

    print(
        "\n--- Model C: Log-price XGBoost ---"
    )

    pre_c = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                ),
                common_categorical,
            ),
            (
                "num",
                "passthrough",
                common_numerical,
            ),
        ]
    )

    Xc_train = pre_c.fit_transform(
        train_basic
    )

    Xc_test = pre_c.transform(
        test_basic
    )

    model_c = build_model()

    model_c.fit(
        Xc_train,
        np.log1p(
            train_raw["price"].values
        )
    )

    pred_c = np.expm1(
        model_c.predict(
            Xc_test
        )
    )

    pred_c = np.maximum(
        pred_c,
        0
    )

    mae_c, rmse_c, r2_c, mape_c = (
        calculate_metrics(
            test_raw["price"].values,
            pred_c
        )
    )

    print(
        f"MAE: ₹{mae_c:,.0f} | "
        f"RMSE: ₹{rmse_c:,.0f} | "
        f"R²: {r2_c:.4f} | "
        f"MAPE: {mape_c:.2f}%"
    )

    # ========================================================
    # STORE RESULTS
    # ========================================================

    rows = [
        {
            "city": city,
            "model": "V3_RawPrice",
            "MAE": mae_a,
            "RMSE": rmse_a,
            "R2": r2_a,
            "MAPE": mape_a,
            "train_rows": len(train_raw),
            "test_rows": len(test_raw),
            "train_locations": len(train_locations),
            "test_locations": len(test_locations),
            "location_overlap": len(overlap),
        },
        {
            "city": city,
            "model": "V4_HierarchicalRawPrice",
            "MAE": mae_b,
            "RMSE": rmse_b,
            "R2": r2_b,
            "MAPE": mape_b,
            "train_rows": len(train_raw),
            "test_rows": len(test_raw),
            "train_locations": len(train_locations),
            "test_locations": len(test_locations),
            "location_overlap": len(overlap),
        },
        {
            "city": city,
            "model": "V5_LogPrice",
            "MAE": mae_c,
            "RMSE": rmse_c,
            "R2": r2_c,
            "MAPE": mape_c,
            "train_rows": len(train_raw),
            "test_rows": len(test_raw),
            "train_locations": len(train_locations),
            "test_locations": len(test_locations),
            "location_overlap": len(overlap),
        },
    ]

    all_results.extend(rows)


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

print("\n\n======================================================")
print("MILLOW V5.1 — FIXED-SPLIT MODEL COMPARISON")
print("======================================================")

print(
    results_df[
        [
            "city",
            "model",
            "MAE",
            "RMSE",
            "R2",
            "MAPE",
            "location_overlap",
        ]
    ].to_string(index=False)
)


# ============================================================
# OVERALL MEAN
# ============================================================

overall = (
    results_df
    .groupby("model")
    .agg(
        MAE=("MAE", "mean"),
        RMSE=("RMSE", "mean"),
        R2=("R2", "mean"),
        MAPE=("MAPE", "mean"),
    )
    .reset_index()
)

print("\n\n==============================")
print("MEAN METRICS ACROSS CITIES")
print("==============================")

print(
    overall.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "v5_1_fixed_split_results.csv"
)

overall_path = os.path.join(
    OUTPUT_DIR,
    "v5_1_fixed_split_overall.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

overall.to_csv(
    overall_path,
    index=False
)

print(
    f"\nSaved: {results_path}"
)

print(
    f"Saved: {overall_path}"
)