import os
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


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/processed/MREID_property.csv"
OUTPUT_DIR = "artifacts/valuation/v5_2_tail"

RANDOM_STATE = 42
TEST_SIZE = 0.20

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

df["price"] = pd.to_numeric(
    df["price"],
    errors="coerce"
)

df["area"] = pd.to_numeric(
    df["area"],
    errors="coerce"
)

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

cities = sorted(
    df["source_city"].unique()
)

print("\nCities:")
print(cities)


# ============================================================
# AMENITIES
# ============================================================

exclude_cols = {
    "mreid_id",
    "price",
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
# FEATURES
# ============================================================

def create_features(data):

    x = data.copy()

    x["log_area"] = np.log1p(
        x["area"]
    )

    x["log_bedrooms"] = np.log1p(
        x["no_of_bedrooms"]
    )

    x["area_per_bedroom"] = (
        x["area"]
        / x["no_of_bedrooms"]
    )

    x["area_bedroom_interaction"] = (
        x["area"]
        * x["no_of_bedrooms"]
    )

    amenity_values = x[
        amenity_cols
    ].apply(
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
# METRICS
# ============================================================

def metrics(
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

    return (
        mae,
        rmse,
        r2,
        mape
    )


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# CITY LOOP
# ============================================================

for city in cities:

    city_df = df[
        df["source_city"] == city
    ].copy()

    print("\n==========================================")
    print(
        f"TAIL SENSITIVITY: {city} "
        f"({len(city_df)} rows)"
    )
    print("==========================================")

    # --------------------------------------------------------
    # FIXED LOCATION SPLIT
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

    train_df = city_df.iloc[
        train_idx
    ].copy()

    test_df = city_df.iloc[
        test_idx
    ].copy()

    train_locations = set(
        train_df["location"]
    )

    test_locations = set(
        test_df["location"]
    )

    overlap = (
        train_locations
        & test_locations
    )

    print(
        f"Train rows:      {len(train_df)}"
    )

    print(
        f"Test rows:       {len(test_df)}"
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

    # --------------------------------------------------------
    # TRAINING 99TH PERCENTILE
    #
    # IMPORTANT:
    # Threshold is calculated ONLY from training data.
    # --------------------------------------------------------

    train_p99 = (
        train_df["price"]
        .quantile(0.99)
    )

    print(
        f"Training price P99: "
        f"₹{train_p99:,.0f}"
    )

    # --------------------------------------------------------
    # TEST TAIL COUNTS
    # --------------------------------------------------------

    test_above_p99 = (
        test_df["price"]
        > train_p99
    ).sum()

    test_within_p99 = (
        test_df["price"]
        <= train_p99
    ).sum()

    print(
        f"Test <= training P99: "
        f"{test_within_p99}"
    )

    print(
        f"Test > training P99:  "
        f"{test_above_p99}"
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    X_train = create_features(
        train_df
    )

    X_test = create_features(
        test_df
    )

    categorical_features = [
        "location"
    ]

    numerical_features = [
        "log_area",
        "log_bedrooms",
        "area_per_bedroom",
        "area_bedroom_interaction",
        "amenity_yes_count",
        "amenity_known_count",
        "amenity_unknown_count",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                ),
                categorical_features,
            ),
            (
                "num",
                "passthrough",
                numerical_features,
            ),
        ]
    )

    X_train_encoded = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_test_encoded = (
        preprocessor.transform(
            X_test
        )
    )

    # --------------------------------------------------------
    # V5 LOG MODEL
    # --------------------------------------------------------

    model = XGBRegressor(
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

    model.fit(
        X_train_encoded,
        np.log1p(
            train_df["price"].values
        )
    )

    predicted_price = np.expm1(
        model.predict(
            X_test_encoded
        )
    )

    predicted_price = np.maximum(
        predicted_price,
        0
    )

    actual_price = (
        test_df["price"].values
    )

    # ========================================================
    # FULL TEST SET
    # ========================================================

    full_mae, full_rmse, full_r2, full_mape = (
        metrics(
            actual_price,
            predicted_price
        )
    )

    # ========================================================
    # <= TRAINING P99
    # ========================================================

    within_mask = (
        actual_price
        <= train_p99
    )

    tail_mask = (
        actual_price
        > train_p99
    )

    if within_mask.sum() > 1:

        p99_mae, p99_rmse, p99_r2, p99_mape = (
            metrics(
                actual_price[within_mask],
                predicted_price[within_mask]
            )
        )

    else:

        p99_mae = np.nan
        p99_rmse = np.nan
        p99_r2 = np.nan
        p99_mape = np.nan

    # ========================================================
    # > TRAINING P99
    # ========================================================

    if tail_mask.sum() > 1:

        tail_mae, tail_rmse, tail_r2, tail_mape = (
            metrics(
                actual_price[tail_mask],
                predicted_price[tail_mask]
            )
        )

    else:

        tail_mae = np.nan
        tail_rmse = np.nan
        tail_r2 = np.nan
        tail_mape = np.nan

    # ========================================================
    # PRINT
    # ========================================================

    print("\nFULL TEST SET")
    print(
        f"MAE:  ₹{full_mae:,.0f}"
    )
    print(
        f"RMSE: ₹{full_rmse:,.0f}"
    )
    print(
        f"R²:   {full_r2:.4f}"
    )
    print(
        f"MAPE: {full_mape:.2f}%"
    )

    print(
        "\nTEST <= TRAINING P99"
    )
    print(
        f"Rows: {within_mask.sum()}"
    )
    print(
        f"MAE:  ₹{p99_mae:,.0f}"
    )
    print(
        f"RMSE: ₹{p99_rmse:,.0f}"
    )
    print(
        f"R²:   {p99_r2:.4f}"
    )
    print(
        f"MAPE: {p99_mape:.2f}%"
    )

    print(
        "\nTEST > TRAINING P99"
    )
    print(
        f"Rows: {tail_mask.sum()}"
    )
    print(
        f"MAE:  ₹{tail_mae:,.0f}"
    )
    print(
        f"RMSE: ₹{tail_rmse:,.0f}"
    )
    print(
        f"R²:   {tail_r2:.4f}"
    )
    print(
        f"MAPE: {tail_mape:.2f}%"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results.append({
        "city": city,

        "train_rows": len(train_df),
        "test_rows": len(test_df),

        "train_locations": len(train_locations),
        "test_locations": len(test_locations),
        "location_overlap": len(overlap),

        "train_price_p99": train_p99,

        "test_within_p99": within_mask.sum(),
        "test_above_p99": tail_mask.sum(),

        "full_MAE": full_mae,
        "full_RMSE": full_rmse,
        "full_R2": full_r2,
        "full_MAPE": full_mape,

        "within_p99_MAE": p99_mae,
        "within_p99_RMSE": p99_rmse,
        "within_p99_R2": p99_r2,
        "within_p99_MAPE": p99_mape,

        "above_p99_MAE": tail_mae,
        "above_p99_RMSE": tail_rmse,
        "above_p99_R2": tail_r2,
        "above_p99_MAPE": tail_mape,
    })


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(
    results
)

print("\n\n======================================================")
print("MILLOW V5.2 — 99TH-PERCENTILE SENSITIVITY")
print("======================================================")

print(
    results_df.to_string(
        index=False
    )
)

# ------------------------------------------------------------
# Mean metrics
# ------------------------------------------------------------

summary = pd.DataFrame({

    "metric": [
        "MAE",
        "RMSE",
        "R2",
        "MAPE",
    ],

    "FULL_TEST_MEAN": [
        results_df["full_MAE"].mean(),
        results_df["full_RMSE"].mean(),
        results_df["full_R2"].mean(),
        results_df["full_MAPE"].mean(),
    ],

    "WITHIN_P99_MEAN": [
        results_df["within_p99_MAE"].mean(),
        results_df["within_p99_RMSE"].mean(),
        results_df["within_p99_R2"].mean(),
        results_df["within_p99_MAPE"].mean(),
    ],

    "ABOVE_P99_MEAN": [
        results_df["above_p99_MAE"].mean(),
        results_df["above_p99_RMSE"].mean(),
        results_df["above_p99_R2"].mean(),
        results_df["above_p99_MAPE"].mean(),
    ],
})

print("\n\n==============================")
print("MEAN SENSITIVITY RESULTS")
print("==============================")

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "v5_2_tail_results.csv"
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "v5_2_tail_summary.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

summary.to_csv(
    summary_path,
    index=False
)

print(
    f"\nSaved: {results_path}"
)

print(
    f"Saved: {summary_path}"
)