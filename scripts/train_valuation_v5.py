import os
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/processed/MREID_property.csv"
MODEL_DIR = "models/valuation/v5"
METRICS_DIR = "artifacts/valuation/v5"

RANDOM_STATE = 42
TEST_SIZE = 0.20

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["area"] = pd.to_numeric(df["area"], errors="coerce")
df["no_of_bedrooms"] = pd.to_numeric(
    df["no_of_bedrooms"], errors="coerce"
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
# AMENITY FEATURES
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

# Keep only numeric amenity columns
amenity_cols = [
    c for c in amenity_cols
    if pd.api.types.is_numeric_dtype(df[c])
]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(data):

    x = data.copy()

    # --------------------------------------------------------
    # Basic numerical features
    # --------------------------------------------------------

    x["log_area"] = np.log1p(x["area"])
    x["log_bedrooms"] = np.log1p(x["no_of_bedrooms"])

    x["area_per_bedroom"] = (
        x["area"] / x["no_of_bedrooms"]
    )

    x["area_bedroom_interaction"] = (
        x["area"] * x["no_of_bedrooms"]
    )

    # --------------------------------------------------------
    # Amenity summary features
    # --------------------------------------------------------

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

    # Do not use raw amenity columns separately.
    # Keep their aggregate representation to reduce noise.

    x = x.drop(columns=amenity_cols)

    return x


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

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

    # Avoid division problems
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
# TRAIN CITY-SPECIFIC MODELS
# ============================================================

all_results = []


for city in cities:

    city_df = df[
        df["source_city"] == city
    ].copy()

    print("\n======================================")
    print(
        f"V5 training: {city} "
        f"({len(city_df)} rows)"
    )
    print("======================================")

    # --------------------------------------------------------
    # Group split by location
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
    # Features
    # --------------------------------------------------------

    X_train = create_features(
        train_df
    )

    X_test = create_features(
        test_df
    )

    # --------------------------------------------------------
    # Target transformation
    #
    # IMPORTANT:
    # Train model on log(price)
    # --------------------------------------------------------

    y_train = np.log1p(
        train_df["price"].values
    )

    y_test = test_df["price"].values

    # --------------------------------------------------------
    # Columns
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Preprocessor
    # --------------------------------------------------------

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
    # XGBoost
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

    print("\nTraining XGBoost on log(price)...")

    model.fit(
        X_train_encoded,
        y_train
    )

    # --------------------------------------------------------
    # Predict log(price)
    # --------------------------------------------------------

    predicted_log_price = model.predict(
        X_test_encoded
    )

    # --------------------------------------------------------
    # Convert back to actual price
    # --------------------------------------------------------

    predicted_price = np.expm1(
        predicted_log_price
    )

    # Safety
    predicted_price = np.maximum(
        predicted_price,
        0
    )

    # --------------------------------------------------------
    # Metrics in original ₹ scale
    # --------------------------------------------------------

    mae, rmse, r2, mape = calculate_metrics(
        y_test,
        predicted_price
    )

    print(
        f"MAE:             ₹{mae:,.0f}"
    )

    print(
        f"RMSE:            ₹{rmse:,.0f}"
    )

    print(
        f"R²:              {r2:.4f}"
    )

    print(
        f"MAPE:            {mape:.2f}%"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        f"{city.lower()}_log_model.json"
    )

    model.save_model(
        model_path
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    all_results.append({
        "city": city,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_locations": len(train_locations),
        "test_locations": len(test_locations),
        "location_overlap": len(overlap),
    })


# ============================================================
# OVERALL RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

print("\n\n==============================================")
print("MILLOW VALUATION V5 RESULTS")
print("==============================================")

print(
    results_df.to_string(
        index=False
    )
)

metrics_path = os.path.join(
    METRICS_DIR,
    "v5_city_grouped_metrics.csv"
)

results_df.to_csv(
    metrics_path,
    index=False
)

print(
    f"\nSaved: {metrics_path}"
)