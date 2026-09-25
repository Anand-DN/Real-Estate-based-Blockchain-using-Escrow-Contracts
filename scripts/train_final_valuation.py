import json
import os
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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/processed/MREID_property.csv"

MODEL_DIR = "models/valuation/final"
ARTIFACT_DIR = "artifacts/valuation/final"

RANDOM_STATE = 42
TEST_SIZE = 0.20

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading MREID dataset...")

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

print(
    f"Valid rows: {len(df)}"
)


# ============================================================
# AMENITY COLUMNS
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
# FEATURE ENGINEERING
# ============================================================

def create_features(data):

    x = data.copy()

    # --------------------------------------------------------
    # Numeric property features
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Amenity summaries
    #
    # 0 = No
    # 1 = Yes
    # 9 = Unknown / not specified
    # --------------------------------------------------------

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

    # Remove raw amenity columns.
    x = x.drop(
        columns=amenity_cols
    )

    return x


# ============================================================
# FIXED CITY/LOCATION SPLIT
# ============================================================

cities = sorted(
    df["source_city"].unique()
)

print("\nCities:")
print(cities)


# We use the same deterministic methodology as V5.1.
# The split is used only for final validation.
#
# After validation, the final production model is trained
# on ALL valid rows.


validation_results = []


for city in cities:

    city_df = df[
        df["source_city"] == city
    ].copy()

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

    # --------------------------------------------------------
    # Features
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
    # Log-price model
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

    mae = mean_absolute_error(
        actual_price,
        predicted_price
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_price,
            predicted_price
        )
    )

    r2 = r2_score(
        actual_price,
        predicted_price
    )

    mape = (
        np.mean(
            np.abs(
                (
                    actual_price
                    - predicted_price
                )
                / actual_price
            )
        )
        * 100
    )

    validation_results.append({
        "city": city,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_locations": train_df[
            "location"
        ].nunique(),
        "test_locations": test_df[
            "location"
        ].nunique(),
    })

    print(
        f"\n{city}"
    )

    print(
        f"  MAE  : ₹{mae:,.0f}"
    )

    print(
        f"  RMSE : ₹{rmse:,.0f}"
    )

    print(
        f"  R²   : {r2:.4f}"
    )

    print(
        f"  MAPE : {mape:.2f}%"
    )


# ============================================================
# VALIDATION SUMMARY
# ============================================================

validation_df = pd.DataFrame(
    validation_results
)

print("\n\n==============================================")
print("FINAL MODEL VALIDATION")
print("==============================================")

print(
    validation_df.to_string(
        index=False
    )
)

print("\nMean metrics:")

print(
    f"MAE  : ₹{validation_df['MAE'].mean():,.0f}"
)

print(
    f"RMSE : ₹{validation_df['RMSE'].mean():,.0f}"
)

print(
    f"R²   : {validation_df['R2'].mean():.4f}"
)

print(
    f"MAPE : {validation_df['MAPE'].mean():.2f}%"
)


# ============================================================
# TRAIN FINAL PRODUCTION MODEL ON ALL DATA
# ============================================================

print("\n\n==============================================")
print("TRAINING FINAL PRODUCTION MODEL")
print("==============================================")

X_full = create_features(
    df
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

final_preprocessor = ColumnTransformer(
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

X_full_encoded = (
    final_preprocessor.fit_transform(
        X_full
    )
)

final_model = XGBRegressor(
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

final_model.fit(
    X_full_encoded,
    np.log1p(
        df["price"].values
    )
)

print(
    f"Production training rows: {len(df)}"
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "millow_valuation_model.json"
)

final_model.save_model(
    model_path
)


# ============================================================
# SAVE PREPROCESSOR
# ============================================================

preprocessor_path = os.path.join(
    MODEL_DIR,
    "millow_valuation_preprocessor.joblib"
)

joblib.dump(
    final_preprocessor,
    preprocessor_path
)


# ============================================================
# TRAINING METADATA
# ============================================================

metadata = {

    "model_name":
        "MILLOW V5 Log-Price XGBoost",

    "target":
        "log1p(price)",

    "inverse_transform":
        "expm1(prediction)",

    "dataset":
        DATA_PATH,

    "training_rows":
        int(len(df)),

    "cities":
        cities,

    "location_count":
        int(df["location"].nunique()),

    "features": {
        "categorical": categorical_features,
        "numerical": numerical_features,
    },

    "excluded_target_derived_features": [
        "derived_price_per_sqft",
    ],

    "amenity_encoding": {
        "0": "No",
        "1": "Yes",
        "9": "Unknown / not specified",
    },

    "model_parameters": {
        "n_estimators": 900,
        "max_depth": 7,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 2,
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
    },

    "validation": {
        "split":
            "GroupShuffleSplit by location",
        "test_size":
            TEST_SIZE,
        "random_state":
            RANDOM_STATE,
        "location_overlap":
            0,
        "mean_MAE":
            float(validation_df["MAE"].mean()),
        "mean_RMSE":
            float(validation_df["RMSE"].mean()),
        "mean_R2":
            float(validation_df["R2"].mean()),
        "mean_MAPE":
            float(validation_df["MAPE"].mean()),
    },
}

metadata_path = os.path.join(
    MODEL_DIR,
    "metadata.json"
)

with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


# ============================================================
# SAVE VALIDATION RESULTS
# ============================================================

validation_path = os.path.join(
    ARTIFACT_DIR,
    "final_validation_metrics.csv"
)

validation_df.to_csv(
    validation_path,
    index=False
)


# ============================================================
# FINISH
# ============================================================

print("\n==============================================")
print("MILLOW FINAL VALUATION MODEL SAVED")
print("==============================================")

print(
    f"Model       : {model_path}"
)

print(
    f"Preprocessor: {preprocessor_path}"
)

print(
    f"Metadata    : {metadata_path}"
)

print(
    f"Validation  : {validation_path}"
)