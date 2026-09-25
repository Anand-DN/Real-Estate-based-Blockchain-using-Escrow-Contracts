"""
===============================================================
MILLOW - M.Tech Data Science Research Model Pipeline
===============================================================

Dataset:
    MREID_property.csv

Current schema:
    price
    area
    location
    no_of_bedrooms
    resale
    amenity/property features
    source_city
    source_file
    derived_price_per_sqft

Research objectives:
    1. Feature engineering
    2. EDA
    3. Model comparison
    4. Feature ablation
    5. City-wise evaluation
    6. Cross-city evaluation
    7. Save reproducible results

IMPORTANT:
    derived_price_per_sqft is NOT used as an input feature
    because it is directly calculated using the target price.

===============================================================
"""

from pathlib import Path
import re
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================
# 1. PATHS
# =============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "MREID_property.csv"
)

RESEARCH_DIR = PROJECT_ROOT / "research"

RESULTS_DIR = (
    RESEARCH_DIR
    / "experiments"
    / "results"
)

EDA_DIR = (
    RESEARCH_DIR
    / "reports"
    / "eda"
)

MODEL_DIR = (
    RESEARCH_DIR
    / "reports"
    / "model_comparison"
)

FEATURE_DIR = (
    RESEARCH_DIR
    / "reports"
    / "feature_analysis"
)


for directory in [
    RESULTS_DIR,
    EDA_DIR,
    MODEL_DIR,
    FEATURE_DIR
]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# =============================================================
# 2. LOAD DATA
# =============================================================

def load_data():

    print("\n" + "=" * 70)
    print("LOADING MREID")
    print("=" * 70)

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"\nMREID dataset not found:\n"
            f"{DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE,
        low_memory=False
    )

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    return df


# =============================================================
# 3. DATA AUDIT
# =============================================================

def audit_data(df):

    print("\n" + "=" * 70)
    print("DATA AUDIT")
    print("=" * 70)

    print("\nColumns:")

    for i, column in enumerate(
        df.columns,
        start=1
    ):

        print(
            f"{i:02d}. {column}"
        )

    print("\nData types:")

    print(
        df.dtypes.to_string()
    )

    print("\nMissing values:")

    missing = (
        df.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(
        missing[
            missing > 0
        ].to_string()
    )

    # Save audit
    audit = pd.DataFrame({

        "column":
            df.columns,

        "dtype":
            [
                str(df[col].dtype)
                for col in df.columns
            ],

        "missing":
            [
                int(df[col].isna().sum())
                for col in df.columns
            ],

        "missing_percentage":
            [
                round(
                    df[col].isna().mean() * 100,
                    2
                )
                for col in df.columns
            ],

        "unique":
            [
                int(df[col].nunique())
                for col in df.columns
            ],
    })

    audit.to_csv(
        FEATURE_DIR /
        "feature_audit.csv",
        index=False
    )

    return audit


# =============================================================
# 4. CLEAN TARGET AND CORE NUMERIC FEATURES
# =============================================================

def clean_core_features(df):

    print("\n" + "=" * 70)
    print("CORE FEATURE CLEANING")
    print("=" * 70)

    df = df.copy()

    # ---------------------------------------------------------
    # Target
    # ---------------------------------------------------------

    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce"
    )

    # ---------------------------------------------------------
    # Area
    # ---------------------------------------------------------

    df["area"] = pd.to_numeric(
        df["area"],
        errors="coerce"
    )

    # ---------------------------------------------------------
    # Bedrooms
    # ---------------------------------------------------------

    df["no_of_bedrooms"] = pd.to_numeric(
        df["no_of_bedrooms"],
        errors="coerce"
    )

    # ---------------------------------------------------------
    # Remove impossible target values
    # ---------------------------------------------------------

    before = len(df)

    df = df[
        df["price"].notna()
        & (df["price"] > 0)
    ]

    print(
        f"Invalid price rows removed: "
        f"{before - len(df):,}"
    )

    # ---------------------------------------------------------
    # Remove impossible areas
    # ---------------------------------------------------------

    before = len(df)

    df.loc[
        df["area"] <= 0,
        "area"
    ] = np.nan

    print(
        f"Invalid area values converted "
        f"to missing: {before - len(df):,}"
    )

    # ---------------------------------------------------------
    # Bedrooms
    # ---------------------------------------------------------

    df.loc[
        (df["no_of_bedrooms"] <= 0)
        |
        (df["no_of_bedrooms"] > 30),
        "no_of_bedrooms"
    ] = np.nan

    return df


# =============================================================
# 5. CLEAN CATEGORICAL FEATURES
# =============================================================

def clean_location(df):

    df = df.copy()

    if "location" in df.columns:

        df["location"] = (
            df["location"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        df["location"] = (
            df["location"]
            .replace(
                {
                    "nan": pd.NA,
                    "none": pd.NA,
                    "": pd.NA
                }
            )
        )

    if "source_city" in df.columns:

        df["source_city"] = (
            df["source_city"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

    return df


# =============================================================
# 6. CONVERT YES/NO / BOOLEAN FEATURES
# =============================================================

def convert_boolean_features(df):

    df = df.copy()

    boolean_columns = [

        "resale",
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
        "vaastucompliant",
        "microwave",
        "golfcourse",
        "tv",
        "diningtable",
        "sofa",
        "wardrobe",
        "refrigerator",
    ]

    yes_values = {
        "yes",
        "y",
        "true",
        "1",
        "available"
    }

    no_values = {
        "no",
        "n",
        "false",
        "0",
        "not available"
    }

    converted = []

    for column in boolean_columns:

        if column not in df.columns:
            continue

        original = df[column]

        # Already numeric
        if pd.api.types.is_numeric_dtype(
            original
        ):

            df[column] = pd.to_numeric(
                original,
                errors="coerce"
            )

        else:

            values = (
                original
                .astype("string")
                .str.strip()
                .str.lower()
            )

            df[column] = values.map(
                lambda x:
                    1
                    if x in yes_values
                    else (
                        0
                        if x in no_values
                        else np.nan
                    )
            )

        converted.append(column)

    print(
        f"Boolean/property features processed: "
        f"{len(converted)}"
    )

    return df


# =============================================================
# 7. AMENITY SCORE
# =============================================================

def create_amenity_features(df):

    df = df.copy()

    amenity_columns = [

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
        "vaastucompliant",
        "microwave",
        "golfcourse",
        "tv",
        "diningtable",
        "sofa",
        "wardrobe",
        "refrigerator",
    ]

    available = [
        col
        for col in amenity_columns
        if col in df.columns
    ]

    if available:

        df["amenity_count"] = (
            df[available]
            .sum(
                axis=1,
                skipna=True
            )
        )

        df["amenity_coverage"] = (
            df[available]
            .notna()
            .sum(axis=1)
        )

        print(
            f"Created amenity_count "
            f"from {len(available)} features."
        )

    return df


# =============================================================
# 8. TARGET LEAKAGE CHECK
# =============================================================

def remove_target_leakage(df):

    df = df.copy()

    leakage_columns = [

        "derived_price_per_sqft",

        "price_per_sqft",

        "price_sqft",

        "listed_price",

        "listed_price_inr",

        "property_price",
    ]

    removed = []

    for column in leakage_columns:

        if column in df.columns:

            removed.append(column)

            df = df.drop(
                columns=[column]
            )

    if removed:

        print(
            "\nTarget leakage columns removed:"
        )

        for column in removed:

            print(
                f"  - {column}"
            )

    return df


# =============================================================
# 9. FEATURE SUMMARY
# =============================================================

def create_feature_summary(df):

    numeric = (
        df.select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    categorical = (
        df.select_dtypes(
            include=[
                "object",
                "string",
                "category"
            ]
        )
        .columns
        .tolist()
    )

    summary = {

        "total_rows":
            len(df),

        "total_columns":
            len(df.columns),

        "numeric_features":
            numeric,

        "categorical_features":
            categorical,

        "target":
            "price",

        "target_leakage_excluded":
            [
                "derived_price_per_sqft",
                "price_per_sqft",
                "price_sqft"
            ],
    }

    output = (
        FEATURE_DIR /
        "feature_summary.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    print(
        f"\nFeature summary saved:\n"
        f"{output}"
    )


# =============================================================
# 10. CREATE FEATURE GROUPS
# =============================================================

def get_feature_groups(df):

    structural = [
        "area",
        "no_of_bedrooms",
    ]

    property_status = [
        "resale",
    ]

    amenities = [

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
        "vaastucompliant",
        "microwave",
        "golfcourse",
        "tv",
        "diningtable",
        "sofa",
        "wardrobe",
        "refrigerator",

        "amenity_count",
        "amenity_coverage",
    ]

    location = [
        "location",
        "source_city",
    ]

    groups = {

        "structural": [
            c
            for c in structural
            if c in df.columns
        ],

        "property_status": [
            c
            for c in property_status
            if c in df.columns
        ],

        "amenities": [
            c
            for c in amenities
            if c in df.columns
        ],

        "location": [
            c
            for c in location
            if c in df.columns
        ],
    }

    groups["structural_only"] = (
        groups["structural"]
    )

    groups["structural_property"] = (
        groups["structural"]
        +
        groups["property_status"]
    )

    groups["structural_property_amenity"] = (
        groups["structural"]
        +
        groups["property_status"]
        +
        groups["amenities"]
    )

    groups["full"] = (
        groups["structural"]
        +
        groups["property_status"]
        +
        groups["amenities"]
        +
        groups["location"]
    )

    return groups


# =============================================================
# 11. PREPARE MODEL DATA
# =============================================================

def prepare_model_data(
    df,
    features
):

    data = df[
        features + ["price"]
    ].copy()

    data = data[
        data["price"].notna()
        &
        (data["price"] > 0)
    ]

    X = data[
        features
    ].copy()

    y = data[
        "price"
    ].copy()

    return X, y


# =============================================================
# 12. MODEL EVALUATION
# =============================================================

def evaluate_model(
    model_name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):

    from sklearn.pipeline import Pipeline

    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # Avoid zero division
    denominator = np.where(
        y_test == 0,
        1,
        y_test
    )

    mape = (
        np.mean(
            np.abs(
                (
                    y_test -
                    predictions
                )
                /
                denominator
            )
        )
        * 100
    )

    return {
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE_percent": mape,
        "pipeline": pipeline
    }


# =============================================================
# 13. RUN ONE MODEL SET
# =============================================================

def run_models_for_features(
    df,
    feature_name,
    features
):

    from sklearn.model_selection import train_test_split

    from sklearn.compose import ColumnTransformer

    from sklearn.pipeline import Pipeline

    from sklearn.preprocessing import (
        OneHotEncoder
    )

    from sklearn.impute import SimpleImputer

    from sklearn.linear_model import (
        LinearRegression
    )

    from sklearn.tree import (
        DecisionTreeRegressor
    )

    from sklearn.ensemble import (
        RandomForestRegressor,
        GradientBoostingRegressor
    )

    print("\n")
    print(
        "-" * 70
    )

    print(
        f"FEATURE GROUP: {feature_name}"
    )

    print(
        "-" * 70
    )

    print(
        "Features:"
    )

    print(
        features
    )

    X, y = prepare_model_data(
        df,
        features
    )

    if len(X) < 100:

        print(
            "Not enough records."
        )

        return []

    categorical_features = (
        X.select_dtypes(
            include=[
                "object",
                "string",
                "category"
            ]
        )
        .columns
        .tolist()
    )

    numeric_features = (
        X.select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    numeric_pipeline = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    transformers = []

    if numeric_features:

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            )
        )

    if categorical_features:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )
    )

    models = {

        "Linear Regression":
            LinearRegression(),

        "Decision Tree":
            DecisionTreeRegressor(
                random_state=42,
                max_depth=20
            ),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                max_features="sqrt"
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(
                random_state=42,
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3
            ),
    }

    results = []

    for model_name, model in models.items():

        print(
            f"\nTraining "
            f"{model_name}..."
        )

        try:

            result = evaluate_model(
                model_name,
                model,
                X_train,
                X_test,
                y_train,
                y_test,
                preprocessor
            )

            result.pop(
                "pipeline"
            )

            result[
                "feature_group"
            ] = feature_name

            result[
                "n_features"
            ] = len(features)

            result[
                "training_rows"
            ] = len(X_train)

            result[
                "testing_rows"
            ] = len(X_test)

            results.append(
                result
            )

            print(
                f"MAE  : "
                f"{result['MAE']:,.2f}"
            )

            print(
                f"RMSE : "
                f"{result['RMSE']:,.2f}"
            )

            print(
                f"R2   : "
                f"{result['R2']:.4f}"
            )

            print(
                f"MAPE : "
                f"{result['MAPE_percent']:.2f}%"
            )

        except Exception as e:

            print(
                f"[ERROR] "
                f"{model_name}: {e}"
            )

    return results


# =============================================================
# 14. FEATURE ABLATION EXPERIMENT
# =============================================================

def run_feature_ablation(df):

    print("\n" + "=" * 70)
    print("FEATURE ABLATION EXPERIMENT")
    print("=" * 70)

    groups = get_feature_groups(
        df
    )

    all_results = []

    for name, features in groups.items():

        # Only run meaningful experimental groups
        if name not in [
            "structural_only",
            "structural_property",
            "structural_property_amenity",
            "full"
        ]:

            continue

        results = run_models_for_features(
            df,
            name,
            features
        )

        all_results.extend(
            results
        )

    if not all_results:

        print(
            "No experiment results."
        )

        return

    results_df = pd.DataFrame(
        all_results
    )

    output = (
        RESULTS_DIR /
        "feature_ablation_results.csv"
    )

    results_df.to_csv(
        output,
        index=False
    )

    print(
        "\n[OK] Feature ablation results:"
    )

    print(output)

    print(
        "\nSummary:"
    )

    print(
        results_df[
            [
                "feature_group",
                "model",
                "MAE",
                "RMSE",
                "R2",
                "MAPE_percent"
            ]
        ].to_string(
            index=False
        )
    )


# =============================================================
# 15. CITY-WISE EXPERIMENT
# =============================================================

def run_city_analysis(df):

    print("\n" + "=" * 70)
    print("CITY-WISE ANALYSIS")
    print("=" * 70)

    if "source_city" not in df.columns:

        print(
            "source_city not available."
        )

        return

    city_summary = (
        df.groupby(
            "source_city"
        )
        .agg(
            records=(
                "mreid_id",
                "count"
            ),
            median_price=(
                "price",
                "median"
            ),
            mean_price=(
                "price",
                "mean"
            ),
            median_area=(
                "area",
                "median"
            ),
            median_bedrooms=(
                "no_of_bedrooms",
                "median"
            )
        )
        .reset_index()
    )

    output = (
        RESULTS_DIR /
        "city_summary.csv"
    )

    city_summary.to_csv(
        output,
        index=False
    )

    print(
        city_summary.to_string(
            index=False
        )
    )

    print(
        f"\n[OK] Saved: {output}"
    )


# =============================================================
# 16. PRICE STATISTICS
# =============================================================

def create_price_statistics(df):

    print("\n" + "=" * 70)
    print("PRICE STATISTICS")
    print("=" * 70)

    stats = (
        df["price"]
        .describe()
    )

    print(
        stats
    )

    # Percentiles
    percentiles = (
        df["price"]
        .quantile(
            [
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99
            ]
        )
    )

    print(
        "\nPrice percentiles:"
    )

    print(
        percentiles
    )

    output = (
        RESULTS_DIR /
        "price_statistics.csv"
    )

    percentiles.to_csv(
        output
    )

    print(
        f"\n[OK] Saved: {output}"
    )


# =============================================================
# 17. EDA
# =============================================================

def run_eda(df):

    print("\n" + "=" * 70)
    print("EDA")
    print("=" * 70)

    try:

        import matplotlib.pyplot as plt

    except ImportError:

        print(
            "Install matplotlib:"
        )

        print(
            "pip install matplotlib"
        )

        return

    # ---------------------------------------------------------
    # Price distribution
    # ---------------------------------------------------------

    price = (
        df["price"]
        .dropna()
    )

    if len(price):

        low = price.quantile(
            0.01
        )

        high = price.quantile(
            0.99
        )

        values = price[
            price.between(
                low,
                high
            )
        ]

        plt.figure(
            figsize=(10, 6)
        )

        plt.hist(
            values,
            bins=60
        )

        plt.title(
            "Property Price Distribution"
        )

        plt.xlabel(
            "Price"
        )

        plt.ylabel(
            "Frequency"
        )

        plt.tight_layout()

        plt.savefig(
            EDA_DIR /
            "price_distribution.png",
            dpi=300
        )

        plt.close()

    # ---------------------------------------------------------
    # Area vs Price
    # ---------------------------------------------------------

    if (
        "area" in df.columns
        and "price" in df.columns
    ):

        sample = df[
            [
                "area",
                "price"
            ]
        ].dropna()

        if len(sample) > 5000:

            sample = sample.sample(
                5000,
                random_state=42
            )

        plt.figure(
            figsize=(10, 6)
        )

        plt.scatter(
            sample["area"],
            sample["price"],
            alpha=0.35
        )

        plt.title(
            "Property Area vs Price"
        )

        plt.xlabel(
            "Area"
        )

        plt.ylabel(
            "Price"
        )

        plt.tight_layout()

        plt.savefig(
            EDA_DIR /
            "area_vs_price.png",
            dpi=300
        )

        plt.close()

    # ---------------------------------------------------------
    # Bedrooms vs Price
    # ---------------------------------------------------------

    if (
        "no_of_bedrooms" in df.columns
        and "price" in df.columns
    ):

        grouped = (
            df.groupby(
                "no_of_bedrooms"
            )["price"]
            .median()
        )

        grouped = grouped[
            grouped.index <= 10
        ]

        plt.figure(
            figsize=(10, 6)
        )

        grouped.plot(
            kind="bar"
        )

        plt.title(
            "Median Property Price by Bedrooms"
        )

        plt.xlabel(
            "Number of Bedrooms"
        )

        plt.ylabel(
            "Median Price"
        )

        plt.tight_layout()

        plt.savefig(
            EDA_DIR /
            "bedrooms_vs_price.png",
            dpi=300
        )

        plt.close()

    # ---------------------------------------------------------
    # City median price
    # ---------------------------------------------------------

    if "source_city" in df.columns:

        city_price = (
            df.groupby(
                "source_city"
            )["price"]
            .median()
            .sort_values(
                ascending=False
            )
        )

        plt.figure(
            figsize=(10, 6)
        )

        city_price.plot(
            kind="bar"
        )

        plt.title(
            "Median Property Price by City"
        )

        plt.xlabel(
            "City"
        )

        plt.ylabel(
            "Median Price"
        )

        plt.tight_layout()

        plt.savefig(
            EDA_DIR /
            "city_median_price.png",
            dpi=300
        )

        plt.close()

    print(
        f"[OK] EDA saved to:\n"
        f"{EDA_DIR}"
    )


# =============================================================
# 18. MAIN
# =============================================================

def main():

    print("\n")
    print("=" * 70)
    print("MILLOW - RESEARCH MODEL PIPELINE")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    df = load_data()

    # ---------------------------------------------------------
    # Audit
    # ---------------------------------------------------------

    audit_data(df)

    # ---------------------------------------------------------
    # Clean
    # ---------------------------------------------------------

    df = clean_core_features(
        df
    )

    df = clean_location(
        df
    )

    df = convert_boolean_features(
        df
    )

    # ---------------------------------------------------------
    # Feature engineering
    # ---------------------------------------------------------

    df = create_amenity_features(
        df
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    # Remove target leakage
    # ---------------------------------------------------------

    df = remove_target_leakage(
        df
    )

    # ---------------------------------------------------------
    # Save research-ready dataset
    # ---------------------------------------------------------

    research_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "MREID_research_ready.csv"
    )

    df.to_csv(
        research_file,
        index=False
    )

    print(
        f"\n[OK] Research-ready dataset:"
    )

    print(
        research_file
    )

    # ---------------------------------------------------------
    # Feature summary
    # ---------------------------------------------------------

    create_feature_summary(
        df
    )

    # ---------------------------------------------------------
    # Price statistics
    # ---------------------------------------------------------

    create_price_statistics(
        df
    )

    # ---------------------------------------------------------
    # City analysis
    # ---------------------------------------------------------

    run_city_analysis(
        df
    )

    # ---------------------------------------------------------
    # EDA
    # ---------------------------------------------------------

    run_eda(
        df
    )

    # ---------------------------------------------------------
    # Feature ablation
    # ---------------------------------------------------------

    run_feature_ablation(
        df
    )

    # ---------------------------------------------------------
    # Finish
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MILLOW RESEARCH PIPELINE COMPLETE")
    print("=" * 70)

    print(
        "\nImportant outputs:"
    )

    print(
        f"Research dataset:\n"
        f"    {research_file}"
    )

    print(
        f"\nFeature audit:\n"
        f"    {FEATURE_DIR}"
    )

    print(
        f"\nEDA:\n"
        f"    {EDA_DIR}"
    )

    print(
        f"\nExperiments:\n"
        f"    {RESULTS_DIR}"
    )

    print(
        "\nNext:"
    )

    print(
        "1. Inspect feature_ablation_results.csv"
    )

    print(
        "2. Inspect city_summary.csv"
    )

    print(
        "3. Check EDA graphs"
    )

    print(
        "4. Perform cross-city validation"
    )

    print(
        "5. Perform Gurgaon external validation"
    )

    print(
        "6. Add SHAP/XAI"
    )

    print(
        "7. Add Horizon features"
    )


# =============================================================
# RUN
# =============================================================

if __name__ == "__main__":

    main()