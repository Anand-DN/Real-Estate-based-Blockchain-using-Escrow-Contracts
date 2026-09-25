"""
===============================================================
MILLOW - M.Tech Data Science Research Data Pipeline
===============================================================

Purpose:
    Complete one-file pipeline for MILLOW real-estate research.

Pipeline:
    Raw Data
       ↓
    Dataset Audit
       ↓
    Extraction
       ↓
    Standardization
       ↓
    Cleaning
       ↓
    MREID Property Dataset
       ↓
    EDA
       ↓
    ML Price Prediction
       ↓
    NHB Market Data
       ↓
    Horizon/RERA Inventory
       ↓
    Research Results

Author:
    MILLOW Project

IMPORTANT:
    Raw datasets are NEVER modified.

Run from project root:
    python scripts/millow_data_pipeline.py

===============================================================
"""

from pathlib import Path
import zipfile
import shutil
import warnings
import json
import re

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================
# 1. PROJECT PATHS
# =============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROPERTY_DIR = RAW_DIR / "property"
VALIDATION_DIR = RAW_DIR / "validation"
MARKET_DIR = RAW_DIR / "market"
HORIZON_DIR = RAW_DIR / "horizon"

HORIZON_EXTRACTED_DIR = HORIZON_DIR / "extracted"

INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

METADATA_DIR = PROJECT_ROOT / "metadata"

RESEARCH_DIR = PROJECT_ROOT / "research"
REPORT_DIR = RESEARCH_DIR / "reports"

EDA_DIR = REPORT_DIR / "eda"
MODEL_DIR = REPORT_DIR / "model_comparison"
XAI_DIR = REPORT_DIR / "xai"

RESULTS_DIR = RESEARCH_DIR / "experiments" / "results"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_ARTIFACT_DIR = ARTIFACTS_DIR / "models"


# =============================================================
# 2. CREATE FOLDERS
# =============================================================

def create_project_folders():

    folders = [
        PROPERTY_DIR,
        VALIDATION_DIR,
        MARKET_DIR,
        HORIZON_DIR,
        HORIZON_EXTRACTED_DIR,

        INTERIM_DIR,
        PROCESSED_DIR,

        METADATA_DIR,

        EDA_DIR,
        MODEL_DIR,
        XAI_DIR,
        RESULTS_DIR,

        MODEL_ARTIFACT_DIR,
    ]

    for folder in folders:
        folder.mkdir(
            parents=True,
            exist_ok=True
        )

    print("[OK] Project data folders created.")


# =============================================================
# 3. EXPECTED FILES
# =============================================================

CITY_FILES = {
    "Bangalore": "Bangalore.csv",
    "Chennai": "Chennai.csv",
    "Delhi": "Delhi.csv",
    "Hyderabad": "Hyderabad.csv",
    "Kolkata": "Kolkata.csv",
    "Mumbai": "Mumbai.csv",
}

NHB_FILE = "MILLOW_NHB_RESIDEX_Master.csv"

GURGAON_FILE = "gurgaon_10k.csv"


# =============================================================
# 4. FILE SEARCH
# =============================================================

def find_file(filename, search_dirs):

    for directory in search_dirs:

        path = directory / filename

        if path.exists():
            return path

    return None


# =============================================================
# 5. COPY DATASETS INTO STANDARD LOCATIONS
# =============================================================

def organize_existing_files():

    """
    Searches the project directories for datasets.

    If files are already in data/raw, nothing happens.

    If you downloaded files elsewhere, you can manually place
    them into the expected folders.
    """

    print("\n" + "=" * 70)
    print("DATASET LOCATION CHECK")
    print("=" * 70)

    for city, filename in CITY_FILES.items():

        destination = PROPERTY_DIR / filename

        if destination.exists():

            print(
                f"[FOUND] {city}: {destination}"
            )

        else:

            print(
                f"[MISSING] {city}: "
                f"Place {filename} inside:\n"
                f"    {PROPERTY_DIR}"
            )

    nhb = MARKET_DIR / NHB_FILE

    if nhb.exists():

        print(
            f"[FOUND] NHB: {nhb}"
        )

    else:

        print(
            f"[MISSING] NHB:\n"
            f"    {nhb}"
        )


# =============================================================
# 6. EXTRACT HORIZON ZIP FILES
# =============================================================

def extract_horizon_archives():

    print("\n" + "=" * 70)
    print("HORIZON ZIP EXTRACTION")
    print("=" * 70)

    zip_files = list(
        HORIZON_DIR.glob("*.zip")
    )

    if not zip_files:

        print(
            "[INFO] No Horizon ZIP files found."
        )

        print(
            "Place archive.zip and archive (1).zip "
            "inside:"
        )

        print(HORIZON_DIR)

        return

    for zip_path in zip_files:

        name = zip_path.stem

        output_dir = (
            HORIZON_EXTRACTED_DIR /
            name.replace(" ", "_")
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        try:

            with zipfile.ZipFile(
                zip_path,
                "r"
            ) as zip_ref:

                zip_ref.extractall(
                    output_dir
                )

            print(
                f"[OK] Extracted: "
                f"{zip_path.name}"
            )

        except zipfile.BadZipFile:

            print(
                f"[ERROR] Invalid ZIP: "
                f"{zip_path.name}"
            )

        except Exception as e:

            print(
                f"[ERROR] {zip_path.name}: {e}"
            )


# =============================================================
# 7. READ CSV SAFELY
# =============================================================

def read_csv_safe(path):

    encodings = [
        "utf-8",
        "utf-8-sig",
        "latin1",
        "cp1252"
    ]

    for encoding in encodings:

        try:

            return pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False
            )

        except UnicodeDecodeError:
            continue

        except Exception as e:

            print(
                f"[WARNING] Could not read "
                f"{path.name}: {e}"
            )

            break

    return None


# =============================================================
# 8. STANDARDIZE COLUMN NAMES
# =============================================================

def standardize_column_names(df):

    df = df.copy()

    new_columns = []

    for column in df.columns:

        column = str(column)

        column = column.strip().lower()

        column = re.sub(
            r"[^a-z0-9]+",
            "_",
            column
        )

        column = column.strip("_")

        new_columns.append(
            column
        )

    df.columns = new_columns

    return df


# =============================================================
# 9. LOAD SIX CITY DATASETS
# =============================================================

def load_property_datasets():

    print("\n" + "=" * 70)
    print("LOADING PROPERTY DATASETS")
    print("=" * 70)

    frames = []

    audit = []

    for city, filename in CITY_FILES.items():

        path = PROPERTY_DIR / filename

        if not path.exists():

            print(
                f"[MISSING] {filename}"
            )

            continue

        df = read_csv_safe(path)

        if df is None:
            continue

        original_shape = df.shape

        df = standardize_column_names(
            df
        )

        # Preserve source
        df["source_city"] = city

        df["source_file"] = filename

        frames.append(df)

        audit.append({
            "city": city,
            "file": filename,
            "rows": original_shape[0],
            "columns": original_shape[1],
        })

        print(
            f"[OK] {city}: "
            f"{original_shape[0]:,} rows × "
            f"{original_shape[1]} columns"
        )

    if not frames:

        raise FileNotFoundError(
            "\nNo city CSV files found.\n"
            "Put Bangalore.csv, Chennai.csv, Delhi.csv, "
            "Hyderabad.csv, Kolkata.csv and Mumbai.csv "
            f"inside:\n{PROPERTY_DIR}"
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False
    )

    audit_df = pd.DataFrame(audit)

    audit_df.to_csv(
        METADATA_DIR /
        "property_dataset_audit.csv",
        index=False
    )

    print(
        "\nCombined dataset:"
    )

    print(
        f"Rows: {len(combined):,}"
    )

    print(
        f"Columns: {len(combined.columns)}"
    )

    return combined


# =============================================================
# 10. DATA QUALITY REPORT
# =============================================================

def create_data_quality_report(
    df,
    filename="data_quality_report.csv"
):

    report = []

    for column in df.columns:

        series = df[column]

        report.append({

            "column": column,

            "dtype":
                str(series.dtype),

            "rows":
                len(series),

            "missing_count":
                int(series.isna().sum()),

            "missing_percentage":
                round(
                    series.isna().mean() * 100,
                    2
                ),

            "unique_values":
                int(series.nunique(
                    dropna=True
                )),

            "duplicate_values":
                int(
                    series.duplicated().sum()
                ),
        })

    report_df = pd.DataFrame(
        report
    )

    report_df.to_csv(
        METADATA_DIR / filename,
        index=False
    )

    print(
        f"[OK] Data quality report saved: "
        f"{filename}"
    )

    return report_df


# =============================================================
# 11. BASIC CLEANING
# =============================================================

def basic_clean_property_data(df):

    print("\n" + "=" * 70)
    print("PROPERTY DATA CLEANING")
    print("=" * 70)

    df = df.copy()

    original_rows = len(df)

    # ---------------------------------------------------------
    # Remove completely empty rows
    # ---------------------------------------------------------

    df = df.dropna(
        how="all"
    )

    # ---------------------------------------------------------
    # Remove completely empty columns
    # ---------------------------------------------------------

    df = df.dropna(
        axis=1,
        how="all"
    )

    # ---------------------------------------------------------
    # Convert empty strings to NaN
    # ---------------------------------------------------------

    df = df.replace(
        r"^\s*$",
        np.nan,
        regex=True
    )

    # ---------------------------------------------------------
    # Remove exact duplicates
    # ---------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates()

    duplicates_removed = (
        before - len(df)
    )

    print(
        f"Exact duplicates removed: "
        f"{duplicates_removed:,}"
    )

    # ---------------------------------------------------------
    # Strip whitespace from string columns
    # ---------------------------------------------------------

    object_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in object_columns:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    # ---------------------------------------------------------
    # Add unique MREID
    # ---------------------------------------------------------

    if "mreid_id" not in df.columns:

        df.insert(
            0,
            "mreid_id",
            [
                f"MREID_{i:07d}"
                for i in range(
                    1,
                    len(df) + 1
                )
            ]
        )

    # ---------------------------------------------------------
    # Convert obvious numeric columns
    # ---------------------------------------------------------

    possible_numeric = [

        "price",
        "listed_price",
        "price_inr",
        "price_sqft",
        "price_per_sqft",

        "bhk",
        "bedroom_num",
        "bathroom_num",
        "bathrooms",

        "total_sqft",
        "area",
        "built_up_area",
        "carpet_area",
        "superbuiltup_sqft",
        "super_built_up_sqft",

        "floor",
        "floor_num",
        "total_floor",
        "total_floors",

        "age",
        "property_age",

        "latitude",
        "longitude",
    ]

    for column in possible_numeric:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # ---------------------------------------------------------
    # Remove obviously impossible numeric values
    # ---------------------------------------------------------

    invalid_rules = {

        "bhk": lambda x:
            (x <= 0) | (x > 30),

        "bedroom_num": lambda x:
            (x <= 0) | (x > 30),

        "bathroom_num": lambda x:
            (x <= 0) | (x > 30),

        "bathrooms": lambda x:
            (x <= 0) | (x > 30),

        "total_sqft": lambda x:
            x <= 0,

        "area": lambda x:
            x <= 0,

        "built_up_area": lambda x:
            x <= 0,

        "carpet_area": lambda x:
            x <= 0,

        "price": lambda x:
            x <= 0,

        "listed_price": lambda x:
            x <= 0,

        "price_inr": lambda x:
            x <= 0,

        "price_sqft": lambda x:
            x <= 0,

        "price_per_sqft": lambda x:
            x <= 0,
    }

    for column, rule in invalid_rules.items():

        if column in df.columns:

            mask = rule(
                df[column]
            )

            invalid_count = int(
                mask.fillna(False).sum()
            )

            if invalid_count:

                print(
                    f"{column}: "
                    f"{invalid_count:,} "
                    f"invalid values → NaN"
                )

                df.loc[
                    mask,
                    column
                ] = np.nan

    # ---------------------------------------------------------
    # Latitude / longitude validation
    # ---------------------------------------------------------

    if "latitude" in df.columns:

        df.loc[
            ~df["latitude"].between(
                -90,
                90
            ),
            "latitude"
        ] = np.nan

    if "longitude" in df.columns:

        df.loc[
            ~df["longitude"].between(
                -180,
                180
            ),
            "longitude"
        ] = np.nan

    print(
        f"\nRows before cleaning: "
        f"{original_rows:,}"
    )

    print(
        f"Rows after cleaning: "
        f"{len(df):,}"
    )

    return df


# =============================================================
# 12. DERIVE PRICE PER SQFT
# =============================================================

def create_price_features(df):

    df = df.copy()

    price_columns = [
        "price",
        "listed_price",
        "price_inr"
    ]

    area_columns = [
        "total_sqft",
        "area",
        "built_up_area",
        "carpet_area",
        "superbuiltup_sqft",
        "super_built_up_sqft"
    ]

    price_col = None
    area_col = None

    for col in price_columns:

        if col in df.columns:

            price_col = col
            break

    for col in area_columns:

        if col in df.columns:

            area_col = col
            break

    if price_col and area_col:

        if "derived_price_per_sqft" not in df.columns:

            df["derived_price_per_sqft"] = (
                df[price_col] /
                df[area_col]
            )

        print(
            "Created: derived_price_per_sqft"
        )

    return df


# =============================================================
# 13. SAVE MREID PROPERTY DATA
# =============================================================

def save_mreid_property(df):

    interim_file = (
        INTERIM_DIR /
        "property_combined_cleaned.csv"
    )

    processed_file = (
        PROCESSED_DIR /
        "MREID_property.csv"
    )

    df.to_csv(
        interim_file,
        index=False
    )

    df.to_csv(
        processed_file,
        index=False
    )

    print(
        "\n[OK] Saved:"
    )

    print(interim_file)
    print(processed_file)


# =============================================================
# 14. NHB PROCESSING
# =============================================================

def process_nhb():

    print("\n" + "=" * 70)
    print("NHB RESIDEX")
    print("=" * 70)

    path = MARKET_DIR / NHB_FILE

    if not path.exists():

        print(
            "[INFO] NHB CSV not found."
        )

        return None

    nhb = read_csv_safe(path)

    if nhb is None:
        return None

    nhb = standardize_column_names(
        nhb
    )

    output = (
        PROCESSED_DIR /
        "MREID_market.csv"
    )

    nhb.to_csv(
        output,
        index=False
    )

    print(
        f"NHB rows: {len(nhb):,}"
    )

    print(
        f"NHB columns: "
        f"{len(nhb.columns)}"
    )

    print(
        f"[OK] Saved: {output}"
    )

    return nhb


# =============================================================
# 15. GURGAON VALIDATION DATA
# =============================================================

def process_gurgaon():

    print("\n" + "=" * 70)
    print("99ACRES / GURGAON VALIDATION DATA")
    print("=" * 70)

    path = (
        VALIDATION_DIR /
        GURGAON_FILE
    )

    if not path.exists():

        print(
            "[INFO] Gurgaon CSV not found."
        )

        return None

    df = read_csv_safe(path)

    if df is None:
        return None

    df = standardize_column_names(
        df
    )

    output = (
        PROCESSED_DIR /
        "validation_gurgaon.csv"
    )

    df.to_csv(
        output,
        index=False
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        f"[OK] Saved: {output}"
    )

    return df


# =============================================================
# 16. HORIZON DATA INVENTORY
# =============================================================

def inventory_horizon():

    print("\n" + "=" * 70)
    print("HORIZON / RERA DATA INVENTORY")
    print("=" * 70)

    if not HORIZON_EXTRACTED_DIR.exists():

        print(
            "[INFO] Horizon extraction directory "
            "does not exist."
        )

        return

    files = []

    for path in HORIZON_EXTRACTED_DIR.rglob("*"):

        if path.is_file():

            try:

                size_mb = (
                    path.stat().st_size /
                    (1024 * 1024)
                )

            except Exception:

                size_mb = 0

            files.append({

                "relative_path":
                    str(
                        path.relative_to(
                            HORIZON_EXTRACTED_DIR
                        )
                    ),

                "extension":
                    path.suffix.lower(),

                "size_mb":
                    round(
                        size_mb,
                        3
                    ),
            })

    if not files:

        print(
            "[INFO] No files found "
            "inside Horizon extraction."
        )

        return

    inventory = pd.DataFrame(
        files
    )

    output = (
        METADATA_DIR /
        "horizon_inventory.csv"
    )

    inventory.to_csv(
        output,
        index=False
    )

    print(
        f"Found {len(inventory)} files."
    )

    print(
        f"[OK] Inventory saved: {output}"
    )

    print("\nFiles:")

    for item in files:

        print(
            f" - {item['relative_path']}"
        )


# =============================================================
# 17. EDA
# =============================================================

def run_eda(df):

    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # ---------------------------------------------------------
    # City distribution
    # ---------------------------------------------------------

    if "source_city" in df.columns:

        counts = (
            df["source_city"]
            .value_counts()
        )

        plt = None

        try:

            import matplotlib.pyplot as plt

            plt.figure(
                figsize=(10, 6)
            )

            counts.plot(
                kind="bar"
            )

            plt.title(
                "MILLOW Property Records by City"
            )

            plt.xlabel(
                "City"
            )

            plt.ylabel(
                "Number of Records"
            )

            plt.tight_layout()

            plt.savefig(
                EDA_DIR /
                "01_records_by_city.png",
                dpi=300
            )

            plt.close()

        except Exception as e:

            print(
                f"[WARNING] City chart failed: {e}"
            )

    # ---------------------------------------------------------
    # Numeric distributions
    # ---------------------------------------------------------

    try:

        import matplotlib.pyplot as plt

        numeric_columns = (
            df.select_dtypes(
                include="number"
            ).columns
        )

        for column in numeric_columns:

            values = (
                df[column]
                .dropna()
            )

            if len(values) == 0:
                continue

            # Avoid massive graphs for IDs
            if (
                "id" in column
                and values.nunique()
                > 0.8 * len(values)
            ):
                continue

            # Clip only for visualization.
            # Original data is NOT changed.
            lower = values.quantile(
                0.01
            )

            upper = values.quantile(
                0.99
            )

            plot_values = values[
                values.between(
                    lower,
                    upper
                )
            ]

            if len(plot_values) < 10:
                continue

            plt.figure(
                figsize=(10, 6)
            )

            plt.hist(
                plot_values,
                bins=50
            )

            plt.title(
                f"Distribution of {column}"
            )

            plt.xlabel(
                column
            )

            plt.ylabel(
                "Frequency"
            )

            plt.tight_layout()

            filename = (
                "distribution_" +
                re.sub(
                    r"[^a-zA-Z0-9_]",
                    "_",
                    column
                ) +
                ".png"
            )

            plt.savefig(
                EDA_DIR / filename,
                dpi=300
            )

            plt.close()

        print(
            f"[OK] EDA charts saved to:\n"
            f"{EDA_DIR}"
        )

    except Exception as e:

        print(
            f"[WARNING] EDA failed: {e}"
        )


# =============================================================
# 18. DETECT TARGET PRICE COLUMN
# =============================================================

def detect_price_column(df):

    candidates = [

        "listed_price_inr",
        "listed_price",
        "price_inr",
        "price",
        "property_price",
        "sale_price"
    ]

    for column in candidates:

        if column in df.columns:

            numeric = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            if numeric.notna().sum() > 100:

                return column

    # Fallback:
    # Search for columns containing "price"

    for column in df.columns:

        if "price" in column:

            numeric = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            if numeric.notna().sum() > 100:

                return column

    return None


# =============================================================
# 19. PREPARE ML DATA
# =============================================================

def prepare_ml_data(df):

    target = detect_price_column(
        df
    )

    if target is None:

        print(
            "[INFO] Could not automatically "
            "identify property price column."
        )

        return None, None

    print(
        f"[ML] Target column: {target}"
    )

    working = df.copy()

    working[target] = pd.to_numeric(
        working[target],
        errors="coerce"
    )

    working = working[
        working[target] > 0
    ]

    # Remove extreme target outliers only
    # for the baseline experiment.
    lower = working[target].quantile(
        0.01
    )

    upper = working[target].quantile(
        0.99
    )

    working = working[
        working[target].between(
            lower,
            upper
        )
    ]

    # Candidate features
    preferred_features = [

        "bhk",
        "bedroom_num",
        "bathroom_num",
        "bathrooms",

        "total_sqft",
        "area",
        "built_up_area",
        "carpet_area",
        "superbuiltup_sqft",
        "super_built_up_sqft",

        "floor",
        "floor_num",
        "total_floor",
        "total_floors",

        "age",
        "property_age",

        "latitude",
        "longitude",
    ]

    features = []

    for column in preferred_features:

        if column in working.columns:

            if pd.api.types.is_numeric_dtype(
                working[column]
            ):

                if working[column].notna().sum() > 100:

                    features.append(
                        column
                    )

    # Include source city
    if "source_city" in working.columns:

        features.append(
            "source_city"
        )

    if not features:

        print(
            "[INFO] No suitable ML "
            "features detected."
        )

        return None, None

    X = working[features].copy()

    y = working[target].copy()

    print(
        f"Features selected: {features}"
    )

    print(
        f"ML dataset: "
        f"{len(X):,} records"
    )

    return X, y


# =============================================================
# 20. TRAIN BASELINE MODELS
# =============================================================

def run_ml_experiments(df):

    print("\n" + "=" * 70)
    print("MACHINE LEARNING EXPERIMENTS")
    print("=" * 70)

    try:

        from sklearn.model_selection import train_test_split

        from sklearn.compose import ColumnTransformer

        from sklearn.pipeline import Pipeline

        from sklearn.preprocessing import (
            OneHotEncoder,
            StandardScaler
        )

        from sklearn.impute import SimpleImputer

        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score
        )

        from sklearn.linear_model import LinearRegression

        from sklearn.tree import DecisionTreeRegressor

        from sklearn.ensemble import (
            RandomForestRegressor,
            GradientBoostingRegressor
        )

    except ImportError:

        print(
            "\n[ERROR] scikit-learn is not installed."
        )

        print(
            "Run:"
        )

        print(
            "pip install pandas numpy matplotlib "
            "scikit-learn openpyxl"
        )

        return

    X, y = prepare_ml_data(
        df
    )

    if X is None:
        return

    categorical_features = (
        X.select_dtypes(
            include=["object", "string"]
        ).columns.tolist()
    )

    numeric_features = (
        X.select_dtypes(
            include=["number"]
        ).columns.tolist()
    )

    numeric_pipeline = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            ),
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
            ),
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
                n_jobs=-1
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(
                random_state=42,
                n_estimators=200
            ),
    }

    results = []

    for model_name, model in models.items():

        print(
            f"\nTraining: {model_name}"
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
                ),
            ]
        )

        try:

            pipeline.fit(
                X_train,
                y_train
            )

            predictions = (
                pipeline.predict(
                    X_test
                )
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

            # MAPE
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
                        ) /
                        denominator
                    )
                ) * 100
            )

            results.append({

                "model":
                    model_name,

                "MAE":
                    mae,

                "RMSE":
                    rmse,

                "R2":
                    r2,

                "MAPE_percent":
                    mape,

                "training_rows":
                    len(X_train),

                "testing_rows":
                    len(X_test),
            })

            print(
                f"MAE  : {mae:,.2f}"
            )

            print(
                f"RMSE : {rmse:,.2f}"
            )

            print(
                f"R2   : {r2:.4f}"
            )

            print(
                f"MAPE : {mape:.2f}%"
            )

            # Save model
            try:

                import joblib

                model_file = (
                    MODEL_ARTIFACT_DIR /
                    (
                        model_name
                        .lower()
                        .replace(
                            " ",
                            "_"
                        ) +
                        ".joblib"
                    )
                )

                joblib.dump(
                    pipeline,
                    model_file
                )

            except Exception as e:

                print(
                    f"[WARNING] Model save failed: {e}"
                )

        except Exception as e:

            print(
                f"[ERROR] "
                f"{model_name}: {e}"
            )

    if not results:

        return

    results_df = pd.DataFrame(
        results
    )

    results_file = (
        RESULTS_DIR /
        "model_comparison.csv"
    )

    results_df.to_csv(
        results_file,
        index=False
    )

    print(
        "\n[OK] Model comparison saved:"
    )

    print(results_file)

    print(
        "\nResults:"
    )

    print(
        results_df.to_string(
            index=False
        )
    )


# =============================================================
# 21. SAVE DATASET SUMMARY
# =============================================================

def save_dataset_summary(df):

    summary = {

        "dataset_name":
            "MREID Property Dataset",

        "rows":
            int(len(df)),

        "columns":
            int(len(df.columns)),

        "cities":
            (
                df["source_city"]
                .dropna()
                .unique()
                .tolist()
                if "source_city"
                in df.columns
                else []
            ),

        "missing_cells":
            int(
                df.isna()
                .sum()
                .sum()
            ),

        "duplicate_rows":
            int(
                df.duplicated()
                .sum()
            ),
    }

    output = (
        METADATA_DIR /
        "MREID_dataset_summary.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print(
        f"[OK] Summary saved: {output}"
    )


# =============================================================
# 22. MAIN PIPELINE
# =============================================================

def main():

    print("\n")
    print("=" * 70)
    print("MILLOW - REAL ESTATE DATA SCIENCE PIPELINE")
    print("=" * 70)

    print(
        f"\nProject root:\n"
        f"{PROJECT_ROOT}"
    )

    # ---------------------------------------------------------
    # Step 1
    # ---------------------------------------------------------

    create_project_folders()

    # ---------------------------------------------------------
    # Step 2
    # ---------------------------------------------------------

    organize_existing_files()

    # ---------------------------------------------------------
    # Step 3
    # ---------------------------------------------------------

    extract_horizon_archives()

    # ---------------------------------------------------------
    # Step 4
    # ---------------------------------------------------------

    inventory_horizon()

    # ---------------------------------------------------------
    # Step 5
    # ---------------------------------------------------------

    property_df = load_property_datasets()

    # ---------------------------------------------------------
    # Step 6
    # ---------------------------------------------------------

    print("\nCreating initial data-quality report...")

    create_data_quality_report(
        property_df,
        "raw_property_quality_report.csv"
    )

    # ---------------------------------------------------------
    # Step 7
    # ---------------------------------------------------------

    property_df = (
        basic_clean_property_data(
            property_df
        )
    )

    # ---------------------------------------------------------
    # Step 8
    # ---------------------------------------------------------

    property_df = (
        create_price_features(
            property_df
        )
    )

    # ---------------------------------------------------------
    # Step 9
    # ---------------------------------------------------------

    create_data_quality_report(
        property_df,
        "clean_property_quality_report.csv"
    )

    # ---------------------------------------------------------
    # Step 10
    # ---------------------------------------------------------

    save_mreid_property(
        property_df
    )

    # ---------------------------------------------------------
    # Step 11
    # ---------------------------------------------------------

    save_dataset_summary(
        property_df
    )

    # ---------------------------------------------------------
    # Step 12
    # ---------------------------------------------------------

    run_eda(
        property_df
    )

    # ---------------------------------------------------------
    # Step 13
    # ---------------------------------------------------------

    process_nhb()

    # ---------------------------------------------------------
    # Step 14
    # ---------------------------------------------------------

    process_gurgaon()

    # ---------------------------------------------------------
    # Step 15
    # ---------------------------------------------------------

    run_ml_experiments(
        property_df
    )

    # ---------------------------------------------------------
    # FINISHED
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MILLOW DATA PIPELINE COMPLETE")
    print("=" * 70)

    print(
        "\nGenerated folders:"
    )

    print(
        f"Processed data:\n"
        f"    {PROCESSED_DIR}"
    )

    print(
        f"Metadata:\n"
        f"    {METADATA_DIR}"
    )

    print(
        f"EDA:\n"
        f"    {EDA_DIR}"
    )

    print(
        f"ML results:\n"
        f"    {RESULTS_DIR}"
    )

    print(
        f"Models:\n"
        f"    {MODEL_ARTIFACT_DIR}"
    )

    print(
        "\nNext research steps:"
    )

    print(
        "1. Inspect MREID_property.csv"
    )

    print(
        "2. Review data-quality reports"
    )

    print(
        "3. Review EDA figures"
    )

    print(
        "4. Review model_comparison.csv"
    )

    print(
        "5. Standardize additional features"
    )

    print(
        "6. Perform feature-ablation experiments"
    )

    print(
        "7. Perform cross-city validation"
    )

    print(
        "8. Perform Gurgaon external validation"
    )

    print(
        "9. Add SHAP/XAI"
    )

    print(
        "10. Integrate NHB/Horizon carefully"
    )


# =============================================================
# RUN
# =============================================================

if __name__ == "__main__":

    main()