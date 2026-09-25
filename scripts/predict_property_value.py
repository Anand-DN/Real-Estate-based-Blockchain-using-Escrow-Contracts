import json
import os

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


# ============================================================
# CONFIG
# ============================================================

MODEL_DIR = "models/valuation/final"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "millow_valuation_model.json"
)

PREPROCESSOR_PATH = os.path.join(
    MODEL_DIR,
    "millow_valuation_preprocessor.joblib"
)

METADATA_PATH = os.path.join(
    MODEL_DIR,
    "metadata.json"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading MILLOW valuation model...")

model = XGBRegressor()
model.load_model(MODEL_PATH)

preprocessor = joblib.load(PREPROCESSOR_PATH)

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as f:
    metadata = json.load(f)


# ============================================================
# AMENITY COLUMNS
# ============================================================

amenity_cols = [
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
# FEATURE ENGINEERING
# ============================================================

def create_features(property_data):

    x = property_data.copy()

    x["area"] = pd.to_numeric(
        x["area"],
        errors="coerce"
    )

    x["no_of_bedrooms"] = pd.to_numeric(
        x["no_of_bedrooms"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Basic features
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
    # Amenity features
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

    x = x.drop(
        columns=amenity_cols,
        errors="ignore"
    )

    return x


# ============================================================
# PRICE FORMATTING
# ============================================================

def format_indian_price(price):

    if price >= 1_00_00_000:
        return f"₹{price / 1_00_00_000:.2f} Crore"

    if price >= 1_00_000:
        return f"₹{price / 1_00_000:.2f} Lakh"

    if price >= 1_000:
        return f"₹{price / 1_000:.2f} Thousand"

    return f"₹{price:,.0f}"


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_property(
    city,
    location,
    area,
    bedrooms,
    amenities=None
):

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not city or not city.strip():
        raise ValueError(
            "City cannot be empty."
        )

    if not location or not location.strip():
        raise ValueError(
            "Location cannot be empty."
        )

    if area <= 0:
        raise ValueError(
            "Area must be greater than 0."
        )

    if bedrooms <= 0:
        raise ValueError(
            "Bedrooms must be greater than 0."
        )

    # --------------------------------------------------------
    # Default amenities
    # 9 = unknown / not specified
    # --------------------------------------------------------

    if amenities is None:
        amenities = {}

    # --------------------------------------------------------
    # Build property record
    # --------------------------------------------------------

    property_data = {
        "source_city": city,
        "location": location,
        "area": area,
        "no_of_bedrooms": bedrooms,
    }

    for amenity in amenity_cols:

        value = amenities.get(
            amenity,
            9
        )

        property_data[amenity] = value

    property_df = pd.DataFrame(
        [property_data]
    )

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    features = create_features(
        property_df
    )

    # --------------------------------------------------------
    # Transform using saved preprocessor
    # --------------------------------------------------------

    encoded = preprocessor.transform(
        features
    )

    # --------------------------------------------------------
    # Predict log(price)
    # --------------------------------------------------------

    predicted_log_price = model.predict(
        encoded
    )[0]

    # --------------------------------------------------------
    # Convert log(price) back to ₹
    # --------------------------------------------------------

    predicted_price = float(
        np.expm1(
            predicted_log_price
        )
    )

    predicted_price = max(
        predicted_price,
        0
    )

    # --------------------------------------------------------
    # Calculate estimated price/sqft
    # --------------------------------------------------------

    predicted_ppsf = (
        predicted_price / area
    )

    # --------------------------------------------------------
    # Return prediction
    # --------------------------------------------------------

    return {
        "city": city,
        "location": location,
        "area_sqft": area,
        "bedrooms": bedrooms,
        "estimated_price": predicted_price,
        "estimated_price_per_sqft": predicted_ppsf,
        "model": metadata.get(
            "model_name",
            "MILLOW V5 Log-Price XGBoost"
        ),
    }


# ============================================================
# INTERACTIVE MODE
# ============================================================

if __name__ == "__main__":

    print()
    print("==============================================")
    print("MILLOW AI PROPERTY VALUATION")
    print("==============================================")

    try:

        city = input(
            "City: "
        ).strip()

        location = input(
            "Location: "
        ).strip()

        area = float(
            input(
                "Area (sqft): "
            )
        )

        bedrooms = int(
            input(
                "Number of bedrooms: "
            )
        )

        result = predict_property(
            city=city,
            location=location,
            area=area,
            bedrooms=bedrooms,
        )

        print()
        print("==============================================")
        print("MILLOW VALUATION RESULT")
        print("==============================================")

        print(
            f"City:                 "
            f"{result['city']}"
        )

        print(
            f"Location:             "
            f"{result['location']}"
        )

        print(
            f"Area:                 "
            f"{result['area_sqft']:,.0f} sqft"
        )

        print(
            f"Bedrooms:             "
            f"{result['bedrooms']}"
        )

        print()
        print("Estimated Property Value:")

        print(
            format_indian_price(
                result["estimated_price"]
            )
        )

        print()
        print("Estimated Price / Sqft:")

        print(
            f"₹{result['estimated_price_per_sqft']:,.0f}"
        )

        print()
        print("Model:")

        print(
            result["model"]
        )

        print()
        print("----------------------------------------------")
        print("IMPORTANT")
        print("----------------------------------------------")
        print(
            "This is an AI-assisted property valuation"
        )
        print(
            "estimate based on historical property data."
        )
        print(
            "Actual market value may vary depending on"
        )
        print(
            "property condition, exact location, amenities,"
        )
        print(
            "market conditions, and other factors."
        )

        print()
        print("==============================================")

    except ValueError as e:

        print()
        print("Input Error:")
        print(e)

    except Exception as e:

        print()
        print("Prediction Error:")
        print(e)