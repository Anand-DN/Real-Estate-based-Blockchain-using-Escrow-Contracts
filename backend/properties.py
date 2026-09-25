"""
MILLOW - Property Listing Backend (Indian MREID dataset)

FastAPI router exposing the processed MREID property data for
discovery/listing. Does NOT include payments, escrow, wallets,
NFT minting, or blockchain (separate phases).

Run inside the same server as backend/app.py (port 8001).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

import chain_index

from predict_property_value import (  # noqa: E402
    amenity_cols,
    create_features,
    format_indian_price,
    model,
    preprocessor,
)

DATA_PATH = ROOT / "data" / "processed" / "MREID_property.csv"

AMENITY_LABELS = {
    0: "No",
    1: "Yes",
    9: "Unknown / Not specified",
}

# Historical dataset columns returned as-is (never used as a model input;
# the valuation feature pipeline uses only engineered features).
HISTORICAL_PPSF_COL = "derived_price_per_sqft"

DISCLAIMER = (
    "AI estimated value is an AI-assisted estimate computed by the MILLOW "
    "V5 Log-Price XGBoost model for research purposes. It is not a "
    "certified appraisal. Actual market value may vary significantly."
)

MODEL_NAME = "MILLOW V5 Log-Price XGBoost"

# ============================================================
# DATA LOAD (once per process)
# ============================================================

def _load_properties():
    df = pd.read_csv(DATA_PATH)

    for col in ["area", "no_of_bedrooms", "price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[
        (df["price"] > 0)
        & (df["area"] > 0)
        & (df["no_of_bedrooms"] > 0)
    ].reset_index(drop=True)

    df["location"] = df["location"].astype(str).str.strip()
    df["source_city"] = df["source_city"].astype(str).str.strip()

    df["price_per_sqft"] = df["price"] / df["area"]

    return df


PROPERTIES = _load_properties()

KNOWN_CITIES = sorted(PROPERTIES["source_city"].unique().tolist())

_id_to_index = {
    v: k
    for k, v in (
        PROPERTIES["mreid_id"]
        .astype(str)
        .to_dict()
        .items()
    )
}

_missing_amenities = set(amenity_cols) - set(PROPERTIES.columns)

# ============================================================
# MARKET SIGNAL / LOCALITY COHORT STATS
# ============================================================

# The band (in percent) around the AI estimate within which a listing is
# considered "Near estimated market range".  Outside the band (tip of the
# AI research estimate) a listing is classified "Potentially undervalued"
# (listed below the estimate) or "Potentially overvalued" (listed above it).
SIGNAL_BAND_PCT = 5.0

SIGNAL_UNDERVALUED = "Potentially undervalued"
SIGNAL_OVERVALUED = "Potentially overvalued"
SIGNAL_IN_RANGE = "Near estimated market range"

SIGNAL_NOTE = (
    "This classification is a model-based market signal computed by "
    "comparing the listed price against the MILLOW AI research estimate. "
    "It is not a certified valuation, not financial/investment advice, and "
    "does not imply any fraud or mispricing claim about the seller."
)

# Locality-level price-per-sqft cohort statistics, computed once at load from
# the real MREID dataset.  Used only as comparable information (not a model
# input and never a guarantee of any individual property's value).
_LOCALITY_STATS = PROPERTIES.groupby(
    ["source_city", "location"], sort=False
)["price_per_sqft"].agg(["median", "mean", "count"]).reset_index()


def locality_stats(city, location):
    rows = _LOCALITY_STATS.loc[
        (_LOCALITY_STATS["source_city"] == city)
        & (_LOCALITY_STATS["location"] == location)
    ]
    if rows.empty:
        return {
            "count": 0,
            "median_price_per_sqft": None,
            "mean_price_per_sqft": None,
            "note": "No comparable records for this locality in the dataset.",
        }
    row = rows.iloc[0]
    return {
        "count": int(row["count"]),
        "median_price_per_sqft": round(float(row["median"]), 2),
        "mean_price_per_sqft": round(float(row["mean"]), 2),
        "note": (
            f"Median and mean listed ₹/sqft across {int(row['count'])} "
            "dataset records for this locality. Comparable information "
            "only — not a valuation of this specific property."
        ),
    }


def ai_market_signal(listed, ai_price):
    """Compare a listed price against the AI estimate (research signal)."""
    if listed <= 0 or ai_price is None:
        return {
            "label": SIGNAL_IN_RANGE,
            "listed_price": listed,
            "ai_estimated_price": None,
            "difference_inr": None,
            "difference_pct": None,
            "band_pct": SIGNAL_BAND_PCT,
            "note": SIGNAL_NOTE,
        }

    difference = round(float(ai_price) - float(listed), 2)
    difference_pct = round(
        (float(ai_price) - float(listed)) / float(listed) * 100.0, 2
    )

    if difference_pct > SIGNAL_BAND_PCT:
        label = SIGNAL_UNDERVALUED
    elif difference_pct < -SIGNAL_BAND_PCT:
        label = SIGNAL_OVERVALUED
    else:
        label = SIGNAL_IN_RANGE

    return {
        "label": label,
        "listed_price": float(listed),
        "ai_estimated_price": round(float(ai_price), 2),
        "difference_inr": difference,
        "difference_pct": difference_pct,
        "band_pct": SIGNAL_BAND_PCT,
        "note": SIGNAL_NOTE,
    }


# ============================================================
# HELPERS
# ============================================================

def _result_meta(page, page_size, total):
    return {
        "page": page,
        "page_size": page_size,
        "total": int(total),
        "total_pages": int(
            np.ceil(total / page_size)
        ) if total else 0,
    }


def _amenity_payload(row):
    return {
        name: AMENITY_LABELS.get(value, "Unknown / Not specified")
        for name, value in row.items()
        if name in amenity_cols
        and not pd.isna(value)
    }


def _batch_ai_predict(rows):
    """Predict AI value for a batch of property rows (single model call)."""
    if len(rows) == 0:
        return [], []

    batch = pd.DataFrame(
        {
            "source_city": rows["source_city"].tolist(),
            "location": rows["location"].tolist(),
            "area": rows["area"].astype(float).tolist(),
            "no_of_bedrooms": rows["no_of_bedrooms"].astype(int).tolist(),
        }
    )

    for a in amenity_cols:
        batch[a] = (
            rows[a].astype(float).fillna(9).tolist()
            if a in rows.columns
            else [9] * len(rows)
        )

    engineered = create_features(batch)

    encoded = preprocessor.transform(engineered)

    pred_log = model.predict(encoded)

    prices = np.maximum(np.expm1(pred_log), 0.0)

    areas = np.asarray(rows["area"].astype(float).tolist())

    ppsf = prices / np.maximum(areas, 1.0)

    return prices.tolist(), ppsf.tolist()


def _ai_fields(row, ai_price, ai_ppsf):
    return {
        "ai_estimated_price": float(round(ai_price, 2)),
        "ai_estimated_price_formatted": format_indian_price(ai_price),
        "ai_estimated_price_per_sqft": float(round(ai_ppsf, 2)),
    }


def _summary(row, ai_price, ai_ppsf):
    listed = float(row["price"])

    return {
        "mreid_id": str(row["mreid_id"]),
        "price": listed,
        "price_formatted": format_indian_price(listed),
        "price_per_sqft": round(float(row["price_per_sqft"]), 2),
        "area": int(row["area"]),
        "location": row["location"],
        "city": row["source_city"],
        "bedrooms": int(row["no_of_bedrooms"]),
        "amenities": _amenity_payload(row),
        **_ai_fields(row, ai_price, ai_ppsf),
        "ai_market_signal": ai_market_signal(listed, ai_price),
        "locality": locality_stats(row["source_city"], row["location"]),
    }


def _detail(row, ai_price, ai_ppsf):
    listed = float(row["price"])

    return {
        "mreid_id": str(row["mreid_id"]),
        "listed_price": listed,
        "listed_price_formatted": format_indian_price(listed),
        "price_per_sqft": round(float(row["price_per_sqft"]), 2),
        "historical_derived_price_per_sqft": (
            float(row[HISTORICAL_PPSF_COL])
            if HISTORICAL_PPSF_COL in row
            else None
        ),
        "property": {
            "area": int(row["area"]),
            "location": row["location"],
            "city": row["source_city"],
            "bedrooms": int(row["no_of_bedrooms"]),
            "resale": int(row["resale"]) if "resale" in row else None,
            "amenities": _amenity_payload(row),
        },
        "ai_estimation": {
            **_ai_fields(row, ai_price, ai_ppsf),
            "model": MODEL_NAME,
            "note": (
                "AI estimated value differs from the listed/historical "
                "dataset price and is provided as an independent estimate."
            ),
        },
        "ai_market_signal": ai_market_signal(listed, ai_price),
        "locality": locality_stats(row["source_city"], row["location"]),
        "disclaimer": DISCLAIMER,
    }


def _chain_metadata():
    return {
        "available": chain_index.available(),
        "exported_at": chain_index.exported_at(),
    }


# ============================================================
# ROUTES
# ============================================================

router = APIRouter()


@router.get("/api/properties/cities")
def property_cities():
    return {"cities": KNOWN_CITIES}


@router.get("/api/properties/locations")
def property_locations(
    city: str | None = None,
):
    if city:
        city_key = city.strip()
        mask = (
            PROPERTIES["source_city"].str.casefold()
            == city_key.casefold()
        )
        if not mask.any():
            raise HTTPException(
                status_code=400,
                detail=f"Unknown city '{city}'. "
                       f"Supported cities: {', '.join(KNOWN_CITIES)}.",
            )
        locations = sorted(
            PROPERTIES.loc[mask, "location"].unique().tolist()
        )
    else:
        locations = sorted(
            PROPERTIES["location"].unique().tolist()
        )

    return {"locations": locations, "count": len(locations)}


@router.get("/api/properties/search")
def search_properties(
    city: str | None = None,
    location: str | None = None,
    min_price: float | None = Query(None, ge=0),
    max_price: float | None = Query(None, ge=0),
    min_area: float | None = Query(None, ge=0),
    max_area: float | None = Query(None, ge=0),
    bedrooms: int | None = Query(None, ge=1),
    ai_signal: str | None = Query(None, pattern=(
        "^(undervalued|overvalued|in_range)$"
    )),
    sort: str = Query("id", pattern=(
        "^(id|price_asc|price_desc|area_asc|area_desc|"
        "price_per_sqft_asc|price_per_sqft_desc|"
        "ai_estimate_asc|ai_estimate_desc|ai_difference_asc|"
        "ai_difference_desc)$"
    )),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if min_price is not None and max_price is not None:
        if min_price > max_price:
            raise HTTPException(
                status_code=400,
                detail="min_price cannot be greater than max_price.",
            )

    if min_area is not None and max_area is not None:
        if min_area > max_area:
            raise HTTPException(
                status_code=400,
                detail="min_area cannot be greater than max_area.",
            )

    mask = pd.Series(True, index=PROPERTIES.index)

    if city:
        city_key = city.strip()
        if not (PROPERTIES["source_city"].str.casefold() == city_key.casefold()).any():
            raise HTTPException(
                status_code=400,
                detail=f"Unknown city '{city}'. "
                       f"Supported cities: {', '.join(KNOWN_CITIES)}.",
            )
        mask &= PROPERTIES["source_city"].str.casefold() == city_key.casefold()

    if location:
        loc_key = location.strip().casefold()
        mask &= PROPERTIES["location"].str.casefold().str.contains(loc_key)

    if min_price is not None:
        mask &= PROPERTIES["price"] >= min_price

    if max_price is not None:
        mask &= PROPERTIES["price"] <= max_price

    if min_area is not None:
        mask &= PROPERTIES["area"] >= min_area

    if max_area is not None:
        mask &= PROPERTIES["area"] <= max_area

    if bedrooms is not None:
        mask &= PROPERTIES["no_of_bedrooms"] == bedrooms

    filtered = PROPERTIES.loc[mask].copy()

    if ai_signal:
        full_ai, full_ppsf = _batch_ai_predict(filtered)
        if ai_signal == "undervalued":
            keep = [
                i for i, l in enumerate(filtered["price"])
                if ai_market_signal(float(l), full_ai[i])["label"] == SIGNAL_UNDERVALUED
            ]
        elif ai_signal == "overvalued":
            keep = [
                i for i, l in enumerate(filtered["price"])
                if ai_market_signal(float(l), full_ai[i])["label"] == SIGNAL_OVERVALUED
            ]
        else:
            keep = [
                i for i, l in enumerate(filtered["price"])
                if ai_market_signal(float(l), full_ai[i])["label"] == SIGNAL_IN_RANGE
            ]
        filtered = filtered.iloc[keep].copy()

    if sort == "id":
        filtered = filtered.sort_values("mreid_id")
    elif sort == "price_asc":
        filtered = filtered.sort_values("price", ascending=True)
    elif sort == "price_desc":
        filtered = filtered.sort_values("price", ascending=False)
    elif sort == "area_asc":
        filtered = filtered.sort_values("area", ascending=True)
    elif sort == "area_desc":
        filtered = filtered.sort_values("area", ascending=False)
    elif sort == "price_per_sqft_asc":
        filtered = filtered.sort_values("price_per_sqft", ascending=True)
    elif sort == "price_per_sqft_desc":
        filtered = filtered.sort_values("price_per_sqft", ascending=False)
    elif sort in (
        "ai_estimate_asc",
        "ai_estimate_desc",
        "ai_difference_asc",
        "ai_difference_desc",
    ):
        full_ai, _ = _batch_ai_predict(filtered)
        full_listed = filtered["price"].to_numpy(dtype=float)
        filtered["_ai_est"] = [float(x) for x in full_ai]
        filtered["_ai_diff"] = [
            float(a) - float(l) for a, l in zip(full_ai, full_listed)
        ]
        if sort.startswith("ai_estimate"):
            col, ascending = "_ai_est", sort.endswith("asc")
        else:
            col, ascending = "_ai_diff", sort.endswith("asc")
        filtered = filtered.sort_values(col, ascending=ascending)

    total = len(filtered)

    start = (page - 1) * page_size
    page_rows = filtered.iloc[start:start + page_size]

    ai_prices, ai_ppsfs = _batch_ai_predict(page_rows)

    results = [
        _summary(row, ai_prices[i], ai_ppsfs[i])
        for i, (_, row) in enumerate(page_rows.iterrows())
    ]

    return {
        **_result_meta(page, page_size, total),
        "filters": {
            "city": city,
            "location": location,
            "min_price": min_price,
            "max_price": max_price,
            "min_area": min_area,
            "max_area": max_area,
            "bedrooms": bedrooms,
            "ai_signal": ai_signal,
            "sort": sort,
        },
        "results": results,
    }


@router.get("/api/properties/{mreid_id}/chain")
def property_chain(mreid_id: str):
    mreid_key = mreid_id.strip()

    if mreid_key not in _id_to_index:
        raise HTTPException(
            status_code=404,
            detail=f"Property '{mreid_id}' not found.",
        )

    return {
        "mreid_id": mreid_key,
        "chain": chain_index.as_summary(mreid_key),
        "chain_snapshot": _chain_metadata(),
    }


@router.get("/api/properties/{mreid_id}")
def property_detail(mreid_id: str):
    mreid_key = mreid_id.strip()

    if mreid_key not in _id_to_index:
        raise HTTPException(
            status_code=404,
            detail=f"Property '{mreid_id}' not found.",
        )

    idx = _id_to_index[mreid_key]
    row = PROPERTIES.iloc[idx]

    ai_price, ai_ppsf = _batch_ai_predict(
        PROPERTIES.iloc[[idx]]
    )

    detail = _detail(row, ai_price[0], ai_ppsf[0])
    detail["chain"] = chain_index.as_summary(mreid_key)
    detail["chain_snapshot"] = _chain_metadata()
    return detail


@router.get("/api/properties")
def list_properties(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    total = len(PROPERTIES)

    start = (page - 1) * page_size
    page_rows = PROPERTIES.iloc[start:start + page_size]

    ai_prices, ai_ppsfs = _batch_ai_predict(page_rows)

    results = [
        _summary(row, ai_prices[i], ai_ppsfs[i])
        for i, (_, row) in enumerate(page_rows.iterrows())
    ]

    return {
        **_result_meta(page, page_size, total),
        "results": results,
    }
