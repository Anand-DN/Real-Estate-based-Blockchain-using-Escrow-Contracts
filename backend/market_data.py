"""
MILLOW - NHB RESIDEX market context data layer (Phase 4)

External, government-published city-level market benchmark. Used only as
context — never as an individual property's value and never as an input to
the MILLOW V5 valuation model.

All data is loaded once at module import, mirroring how the MREID property
data is loaded by backend/properties.py.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

NHB_DATA_PATH = ROOT / "data" / "raw" / "market" / "MILLOW_NHB_RESIDEX_Master.csv"

# MREID source city -> NHB RESIDEX city (only where names differ).
NHB_CITY_MAP = {
    "Bangalore": "Bengaluru",
}

MREID_CITIES = [
    "Bangalore",
    "Chennai",
    "Delhi",
    "Hyderabad",
    "Kolkata",
    "Mumbai",
]

# NHB quarter labels are the calendar quarter-end month and year
# (e.g. "Jun 2025", "Sep 2025", "Dec 2025", "Mar 2026", "Jun 2026").
MONTH_INDEX = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

SQM_PER_SQFT = 1.0 / 10.7639


def _quarter_sort_key(label):
    """Chronological sort key derived strictly from the supplied quarter label."""
    parts = str(label).strip().split()
    if len(parts) != 2 or not parts[1].isdigit():
        return (0, 0)
    return (int(parts[1]), MONTH_INDEX.get(parts[0].lower()[:3], 0))


def _load_nhb():
    if not NHB_DATA_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(NHB_DATA_PATH)

    for col in [
        "composite_price_rs_sqft",
        "price_le_60sqm_rs_sqft",
        "price_60_110sqm_rs_sqft",
        "price_gt_110sqm_rs_sqft",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["city"] = df["city"].astype(str).str.strip()
    df["quarter"] = df["quarter"].astype(str).str.strip()
    df["quarter_key"] = df["quarter"].apply(_quarter_sort_key)

    return df


NHB = _load_nhb()


def nhb_cities():
    """All distinct NHB cities present in the loaded dataset."""
    if NHB.empty:
        return []
    return sorted(NHB["city"].unique().tolist())


def nhb_city_for(mreid_city):
    """Map an MREID city to its NHB RESIDEX city name."""
    return NHB_CITY_MAP.get(str(mreid_city).strip(), str(mreid_city).strip())


def latest_observation(nhb_city):
    """
    Latest chronological NHB observation for a city.

    Returns a plain dict with the original quarter label preserved, or None
    when the city has no NHB data.
    """
    if NHB.empty:
        return None

    city = str(nhb_city).strip()
    sub = NHB[NHB["city"] == city]
    if sub.empty:
        return None

    latest_key = sub["quarter_key"].max()
    row = sub[sub["quarter_key"] == latest_key].iloc[0]

    return {
        "quarter": row["quarter"],
        "composite_price_rs_sqft": int(row["composite_price_rs_sqft"]),
        "price_le_60sqm_rs_sqft": int(row["price_le_60sqm_rs_sqft"]),
        "price_60_110sqm_rs_sqft": int(row["price_60_110sqm_rs_sqft"]),
        "price_gt_110sqm_rs_sqft": int(row["price_gt_110sqm_rs_sqft"]),
        "hpi_assessment": float(row["hpi_assessment"]),
        "source": str(row["source"]),
        "source_url": str(row["source_url"]),
        "series": str(row["series"]),
        "access_period": str(row["access_period"]),
    }


def size_band_for(area_sqft, observation):
    """
    NHB size-band benchmark matching the property's area, when available.

    NHB reports prices per size band (<=60 sqm, 60-110 sqm, >110 sqm).
    The band is context only and carries the same city-level caveat.
    """
    if observation is None:
        return None

    sqm = float(area_sqft) * SQM_PER_SQFT

    if sqm <= 60:
        band = "≤ 60 sqm"
        price = observation["price_le_60sqm_rs_sqft"]
    elif sqm <= 110:
        band = "60 – 110 sqm"
        price = observation["price_60_110sqm_rs_sqft"]
    else:
        band = "> 110 sqm"
        price = observation["price_gt_110sqm_rs_sqft"]

    return {
        "label": band,
        "price_rs_sqft": int(price),
        "property_area_sqft": float(area_sqft),
        "equivalent_sqm": round(sqm, 2),
    }