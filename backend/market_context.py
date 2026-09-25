"""
MILLOW - Property Market Context (NHB RESIDEX) router (Phase 4)

Adds a per-property market-context endpoint that pairs the MREID property
listing with the external NHB RESIDEX city-level benchmark.

Three clearly separated layers:
  - property   : listed market data from the MREID dataset
  - millow_ai  : MILLOW V5 model estimate (unchanged pipeline)
  - market_context : NHB RESIDEX city-level external benchmark (context only)

Routes live in their own module so the listing router stays untouched.
Registered by backend/app.py on the same FastAPI server (port 8001).
"""

from fastapi import APIRouter, HTTPException

import market_data
import properties

router = APIRouter()

MODEL_NAME = properties.MODEL_NAME

AI_NOTE = (
    "MILLOW AI estimate is an AI-assisted estimate computed with the "
    "MILLOW V5 model from the property's own attributes (MREID dataset). "
    "It is not a certified appraisal."
)

NHB_NOTE = (
    "NHB RESIDEX is an external, government-published city-level market "
    "benchmark. It reflects an aggregate index for the whole city and is "
    "not a valuation of this specific property. Do not treat it as "
    "identical to the property's listed or AI-estimated price."
)

DISCLAIMER = (
    "Listed prices come from the MREID property dataset. The MILLOW AI "
    "estimate is an AI-assisted estimate for research purposes only — not "
    "a certified appraisal. NHB RESIDEX values are city-level market "
    "context and not the property's true market value."
)


@router.get("/api/properties/{mreid_id}/market-context")
def property_market_context(mreid_id: str):
    mreid_key = mreid_id.strip()

    if mreid_key not in properties._id_to_index:
        raise HTTPException(
            status_code=404,
            detail=f"Property '{mreid_id}' not found.",
        )

    idx = properties._id_to_index[mreid_key]
    row = properties.PROPERTIES.iloc[idx]

    city = str(row["source_city"])
    nhb_city = market_data.nhb_city_for(city)

    observation = market_data.latest_observation(nhb_city)
    if observation is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No NHB RESIDEX market context available for city '{city}'."
            ),
        )

    ai_price, ai_ppsf = properties._batch_ai_predict(
        properties.PROPERTIES.iloc[[idx]]
    )

    listed = float(row["price"])
    area = float(row["area"])
    listed_ppsf = float(row["price_per_sqft"])

    return {
        "mreid_id": mreid_key,
        "city": city,
        "nhb_city": nhb_city,
        "property": {
            "listed_price": listed,
            "listed_price_formatted": properties.format_indian_price(listed),
            "area_sqft": int(area),
            "listed_price_per_sqft": round(listed_ppsf, 2),
        },
        "millow_ai": {
            "ai_estimated_price": float(round(ai_price[0], 2)),
            "ai_estimated_price_formatted": properties.format_indian_price(
                ai_price[0]
            ),
            "ai_estimated_price_per_sqft": float(round(ai_ppsf[0], 2)),
            "model": MODEL_NAME,
            "note": AI_NOTE,
        },
        "market_context": {
            "source": str(observation["source"]),
            "source_url": str(observation["source_url"]),
            "series": str(observation["series"]),
            "access_period": str(observation["access_period"]),
            "benchmark_level": "city",
            "city": nhb_city,
            "quarter": str(observation["quarter"]),
            "observation_period": (
                f"Quarter ending {observation['quarter']}"
            ),
            "composite_price_rs_sqft": int(
                observation["composite_price_rs_sqft"]
            ),
            "price_le_60sqm_rs_sqft": int(
                observation["price_le_60sqm_rs_sqft"]
            ),
            "price_60_110sqm_rs_sqft": int(
                observation["price_60_110sqm_rs_sqft"]
            ),
            "price_gt_110sqm_rs_sqft": int(
                observation["price_gt_110sqm_rs_sqft"]
            ),
            "hpi_assessment": float(observation["hpi_assessment"]),
            "applicable_size_band": market_data.size_band_for(
                area, observation
            ),
            "note": NHB_NOTE,
        },
        "note": (
            "Three distinct layers: listed market data (MREID), MILLOW AI "
            "estimate (computed from property attributes), and NHB RESIDEX "
            "external city-level market context."
        ),
        "disclaimer": DISCLAIMER,
    }