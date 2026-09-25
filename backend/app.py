"""
MILLOW - AI Property Valuation API (Indian market)

Phase 1 backend:
    Exposes the V5 Log-Price XGBoost valuation model
    (models/valuation/final) as a FastAPI service.

Run from project root:
    python backend/app.py
    # or
    uvicorn backend.app:app --host 0.0.0.0 --port 8001
"""

import json
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from predict_property_value import (  # noqa: E402  (reuses tested CLI logic)
    format_indian_price,
    predict_property,
)

MODEL_DIR = ROOT / "models" / "valuation" / "final"
METADATA_PATH = MODEL_DIR / "metadata.json"

APP_TITLE = "MILLOW Property Valuation API"
PORT = 8001

with open(METADATA_PATH, "r", encoding="utf-8") as fh:
    METADATA = json.load(fh)

CITIES = sorted(METADATA.get("cities", []))
VALIDATION = METADATA.get("validation", {})
MODEL_NAME = METADATA.get(
    "model_name",
    "MILLOW V5 Log-Price XGBoost",
)

PRICE_UNITS = ["9", "0", "1"]

DISCLAIMER = (
    "This is an AI-assisted property valuation estimate based on "
    "historical property data for educational/research purposes. "
    "It is not a certified appraisal. Actual market value may vary "
    "significantly depending on property condition, exact micro-location, "
    "amenities, market conditions and other factors."
)

app = FastAPI(
    title=APP_TITLE,
    version="1.0.0",
    description="MILLOW AI-assisted real estate valuation for Indian cities.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost",  # jsdom/live-test origin (bare localhost)
    ],
    allow_origin_regex=r"http://localhost:\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)

from properties import router as properties_router  # noqa: E402
from market_context import router as market_context_router  # noqa: E402
from risk_context import router as risk_context_router  # noqa: E402
from recommendation_context import router as recommendation_context_router  # noqa: E402
from dashboard import router as dashboard_router  # noqa: E402

app.include_router(properties_router)
app.include_router(market_context_router)
app.include_router(risk_context_router)
app.include_router(recommendation_context_router)
app.include_router(dashboard_router)


# ============================================================
# REQUESTS / RESPONSES
# ============================================================

class ValuationRequest(BaseModel):
    city: str = Field(..., min_length=1)
    location: str = Field(..., min_length=1)
    area_sqft: float = Field(..., gt=0)
    bedrooms: int = Field(..., gt=0)
    amenities: dict[str, int] | None = None


class ValuationResponse(BaseModel):
    city: str
    location: str
    area_sqft: float
    bedrooms: int
    estimated_price: float
    estimated_price_formatted: str
    estimated_price_per_sqft: float
    estimated_price_range: dict[str, float]
    model: str
    disclaimer: str


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "service": APP_TITLE,
        "model": MODEL_NAME,
        "cities": CITIES,
        "endpoints": [
            "/health",
            "/api/valuation",
            "/api/valuation/cities",
            "/api/valuation/model",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_loaded": True,
    }


@app.get("/api/valuation/cities")
def valuation_cities():
    return {
        "cities": CITIES,
        "total_locations": METADATA.get("location_count"),
    }


@app.get("/api/valuation/model")
def valuation_model():
    return {
        "model_name": MODEL_NAME,
        "target": METADATA.get("target"),
        "inverse_transform": METADATA.get("inverse_transform"),
        "training_rows": METADATA.get("training_rows"),
        "features": METADATA.get("features"),
        "validation": {
            "split": VALIDATION.get("split"),
            "location_overlap": VALIDATION.get("location_overlap"),
            "mean_MAE": VALIDATION.get("mean_MAE"),
            "mean_RMSE": VALIDATION.get("mean_RMSE"),
            "mean_R2": VALIDATION.get("mean_R2"),
            "mean_MAPE": VALIDATION.get("mean_MAPE"),
        },
        "amenity_encoding": METADATA.get("amenity_encoding"),
    }


@app.post("/api/valuation")
def valuation(req: ValuationRequest):
    city = req.city.strip()
    location = req.location.strip()

    if city.casefold() not in {c.casefold() for c in CITIES}:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported city '{req.city}'. "
                f"Supported cities: {', '.join(CITIES)}."
            ),
        )

    try:
        result = predict_property(
            city=city,
            location=location,
            area=req.area_sqft,
            bedrooms=req.bedrooms,
            amenities=req.amenities,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )

    price = result["estimated_price"]

    mae = VALIDATION.get("mean_MAE")
    price_range = {
        "low": max(round(price - mae, 2), 0.0),
        "high": round(price + mae, 2)
    } if mae else {}

    return ValuationResponse(
        city=result["city"],
        location=result["location"],
        area_sqft=result["area_sqft"],
        bedrooms=result["bedrooms"],
        estimated_price=round(price, 2),
        estimated_price_formatted=format_indian_price(price),
        estimated_price_per_sqft=round(
            result["estimated_price_per_sqft"], 2
        ),
        estimated_price_range=price_range,
        model=result["model"],
        disclaimer=DISCLAIMER,
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
    )