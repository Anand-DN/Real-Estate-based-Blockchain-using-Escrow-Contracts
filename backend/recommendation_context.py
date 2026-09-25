"""
MILLOW - Property Recommendation / Similar Properties router (Phase 6)

Two deterministic, explainable endpoints:

  * GET /api/properties/{mreid_id}/recommendations
      Similar properties in the same city (city = eligibility only).
  * GET /api/recommendations
      Filter / requirement-based recommendations (hard filters, then
      relevance ranking).

Recommendation scores are similarity/relevance (0-100). They are NOT
quality scores, investment advice, 'best property' rankings, or anything
derived from AI valuation error.
"""

from fastapi import APIRouter, HTTPException, Query

import recommendation

router = APIRouter()


@router.get("/api/properties/{mreid_id}/recommendations")
def similar_properties(
    mreid_id: str,
    limit: int = Query(6, ge=1, le=recommendation.LIMIT_MAX),
):
    payload = recommendation.recommend_similar(mreid_id, limit)
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=f"Property '{mreid_id}' not found.",
        )
    return payload


@router.get("/api/recommendations")
def recommendations(
    city: str | None = None,
    location: str | None = None,
    bedrooms: int | None = Query(None, ge=1),
    min_price: float | None = Query(None, ge=0),
    max_price: float | None = Query(None, ge=0),
    min_area: float | None = Query(None, ge=0),
    max_area: float | None = Query(None, ge=0),
    amenities: str | None = None,
    limit: int = Query(6, ge=1, le=recommendation.LIMIT_MAX),
):
    amenity_list = (
        [a.strip() for a in amenities.split(",") if a.strip()]
        if amenities
        else []
    )
    req = {
        "city": city,
        "location": location,
        "bedrooms": bedrooms,
        "min_price": min_price,
        "max_price": max_price,
        "min_area": min_area,
        "max_area": max_area,
        "amenities": amenity_list,
    }
    try:
        return recommendation.recommend_requirements(req, limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))