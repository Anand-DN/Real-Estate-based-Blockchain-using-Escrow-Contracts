"""
MILLOW - Property Listing API tests (Phase 2)

Asserts listing discovery/search behavior for backend/app.py
using FastAPI's TestClient.

Run from project root:
    python backend/test_properties_api.py
"""

import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

import app  # noqa: E402
import properties  # noqa: E402

client = TestClient(app.app)

PASSED = 0
FAILED = 0


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name}  {detail}")


def main():

    print("=" * 60)
    print("MILLOW PROPERTY API TESTS (PHASE 2)")
    print("=" * 60)

    # ----------------------------------------------------------
    # Listing retrieval
    # ------------------------------------------------------------

    r = client.get("/api/properties")
    check("GET /api/properties 200", r.status_code == 200, r.text)
    body = r.json()
    check("default page_size 20", len(body["results"]) == 20, r.text)
    check("total equals dataset size",
          body["total"] == len(properties.PROPERTIES), r.text)
    check("total_pages computed",
          body["total_pages"] == 1457, r.text)  # ceil(29135/20)
    required = {
        "mreid_id", "price", "price_formatted", "area", "location",
        "city", "bedrooms", "amenities",
    }
    first = body["results"][0]
    check("summary contains required fields",
          required.issubset(first.keys()), str(list(first.keys())))
    check("summary has AI estimate distinct from price",
          "ai_estimated_price" in first
          and abs(first["ai_estimated_price"] - first["price"]) > 0,
          str(first))

    # ----------------------------------------------------------
    # Pagination
    # ------------------------------------------------------------

    r = client.get("/api/properties", params={"page": 2, "page_size": 10})
    body = r.json()
    check("page 2 / size 10 returns 10 rows",
          len(body["results"]) == 10, r.text)
    check("page 2 offset", body["results"][0]["mreid_id"] == "MREID_0000011",
          body["results"][0]["mreid_id"])

    r = client.get("/api/properties", params={"page": 99999, "page_size": 20})
    body = r.json()
    check("out-of-range page returns empty results",
          body["results"] == [], r.text)

    r = client.get("/api/properties", params={"page": 0})
    check("page 0 rejected (422)", r.status_code == 422, r.text)

    r = client.get("/api/properties", params={"page_size": 0})
    check("page_size 0 rejected (422)", r.status_code == 422, r.text)

    r = client.get("/api/properties", params={"page_size": 101})
    check("page_size > 100 rejected (422)", r.status_code == 422, r.text)

    # ----------------------------------------------------------
    # City filtering
    # ------------------------------------------------------------

    r = client.get("/api/properties/cities")
    cities = r.json()["cities"]
    check("cities endpoint returns six cities",
          cities == sorted([
              "Bangalore", "Chennai", "Delhi",
              "Hyderabad", "Kolkata", "Mumbai",
          ]), str(cities))

    r = client.get("/api/properties/search", params={"city": "Bangalore"})
    body = r.json()
    check("city filter Bangalore only",
          len(body["results"]) == 20
          and all(x["city"] == "Bangalore" for x in body["results"]), r.text)
    check("city filter total == Bangalore count",
          body["total"] == (properties.PROPERTIES["source_city"]
                            == "Bangalore").sum(), str(body["total"]))

    r = client.get("/api/properties/search", params={"city": "mumbai"})
    body = r.json()
    check("city filter is case-insensitive",
          all(x["city"] == "Mumbai" for x in body["results"]), r.text)

    r = client.get("/api/properties/search", params={"city": "Paris"})
    check("unknown city rejected (400)", r.status_code == 400, r.text)

    # ----------------------------------------------------------
    # Location filtering
    # ------------------------------------------------------------

    r = client.get("/api/properties/locations", params={"city": "Bangalore"})
    body = r.json()
    check("locations endpoint works for city",
          body["count"] > 0
          and "Banashankari" in body["locations"], r.text)

    r = client.get("/api/properties/locations")
    body = r.json()
    check("locations endpoint without city",
          body["count"] == properties.PROPERTIES["location"].nunique(), r.text)

    r = client.get("/api/properties/search", params={"location": "whitefield"})
    body = r.json()
    check("location filter, case-insensitive substring",
          body["total"] > 0
          and all("whitefield" in x["location"].lower()
                  for x in body["results"]), r.text)

    # ----------------------------------------------------------
    # Price filtering
    # ------------------------------------------------------------

    r = client.get("/api/properties/search",
                   params={"city": "Bengaluru"})
    check("typo city rejected", r.status_code == 400, r.text)

    r = client.get("/api/properties/search",
                   params={"min_price": 4000000, "max_price": 6000000})
    body = r.json()
    check("price range respected",
          all(4000000 <= x["price"] <= 6000000 for x in body["results"]), r.text)

    r = client.get("/api/properties/search",
                   params={"min_price": 9000000, "max_price": 1000000})
    check("min_price > max_price rejected (400)",
          r.status_code == 400, r.text)

    r = client.get("/api/properties/search", params={"min_price": -1})
    check("negative price rejected (422)", r.status_code == 422, r.text)

    # ----------------------------------------------------------
    # Area filtering
    # ------------------------------------------------------------

    r = client.get("/api/properties/search",
                   params={"min_area": 1000, "max_area": 1200})
    body = r.json()
    check("area range respected",
          all(1000 <= x["area"] <= 1200 for x in body["results"]), r.text)

    r = client.get("/api/properties/search",
                   params={"min_area": 2000, "max_area": 500})
    check("min_area > max_area rejected (400)",
          r.status_code == 400, r.text)

    # ----------------------------------------------------------
    # Bedroom filtering
    # ------------------------------------------------------------

    r = client.get("/api/properties/search", params={"bedrooms": 3})
    body = r.json()
    check("bedrooms filter respected",
          len(body["results"]) > 0
          and all(x["bedrooms"] == 3 for x in body["results"]), r.text)

    r = client.get("/api/properties/search", params={"bedrooms": -2})
    check("negative bedrooms rejected", r.status_code == 422, r.text)

    # ----------------------------------------------------------
    # Sorting
    # ------------------------------------------------------------

    r = client.get("/api/properties/search",
                   params={"sort": "price_asc", "page_size": 5})
    body = r.json()
    prices = [x["price"] for x in body["results"]]
    check("price_asc sorted", prices == sorted(prices), r.text)

    r = client.get("/api/properties/search",
                   params={"sort": "price_desc", "page_size": 5})
    body = r.json()
    prices = [x["price"] for x in body["results"]]
    check("price_desc sorted", prices == sorted(prices, reverse=True), r.text)

    r = client.get("/api/properties/search",
                   params={"sort": "area_asc", "page_size": 5})
    body = r.json()
    areas = [x["area"] for x in body["results"]]
    check("area_asc sorted", areas == sorted(areas), r.text)

    r = client.get("/api/properties/search", params={"sort": "bogus"})
    check("invalid sort rejected (422)", r.status_code == 422, r.text)

    # ----------------------------------------------------------
    # Property detail
    # ------------------------------------------------------------

    r = client.get("/api/properties/MREID_0000001")
    check("detail 200 for valid id", r.status_code == 200, r.text)
    detail = r.json()
    check("detail has property block", "property" in detail, r.text)
    check("detail has ai_estimation block", "ai_estimation" in detail, r.text)
    check("detail distinguishes listed vs ai",
          "listed_price" in detail
          and "ai_estimated_price" in detail["ai_estimation"], r.text)
    check("detail amenities preserve 0/1/9 labels",
          set(detail["property"]["amenities"].values())
          <= {"No", "Yes", "Unknown / Not specified"}, r.text)
    check("historical ppsf returned as dataset info",
          "historical_derived_price_per_sqft" in detail, r.text)

    r = client.get("/api/properties/MREID_9999999")
    check("unknown id rejected (404)", r.status_code == 404, r.text)

    r = client.get("/api/properties/search")
    check("route order: search not captured by {id}",
          r.status_code == 200, r.text)

    r = client.get("/api/properties/cities")
    check("route order: cities not captured by {id}",
          r.status_code == 200, r.text)

    # ----------------------------------------------------------
    # AI valuation distinct from listed price
    # ------------------------------------------------------------

    detail = client.get("/api/properties/MREID_0000001").json()
    listed = detail["listed_price"]
    ai = detail["ai_estimation"]["ai_estimated_price"]
    check("AI block is never the listed price",
          abs(ai - listed) > 0, f"listed={listed} ai={ai}")
    check("AI model named",
          detail["ai_estimation"]["model"] == "MILLOW V5 Log-Price XGBoost",
          detail["ai_estimation"]["model"])

    print()
    print("=" * 60)
    print(f"RESULT: {PASSED} passed, {FAILED} failed")
    print("=" * 60)

    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()