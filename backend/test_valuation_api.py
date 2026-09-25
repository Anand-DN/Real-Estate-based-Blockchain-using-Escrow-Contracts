"""
MILLOW - Valuation API tests

Verifies the Phase 1 backend (backend/app.py) end to end using
FastAPI's TestClient, including real MREID properties.

Run from project root:
    python backend/test_valuation_api.py
"""

import os
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

import app  # noqa: E402
from predict_property_value import (  # noqa: E402
    predict_property,
    format_indian_price,
)

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
    print("MILLOW VALUATION API TESTS")
    print("=" * 60)

    # ----------------------------------------------------------
    # Root / health
    # ------------------------------------------------------------

    r = client.get("/")
    check("GET / returns service info", r.status_code == 200, r.text)
    check("GET / lists cities", len(r.json()["cities"]) == 6, r.text)

    r = client.get("/health")
    check("GET /health is ok", r.json()["status"] == "ok", r.text)

    r = client.get("/api/valuation/cities")
    cities = r.json()["cities"]
    check("GET cities includes Bangalore", "Bangalore" in cities, r.text)

    r = client.get("/api/valuation/model")
    body = r.json()
    check("GET model returns validation", "validation" in body, r.text)

    # ----------------------------------------------------------
    # Valid single-property requests
    # ------------------------------------------------------------

    cases = [
        ("Bangalore", "Horamavu", 930, 1, 4371000),
        ("Bangalore", "Begur", 1324, 3, 8700000),
        ("Bangalore", "Banashankari", 981, 2, 5900000),
        ("Mumbai", "Andheri West", 1200, 3, None),
        ("Delhi", "Greater Kailash", 2200, 4, None),
    ]

    for city, loc, area, beds, known_price in cases:
        payload = {
            "city": city,
            "location": loc,
            "area_sqft": area,
            "bedrooms": beds,
        }
        r = client.post("/api/valuation", json=payload)
        check(
            f"POST /api/valuation {city}/{loc}",
            r.status_code == 200,
            r.text,
        )
        if r.status_code != 200:
            continue
        body = r.json()
        check(
            f"  returns formatted price ({loc})",
            body["estimated_price_formatted"].startswith("₹"),
            body["estimated_price_formatted"],
        )
        check(
            f"  estimated_price > 0 ({loc})",
            body["estimated_price"] > 0,
            str(body["estimated_price"]),
        )
        check(
            f"  price/sqft > 0 ({loc})",
            body["estimated_price_per_sqft"] > 0,
            str(body["estimated_price_per_sqft"]),
        )
        check(
            f"  disclaimer present ({loc})",
            "AI-assisted" in body["disclaimer"],
            body["disclaimer"][:60],
        )
        if known_price:
            ratio = (
                abs(body["estimated_price"] - known_price)
                / known_price
            )
            check(
                f"  plausible vs known MREID price ({loc}, "
                f"ratio={ratio:.2f})",
                ratio < 3.0,
                str(body["estimated_price"]),
            )

    # ----------------------------------------------------------
    # Amenity-aware request
    # ------------------------------------------------------------

    r = client.post(
        "/api/valuation",
        json={
            "city": "Bangalore",
            "location": "Begur",
            "area_sqft": 1324,
            "bedrooms": 3,
            "amenities": {"ac": 1, "gymnasium": 1, "swimmingpool": 0},
        },
    )
    check(
        "amenity-rich request accepted",
        r.status_code == 200,
        r.text,
    )

    # ----------------------------------------------------------
    # Parity with the CLI prediction path
    # ------------------------------------------------------------

    direct = predict_property(
        city="Bangalore",
        location="Begur",
        area=1324,
        bedrooms=3,
        amenities={"ac": 1},
    )
    api = client.post(
        "/api/valuation",
        json={
            "city": "Bangalore",
            "location": "Begur",
            "area_sqft": 1324,
            "bedrooms": 3,
            "amenities": {"ac": 1},
        },
    ).json()
    check(
        "API == CLI prediction",
        abs(api["estimated_price"] - direct["estimated_price"]) < 1.0,
        f"api={api['estimated_price']} cli={direct['estimated_price']}",
    )

    # ----------------------------------------------------------
    # Formatting helper
    # ------------------------------------------------------------

    check(
        "format 7420099 -> '₹74.20 Lakh'",
        format_indian_price(7420099) == "₹74.20 Lakh",
        format_indian_price(7420099),
    )

    # ----------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------

    r = client.post(
        "/api/valuation",
        json={"city": "", "location": "Begur", "area_sqft": 1, "bedrooms": 1},
    )
    check("empty city rejected (422)", r.status_code == 422, r.text)

    r = client.post(
        "/api/valuation",
        json={"city": "Paris", "location": "x", "area_sqft": 100, "bedrooms": 2},
    )
    check(
        "unsupported city rejected (400)",
        r.status_code == 400,
        r.text,
    )

    r = client.post(
        "/api/valuation",
        json={"city": "Bangalore", "location": "Begur", "area_sqft": 0, "bedrooms": 2},
    )
    check("area 0 rejected (422)", r.status_code == 422, r.text)

    r = client.post(
        "/api/valuation",
        json={"city": "Bangalore", "location": "Begur", "area_sqft": 100, "bedrooms": 0},
    )
    check("bedrooms 0 rejected (422)", r.status_code == 422, r.text)

    print()
    print("=" * 60)
    print(f"RESULT: {PASSED} passed, {FAILED} failed")
    print("=" * 60)

    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()