"""
MILLOW - Property Market Context (NHB) API tests (Phase 4)

Asserts GET /api/properties/{mreid_id}/market-context for backend/app.py.

Run from project root:
    python backend/test_market_context_api.py
"""

import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

import app  # noqa: E402
import market_data  # noqa: E402
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


def sample_id_for(city):
    """A real MREID id for a given city (deterministic first match)."""
    rows = properties.PROPERTIES[
        properties.PROPERTIES["source_city"] == city
    ]
    assert not rows.empty, f"no MREID rows for {city}"
    return str(rows["mreid_id"].iloc[0])


def require_fields(check_name, body):
    ok = (
        "mreid_id" in body
        and "city" in body
        and "nhb_city" in body
        and "property" in body
        and "millow_ai" in body
        and "market_context" in body
        and "disclaimer" in body
    )
    check(check_name, ok, f"missing top-level keys in {sorted(body)}")

    mctx = body.get("market_context", {})
    for field in [
        "source",
        "source_url",
        "series",
        "access_period",
        "benchmark_level",
        "city",
        "quarter",
        "observation_period",
        "composite_price_rs_sqft",
        "price_le_60sqm_rs_sqft",
        "price_60_110sqm_rs_sqft",
        "price_gt_110sqm_rs_sqft",
        "hpi_assessment",
        "note",
    ]:
        ok = ok and field in mctx

    for field in [
        "listed_price",
        "area_sqft",
        "listed_price_per_sqft",
    ]:
        ok = ok and field in body.get("property", {})

    for field in [
        "ai_estimated_price",
        "ai_estimated_price_per_sqft",
        "model",
    ]:
        ok = ok and field in body.get("millow_ai", {})

    check(check_name + " -> full schema", ok, str(body)[:400])


def main():

    print("=" * 60)
    print("MILLOW MARKET CONTEXT API TESTS (PHASE 4)")
    print("=" * 60)

    # ----------------------------------------------------------
    # Per-city valid properties
    # ----------------------------------------------------------

    for city in [
        "Bangalore",
        "Chennai",
        "Delhi",
        "Hyderabad",
        "Kolkata",
        "Mumbai",
    ]:
        mid = sample_id_for(city)
        r = client.get(f"/api/properties/{mid}/market-context")
        check(f"valid {city} property -> 200", r.status_code == 200, r.text)
        body = r.json()
        require_fields(f"{city} response schema", body)
        check(
            f"{city}: property city echoed",
            body["city"] == city,
            body,
        )

    # ----------------------------------------------------------
    # Bangalore -> Bengaluru mapping
    # ----------------------------------------------------------

    bang = client.get(
        f"/api/properties/{sample_id_for('Bangalore')}/market-context"
    ).json()
    check(
        "Bangalore -> NHB 'Bengaluru' mapping",
        bang["nhb_city"] == "Bengaluru"
        and bang["market_context"]["city"] == "Bengaluru",
        bang,
    )
    check(
        "nhb_city_for('Bangalore') == 'Bengaluru'",
        market_data.nhb_city_for("Bangalore") == "Bengaluru",
    )
    check(
        "identity mapping kept for Chennai",
        market_data.nhb_city_for("Chennai") == "Chennai",
    )

    # ----------------------------------------------------------
    # Unknown id -> 404
    # ----------------------------------------------------------

    r = client.get("/api/properties/MREID_9999999/market-context")
    check("unknown property id -> 404", r.status_code == 404, r.text)

    # ----------------------------------------------------------
    # Quarter is an actual NHB observation for the city
    # ----------------------------------------------------------

    qs = set(market_data.NHB.loc[
        market_data.NHB["city"] == "Bengaluru", "quarter"
    ].tolist())
    check(
        "Bangalore quarter from actual NHB observations",
        bang["market_context"]["quarter"] in qs,
        bang["market_context"]["quarter"],
    )

    # ----------------------------------------------------------
    # NHB values come from the CSV (not fabricated)
    # ----------------------------------------------------------

    expected = market_data.latest_observation("Bengaluru")
    mc = bang["market_context"]
    matches = True
    for key in [
        "composite_price_rs_sqft",
        "price_le_60sqm_rs_sqft",
        "price_60_110sqm_rs_sqft",
        "price_gt_110sqm_rs_sqft",
        "hpi_assessment",
        "quarter",
    ]:
        matches = matches and mc[key] == expected[key]
    check(
        "NHB values match CSV-derived latest observation",
        matches,
        {"endpoint": mc, "csv": {k: expected[k] for k in [
            "composite_price_rs_sqft", "quarter",
        ]}},
    )
    check(
        "NHB source metadata present",
        mc["source"] == "NHB RESIDEX"
        and mc["source_url"].startswith("http")
        and bool(mc["series"])
        and bool(mc["access_period"]),
        mc,
    )

    # ----------------------------------------------------------
    # Observation period / quarter surfaced
    # ----------------------------------------------------------

    check(
        "observation period surfaced",
        mc["observation_period"] == f"Quarter ending {mc['quarter']}",
        mc,
    )

    # ----------------------------------------------------------
    # NHB must never feed the valuation model
    # ----------------------------------------------------------

    model_leaks = {
        name
        for name in ("model", "preprocessor", "create_features")
        if name in vars(market_data)
    }
    check(
        "market_data module has no model/preprocessor imports",
        not model_leaks,
        str(model_leaks),
    )

    detail = client.get(
        f"/api/properties/{sample_id_for('Bangalore')}"
    ).json()
    check(
        "AI estimate unchanged (identical to detail endpoint)",
        float(bang["millow_ai"]["ai_estimated_price_per_sqft"])
        == float(detail["ai_estimation"]["ai_estimated_price_per_sqft"]),
        {
            "context": bang["millow_ai"]["ai_estimated_price_per_sqft"],
            "detail": detail["ai_estimation"]["ai_estimated_price_per_sqft"],
        },
    )

    # ----------------------------------------------------------
    # Encoding / no internal paths
    # ----------------------------------------------------------

    check(
        "response is UTF-8 safe (rupee symbol round-trips)",
        "₹" in bang["property"]["listed_price_formatted"],
        bang["property"]["listed_price_formatted"],
    )

    text = r.text if False else str(bang)
    check(
        "no internal filesystem paths exposed",
        "C:" not in text and ROOT.as_posix() not in text and "\\" not in text,
        text[:200],
    )

    # ----------------------------------------------------------
    print()
    print(f"RESULT: {PASSED} passed, {FAILED} failed")
    print("=" * 60)

    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())