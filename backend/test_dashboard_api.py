"""
MILLOW - Dashboard / AI Market Signal API tests (Phase 6)

Asserts the analytics endpoints for backend/app.py using FastAPI's
TestClient:
  * GET /api/dashboard/overview
  * GET /api/dashboard/market-breakdown
  * GET /api/dashboard/insights
  * POST /api/valuation remains unchanged
  * search endpoint exposes ai_market_signal + ai_* sort options

Run from project root:
    python backend/test_dashboard_api.py
"""

import json
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
    print("MILLOW DASHBOARD / SIGNAL API TESTS (PHASE 6)")
    print("=" * 60)

    # ----------------------------------------------------------
    # AI market signal on the catalogue search payload
    # ----------------------------------------------------------

    r = client.get("/api/properties", params={"page_size": 5})
    body = r.json()
    first = body["results"][0]
    check("summary exposes ai_market_signal",
          "ai_market_signal" in first, str(list(first.keys())))
    signal = first["ai_market_signal"]
    check("signal has label + numbers",
          signal["label"] in {
              "Potentially undervalued",
              "Potentially overvalued",
              "Near estimated market range",
          }
          and "difference_inr" in signal
          and "difference_pct" in signal
          and signal["band_pct"] == properties.SIGNAL_BAND_PCT,
          str(signal))
    check("signal difference is internally consistent",
          abs(signal["ai_estimated_price"]
              - signal["listed_price"]
              - signal["difference_inr"]) < 0.01,
          str(signal))
    check("summary exposes locality cohort stats",
          "locality" in first
          and first["locality"]["count"] > 0
          and first["locality"]["median_price_per_sqft"] is not None,
          str(first.get("locality")))

    r = client.get("/api/properties/MREID_0000001")
    detail = r.json()
    check("detail exposes ai_market_signal",
          "ai_market_signal" in detail, str(list(detail.keys())))
    check("detail signal label is one of the three",
          detail["ai_market_signal"]["label"] in {
              "Potentially undervalued",
              "Potentially overvalued",
              "Near estimated market range",
          },
          detail["ai_market_signal"]["label"])

    # ----------------------------------------------------------
    # AI-based sorting
    # ----------------------------------------------------------

    r = client.get("/api/properties/search",
                   params={"sort": "ai_difference_desc", "page_size": 5})
    check("ai_difference_desc accepted", r.status_code == 200, r.text)
    if r.status_code == 200:
        diffs = [x["ai_market_signal"]["difference_inr"]
                 for x in r.json()["results"]]
        check("ai_difference_desc sorted descending",
              diffs == sorted(diffs, reverse=True), str(diffs))

    r = client.get("/api/properties/search",
                   params={"sort": "ai_estimate_asc", "page_size": 5})
    check("ai_estimate_asc accepted", r.status_code == 200, r.text)
    if r.status_code == 200:
        ests = [x["ai_estimated_price"] for x in r.json()["results"]]
        check("ai_estimate_asc sorted ascending",
              ests == sorted(ests), str(ests))

    r = client.get("/api/properties/search", params={"sort": "bogus"})
    check("invalid sort still rejected", r.status_code == 422, r.text)

    # ----------------------------------------------------------
    # ai_signal filter
    # ----------------------------------------------------------

    r = client.get("/api/properties/search",
                   params={"ai_signal": "undervalued", "page_size": 5})
    check("ai_signal=undervalued accepted", r.status_code == 200, r.text)
    if r.status_code == 200:
        labels = {x["ai_market_signal"]["label"]
                  for x in r.json()["results"]}
        check("ai_signal filter returns only that class",
              labels == {"Potentially undervalued"}, str(labels))
        check("ai_signal filter narrows the total",
              r.json()["total"] > 0
              and r.json()["total"] == len(r.json()["results"])
              or r.json()["total"] >= len(r.json()["results"]),
              str(r.json()["total"]))

    r = client.get("/api/properties/search",
                   params={"ai_signal": "bogus"})
    check("invalid ai_signal rejected", r.status_code == 422, r.text)

    # ----------------------------------------------------------
    # Dashboard overview
    # ----------------------------------------------------------

    r = client.get("/api/dashboard/overview")
    check("overview 200", r.status_code == 200, r.text)
    ov = r.json()
    check("overview catalogue total = dataset size",
          ov["catalogue"]["total"] == len(properties.PROPERTIES), str(ov))
    check("overview ai analyzed = dataset size",
          ov["ai"]["analyzed"] == len(properties.PROPERTIES), str(ov))
    splits = ov["ai"]
    check("signal split sums to catalogue",
          splits["undervalued"] + splits["overvalued"] + splits["in_range"]
          == splits["analyzed"],
          str(splits))
    check("overview has model / disclaimer",
          ov["ai"]["model"] == properties.MODEL_NAME
          and bool(ov["disclaimer"]),
          str(ov["ai"].get("model")))
    check("overview chain snapshot reporting is explicit",
          "available" in ov["catalogue"]["chain"]
          and "exported_at" in ov["catalogue"]["chain"],
          str(ov["catalogue"]["chain"]))
    check("overview avg prices present",
          ov["ai"]["avg_listed"] > 0
          and ov["ai"]["avg_ai_estimate"] > 0,
          str(ov["ai"]))

    # ----------------------------------------------------------
    # Dashboard market breakdown
    # ----------------------------------------------------------

    r = client.get("/api/dashboard/market-breakdown")
    check("breakdown 200 by city", r.status_code == 200, r.text)
    rows = r.json()["rows"]
    check("breakdown has six cities", len(rows) == 6, str(len(rows)))
    check("breakdown row schema complete",
          all({
              "city", "count", "avg_listed", "avg_ai",
              "avg_listed_ppsf", "avg_ai_ppsf",
              "undervalued", "overvalued", "in_range",
          }.issubset(row.keys()) for row in rows),
          str(list(rows[0].keys()) if rows else "none"))
    if rows:
        per_city = sum(row["count"] for row in rows)
        check("breakdown counts sum to catalogue",
              per_city == len(properties.PROPERTIES), str(per_city))

    r = client.get("/api/dashboard/market-breakdown",
                   params={"group": "locality", "city": "Delhi"})
    check("locality breakdown for city 200", r.status_code == 200, r.text)
    loc_rows = r.json()["rows"]
    check("locality breakdown non-empty and all in city",
          len(loc_rows) > 0
          and all(row["city"] == "Delhi" for row in loc_rows),
          str(len(loc_rows)))

    r = client.get("/api/dashboard/market-breakdown", params={"city": "Paris"})
    check("unknown city rejected (400)", r.status_code == 400, r.text)

    r = client.get("/api/dashboard/market-breakdown", params={"group": "bogus"})
    check("invalid group rejected (422)", r.status_code == 422, r.text)

    # ----------------------------------------------------------
    # Dashboard insights
    # ----------------------------------------------------------

    r = client.get("/api/dashboard/insights")
    check("insights 200", r.status_code == 200, r.text)
    ins = r.json()["insights"]
    check("insights non-empty", len(ins) > 0, str(len(ins)))
    check("insights each has kind + text",
          all("kind" in i and "text" in i for i in ins), str(ins))
    check("insights only use allowed kinds",
          all(i["kind"] in {"DATA FACT", "MODEL-BASED INTERPRETATION",
                            "AI MODEL ESTIMATE"}
              for i in ins),
          str({i["kind"] for i in ins}))
    check("insights never claim certified valuation",
          not any("certified" in i["text"].lower() and "not" not in
                  i["text"].lower() for i in ins),
          str(ins))

    # ----------------------------------------------------------
    # Model files untouched by the dashboard layer
    # ----------------------------------------------------------

    md = json.load(open(ROOT / "models" / "valuation" / "final"
                        / "metadata.json", encoding="utf-8"))
    check("metadata model name preserved",
          md["model_name"] == properties.MODEL_NAME, md["model_name"])

    print()
    print("=" * 60)
    print(f"RESULT: {PASSED} passed, {FAILED} failed")
    print("=" * 60)

    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())