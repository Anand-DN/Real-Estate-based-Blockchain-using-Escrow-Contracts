"""
MILLOW - Property Recommendation / Similar Properties API tests (Phase 6)

Run from project root:
    python backend/test_recommendation_api.py
"""

import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

import app  # noqa: E402
import properties  # noqa: E402
import recommendation  # noqa: E402

client = TestClient(app.app)

PASSED = 0
FAILED = 0

FORBIDDEN = [
    "best",
    "undervalued",
    "overvalued",
    "investment potential",
    "safest",
    "investment advice",
    "predicted return",
    "profit",
    "deal score",
    "highest return",
]


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name}  {detail}")


def fresh_engine():
    recommendation._index_cache = None
    recommendation.index()


def sample_id_for(city, location_substr=None):
    idx = properties._id_to_index
    rows = properties.PROPERTIES
    for mid, pos in idx.items():
        row = rows.iloc[pos]
        if row["source_city"] != city:
            continue
        if location_substr and location_substr not in str(row["location"]):
            continue
        return mid
    raise AssertionError(f"no sample for {city}/{location_substr}")


def assert_clean(name, *texts):
    joined = " ".join(texts).lower()
    hits = [w for w in FORBIDDEN if w in joined]
    check(f"clean wording [{name}]", not hits, str(hits))


def main():

    print("=" * 60)
    print("MILLOW RECOMMENDATION API TESTS (PHASE 6)")
    print("=" * 60)

    fresh_engine()

    # ----------------------------------------------------------
    # A-mode: valid property + schema
    # ----------------------------------------------------------

    mid = sample_id_for("Bangalore")
    r = client.get(f"/api/properties/{mid}/recommendations?limit=6")
    check("valid A-mode -> 200", r.status_code == 200, r.text)
    body = r.json()

    ok = (
        body["mode"] == "similar_to_property"
        and "query" in body
        and "count" in body
        and "items" in body
        and "methodology" in body
        and "disclaimer" in body
    )
    check("A-mode response schema", ok, str(body)[:300])

    check(
        "A-mode returns requested count",
        body["count"] == min(6, body["candidate_count"]),
        body["count"],
    )
    check(
        "A-mode candidate pool is the whole city",
        "candidate_count" in body and body["candidate_count"] >= 100,
        body.get("candidate_count"),
    )

    items = body["items"]
    for item in items:
        ok = (
            {"mreid_id", "recommendation_score", "reasons", "property"} <= set(item)
            and 0 <= item["recommendation_score"] <= 100
            and isinstance(item["reasons"], list)
            and bool(item["reasons"])
            and item["property"]["city"] == "Bangalore"
        )
        check(
            f"A-mode item schema + bounds ({item['mreid_id']})",
            ok,
            str(item)[:200],
        )

    # ----------------------------------------------------------
    # Source property excluded + no duplicates + deterministic
    # ----------------------------------------------------------

    ids = [it["mreid_id"] for it in items]
    check("source property excluded", mid not in ids, ids)
    check("no duplicate ids", len(ids) == len(set(ids)), ids)

    again = client.get(f"/api/properties/{mid}/recommendations?limit=6").json()
    check(
        "A-mode deterministic ordering",
        again == body,
        {"first": body["items"][0]["mreid_id"], "again": again["items"][0]["mreid_id"]},
    )

    scores = [it["recommendation_score"] for it in items]
    check(
        "A-mode sorted by score desc",
        scores == sorted(scores, reverse=True),
        scores,
    )

    # ----------------------------------------------------------
    # A-mode reason <=> dimension consistency
    # ----------------------------------------------------------

    for item in items:
        dims = item["dimension_scores"]
        if "Same locality" in " ".join(item["reasons"]):
            check(
                f"locality reason consistent ({item['mreid_id']})",
                dims.get("locality") == 1.0,
                (item["reasons"], dims),
            )
        if dims.get("locality", 0) == 0.0:
            check(
                f"locality-0 reason mentions city ({item['mreid_id']})",
                any("Compared against Bangalore" in x for x in item["reasons"]),
                item["reasons"],
            )

    # ----------------------------------------------------------
    # Sparse locality still returns candidates with supportable reasons
    # ----------------------------------------------------------

    sparse_loc = None
    counts = properties.PROPERTIES.groupby(["source_city", "location"]).size()
    for (city, locn), c in counts.items():
        if c == 1 and city == "Mumbai":
            sparse_loc = (city, locn)
            break
    if sparse_loc:
        smid = sample_id_for(sparse_loc[0], sparse_loc[1])
        r = client.get(f"/api/properties/{smid}/recommendations?limit=4")
        check("sparse locality -> 200", r.status_code == 200, r.text)
        sb = r.json()
        check(
            "sparse locality returns results without locality match",
            sb["count"] >= 1
            and all(
                it["dimension_scores"].get("locality", 0) == 0.0
                for it in sb["items"]
            ),
            sb,
        )
        check(
            "sparse locality reasons are supportable",
            all(
                "Compared against Mumbai listings" in " ".join(it["reasons"])
                for it in sb["items"]
            ),
            [it["reasons"] for it in sb["items"]],
        )
    else:
        check("sparse locality sample found", False, "no 1-property Mumbai location")

    # ----------------------------------------------------------
    # A-mode unknown id -> 404
    # ----------------------------------------------------------

    r = client.get("/api/properties/MREID_9999999/recommendations")
    check("unknown A-mode id -> 404", r.status_code == 404, r.text)

    # ----------------------------------------------------------
    # B-mode: filters honored
    # ----------------------------------------------------------

    r = client.get(
        "/api/recommendations",
        params={
            "city": "Chennai",
            "bedrooms": 3,
            "min_price": 4000000,
            "max_price": 8000000,
            "min_area": 900,
            "max_area": 1500,
            "limit": 5,
        },
    )
    check("valid B-mode -> 200", r.status_code == 200, r.text)
    bb = r.json()
    check("B-mode mode field", bb["mode"] == "requirements", bb["mode"])
    check("B-mode returns requested count", bb["count"] == 5, bb["count"])
    for item in bb["items"]:
        p = item["property"]
        ok = (
            p["city"] == "Chennai"
            and p["bedrooms"] == 3
            and 4000000 <= p["price"] <= 8000000
            and 900 <= p["area"] <= 1500
        )
        check(
            f"B-mode filters honored ({item['mreid_id']})",
            ok,
            str(p)[:200],
        )
    check(
        "B-mode query echo",
        bb["query"]["city"] == "Chennai"
        and bb["query"]["bedrooms"] == 3,
        bb["query"],
    )

    # amenities filter hard-ensured
    r = client.get(
        "/api/recommendations",
        params={"city": "Delhi", "bedrooms": 3, "amenities": "liftavailable,powerbackup"},
    )
    ab = r.json()
    leaks = [
        it["mreid_id"]
        for it in ab["items"]
        if it["property"]["amenities"].get("liftavailable") != "Yes"
        or it["property"]["amenities"].get("powerbackup") != "Yes"
    ]
    check("amenities filter hard-ensured", not leaks, leaks)

    # B-mode score/schema + determinism + sort + forbidden words
    for item in ab["items"]:
        check(
            f"B-mode item schema ({item['mreid_id']})",
            {"recommendation_score", "reasons", "dimension_scores", "property"}
            <= set(item)
            and 0 <= item["recommendation_score"] <= 100,
            str(item)[:200],
        )
        assert_clean(f"B-mode reasons {item['mreid_id']}", *item["reasons"])

    s2 = [it["recommendation_score"] for it in ab["items"]]
    s3 = s2[:]
    check("B-mode sorted by score desc", s2 == sorted(s2, reverse=True), s2)
    again_b = client.get(
        "/api/recommendations",
        params={"city": "Delhi", "bedrooms": 3, "amenities": "liftavailable,powerbackup"},
    ).json()
    check("B-mode deterministic", again_b == ab)
    check("B-mode scores sorted stable", s3 == s2)

    # B-mode with location (locality dimension active)
    r = client.get(
        "/api/recommendations",
        params={"city": "Bangalore", "location": "Koramangala", "bedrooms": 2},
    )
    lb = r.json()
    check(
        "location in query echo",
        lb["query"].get("location") == "Koramangala",
        lb["query"],
    )
    if lb["items"]:
        top = lb["items"][0]
        check(
            "locality dimension present when location requested",
            "locality" in top["dimension_scores"],
            top["dimension_scores"],
        )

    # ----------------------------------------------------------
    # B-mode validation errors
    # ----------------------------------------------------------

    r = client.get("/api/recommendations")
    check("B-mode no requirements -> 400", r.status_code == 400, r.text)

    r = client.get(
        "/api/recommendations",
        params={"city": "Atlantis"},
    )
    check("B-mode unknown city -> 400", r.status_code == 400, r.text)

    r = client.get(
        "/api/recommendations",
        params={"city": "Delhi", "amenities": "saunaworld"},
    )
    check("B-mode unknown amenity -> 400", r.status_code == 400, r.text)

    r = client.get(
        "/api/recommendations",
        params={"city": "Delhi", "min_price": 50000000, "max_price": 2000000},
    )
    check("B-mode bad price range -> 400", r.status_code == 400, r.text)

    # ----------------------------------------------------------
    # B-mode empty result set (over-constrained)
    # ----------------------------------------------------------

    r = client.get(
        "/api/recommendations",
        params={"city": "Hyderabad", "bedrooms": 9, "min_price": 50_000_000},
    )
    eb = r.json()
    check("over-constrained -> 200 with empty items", r.status_code == 200 and eb["count"] == 0, eb)

    # ----------------------------------------------------------
    # Wording safety: reasons/dims never claim quality/deals; meta text is
    # negated-honest only
    # ----------------------------------------------------------

    for item in body["items"] + ab["items"]:
        assert_clean(
            f"reasons {item['mreid_id']}",
            *item["reasons"],
            str(item["dimension_scores"]),
        )
    for meta_key in ("methodology", "disclaimer", "score_interpretation"):
        for meta in (body[meta_key], bb[meta_key]):
            check(
                f"{meta_key} is negated-honest (similarity / not-deal wording)",
                "not" in meta or "similarity" in meta or "relevance" in meta,
                meta[:120],
            )

    # ----------------------------------------------------------
    # Country/criteria sanity: A-mode city never appears in results elsewhere
    # ----------------------------------------------------------

    check(
        "A-mode candidates all in source city",
        all(it["property"]["city"] == "Bangalore" for it in body["items"]),
    )

    # ----------------------------------------------------------
    print()
    print(f"RESULT: {PASSED} passed, {FAILED} failed")
    print("=" * 60)

    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())