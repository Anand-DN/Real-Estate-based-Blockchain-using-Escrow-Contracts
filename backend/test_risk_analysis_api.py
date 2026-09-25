"""
MILLOW - Transaction Risk & Anomaly Analysis API tests (Phase 5)

Run from project root:
    python backend/test_risk_analysis_api.py
"""

import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

import app  # noqa: E402
import properties  # noqa: E402
import risk_analysis  # noqa: E402

client = TestClient(app.app)

PASSED = 0
FAILED = 0

FORBIDDEN = [
    "fraud detected",
    "fraud probability",
    "safe transaction",
    "guaranteed safe",
    "guaranteed risky",
    "certified valuation",
    "certified detection",
    "undervalued",
    "overvalued",
    "good deal",
    "bad deal",
    "fair price",
    "model confidence",
]


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name}  {detail}")


def sample_id_for(city):
    rows = properties.PROPERTIES[
        properties.PROPERTIES["source_city"] == city
    ]
    assert not rows.empty, f"no MREID rows for {city}"
    return str(rows["mreid_id"].iloc[0])


def first_transaction_id():
    df = risk_analysis._load_risk()["df"]
    return str(df["txn_id"].iloc[0])


def assert_clean(name, *texts):
    joined = " ".join(texts).lower()
    hits = [w for w in FORBIDDEN if w in joined]
    check(f"clean wording [{name}]", not hits, str(hits))


def main():

    print("=" * 60)
    print("MILLOW RISK & ANOMALY ANALYSIS API TESTS (PHASE 5)")
    print("=" * 60)

    # ----------------------------------------------------------
    # Property-level endpoint (MREID) - valid ids per city
    # ----------------------------------------------------------

    scores = {}
    for city in [
        "Bangalore",
        "Chennai",
        "Delhi",
        "Hyderabad",
        "Kolkata",
        "Mumbai",
    ]:
        mid = sample_id_for(city)
        r = client.get(f"/api/properties/{mid}/risk-analysis")
        check(f"valid {city} property risk -> 200", r.status_code == 200, r.text)
        body = r.json()
        scores[city] = body["anomaly_score"]

        ok = (
            "mreid_id" in body
            and "city" in body
            and "analysis_subject" in body
            and "anomaly_score" in body
            and "indicators" in body
            and "horizon_context" in body
            and "methodology" in body
            and "disclaimer" in body
        )
        check(f"{city} response schema", ok, str(body)[:300])

        check(
            f"{city}: score within 0-100",
            0 <= body["anomaly_score"] <= 100,
            body["anomaly_score"],
        )
        check(
            f"{city}: indicators non-empty and bounded",
            0 < len(body["indicators"]) <= risk_analysis.INDICATOR_COUNT_PROP,
            len(body["indicators"]),
        )
        for ind in body["indicators"]:
            ok = (
                {"id", "title", "dimension", "status", "severity_points"}
                <= set(ind)
                and 0 <= ind["severity_points"] <= 3
            )
            check(f"{city} indicator {ind['id']} shape", ok, str(ind)[:200])

    check(
        "anomaly score is deterministic",
        client.get(
            f"/api/properties/{sample_id_for('Bangalore')}/risk-analysis"
        ).json()["anomaly_score"] == scores["Bangalore"],
    )

    # ----------------------------------------------------------
    # Score genuinely varies (not constant)
    # ----------------------------------------------------------

    check(
        "property scores vary across cities",
        len(set(scores.values())) > 1,
        scores,
    )

    # ----------------------------------------------------------
    # Horizon context: mapping + labelled, NOT a score input
    # ----------------------------------------------------------

    bang = client.get(
        f"/api/properties/{sample_id_for('Bangalore')}/risk-analysis"
    ).json()
    hc = bang["horizon_context"]
    check(
        "Bangalore horizon context maps to Bengaluru",
        isinstance(hc, dict) and hc["city"] == "Bengaluru",
        hc,
    )
    check(
        "horizon context bounded values (n/props/n_txn)",
        hc["n_transactions"] > 0 and hc["n_properties"] > 0,
        hc,
    )
    check(
        "horizon context carries synthetic note",
        "synthetic_note" in hc and "procedurally generated" in hc["synthetic_note"],
        hc.get("synthetic_note"),
    )
    hc_score_used = any(
        i["id"] == "listing_price_gap" and "Horizon" in i["explanation"]
        for i in bang["indicators"]
    )
    check(
        "horizon data not fed into listing indicators",
        not hc_score_used,
        [i["explanation"] for i in bang["indicators"]],
    )

    check(
        "property methodology excludes Horizon from the score",
        "No valuation" in bang["methodology"],
        bang["methodology"],
    )

    # ----------------------------------------------------------
    # Transaction-level endpoint - valid + schema
    # ----------------------------------------------------------

    txn_id = first_transaction_id()
    r = client.get(f"/api/transactions/{txn_id}/risk-analysis")
    check("valid transaction risk -> 200", r.status_code == 200, r.text)
    body = r.json()

    ok = (
        "transaction_id" in body
        and "property_id" in body
        and "city" in body
        and "transaction_type" in body
        and "quarter" in body
        and "anomaly_score" in body
        and "indicators" in body
        and "comparable_group" in body
        and "methodology" in body
        and "disclaimer" in body
    )
    check("transaction response schema", ok, str(body)[:300])

    check(
        "transaction score within 0-100",
        0 <= body["anomaly_score"] <= 100,
        body["anomaly_score"],
    )
    check(
        "indicator count is exactly INDICATOR_COUNT_TRX",
        len(body["indicators"]) == risk_analysis.INDICATOR_COUNT_TRX,
        len(body["indicators"]),
    )

    ids = [i["id"] for i in body["indicators"]]
    for expected in [
        "comparable_price",
        "circle_ratio",
        "negotiation",
        "days_on_market",
        "listed_gap",
        "rapid_repeat_sale",
        "price_jump",
        "financing",
    ]:
        check(f"transaction has indicator {expected}", expected in ids, ids)

    for ind in body["indicators"]:
        ok = (
            0 <= ind["severity_points"] <= 3
            and ind["status"]
            in {"none", "low", "medium", "high"}
            and "explanation" in ind
            and bool(ind["explanation"])
        )
        check(f"transaction indicator {ind['id']} shape", ok, str(ind)[:200])
        assert_clean(f"txn indicator {ind['id']}", ind.get("explanation", ""))

    # ----------------------------------------------------------
    # Determinism / same payload on repeat call
    # ----------------------------------------------------------

    again = client.get(f"/api/transactions/{txn_id}/risk-analysis").json()
    check(
        "transaction analysis is deterministic",
        again == body,
        {"first": body["anomaly_score"], "again": again["anomaly_score"]},
    )

    # ----------------------------------------------------------
    # Some transaction in the sample set is flagged (signal present)
    # ----------------------------------------------------------

    flagged = 0
    seen = 0
    ids_all = risk_analysis._load_risk()["df"]["txn_id"].to_numpy()
    for i in range(0, len(ids_all), max(1, len(ids_all) // 200)):
        b = client.get(
            f"/api/transactions/{ids_all[i]}/risk-analysis"
        ).json()
        seen += 1
        if any(ind["severity_points"] > 0 for ind in b["indicators"]):
            flagged += 1
    check(
        "sampled transactions include flagged indicators",
        flagged > 0,
        f"{flagged}/{seen} flagged",
    )

    # ----------------------------------------------------------
    # 404s
    # ----------------------------------------------------------

    r = client.get("/api/properties/MREID_9999999/risk-analysis")
    check("unknown property id -> 404", r.status_code == 404, r.text)

    r = client.get("/api/transactions/TRX_does_not_exist/risk-analysis")
    check("unknown transaction id -> 404", r.status_code == 404, r.text)

    # ----------------------------------------------------------
    # Wording safety across full payloads
    # ----------------------------------------------------------

    full_prop = (
        str(bang)
        + bang["methodology"]
        + bang["disclaimer"]
        + (bang["horizon_context"]["synthetic_note"] if bang["horizon_context"] else "")
    )
    assert_clean("property payload", full_prop)

    full_txn = str(body) + body["methodology"] + body["disclaimer"]
    assert_clean("transaction payload", full_txn)

    # ----------------------------------------------------------
    # No internal paths / deterministic scoring source
    # ----------------------------------------------------------

    text = str(bang) + str(body)
    check(
        "no internal filesystem paths exposed",
        "C:" not in text and ROOT.as_posix() not in text,
        text[:200],
    )

    check(
        "transaction summary is sorted by id (searchsound)",
        bool(
            (
                risk_analysis._load_risk()["ids"][1:]
                >= risk_analysis._load_risk()["ids"][:-1]
            ).all()
        ),
    )

    # ----------------------------------------------------------
    print()
    print(f"RESULT: {PASSED} passed, {FAILED} failed")
    print("=" * 60)

    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())