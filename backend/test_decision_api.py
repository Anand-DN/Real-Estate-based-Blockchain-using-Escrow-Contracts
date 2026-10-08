"""
MILLOW - Decision Policy (v1.0) tests

Validation plan covered here:

    V1  rule table + frozen precedence (R1 > R2 > ... > R8)
    V2  threshold boundaries - existing production constants only
    V3  degradation - missing evidence never silently becomes PROCEED
    V4  consistency / contradiction invariants
    V5  workflow can never change the decision
    V6  forbidden language + exact boundary statement
    V7  API contract, provenance, purity and read-only guarantees

Run from project root:
    python backend/test_decision_api.py
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

import app  # noqa: E402
import chain_index  # noqa: E402
import decision_context as dc  # noqa: E402
import decision_policy as dp  # noqa: E402
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

BOUNDARY_TEXT = (
    "This panel can prevent the MILLOW application from advancing the "
    "workflow. It cannot prevent a direct wallet transaction to "
    "MillowEscrow, and an on-chain execution does not by itself certify "
    "legality, title or valuation."
)

STATUS_BY_POINTS = {0: "none", 1: "low", 2: "medium", 3: "high"}


# ============================================================
# HARNESS
# ============================================================

def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name}  {detail}")


def section(title):
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


def assert_clean(name, *texts):
    joined = " ".join(str(t) for t in texts).lower()
    hits = [w for w in FORBIDDEN if w in joined]
    check(f"clean wording [{name}]", not hits, str(hits))


def codes(result):
    return [r["code"] for r in result["reasons"]]


def ind(ind_id, points, dimension="pricing", z=None):
    return {
        "id": ind_id,
        "title": ind_id,
        "dimension": dimension,
        "status": STATUS_BY_POINTS[points],
        "direction": "n/a",
        "z": z,
        "explanation": "fixture",
        "severity_points": points,
    }


def make_signals(
    indicators=(),
    diff=0.0,
    band=5.0,
    listed=1000000.0,
    use_risk=True,
    use_chain=True,
    label="Near estimated market range",
    mreid="MREID-TEST",
    ai_price=None,
    ai_signal=True,
):
    if ai_price is None and isinstance(listed, (int, float)) and listed:
        ai_price = listed * (1.0 + diff / 100.0)
    signal = None
    if ai_signal:
        signal = {
            "label": label,
            "listed_price": listed,
            "ai_estimated_price": ai_price,
            "difference_pct": diff,
            "band_pct": band,
            "note": "fixture",
        }
    chain = None
    if use_chain:
        chain = {
            "tokenized": True,
            "token_id": 1,
            "owner": "0x0000000000000000000000000000000000000001",
            "listed": True,
            "active_sale": False,
            "finalized": False,
            "sale_status": "None",
            "available": True,
            "exported_at": "2026-09-26T16:23:37.384Z",
        }
    return {
        "mreid_id": mreid,
        "listed_price": listed,
        "ai_estimated_price": ai_price,
        "ai_market_signal": signal,
        "risk": {"mreid_id": mreid, "indicators": list(indicators)}
        if use_risk else None,
        "chain": chain,
    }


def sample_id():
    return str(properties.PROPERTIES["mreid_id"].iloc[0])


# ============================================================
# V1 - RULE TABLE + FROZEN PRECEDENCE
# ============================================================

def v1_rule_table():
    section("V1 - rule table and frozen precedence")

    check("PROCEED state", dp.PROCEED == "PROCEED")
    check("REVIEW_REQUIRED state", dp.REVIEW_REQUIRED == "REVIEW_REQUIRED")
    check("ENHANCED_REVIEW state", dp.ENHANCED_REVIEW == "ENHANCED_REVIEW")
    check("HOLD state", dp.HOLD == "HOLD")
    check("exactly four decision states", len(dp.DECISIONS) == 4, dp.DECISIONS)

    check(
        "precedence order frozen",
        dp.POLICY_CONFIG["rule_precedence"]
        == ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"],
        dp.POLICY_CONFIG["rule_precedence"],
    )
    check(
        "rule decisions frozen",
        dp.RULE_DECISION == {
            "R1": "HOLD",
            "R2": "ENHANCED_REVIEW",
            "R3": "REVIEW_REQUIRED",
            "R4": "ENHANCED_REVIEW",
            "R5": "ENHANCED_REVIEW",
            "R6": "REVIEW_REQUIRED",
            "R7": "REVIEW_REQUIRED",
            "R8": "PROCEED",
        },
        dp.RULE_DECISION,
    )
    check(
        "rule severities frozen",
        dp.RULE_SEVERITY == {
            "R1": "hold",
            "R2": "enhanced",
            "R3": "review",
            "R4": "enhanced",
            "R5": "enhanced",
            "R6": "review",
            "R7": "review",
            "R8": "info",
        },
        dp.RULE_SEVERITY,
    )

    # ---- individual rules -------------------------------------------------
    r = dp.evaluate(make_signals([ind("listing_price_gap", 3)]))
    check("R1 high severity price -> HOLD", r["decision"] == dp.HOLD, r["decision"])
    check("R1 reason", "PRICE_HIGH_SEVERITY" in codes(r), codes(r))
    check(
        "R1 severity hold",
        [x["severity"] for x in r["reasons"]][0] == "hold",
        r["reasons"],
    )

    r = dp.evaluate(
        make_signals([ind("bedroom_area", 3, "structure")])
    )
    check("R2 high severity structure -> ENHANCED_REVIEW",
          r["decision"] == dp.ENHANCED_REVIEW, r["decision"])
    check("R2 reason", "STRUCTURE_HIGH_SEVERITY" in codes(r), codes(r))

    r = dp.evaluate(
        make_signals([ind("amenity_disclosure", 3, "records")])
    )
    check("R3 high severity disclosure -> REVIEW_REQUIRED",
          r["decision"] == dp.REVIEW_REQUIRED, r["decision"])
    check("R3 reason", "DISCLOSURE_HIGH_SEVERITY" in codes(r), codes(r))

    r = dp.evaluate(
        make_signals([ind("listing_price_gap", 2)], diff=6.0)
    )
    check("R4 elevated + outside band -> ENHANCED_REVIEW",
          r["decision"] == dp.ENHANCED_REVIEW, r["decision"])
    check("R4 reason", "ELEVATED_RISK_WITH_PRICE_DEVIATION" in codes(r), codes(r))

    r = dp.evaluate(
        make_signals([ind("listing_price_gap", 2), ind("bedroom_area", 2, "structure")])
    )
    check("R5 two elevated -> ENHANCED_REVIEW",
          r["decision"] == dp.ENHANCED_REVIEW, r["decision"])
    check("R5 reason", "MULTIPLE_ELEVATED_INDICATORS" in codes(r), codes(r))

    r = dp.evaluate(make_signals([ind("amenity_disclosure", 2, "records")]))
    check("R6 one elevated -> REVIEW_REQUIRED",
          r["decision"] == dp.REVIEW_REQUIRED, r["decision"])
    check("R6 reason", "SINGLE_ELEVATED_INDICATOR" in codes(r), codes(r))

    r = dp.evaluate(make_signals([], diff=6.0))
    check("R7 outside band -> REVIEW_REQUIRED",
          r["decision"] == dp.REVIEW_REQUIRED, r["decision"])
    check("R7 reason", "PRICE_OUTSIDE_BAND" in codes(r), codes(r))

    r = dp.evaluate(make_signals([]))
    check("R8 clean -> PROCEED", r["decision"] == dp.PROCEED, r["decision"])
    check("R8 reason",
          "WITHIN_COMPARISON_BAND_NO_ELEVATED_INDICATORS" in codes(r), codes(r))

    # ---- lower precedence rules still reported ---------------------------
    r = dp.evaluate(make_signals([ind("listing_price_gap", 3)], diff=6.0))
    c = codes(r)
    check("lower precedence rules retained",
          "PRICE_HIGH_SEVERITY" in c and "SINGLE_ELEVATED_INDICATOR" in c
          and "PRICE_OUTSIDE_BAND" in c, c)
    check(
        "precedence R1 > R6 > R7",
        c.index("PRICE_HIGH_SEVERITY")
        < c.index("SINGLE_ELEVATED_INDICATOR")
        < c.index("PRICE_OUTSIDE_BAND"),
        c,
    )
    check("R1 decision wins", r["decision"] == dp.HOLD, r["decision"])

    # ---- precedence P1: all three high-severity rules --------------------
    r = dp.evaluate(make_signals([
        ind("amenity_disclosure", 3, "records"),
        ind("listing_price_gap", 3),
        ind("bedroom_area", 3, "structure"),
    ], diff=6.0))
    c = codes(r)
    check("P1 all three high severity -> HOLD",
          r["decision"] == dp.HOLD, r["decision"])
    check(
        "P1 order R1 < R2 < R3 < R4 < R5 < R7",
        c.index("PRICE_HIGH_SEVERITY")
        < c.index("STRUCTURE_HIGH_SEVERITY")
        < c.index("DISCLOSURE_HIGH_SEVERITY")
        < c.index("ELEVATED_RISK_WITH_PRICE_DEVIATION")
        < c.index("MULTIPLE_ELEVATED_INDICATORS")
        < c.index("PRICE_OUTSIDE_BAND"),
        c,
    )

    # ---- precedence P2: R3 (review) beats R4 (enhanced) ------------------
    r = dp.evaluate(make_signals([
        ind("amenity_disclosure", 3, "records"),
        ind("listing_price_gap", 2),
    ], diff=6.0))
    c = codes(r)
    check("P2 R3 beats R4 decision",
          r["decision"] == dp.REVIEW_REQUIRED, r["decision"])
    check("P2 reason order", c.index("DISCLOSURE_HIGH_SEVERITY")
          < c.index("ELEVATED_RISK_WITH_PRICE_DEVIATION"), c)

    # ---- precedence P3: R4 beats R6/R7 ----------------------------------
    r = dp.evaluate(make_signals([ind("listing_price_gap", 2)], diff=6.0))
    c = codes(r)
    check("P3 R4 beats R6/R7 decision",
          r["decision"] == dp.ENHANCED_REVIEW, r["decision"])
    check("P3 reason order", c.index("ELEVATED_RISK_WITH_PRICE_DEVIATION")
          < c.index("SINGLE_ELEVATED_INDICATOR")
          < c.index("PRICE_OUTSIDE_BAND"), c)

    # ---- first match decides --------------------------------------------
    code_to_rule = {
        "PRICE_HIGH_SEVERITY": "R1",
        "STRUCTURE_HIGH_SEVERITY": "R2",
        "DISCLOSURE_HIGH_SEVERITY": "R3",
        "ELEVATED_RISK_WITH_PRICE_DEVIATION": "R4",
        "MULTIPLE_ELEVATED_INDICATORS": "R5",
        "SINGLE_ELEVATED_INDICATOR": "R6",
        "PRICE_OUTSIDE_BAND": "R7",
        "WITHIN_COMPARISON_BAND_NO_ELEVATED_INDICATORS": "R8",
    }
    firsts = set()
    fixtures = [
        [ind("listing_price_gap", 3)],
        [ind("bedroom_area", 3, "structure")],
        [ind("amenity_disclosure", 3, "records")],
        [ind("listing_price_gap", 2)],
        [ind("listing_price_gap", 2), ind("bedroom_area", 2, "structure")],
        [ind("amenity_disclosure", 2, "records")],
        [],
    ]
    for diff in (0.0, 6.0):
        for fixture in fixtures:
            res = dp.evaluate(make_signals(fixture, diff=diff))
            first = codes(res)[0]
            firsts.add(first)
            check(
                f"first match {first} decides (diff {diff})",
                res["decision"] == dp.RULE_DECISION[code_to_rule[first]],
                res["decision"],
            )
    check("every rule can be the first match",
          firsts == set(code_to_rule), firsts)


# ============================================================
# V2 - THRESHOLD BOUNDARIES
# ============================================================

def v2_thresholds():
    section("V2 - threshold boundaries (existing constants only)")

    check("price band is exactly 5.0",
          dp.POLICY_CONFIG["price_band_pct"] == 5.0)

    band = dp.POLICY_CONFIG["price_band_pct"]
    for diff, expect_in_band in [
        (0.0, True),
        (4.99, True),
        (5.0, True),          # exactly on the band stays inside
        (5.01, False),
        (-4.99, True),
        (-5.0, True),
        (-5.01, False),
    ]:
        res = dp.evaluate(make_signals([], diff=diff))
        got_in_band = res["decision"] == dp.PROCEED
        check(
            f"difference_pct {diff} in_band={expect_in_band}",
            got_in_band == expect_in_band,
            res["decision"],
        )

    check("5.01 uses R7", "PRICE_OUTSIDE_BAND" in codes(
        dp.evaluate(make_signals([], diff=5.01))))
    check("5.0 uses R8", "WITHIN_COMPARISON_BAND_NO_ELEVATED_INDICATORS" in
          codes(dp.evaluate(make_signals([], diff=5.0))))

    # ---- z-score boundaries ---------------------------------------------
    status = risk_analysis._status_for_z
    check("z == 2.5 -> medium (2)", status(2.5) == 2, status(2.5))
    check("z == 2.49 -> none (0)", status(2.49) == 0, status(2.49))
    check("z == -2.5 -> medium (2)", status(-2.5) == 2, status(-2.5))
    check("z == 4.0 -> high (3)", status(4.0) == 3, status(4.0))
    check("z == 3.99 -> medium (2)", status(3.99) == 2, status(3.99))
    check("z == -4.0 -> high (3)", status(-4.0) == 3, status(-4.0))
    check("z is None -> none (0)", status(None) == 0)
    check("z is NaN -> none (0)", status(float("nan")) == 0)

    check("Z_MEDIUM unchanged", risk_analysis.Z_MEDIUM == 2.5,
          risk_analysis.Z_MEDIUM)
    check("Z_HIGH unchanged", risk_analysis.Z_HIGH == 4.0,
          risk_analysis.Z_HIGH)
    check("SEVERITY_LABEL unchanged",
          dict(risk_analysis.SEVERITY_LABEL) == STATUS_BY_POINTS,
          dict(risk_analysis.SEVERITY_LABEL))

    # ---- completeness boundaries (amenity disclosure) --------------------
    def disclosure(n_known, n_total=100):
        real_cols = list(properties.amenity_cols)
        fake = [f"decision_test_amenity_{i}" for i in range(n_total)]
        sample = properties.PROPERTIES.iloc[0]
        row_dict = {
            c: (1 if i < n_known else np.nan)
            for i, c in enumerate(fake)
        }
        row_dict.update({
            "price": float(sample["price"]),
            "area": float(sample["area"]),
            "source_city": str(sample["source_city"]),
            "location": str(sample["location"]),
            "no_of_bedrooms": int(sample["no_of_bedrooms"]),
        })
        row = pd.Series(row_dict)
        properties.amenity_cols = fake
        try:
            outs = risk_analysis._property_indicators(
                row, properties.PROPERTIES
            )
        finally:
            properties.amenity_cols = real_cols
        for item in outs:
            if item["id"] == "amenity_disclosure":
                return item
        raise AssertionError("amenity_disclosure indicator not produced")

    check("COMPLETENESS_HIGH unchanged",
          risk_analysis.COMPLETENESS_HIGH == 0.25,
          risk_analysis.COMPLETENESS_HIGH)
    check("COMPLETENESS_LOW unchanged",
          risk_analysis.COMPLETENESS_LOW == 0.50,
          risk_analysis.COMPLETENESS_LOW)

    for n_known, expect in [(25, 3), (26, 2), (50, 2), (51, 0)]:
        got = disclosure(n_known)
        check(
            f"completeness {n_known}/100 -> severity {expect}",
            got["severity_points"] == expect and got["status"]
            == STATUS_BY_POINTS[expect],
            got,
        )

    # ---- production constants match POLICY_CONFIG provenance -------------
    check("POLICY_CONFIG price band == properties.SIGNAL_BAND_PCT",
          dp.POLICY_CONFIG["price_band_pct"] == properties.SIGNAL_BAND_PCT,
          (dp.POLICY_CONFIG["price_band_pct"], properties.SIGNAL_BAND_PCT))
    check("POLICY_CONFIG elevated minimum == 2",
          dp.POLICY_CONFIG["elevated_min_severity_points"] == 2)
    check("POLICY_CONFIG high severity == 3",
          dp.POLICY_CONFIG["high_severity_points"] == 3)
    check("POLICY_CONFIG declares no new thresholds",
          dp.POLICY_CONFIG["no_new_thresholds"] is True)

    # ---- real payload: severity/status stay consistent -------------------
    payload = risk_analysis.property_payload(sample_id())
    for item in payload["indicators"]:
        ok = (
            item["status"] == STATUS_BY_POINTS[item["severity_points"]]
            and 0 <= item["severity_points"] <= 3
        )
        check(f"payload {item['id']} status/points consistent", ok, item)

    # ---- policy consumes the signal band, never a derived copy -----------
    res = dp.evaluate(make_signals([], diff=5.0, band=properties.SIGNAL_BAND_PCT))
    check("policy consumes SIGNAL_BAND_PCT value",
          res["decision"] == dp.PROCEED, res["decision"])


# ============================================================
# V3 - DEGRADATION
# ============================================================

def v3_degradation():
    section("V3 - degradation (missing evidence never silently PROCEEDs)")

    # missing / invalid property identity
    r = dp.evaluate(make_signals(mreid=""))
    check("missing mreid -> HOLD", r["decision"] == dp.HOLD, r["decision"])
    check("missing mreid reason",
          "LISTING_DATA_INCOMPLETE" in codes(r), codes(r))

    r = dp.evaluate(make_signals(listed=0.0))
    check("listed price 0 -> HOLD", r["decision"] == dp.HOLD, r["decision"])
    check("listed price 0 reason",
          "LISTING_DATA_INCOMPLETE" in codes(r), codes(r))

    r = dp.evaluate(make_signals(listed=None))
    check("listed price None -> HOLD", r["decision"] == dp.HOLD, r["decision"])

    # missing AI estimate
    r = dp.evaluate(make_signals(ai_signal=False, ai_price=None))
    check("missing AI estimate -> HOLD", r["decision"] == dp.HOLD, r["decision"])
    check("missing AI estimate reason",
          "AI_ESTIMATE_UNAVAILABLE" in codes(r), codes(r))
    check("missing AI estimate degraded", r["degraded"] is True)
    check("missing AI estimate is not PROCEED",
          r["decision"] != dp.PROCEED)

    # missing risk context
    r = dp.evaluate(make_signals(use_risk=False))
    check("missing risk -> HOLD", r["decision"] == dp.HOLD, r["decision"])
    check("missing risk reason",
          "RISK_CONTEXT_UNAVAILABLE" in codes(r), codes(r))
    check("missing risk degraded", r["degraded"] is True)

    # missing / stale chain snapshot
    r = dp.evaluate(make_signals(use_chain=False))
    check("missing chain snapshot -> HOLD", r["decision"] == dp.HOLD,
          r["decision"])
    check("missing chain snapshot reason",
          "CHAIN_SNAPSHOT_UNAVAILABLE" in codes(r), codes(r))
    check("missing chain snapshot degraded", r["degraded"] is True)
    check("missing chain snapshot never PROCEEDs",
          r["decision"] != dp.PROCEED)

    stale = dict(make_signals()["chain"])
    stale.pop("exported_at", None)
    sig = make_signals()
    sig["chain"] = stale
    r = dp.evaluate(sig)
    check("stale snapshot (no exported_at) -> HOLD",
          r["decision"] == dp.HOLD, r["decision"])
    check("stale snapshot reason",
          "CHAIN_SNAPSHOT_UNAVAILABLE" in codes(r), codes(r))

    # only the chain snapshot missing: escalation evidence still reported
    r = dp.evaluate(make_signals([ind("listing_price_gap", 3)],
                                 use_chain=False))
    c = codes(r)
    check("chain missing keeps escalation reasons",
          "CHAIN_SNAPSHOT_UNAVAILABLE" in c and "PRICE_HIGH_SEVERITY" in c, c)
    check("chain missing still HOLD", r["decision"] == dp.HOLD, r["decision"])

    # everything missing
    r = dp.evaluate({
        "mreid_id": "", "listed_price": None, "ai_estimated_price": None,
        "ai_market_signal": None, "risk": None, "chain": None,
    })
    c = codes(r)
    check("all evidence missing -> HOLD", r["decision"] == dp.HOLD,
          r["decision"])
    check("all four prerequisite reasons reported", len(c) == 4, c)
    check("all evidence missing degraded", r["degraded"] is True)
    check("no escalation rule ran on empty evidence",
          not any(x in c for x in (
              "WITHIN_COMPARISON_BAND_NO_ELEVATED_INDICATORS",
              "PRICE_OUTSIDE_BAND",
          )), c)

    # no degraded response may be PROCEED
    degraded_cases = [
        make_signals(mreid=""),
        make_signals(listed=0.0),
        make_signals(ai_signal=False),
        make_signals(use_risk=False),
        make_signals(use_chain=False),
        {"mreid_id": "", "listed_price": None, "ai_estimated_price": None,
         "ai_market_signal": None, "risk": None, "chain": None},
    ]
    for i, sig in enumerate(degraded_cases):
        res = dp.evaluate(sig)
        check(
            f"degraded case {i} is never PROCEED",
            res["decision"] != dp.PROCEED and res["degraded"] is True,
            res["decision"],
        )

    # degraded responses always explain themselves
    for sig in degraded_cases:
        res = dp.evaluate(sig)
        check(
            "degraded case carries an explanation and disclaimer",
            bool(res["display_explanation"]) and bool(res["disclaimer"]),
            res,
        )


# ============================================================
# V4 - CONSISTENCY / CONTRADICTIONS
# ============================================================

def v4_consistency():
    section("V4 - consistency and contradiction invariants")

    fixtures = [
        make_signals([]),
        make_signals([], diff=6.0),
        make_signals([ind("listing_price_gap", 3)]),
        make_signals([ind("bedroom_area", 2, "structure")]),
        make_signals(use_risk=False),
        make_signals(use_chain=False),
        make_signals([ind("amenity_disclosure", 3, "records")]),
        make_signals([ind("listing_price_gap", 2), ind("bedroom_area", 2,
                                                       "structure")]),
    ]

    for i, sig in enumerate(fixtures):
        res = dp.evaluate(sig)
        c = codes(res)

        check(f"[{i}] decision is one of four states",
              res["decision"] in dp.DECISIONS, res["decision"])
        check(f"[{i}] decision_version is 1.0",
              res["decision_version"] == "1.0", res["decision_version"])
        check(f"[{i}] PROCEED iff human review false",
              (res["decision"] == dp.PROCEED) == (
                  res["human_review_required"] is False), res)
        check(f"[{i}] HOLD implies no blockchain actions",
              res["decision"] != dp.HOLD
              or res["blockchain_actions_allowed"] == [],
              res["blockchain_actions_allowed"])
        check(f"[{i}] degraded implies HOLD",
              (not res["degraded"]) or res["decision"] == dp.HOLD,
              res["decision"])
        check(f"[{i}] degraded implies a prerequisite reason",
              (not res["degraded"]) or bool(set(c) & {
                  "LISTING_DATA_INCOMPLETE",
                  "AI_ESTIMATE_UNAVAILABLE",
                  "RISK_CONTEXT_UNAVAILABLE",
                  "CHAIN_SNAPSHOT_UNAVAILABLE",
              }), c)
        check(f"[{i}] reason codes are unique", len(c) == len(set(c)), c)
        check(f"[{i}] reasons are non-empty", bool(res["reasons"]), res)
        check(f"[{i}] REVIEW_WITH_OPERATOR iff human review",
              ("REVIEW_WITH_OPERATOR" in res["required_actions"])
              == res["human_review_required"], res["required_actions"])
        check(f"[{i}] display_explanation is non-empty",
              bool(res["display_explanation"].strip()))
        check(f"[{i}] workflow has four keys",
              set(res["workflow"]) == {
                  "state", "stage", "unmet_conditions", "allowed_actions"},
              res["workflow"])

        for reason in res["reasons"]:
            shape_ok = set(reason) == {"code", "severity", "evidence", "text"}
            check(f"[{i}] {reason.get('code')} reason shape", shape_ok, reason)
            check(f"[{i}] {reason.get('code')} severity vocabulary",
                  reason.get("severity") in {"hold", "enhanced", "review",
                                             "info"}, reason)
            check(f"[{i}] {reason.get('code')} has text",
                  bool(str(reason.get("text", "")).strip()), reason)
            check(f"[{i}] {reason.get('code')} has evidence",
                  isinstance(reason.get("evidence"), dict), reason)

        # the deciding reason is always the first reason
        first_severity = res["reasons"][0]["severity"]
        expected_first = {
            dp.PROCEED: "info",
            dp.REVIEW_REQUIRED: "review",
            dp.ENHANCED_REVIEW: "enhanced",
            dp.HOLD: "hold",
        }[res["decision"]]
        check(f"[{i}] decision matches the first reason severity",
              first_severity == expected_first,
              (res["decision"], first_severity))

    # determinism
    a = dp.evaluate(make_signals([ind("listing_price_gap", 2)], diff=6.0))
    b = dp.evaluate(make_signals([ind("listing_price_gap", 2)], diff=6.0))
    check("policy is deterministic", a == b)

    # a blocked contract action is never advertised
    r = dp.evaluate(make_signals([ind("listing_price_gap", 3)]))
    check("HOLD never advertises contract actions",
          r["blockchain_actions_allowed"] == [],
          r["blockchain_actions_allowed"])

    # market context must never reach the policy
    sig = make_signals([])
    sig["market_context"] = {"median_price_per_sqft": 1}
    check("extra market context is ignored",
          dp.evaluate(sig) == dp.evaluate(make_signals([])))


# ============================================================
# V5 - WORKFLOW CAN NEVER CHANGE THE DECISION
# ============================================================

def v5_workflow_independence():
    section("V5 - workflow never changes the decision")

    workflows = [
        None,
        {"state": "unavailable", "stage": None,
         "unmet_conditions": ["CHAIN_SNAPSHOT_UNAVAILABLE"],
         "allowed_actions": []},
        {"state": "snapshot", "stage": "None",
         "unmet_conditions": ["NOT_TOKENIZED"], "allowed_actions": []},
        {"state": "snapshot", "stage": "Listed",
         "unmet_conditions": ["NOT_LISTED"], "allowed_actions": []},
        {"state": "live", "stage": "Finalized",
         "unmet_conditions": ["SALE_ALREADY_FINALIZED"],
         "allowed_actions": []},
        {"state": "live", "stage": "UnderContract",
         "unmet_conditions": ["INSPECTION_REQUIRED_NOT_PASSED",
                               "LENDER_APPROVAL_MISSING",
                               "FUNDING_INCOMPLETE",
                               "BUYER_APPROVAL_MISSING",
                               "SELLER_APPROVAL_MISSING"],
         "allowed_actions": list(dc.IN_PROGRESS_ACTIONS) + ["FINALIZE_SALE"]},
        {"state": "live", "stage": "Approved",
         "unmet_conditions": ["BUYER_APPROVAL_MISSING"],
         "allowed_actions": ["APPROVE_BUYER", "APPROVE_SELLER",
                              "FINALIZE_SALE", "CANCEL_SALE"]},
    ]

    signals_sets = [
        ("PROCEED", make_signals([])),
        ("REVIEW", make_signals([], diff=6.0)),
        ("ENHANCED", make_signals(
            [ind("listing_price_gap", 2), ind("bedroom_area", 2,
                                              "structure")])),
        ("HOLD", make_signals([ind("listing_price_gap", 3)])),
    ]

    for label, sig in signals_sets:
        decisions = set()
        for wf in workflows:
            res = dp.evaluate(sig, wf)
            decisions.add(res["decision"])
            expected_workflow = wf if wf is not None else dp._empty_workflow()
            check(
                f"{label}: workflow echoed verbatim",
                res["workflow"] == expected_workflow,
                res["workflow"],
            )
        check(f"{label}: decision identical across all workflows",
              len(decisions) == 1, decisions)

    # ---- gating is additive and never overrides the contract -------------
    approved = {
        "state": "live", "stage": "Approved",
        "unmet_conditions": ["BUYER_APPROVAL_MISSING"],
        "allowed_actions": ["APPROVE_BUYER", "APPROVE_SELLER",
                             "FINALIZE_SALE", "CANCEL_SALE"],
    }

    res = dp.evaluate(make_signals([]), approved)
    check("PROCEED keeps contract actions unchanged",
          res["blockchain_actions_allowed"] == approved["allowed_actions"],
          res["blockchain_actions_allowed"])

    res = dp.evaluate(make_signals([], diff=6.0), approved)
    gated = res["blockchain_actions_allowed"]
    check("REVIEW holds back gated actions only",
          "APPROVE_BUYER" not in gated
          and "APPROVE_SELLER" not in gated
          and "FINALIZE_SALE" not in gated
          and "CANCEL_SALE" in gated,
          gated)

    res = dp.evaluate(make_signals([ind("listing_price_gap", 3)]), approved)
    check("HOLD clears every contract action",
          res["blockchain_actions_allowed"] == [],
          res["blockchain_actions_allowed"])

    # ---- workflow conditions map to actions ------------------------------
    mapping = [
        ("NOT_TOKENIZED", "REGISTRATION_OR_LISTING_REQUIRED"),
        ("NOT_LISTED", "REGISTRATION_OR_LISTING_REQUIRED"),
        ("INSPECTION_REQUIRED_NOT_PASSED", "OBTAIN_INSPECTION"),
        ("LENDER_APPROVAL_MISSING", "OBTAIN_LENDER_APPROVAL"),
        ("FUNDING_INCOMPLETE", "COMPLETE_FUNDING"),
        ("BUYER_APPROVAL_MISSING", "BUYER_APPROVAL"),
        ("SELLER_APPROVAL_MISSING", "SELLER_APPROVAL"),
        ("SALE_ALREADY_FINALIZED", "SALE_ALREADY_FINALIZED"),
        ("CHAIN_SNAPSHOT_UNAVAILABLE", "VERIFY_ON_CHAIN_MANUALLY"),
        ("LIVE_CHAIN_READ_FAILED", "VERIFY_ON_CHAIN_MANUALLY"),
    ]
    for condition, action in mapping:
        wf = {"state": "snapshot", "stage": "None",
              "unmet_conditions": [condition], "allowed_actions": []}
        res = dp.evaluate(make_signals([]), wf)
        check(f"{condition} -> {action}",
              action in res["required_actions"], res["required_actions"])

    # a fully met workflow does not by itself force a review
    res = dp.evaluate(
        make_signals([]),
        {"state": "live", "stage": "Listed", "unmet_conditions": [],
         "allowed_actions": ["CLOSE_LISTING", "COMMIT_AND_DEPOSIT"]},
    )
    check("no unmet condition, clean evidence -> PROCEED",
          res["decision"] == dp.PROCEED, res["decision"])
    check("workflow conditions do not add a review action",
          "REVIEW_WITH_OPERATOR" not in res["required_actions"],
          res["required_actions"])


# ============================================================
# V6 - FORBIDDEN LANGUAGE + BOUNDARY STATEMENT
# ============================================================

def v6_language():
    section("V6 - forbidden language and boundary statement")

    check("boundary statement matches the specification",
          dp.BOUNDARY_STATEMENT == BOUNDARY_TEXT, dp.BOUNDARY_STATEMENT)
    check("boundary statement appears in the disclaimer",
          BOUNDARY_TEXT in dp.DISCLAIMER)
    assert_clean("BOUNDARY_STATEMENT", dp.BOUNDARY_STATEMENT)
    assert_clean("DISCLAIMER", dp.DISCLAIMER)
    assert_clean("POLICY_CONFIG", str(dp.POLICY_CONFIG))
    assert_clean("BOUNDARY", str(dc.BOUNDARY))
    assert_clean("CONDITION_ACTIONS", str(dp.CONDITION_ACTIONS))
    assert_clean("DEFAULT_WORKFLOW", str(dp.DEFAULT_WORKFLOW))

    # every explanation, for every decision state
    for label, sig in [
        ("PROCEED", make_signals([])),
        ("REVIEW", make_signals([], diff=6.0)),
        ("ENHANCED", make_signals(
            [ind("listing_price_gap", 2), ind("bedroom_area", 2,
                                              "structure")])),
        ("HOLD", make_signals([ind("listing_price_gap", 3)])),
        ("DEGRADED", make_signals(use_risk=False)),
        ("NO-AI", make_signals(ai_signal=False)),
    ]:
        res = dp.evaluate(sig)
        texts = [res["display_explanation"], res["disclaimer"]] + [
            str(r) for r in res["reasons"]
        ]
        assert_clean(f"{label} response", *texts)

    # passthrough label must never be echoed into policy prose
    sig = make_signals([], diff=6.0, label="Potentially undervalued")
    res = dp.evaluate(sig)
    policy_prose = " ".join(
        [res["display_explanation"], res["disclaimer"]]
        + [r["text"] for r in res["reasons"]]
    ).lower()
    check("passthrough label exists in the input",
          sig["ai_market_signal"]["label"] == "Potentially undervalued")
    check("passthrough label is not echoed by the policy",
          "undervalued" not in policy_prose, policy_prose[:300])
    assert_clean("passthrough label never leaks",
                 res["display_explanation"], res["disclaimer"],
                 [r["text"] for r in res["reasons"]])

    # the disclaimer states what the policy is not
    for phrase in [
        "not legal advice",
        "not fraud detection",
        "not investment advice",
        "not a replacement for MillowEscrow.sol",
        "workflow gating only",
    ]:
        check(f"disclaimer states '{phrase}'", phrase in dp.DISCLAIMER,
              dp.DISCLAIMER)

    # boundary booleans
    check("conformal_uncertainty_used is False",
          dc.BOUNDARY["conformal_uncertainty_used"] is False,
          dc.BOUNDARY)
    check("ai decision is not approval",
          dc.BOUNDARY["ai_decision_is_not_approval"] is True)
    check("chain execution is not lawfulness",
          dc.BOUNDARY["chain_execution_is_not_lawfulness"] is True)


# ============================================================
# V7 - API CONTRACT, PROVENANCE, PURITY
# ============================================================

def v7_api_and_purity():
    section("V7 - API contract, provenance and purity")

    openapi = app.app.openapi()
    paths = sorted(openapi.get("paths", {}))
    check("decision route is registered",
          "/api/properties/{mreid_id}/decision" in paths, paths)
    check("decision route is GET only",
          list(openapi["paths"]["/api/properties/{mreid_id}/decision"])
          == ["get"],
          list(openapi["paths"]["/api/properties/{mreid_id}/decision"]))

    mid = sample_id()
    resp = client.get(f"/api/properties/{mid}/decision")
    check("decision endpoint -> 200", resp.status_code == 200, resp.text)
    body = resp.json()

    expected_keys = {
        "mreid_id", "decision", "decision_version",
        "human_review_required", "degraded", "display_explanation",
        "disclaimer", "reasons", "required_actions",
        "blockchain_actions_allowed", "signals", "workflow", "boundary",
    }
    check("response shape", expected_keys <= set(body),
          sorted(set(body) ^ expected_keys))
    check("mreid_id echoed", body["mreid_id"] == mid, body["mreid_id"])
    check("decision in the four states", body["decision"] in dp.DECISIONS,
          body["decision"])
    check("decision_version 1.0", body["decision_version"] == "1.0",
          body["decision_version"])
    check("boundary statement present",
          BOUNDARY_TEXT in body["disclaimer"], body["disclaimer"][:200])
    check("conformal_uncertainty_used is False in the response",
          body["boundary"]["conformal_uncertainty_used"] is False,
          body["boundary"])
    check("signals block shape",
          {"ai_estimated_price", "ai_market_signal", "risk",
           "market_context", "chain"} <= set(body["signals"]),
          sorted(body["signals"]))
    check("workflow block shape",
          {"state", "stage", "unmet_conditions", "allowed_actions"}
          <= set(body["workflow"]), body["workflow"])
    check("workflow state vocabulary",
          body["workflow"]["state"] in {"snapshot", "live", "unavailable"},
          body["workflow"])
    check("live chain state is not fabricated",
          body["workflow"]["state"] in {"snapshot", "unavailable"}
          or body["workflow"]["stage"] is not None, body["workflow"])
    check("reasons carry the four fields",
          all(set(r) == {"code", "severity", "evidence", "text"}
              for r in body["reasons"]), body["reasons"])

    again = client.get(f"/api/properties/{mid}/decision").json()
    check("endpoint is deterministic", again == body)

    # policy-authored fields only - the risk/methodology prose is pre-existing
    authored = [body["display_explanation"], body["disclaimer"]] + [
        str(r) for r in body["reasons"]
    ] + list(body["required_actions"])
    assert_clean("API policy-authored fields", *authored)

    text = str(body)
    check("no internal filesystem paths exposed",
          "C:" not in text and ROOT.as_posix() not in text, text[:200])
    check("no Experiment 4 artifact path in the payload",
          "v2_4" not in text and "artifacts/valuation" not in text,
          text[:300])

    r = client.get("/api/properties/MREID_9999999/decision")
    check("unknown property -> 404", r.status_code == 404, r.text)

    # ---- degradation through the API ------------------------------------
    real = (chain_index.available, chain_index.exported_at,
            chain_index.as_summary)
    try:
        chain_index.available = lambda: False
        chain_index.exported_at = lambda: None
        chain_index.as_summary = lambda key: None
        degraded_body = client.get(f"/api/properties/{mid}/decision").json()
        check("API degraded -> HOLD",
              degraded_body["decision"] == dp.HOLD,
              degraded_body["decision"])
        check("API degraded flag", degraded_body["degraded"] is True)
        check("API degraded reason",
              any(r["code"] == "CHAIN_SNAPSHOT_UNAVAILABLE"
                  for r in degraded_body["reasons"]),
              degraded_body["reasons"])
        check("API degraded workflow unavailable",
              degraded_body["workflow"]["state"] == "unavailable",
              degraded_body["workflow"])
        check("API degraded clears blockchain actions",
              degraded_body["blockchain_actions_allowed"] == [],
              degraded_body["blockchain_actions_allowed"])
        check("API degraded never PROCEEDs",
              degraded_body["decision"] != dp.PROCEED)
        assert_clean("API degraded policy fields",
                     degraded_body["display_explanation"],
                     degraded_body["disclaimer"],
                     [str(r) for r in degraded_body["reasons"]])
    finally:
        chain_index.available, chain_index.exported_at, \
            chain_index.as_summary = real

    # ---- read-only RPC guarantees ---------------------------------------
    check("RPC allowlist is eth_call only",
          dc.READ_ONLY_METHODS == frozenset({"eth_call"}),
          dc.READ_ONLY_METHODS)
    rpc = dc.ReadOnlyRpcClient("http://127.0.0.1:8545",
                               post=lambda url, payload: {})
    for method in ["eth_sendTransaction", "personal_sign", "evm_mine",
                   "eth_sendRawTransaction", "miner_start"]:
        try:
            rpc._request(method, [])
            check(f"RPC rejects {method}", False, "no exception")
        except dc.ReadOnlyRpcError:
            check(f"RPC rejects {method}", True)

    # ---- calldata + ABI decode ------------------------------------------
    check("sales selector is correct",
          dc.encode_sale_call(7) == "0xb5f522f7" + f"{7:064x}",
          dc.encode_sale_call(7))

    addr_a = "11" * 20
    addr_b = "22" * 20

    def _word(value):
        return f"{int(value):064x}"

    words = [
        _word(3),                       # status = Approved
        ("0" * 24) + addr_a,            # seller
        ("0" * 24) + addr_b,            # buyer
        _word(1000),                    # priceWei
        _word(100),                     # earnestWei
        _word(1000),                    # buyerFundedWei
        _word(0),                       # lenderFundedWei
        _word(1),                       # sellerApproved
        _word(0),                       # buyerApproved
        _word(0),                       # inspectionPassed
        _word(1),                       # inspectionRequired
        _word(1),                       # lenderRequired
        ("0" * 24) + addr_a,            # financing.lender
        _word(1),                       # financing.requested
        _word(0),                       # financing.approved
        _word(0),                       # financing.rejected
        _word(2000),                    # downPaymentPctBps
        _word(800),                     # interestRateBps
        _word(120),                     # loanTenureMonths
        _word(200),                     # downPaymentWei
        _word(800),                     # loanAmountWei
    ]
    blob = "0x" + "".join(words)
    sale = dc.decode_outputs(dc.sales_outputs(), blob)
    check("ABI decode: status", sale["status"] == 3, sale["status"])
    check("ABI decode: seller address",
          sale["seller"].lower() == "0x" + addr_a, sale["seller"])
    check("ABI decode: price", sale["priceWei"] == 1000, sale["priceWei"])
    check("ABI decode: nested tuple",
          isinstance(sale["financing"], dict)
          and sale["financing"]["approved"] is False
          and sale["financing"]["interestRateBps"] == 800,
          sale["financing"])
    check("ABI decode: bools are booleans",
          sale["sellerApproved"] is True and sale["buyerApproved"] is False,
          (sale["sellerApproved"], sale["buyerApproved"]))

    for bad in ["0x", "0x" + "00" * 20, "0x" + "00" * 65]:
        try:
            dc.decode_outputs(dc.sales_outputs(), bad)
            check(f"malformed return data rejected ({len(bad)})",
                  False, "no exception")
        except dc.ReadOnlyRpcError:
            check(f"malformed return data rejected ({len(bad)})", True)

    # ---- workflow builder ------------------------------------------------
    wf = dc.build_workflow()
    check("no snapshot -> unavailable workflow",
          wf["state"] == "unavailable"
          and wf["unmet_conditions"] == ["CHAIN_SNAPSHOT_UNAVAILABLE"]
          and wf["allowed_actions"] == [], wf)

    not_tokenized = {"tokenized": False, "token_id": None, "listed": False,
                     "active_sale": False, "finalized": False,
                     "sale_status": None}
    wf = dc.build_workflow(summary=not_tokenized, snapshot_available=True,
                           exported_at="2026-09-26T16:23:37.384Z")
    check("not tokenized -> registration action",
          wf["state"] == "snapshot"
          and wf["unmet_conditions"] == ["NOT_TOKENIZED"], wf)

    listed = dict(not_tokenized, tokenized=True, listed=True)
    wf = dc.build_workflow(summary=listed, snapshot_available=True,
                           exported_at="2026-09-26T16:23:37.384Z")
    check("tokenized and listed -> LIST_FOR_SALE allowed",
          wf["allowed_actions"] == ["LIST_FOR_SALE"], wf)

    finalized = dict(listed, active_sale=True, finalized=True,
                     sale_status="Finalized")
    wf = dc.build_workflow(summary=finalized, snapshot_available=True,
                           exported_at="2026-09-26T16:23:37.384Z")
    check("finalized sale -> SALE_ALREADY_FINALIZED",
          wf["unmet_conditions"] == ["SALE_ALREADY_FINALIZED"]
          and wf["allowed_actions"] == [], wf)

    active = dict(listed, active_sale=True, sale_status="UnderContract")
    wf = dc.build_workflow(summary=active, snapshot_available=True,
                           exported_at="2026-09-26T16:23:37.384Z",
                           live_error="connection refused")
    check("failed live read -> LIVE_CHAIN_READ_FAILED",
          wf["state"] == "unavailable"
          and wf["unmet_conditions"] == ["LIVE_CHAIN_READ_FAILED"], wf)

    wf = dc.build_workflow(
        summary=active, snapshot_available=True,
        exported_at="2026-09-26T16:23:37.384Z", live_sale=sale)
    check("live sale conditions derived",
          wf["state"] == "live"
          and wf["stage"] == "Approved"
          and wf["unmet_conditions"] == [
              "INSPECTION_REQUIRED_NOT_PASSED",
              "LENDER_APPROVAL_MISSING",
              "BUYER_APPROVAL_MISSING",
          ], wf)
    check("live sale actions include stage actions and FINALIZE_SALE",
          "CANCEL_SALE" in wf["allowed_actions"]
          and "FINALIZE_SALE" in wf["allowed_actions"], wf)

    # ---- read-only helper with an injected transport ---------------------
    captured = {}

    def fake_post(url, payload):
        captured["url"] = url
        captured["payload"] = payload
        return {"jsonrpc": "2.0", "id": 1, "result": blob}

    client_rpc = dc.ReadOnlyRpcClient("http://127.0.0.1:8545",
                                      post=fake_post)
    got = client_rpc.eth_call("0x" + "ab" * 20, dc.encode_sale_call(1))
    check("eth_call returns the node result", got == blob, got[:40])
    check("eth_call is read-only in the payload",
          captured["payload"]["method"] == "eth_call"
          and captured["payload"]["params"][1] == "latest",
          captured["payload"])
    check("no state-changing field in the payload",
          not any(k in captured["payload"] for k in ("value", "gas")),
          captured["payload"])

    # ---- purity: decision_policy is a pure stdlib module -----------------
    dp_src = (ROOT / "backend" / "decision_policy.py").read_text(
        encoding="utf-8")
    dc_src = (ROOT / "backend" / "decision_context.py").read_text(
        encoding="utf-8")

    tree = ast.parse(dp_src)
    dp_imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            dp_imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            dp_imports.add(node.module.split(".")[0])
    check("decision_policy imports only __future__",
          dp_imports <= {"__future__"}, dp_imports)

    tree = ast.parse(dc_src)
    dc_imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            dc_imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            dc_imports.add(node.module.split(".")[0])
    check("decision_context uses no web3 / requests / subprocess",
          not ({"web3", "requests", "subprocess", "socket", "artifacts"}
               & dc_imports), dc_imports)

    for name, source in (("decision_policy", dp_src),
                         ("decision_context", dc_src)):
        check(f"{name}: no Experiment 4 artifact reference",
              "v2_4" not in source, source[:200])
        check(f"{name}: no src/config.json reference",
              "src/config.json" not in source, source[:200])
        check(f"{name}: no ai module import",
              "import ai" not in source and "from ai " not in source,
              source[:200])
        check(f"{name}: no model retraining reference",
              ".fit(" not in source and "joblib" not in source, source[:200])

    check("decision_version is 1.0", dp.DECISION_VERSION == "1.0")
    check("policy declares conformal is not used",
          dp.POLICY_CONFIG["conformal_uncertainty_used"] is False)
    check("policy config carries provenance",
          bool(dp.POLICY_CONFIG.get("price_band_source"))
          and bool(dp.POLICY_CONFIG.get("severity_source"))
          and bool(dp.POLICY_CONFIG.get("risk_threshold_source")),
          dp.POLICY_CONFIG)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("MILLOW DECISION POLICY API TESTS (decision_version 1.0)")
    print("=" * 70)

    v1_rule_table()
    v2_thresholds()
    v3_degradation()
    v4_consistency()
    v5_workflow_independence()
    v6_language()
    v7_api_and_purity()

    print()
    print(f"RESULT: {PASSED} passed, {FAILED} failed")
    print("=" * 70)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
