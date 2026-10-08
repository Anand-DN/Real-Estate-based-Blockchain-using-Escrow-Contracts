"""
MILLOW - Explainable Decision Policy (decision_version "1.0")

Deterministic, rule-based transaction decision support for the MILLOW
application workflow.  It combines evidence that already exists in this
repository:

  * listed price           (MREID dataset, backend/properties.py)
  * AI research estimate   (frozen V5 model, scripts/predict_property_value.py)
  * price comparison band  (properties.SIGNAL_BAND_PCT, backend/properties.py)
  * listing risk indicators (backend/risk_analysis.py, threshold-defined)
  * chain snapshot facts   (backend/chain_index.py)

Decision states:

    PROCEED | REVIEW_REQUIRED | ENHANCED_REVIEW | HOLD

HOLD means the MILLOW application workflow should not advance.  It does
NOT and CANNOT prevent a direct wallet transaction to MillowEscrow; the
contract remains the only authority for on-chain execution.

This module is a PURE FUNCTION module:

  * no network access
  * no blockchain RPC calls
  * no randomness
  * no model training
  * no filesystem access
  * no imports outside the Python standard library

Thresholds are not re-derived here.  The price band and the risk
thresholds are consumed from the existing production implementations;
POLICY_CONFIG records their provenance and backend/test_decision_api.py
asserts that these values still equal the production constants.  No new
threshold, no learned threshold and no tuned threshold is introduced.

Conformal prediction / Experiment 4 output is not an input to this policy.
"""

from __future__ import annotations

# ============================================================
# DECISION STATES
# ============================================================

PROCEED = "PROCEED"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
ENHANCED_REVIEW = "ENHANCED_REVIEW"
HOLD = "HOLD"

DECISIONS = (PROCEED, REVIEW_REQUIRED, ENHANCED_REVIEW, HOLD)

DECISION_VERSION = "1.0"

# Exact boundary statement required by the Decision Policy specification.
BOUNDARY_STATEMENT = (
    "This panel can prevent the MILLOW application from advancing the "
    "workflow. It cannot prevent a direct wallet transaction to "
    "MillowEscrow, and an on-chain execution does not by itself certify "
    "legality, title or valuation."
)

DISCLAIMER = (
    BOUNDARY_STATEMENT
    + " This is application transaction decision support and workflow "
    "gating only. It is not legal advice, not a legal ownership "
    "determination, not a certified property valuation, not fraud "
    "detection and not a probability, not investment advice, not loan "
    "approval, not lender approval, not blockchain authorization, and not "
    "a replacement for MillowEscrow.sol. The smart contract remains "
    "authoritative for actual on-chain execution."
)

# Language that must never appear in policy-authored prose
# (backend/test_risk_analysis_api.py:27-41, applied here as well).
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

# ============================================================
# POLICY CONFIG  (single, centralised, provenance-annotated)
# ============================================================

POLICY_CONFIG = {
    "decision_version": DECISION_VERSION,
    # Price comparison band.  Consumed from the ai_market_signal payload,
    # which properties.py computes with SIGNAL_BAND_PCT.
    "price_band_pct": 5.0,
    "price_band_source": "backend/properties.py:SIGNAL_BAND_PCT (L103)",
    # Indicator severity levels.  risk_analysis.SEVERITY_LABEL maps
    # none=0, low=1, medium=2, high=3; statuses are produced by
    # risk_analysis._status_for_z using Z_MEDIUM=2.5 / Z_HIGH=4.0 and by
    # the disclosure rule using COMPLETENESS_LOW=0.50 / COMPLETENESS_HIGH=0.25.
    "elevated_min_severity_points": 2,
    "high_severity_points": 3,
    "severity_source": (
        "backend/risk_analysis.py:SEVERITY_LABEL (L55) none=0 low=1 "
        "medium=2 high=3"
    ),
    "risk_threshold_source": (
        "backend/risk_analysis.py Z_MEDIUM=2.5 (L41), Z_HIGH=4.0 (L42), "
        "COMPLETENESS_LOW=0.50 (L43), COMPLETENESS_HIGH=0.25 (L44)"
    ),
    # Frozen rule table for v1.  Do not change.
    "rule_precedence": ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"],
    "indicator_ids": {
        "R1": "listing_price_gap",
        "R2": "bedroom_area",
        "R3": "amenity_disclosure",
    },
    "deviation_definition": (
        "consumed unchanged from ai_market_signal.difference_pct "
        "(properties.ai_market_signal, denominator = listed price)"
    ),
    "no_new_thresholds": True,
    "conformal_uncertainty_used": False,
}

# Frozen decision for each escalation rule.
RULE_DECISION = {
    "R1": HOLD,
    "R2": ENHANCED_REVIEW,
    "R3": REVIEW_REQUIRED,
    "R4": ENHANCED_REVIEW,
    "R5": ENHANCED_REVIEW,
    "R6": REVIEW_REQUIRED,
    "R7": REVIEW_REQUIRED,
    "R8": PROCEED,
}

RULE_SEVERITY = {
    "R1": "hold",
    "R2": "enhanced",
    "R3": "review",
    "R4": "enhanced",
    "R5": "enhanced",
    "R6": "review",
    "R7": "review",
    "R8": "info",
}

# Workflow facts -> application actions (workflow never changes decision).
CONDITION_ACTIONS = {
    "NOT_TOKENIZED": "REGISTRATION_OR_LISTING_REQUIRED",
    "NOT_LISTED": "REGISTRATION_OR_LISTING_REQUIRED",
    "INSPECTION_REQUIRED_NOT_PASSED": "OBTAIN_INSPECTION",
    "LENDER_APPROVAL_MISSING": "OBTAIN_LENDER_APPROVAL",
    "FUNDING_INCOMPLETE": "COMPLETE_FUNDING",
    "BUYER_APPROVAL_MISSING": "BUYER_APPROVAL",
    "SELLER_APPROVAL_MISSING": "SELLER_APPROVAL",
    "SALE_ALREADY_FINALIZED": "SALE_ALREADY_FINALIZED",
    "CHAIN_SNAPSHOT_UNAVAILABLE": "VERIFY_ON_CHAIN_MANUALLY",
    "LIVE_CHAIN_READ_FAILED": "VERIFY_ON_CHAIN_MANUALLY",
}

# Application-side actions held back until an operator review is recorded.
REVIEW_GATED_ACTIONS = frozenset(
    {"APPROVE_BUYER", "APPROVE_SELLER", "FINALIZE_SALE"}
)

DEFAULT_WORKFLOW = {
    "state": "unavailable",
    "stage": None,
    "unmet_conditions": ["CHAIN_SNAPSHOT_UNAVAILABLE"],
    "allowed_actions": [],
}


# ============================================================
# HELPERS
# ============================================================

def _num(value):
    """Finite float or None.  Never raises."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _points(indicator):
    """severity_points of one indicator (0 when absent)."""
    if not isinstance(indicator, dict):
        return 0
    points = _num(indicator.get("severity_points"))
    if points is None:
        return 0
    return int(points)


def _reason(code, severity, evidence, text):
    return {
        "code": code,
        "severity": severity,
        "evidence": evidence,
        "text": text,
    }


def _matched_rules(signals):
    """Return (ai_signal, risk_payload, matched_rule_ids, evidence_facts)."""
    ai = signals.get("ai_market_signal")
    risk = signals.get("risk")
    indicators = list((risk or {}).get("indicators") or [])

    elevated = [
        i for i in indicators
        if _points(i) >= POLICY_CONFIG["elevated_min_severity_points"]
    ]
    by_id = {
        str(i.get("id")): i for i in indicators
        if isinstance(i, dict) and i.get("id")
    }

    band = _num(ai.get("band_pct")) if isinstance(ai, dict) else None
    diff = _num(ai.get("difference_pct")) if isinstance(ai, dict) else None
    outside = diff is not None and band is not None and abs(diff) > band

    high_ids = [
        rule
        for rule, ind_id in POLICY_CONFIG["indicator_ids"].items()
        if _points(by_id.get(ind_id)) >= POLICY_CONFIG["high_severity_points"]
    ]

    matched = []
    # Frozen precedence: R1 > R2 > R3 > R4 > R5 > R6 > R7 > R8
    if "R1" in high_ids:
        matched.append("R1")
    if "R2" in high_ids:
        matched.append("R2")
    if "R3" in high_ids:
        matched.append("R3")
    if elevated and outside:
        matched.append("R4")
    if len(elevated) >= 2:
        matched.append("R5")
    if len(elevated) == 1:
        matched.append("R6")
    if outside:
        matched.append("R7")
    if not matched:
        matched.append("R8")

    facts = {
        "ai": ai,
        "risk": risk,
        "band": band,
        "difference_pct": diff,
        "outside_band": outside,
        "elevated": elevated,
        "by_id": by_id,
    }
    return matched, facts


def _rule_reason(rule, facts):
    """Policy-authored reason text for one escalation rule."""
    ai = facts["ai"]
    band = facts["band"]
    diff = facts["difference_pct"]
    elevated = facts["elevated"]
    by_id = facts["by_id"]
    indicator = by_id.get(POLICY_CONFIG["indicator_ids"].get(rule))

    if rule in ("R1", "R2", "R3"):
        evidence = {
            "indicator_id": indicator.get("id"),
            "dimension": indicator.get("dimension"),
            "status": indicator.get("status"),
            "severity_points": _points(indicator),
            "z": indicator.get("z"),
        }
        title = {
            "R1": "Listing price per square foot against comparable "
                  "listings is high severity.",
            "R2": "Area per bedroom against city norms is high severity.",
            "R3": "Amenity disclosure completeness is high severity.",
        }[rule]
        return _reason(
            {
                "R1": "PRICE_HIGH_SEVERITY",
                "R2": "STRUCTURE_HIGH_SEVERITY",
                "R3": "DISCLOSURE_HIGH_SEVERITY",
            }[rule],
            RULE_SEVERITY[rule],
            evidence,
            title,
        )

    if rule == "R4":
        return _reason(
            "ELEVATED_RISK_WITH_PRICE_DEVIATION",
            RULE_SEVERITY[rule],
            {
                "difference_pct": diff,
                "band_pct": band,
                "elevated_indicators": [
                    i.get("id") for i in elevated
                ],
            },
            "At least one listing risk indicator is elevated and the listed "
            "price differs from the AI research estimate by "
            f"{abs(diff)}%, outside the {band}% comparison band.",
        )

    if rule == "R5":
        return _reason(
            "MULTIPLE_ELEVATED_INDICATORS",
            RULE_SEVERITY[rule],
            {
                "elevated_count": len(elevated),
                "elevated_indicators": [i.get("id") for i in elevated],
            },
            f"{len(elevated)} listing risk indicators are elevated.",
        )

    if rule == "R6":
        return _reason(
            "SINGLE_ELEVATED_INDICATOR",
            RULE_SEVERITY[rule],
            {
                "elevated_indicators": [i.get("id") for i in elevated],
                "status": elevated[0].get("status") if elevated else None,
            },
            "One listing risk indicator is elevated.",
        )

    if rule == "R7":
        return _reason(
            "PRICE_OUTSIDE_BAND",
            RULE_SEVERITY[rule],
            {"difference_pct": diff, "band_pct": band},
            "Listed price differs from the AI research estimate by "
            f"{abs(diff)}%, outside the {band}% comparison band.",
        )

    # R8
    band_text = band if band is not None else POLICY_CONFIG["price_band_pct"]
    return _reason(
        "WITHIN_COMPARISON_BAND_NO_ELEVATED_INDICATORS",
        RULE_SEVERITY[rule],
        {
            "difference_pct": diff,
            "band_pct": band_text,
            "elevated_indicators": [],
        },
        f"The listed price is within the {band_text}% comparison band and "
        "no listing risk indicator is elevated.",
    )


def _prerequisite_reasons(signals):
    """D1-D4.  Evaluated before any escalation rule."""
    reasons = []

    mreid_id = signals.get("mreid_id")
    listed = _num(signals.get("listed_price"))
    if not mreid_id or listed is None or listed <= 0:
        reasons.append(_reason(
            "LISTING_DATA_INCOMPLETE",
            "hold",
            {"listed_price": signals.get("listed_price")},
            "The property listing record is missing or its listed price is "
            "not a positive number.",
        ))

    ai = signals.get("ai_market_signal")
    ai_price = _num(signals.get("ai_estimated_price"))
    signal_price = (
        _num(ai.get("ai_estimated_price")) if isinstance(ai, dict) else None
    )
    if (
        not isinstance(ai, dict)
        or ai_price is None
        or signal_price is None
    ):
        reasons.append(_reason(
            "AI_ESTIMATE_UNAVAILABLE",
            "hold",
            {"ai_estimated_price": signals.get("ai_estimated_price")},
            "The AI research estimate is not available for this listing.",
        ))

    if not isinstance(signals.get("risk"), dict):
        reasons.append(_reason(
            "RISK_CONTEXT_UNAVAILABLE",
            "hold",
            {"risk_available": False},
            "The listing risk analysis payload is not available.",
        ))

    chain = signals.get("chain")
    if (
        not isinstance(chain, dict)
        or not chain.get("available")
        or chain.get("exported_at") is None
    ):
        reasons.append(_reason(
            "CHAIN_SNAPSHOT_UNAVAILABLE",
            "hold",
            {
                "chain_available": bool(
                    isinstance(chain, dict) and chain.get("available")
                ),
                "exported_at": (
                    chain.get("exported_at")
                    if isinstance(chain, dict) else None
                ),
            },
            "The chain snapshot is unavailable or carries no export "
            "timestamp, so on-chain workflow facts cannot be reported and "
            "the application workflow is held for review.",
        ))

    return reasons


def _explanation(decision, codes, band):
    """Policy-authored display explanation (forbidden-language clean)."""
    listed = ", ".join(codes) if codes else "no rule"

    if decision == PROCEED:
        return (
            f"No decision rule fired: the listed price is within the "
            f"{band}% comparison band and none of the listing risk "
            "indicators is elevated (matched: "
            f"{listed}). This is application workflow support, not a "
            "recommendation to buy or sell."
        )
    if decision == REVIEW_REQUIRED:
        return (
            f"Decision rule(s) {listed} fired. An operator review is "
            "required before the MILLOW application advances the "
            "workflow."
        )
    if decision == ENHANCED_REVIEW:
        return (
            f"Decision rule(s) {listed} fired. Enhanced operator review "
            "is required, and the supporting evidence should be checked "
            "before the MILLOW application advances the workflow."
        )
    # HOLD
    if any(
        code in (
            "LISTING_DATA_INCOMPLETE",
            "AI_ESTIMATE_UNAVAILABLE",
            "RISK_CONTEXT_UNAVAILABLE",
            "CHAIN_SNAPSHOT_UNAVAILABLE",
        )
        for code in codes
    ):
        return (
            f"Evidence needed to evaluate this listing is incomplete "
            f"({listed}). The MILLOW application workflow should not "
            "advance until an operator reviews it."
        )
    return (
        f"Decision rule(s) {listed} fired at high severity. The MILLOW "
        "application workflow should not advance until an operator "
        "reviews it."
    )


def _empty_workflow():
    return {
        "state": DEFAULT_WORKFLOW["state"],
        "stage": None,
        "unmet_conditions": list(DEFAULT_WORKFLOW["unmet_conditions"]),
        "allowed_actions": [],
    }


# ============================================================
# PUBLIC API
# ============================================================

def evaluate(signals, workflow=None):
    """
    Pure decision function.

    signals:
        mreid_id, listed_price, ai_estimated_price, ai_market_signal,
        risk, chain            (chain may be None)
    workflow:
        {"state", "stage", "unmet_conditions", "allowed_actions"}
        Descriptive only - it can never change the decision.

    Returns the decision block of the API response.
    """
    signals = signals if isinstance(signals, dict) else {}
    wf = workflow if isinstance(workflow, dict) else _empty_workflow()

    prereq = _prerequisite_reasons(signals)
    # D1 / D2 / D3 remove the evidence the escalation rules read, so the
    # rules cannot run.  D4 (chain snapshot) never silently PROCEEDs.
    blocking = [r for r in prereq if r["code"] != "CHAIN_SNAPSHOT_UNAVAILABLE"]

    degraded = bool(prereq)
    band = POLICY_CONFIG["price_band_pct"]

    if blocking:
        decision = HOLD
        reasons = list(prereq)
    else:
        matched, facts = _matched_rules(signals)
        decision = HOLD if prereq else RULE_DECISION[matched[0]]
        reasons = list(prereq)
        reasons += [_rule_reason(rule, facts) for rule in matched]
        if facts["band"] is not None:
            band = facts["band"]

    matched_codes = [r["code"] for r in reasons]

    human_review_required = decision != PROCEED

    # ---- required actions: decision actions first, then workflow facts ----
    required_actions = []
    if human_review_required:
        required_actions.append("REVIEW_WITH_OPERATOR")
    for condition in wf.get("unmet_conditions") or []:
        action = CONDITION_ACTIONS.get(condition)
        if action and action not in required_actions:
            required_actions.append(action)

    # ---- blockchain actions allowed (application level, additive only) ----
    allowed = list(wf.get("allowed_actions") or [])
    if decision == HOLD:
        allowed = []
    elif decision != PROCEED:
        allowed = [a for a in allowed if a not in REVIEW_GATED_ACTIONS]

    return {
        "decision": decision,
        "decision_version": DECISION_VERSION,
        "human_review_required": human_review_required,
        "degraded": degraded,
        "display_explanation": _explanation(decision, matched_codes, band),
        "disclaimer": DISCLAIMER,
        "reasons": reasons,
        "required_actions": required_actions,
        "blockchain_actions_allowed": allowed,
        "workflow": wf,
    }
