"""
MILLOW - MREID (INR) ground-truth tools for the AI assistant (Phase 7)

These tools answer ONLY from real MILLOW backend data (backend :8001):
property records, the V5 AI research estimate and market signal, locality
cohort stats, market breakdowns, and dashboard analytics.  The LLM never
invents numbers: a tool either returns the actual record from the backend
or an explicit error, and every response must be labelled
DATA FACT / AI MODEL ESTIMATE / MODEL-BASED INTERPRETATION.

The ETH demo tools in app.py are untouched; this module adds the INR/MREID
domain on a separate agent instance so the DApp chat and the dashboard chat
can be switched independently.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import httpx

# 127.0.0.1, not "localhost": the backend binds IPv4 only, and on this machine
# "localhost" resolves to ::1 first, so every httpx.get() took ~2.1s to fail
# with WinError 10061 before retrying on 127.0.0.1 and finally succeeding. That
# single character was most of the perceived chat latency, because a risk answer
# makes three backend calls.
BACKEND_BASE = os.getenv(
    "MILLOW_PROPERTY_API_URL",
    "http://127.0.0.1:8001",
)

_TIMEOUT = httpx.Timeout(30.0)

# One pooled client instead of httpx.get() per call.  A throwaway client per
# request meant a fresh TCP handshake every time; the shared client keeps the
# connection warm across a chat turn's tool calls.
_CLIENT = httpx.Client(base_url=BACKEND_BASE, timeout=_TIMEOUT)


def _get(path):
    response = _CLIENT.get(path)
    response.raise_for_status()
    return response.json()


def _get_all(paths):
    """GET several independent paths concurrently, preserving order."""
    if len(paths) == 1:
        return [_get(paths[0])]
    with ThreadPoolExecutor(max_workers=len(paths)) as pool:
        return [f.result() for f in [pool.submit(_get, p) for p in paths]]


# ============================================================
# TOOL HANDLERS (all must return JSON-serialisable dicts)
# ============================================================


def handle_property(mreid_id):
    """Full detail record for one MREID property (real backend data)."""
    mreid_id = str(mreid_id or "").strip()
    if not mreid_id:
        return {"error": "MREID id is required."}
    try:
        return _get(f"/api/properties/{mreid_id}")
    except httpx.HTTPStatusError as exc:
        return {"error": f"Property '{mreid_id}' not found (HTTP {exc.response.status_code})."}
    except Exception as exc:
        return {"error": f"Could not reach the MILLOW property backend at {BACKEND_BASE}: {exc}"}


def handle_search(kwargs):
    """Search the real MREID catalogue with optional filters."""
    params = {}
    for key in ("city", "location", "min_price", "max_price",
                "min_area", "max_area", "bedrooms", "ai_signal",
                "sort", "page_size"):
        value = kwargs.get(key)
        if value in (None, ""):
            continue
        if key == "page_size":
            value = int(value)
            if value < 1 or value > 100:
                value = 20
        params[key] = value
    # One request, not two.  This used to fetch the unfiltered page first and
    # then immediately overwrite it whenever any filter was present, so every
    # filtered search paid for a full page fetch it threw away.
    path = "/api/properties/search"
    if params:
        path += "?" + "&".join(f"{k}={v}" for k, v in params.items())
    try:
        data = _get(path)
    except httpx.HTTPStatusError as exc:
        return {"error": f"Search failed (HTTP {exc.response.status_code})."}
    except Exception as exc:
        return {"error": f"Search failed: {exc}"}
    results = []
    for row in (data.get("results") or [])[:10]:
        signal = row.get("ai_market_signal") or {}
        results.append({
            "mreid_id": row["mreid_id"],
            "city": row["city"],
            "location": row["location"],
            "area": row["area"],
            "bedrooms": row["bedrooms"],
            "listed_price": row["price"],
            "listed_price_formatted": row["price_formatted"],
            "ai_estimated_price": row["ai_estimated_price"],
            "ai_estimated_price_formatted": row["ai_estimated_price_formatted"],
            "signal": signal.get("label"),
            "difference_pct": signal.get("difference_pct"),
        })
    return {
        "total": data.get("total"),
        "page": data.get("page"),
        "returned": len(results),
        "results": results,
        "note": "All values above are real records from the MILLOW MREID catalogue.",
    }


def handle_market_breakdown(kwargs):
    """Market breakdown by city or by locality within a city."""
    group = kwargs.get("group") or "city"
    city = kwargs.get("city")
    if group not in ("city", "locality"):
        return {"error": "group must be 'city' or 'locality'."}
    qs = f"?group={group}"
    if city:
        qs += f"&city={city}"
    try:
        data = _get(f"/api/dashboard/market-breakdown{qs}")
    except httpx.HTTPStatusError as exc:
        return {"error": f"Market breakdown failed (HTTP {exc.response.status_code})."}
    except Exception as exc:
        return {"error": f"Market breakdown failed: {exc}"}
    rows = []
    for row in (data.get("rows") or [])[:15]:
        entry = {
            "city": row["city"],
            "count": row["count"],
            "avg_listed": row["avg_listed"],
            "avg_ai": row["avg_ai"],
            "avg_listed_ppsf": row["avg_listed_ppsf"],
            "avg_ai_ppsf": row["avg_ai_ppsf"],
            "undervalued": row["undervalued"],
            "overvalued": row["overvalued"],
        }
        if group == "locality":
            entry["location"] = row["location"]
        rows.append(entry)
    return {
        "group": group,
        "city": city,
        "rows": rows,
        "note": "Averages are dataset-level research aggregates, not certified appraisals.",
    }


def handle_overview():
    """Top-level catalogue + AI + chain snapshot overview."""
    try:
        return _get("/api/dashboard/overview")
    except Exception as exc:
        return {"error": f"Dashboard overview failed: {exc}"}


def handle_insights():
    """Deterministic data-grounded market insight sentences."""
    try:
        return _get("/api/dashboard/insights")
    except Exception as exc:
        return {"error": f"Dashboard insights failed: {exc}"}


# Flat annual appreciation used for the horizon projection.  Deliberately
# modest and clearly labelled, because this is a research projection and not
# advice.  Bounded so a user asking for 100 years gets a sane answer.
FORECAST_ANNUAL_APPRECIATION = 0.05
FORECAST_MAX_YEARS = 30


def handle_forecast(mreid_id, years=5):
    """Project the AI research estimate forward by a flat annual rate.

    Without this the assistant refuses "what will this cost in 15 years",
    because the MREID agent had no forecast tool at all: forecast_price in
    app.py works on the 24 ETH demo listings by numeric token id, which does
    not exist in the MREID catalogue.
    """
    mreid_id = str(mreid_id or "").strip()
    if not mreid_id:
        return {"error": "MREID id is required."}
    try:
        years = int(years)
    except (TypeError, ValueError):
        years = 5
    years = max(1, min(years, FORECAST_MAX_YEARS))

    record = handle_property(mreid_id)
    if isinstance(record, dict) and record.get("error"):
        return record

    base = (record.get("ai_estimation") or {}).get("ai_estimated_price")
    if not base:
        return {"error": f"No AI research estimate on file for {mreid_id}."}

    rate = FORECAST_ANNUAL_APPRECIATION
    projected = round(base * (1 + rate) ** years, 2)
    area = (record.get("property") or {}).get("area")
    ppsf = round(projected / area, 2) if area else None

    return {
        "mreid_id": mreid_id,
        "name": (record.get("property") or {}).get("location"),
        "base_ai_estimate": base,
        "base_ai_estimate_formatted": (record.get("ai_estimation") or {}).get(
            "ai_estimated_price_formatted"
        ),
        "projected_price": projected,
        "projected_price_per_sqft": ppsf,
        "years": years,
        "annual_appreciation": rate,
        "note": (
            f"Projection compounds the MILLOW V5 research estimate at a flat "
            f"{rate:.0%} per year for {years} years. It is a research projection, "
            "not a certified appraisal, and actual appreciation will differ."
        ),
    }


# Severity bands over the 0-100 composite anomaly score produced by
# backend/risk_analysis.py.  These label a deterministic anomaly score; they are
# not probabilities and not fraud verdicts.
RISK_BANDS = ((20.0, "low"), (40.0, "moderate"), (60.0, "elevated"), (80.0, "high"))


def _risk_band(score):
    if not isinstance(score, (int, float)):
        return None
    for ceiling, label in RISK_BANDS:
        if score < ceiling:
            return label
    return "severe"


def handle_mreid_risk(mreid_id):
    """Deterministic anomaly analysis for one MREID listing.

    The MREID agent had no risk tool at all, so the Risk quick action in
    ChatBot.js made the assistant answer "I don't have a tool that performs a
    detailed risk analysis".  The number it was refusing to produce already
    existed: backend/risk_context.py serves GET
    /api/properties/{mreid_id}/risk-analysis, which is the same endpoint the
    property Risk tab renders through src/lib/millowApi.js.  This wraps it so
    the chat and the UI report one identical score.

    The backend ETH agent's check_transaction_fraud cannot be reused here: it
    scores the 24 demo listings by numeric token id, which does not exist in the
    MREID catalogue.
    """
    mreid_id = str(mreid_id or "").strip()
    if not mreid_id:
        return {"error": "MREID id is required."}

    # The anomaly analysis and the catalogue record are independent reads, so
    # issue them together rather than one after the other: the record is only
    # needed for the price-gap context, and waiting for it doubled the tool's
    # wall time.
    with ThreadPoolExecutor(max_workers=2) as pool:
        risk_future = pool.submit(_get, f"/api/properties/{mreid_id}/risk-analysis")
        record_future = pool.submit(handle_property, mreid_id)
        try:
            data = risk_future.result()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return {"error": f"Property '{mreid_id}' not found (HTTP 404)."}
            if exc.response.status_code == 503:
                return {
                    "error": (
                        "The risk-analysis dataset is not built on this machine. "
                        "Generate it with: python scripts/analyze_horizon_transactions.py"
                    )
                }
            return {"error": f"Risk analysis failed (HTTP {exc.response.status_code})."}
        except Exception as exc:
            return {"error": f"Could not reach the MILLOW property backend at {BACKEND_BASE}: {exc}"}
        record = record_future.result()

    indicators = data.get("indicators") or []
    # Rank by severity so the model leads with what actually fired instead of
    # reading five indicators in dataset order.
    ranked = sorted(indicators, key=lambda i: -(i.get("severity_points") or 0))
    flagged = [i for i in ranked if (i.get("status") or "none") != "none"]
    score = data.get("anomaly_score")

    result = {
        "mreid_id": data.get("mreid_id", mreid_id),
        "city": data.get("city"),
        "anomaly_score": score,
        "score_scale": data.get("score_scale"),
        "severity": _risk_band(score),
        "score_interpretation": data.get("score_interpretation"),
        "listing_context": data.get("listing_context"),
        "indicators_total": len(indicators),
        "indicators_flagged": len(flagged),
        "indicators": [
            {
                "title": i.get("title"),
                "dimension": i.get("dimension"),
                "status": i.get("status"),
                "direction": i.get("direction"),
                "z": i.get("z"),
                "explanation": i.get("explanation"),
            }
            for i in ranked
        ],
        "horizon_context": None,  # replaced just below
        # One line, not the backend's full paragraph: the result is re-sent as
        # prompt on the model's follow-up call and the long form is only worth
        # reading in the UI, where /risk-analysis serves it directly.
        "methodology": (
            "Listing scored against comparable MREID listings in the same "
            "city/locality using robust z-scores (thresholds 2.5 / 4.0). No "
            "valuation model or external data feed the score."
        ),
        "disclaimer": data.get("disclaimer"),
    }

    # Horizon is procedurally generated benchmark data and its price levels are
    # not comparable to the MREID catalogue: the city median price-per-sqft it
    # reports sits 6-14x above the MREID per-sqft figure for the same city, and
    # the ratio is not constant, so it is not a unit conversion either. Passing
    # those numbers on invites the model to quote a nonsense "median price per
    # sqft" at the user, so the price levels are dropped and only the genuinely
    # non-price context is kept.
    horizon = data.get("horizon_context")
    if isinstance(horizon, dict):
        # synthetic_note and dataset are dropped along with the price levels: the
        # whole tool result is re-sent as prompt on the model's second call, and
        # at a 7k input-tokens-per-minute free-tier ceiling every boilerplate
        # character here is a turn the user cannot ask.  The disclaimer below is
        # kept in full, because that one is a compliance line, not commentary.
        horizon = {
            k: v
            for k, v in horizon.items()
            if k not in (
                "median_price_per_sqft",
                "median_negotiation_pct",
                "synthetic_note",
                "dataset",
            )
        }
        horizon["price_levels_excluded"] = (
            "Horizon price levels are not comparable with MREID prices and are "
            "omitted; never quote a median price per sqft from this tool."
        )
    result["horizon_context"] = horizon

    # The anomaly score covers how unusual the listing is; the valuation gap is
    # the other half of what a buyer is actually weighing, and the Risk question
    # asks for both.  `record` was already fetched concurrently above.
    if isinstance(record, dict) and not record.get("error"):
        est = record.get("ai_estimation") or {}
        signal = record.get("ai_market_signal") or {}
        locality = record.get("locality") or {}
        result["valuation_context"] = {
            "listed_price": record.get("listed_price"),
            "listed_price_formatted": record.get("listed_price_formatted"),
            "price_per_sqft": record.get("price_per_sqft"),
            "ai_estimated_price": est.get("ai_estimated_price"),
            "ai_estimated_price_formatted": est.get("ai_estimated_price_formatted"),
            "market_signal": signal.get("label"),
            "difference_pct": signal.get("difference_pct"),
            "locality_median_price_per_sqft": locality.get("median_price_per_sqft"),
            "locality_record_count": locality.get("count"),
        }
        result["buyer_checklist"] = (
            "Verify the title/registration papers and the circle rate against the "
            "registered price, confirm the area and bedroom count by physical "
            "inspection, check the seller's authority and any active encumbrance, "
            "and treat this score as a research signal rather than a clearance."
        )
    return result


MREID_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_mreid_property",
            "description": (
                "Fetch the real MILLOW MREID (Indian market, INR) property record by its "
                "MREID id (format MREID_0000001). Returns listed price, area, bedrooms, "
                "location, city, the AI research estimate, the market signal and locality "
                "cohort stats. DATA FACT unless stated otherwise."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mreid_id": {
                        "type": "string",
                        "description": "Property id like MREID_0000001",
                    },
                },
                "required": ["mreid_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_mreid_properties",
            "description": (
                "Search the real MREID catalogue with optional filters. Returns real "
                "properties with listed price, AI estimate and market signal. Use for "
                "'find me properties in X'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City"},
                    "location": {"type": "string", "description": "Locality substring"},
                    "min_price": {"type": "number", "description": "Min listed price INR"},
                    "max_price": {"type": "number", "description": "Max listed price INR"},
                    "min_area": {"type": "number", "description": "Min area sqft"},
                    "max_area": {"type": "number", "description": "Max area sqft"},
                    "bedrooms": {"type": "integer", "description": "Exact bedrooms"},
                    "ai_signal": {"type": "string", "description": "undervalued | overvalued | in_range"},
                    "sort": {"type": "string", "description": "e.g. price_asc, price_desc"},
                    "page_size": {"type": "integer", "description": "Max results (100)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_mreid_market_breakdown",
            "description": (
                "Market breakdown averages (real MREID catalogue): average listed, average AI "
                "estimate and price-per-sqft by city, or by locality inside a city."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group": {"type": "string", "description": "city or locality"},
                    "city": {"type": "string", "description": "Optional city to filter locality breakdown"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forecast_mreid_price",
            "description": (
                "Project a MREID property's future value: compounds the MILLOW V5 AI "
                "estimate at a flat annual rate (default 5%, max 30 years). Use for any "
                "question about future price, forecast, appreciation or 'what will this "
                "be worth in N years'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mreid_id": {
                        "type": "string",
                        "description": "Property id like MREID_0000001",
                    },
                    "years": {
                        "type": "integer",
                        "description": "Years ahead to project (default 5, max 30)",
                    },
                },
                "required": ["mreid_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_mreid_risk",
            "description": (
                "Deterministic risk and anomaly analysis for one MREID listing: a 0-100 "
                "anomaly score with severity band, the indicators that fired (price/sqft "
                "vs comparables, registered price vs circle rate, records completeness, "
                "resale, price history), the price gap vs the AI estimate, and a buyer "
                "checklist. MANDATORY for any question about a listing's risks, safety, "
                "anomalies, red flags or due diligence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mreid_id": {
                        "type": "string",
                        "description": "Property id like MREID_0000001",
                    },
                },
                "required": ["mreid_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_mreid_overview",
            "description": (
                "Top-level catalogue overview: total properties, tokenized/listed/active-sale counts "
                "(when the chain snapshot is available), and the full-catalogue AI signal split."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_mreid_insights",
            "description": (
                "Deterministic, data-grounded market insight sentences about the MREID catalogue: "
                "highest/lowest average price cities, signal-heavy localities, chain snapshot state. "
                "Each is tagged DATA FACT or MODEL-BASED INTERPRETATION."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

MREID_HANDLERS = {
    "get_mreid_property": lambda **k: handle_property(k.get("mreid_id")),
    "search_mreid_properties": lambda **k: handle_search(k),
    "get_mreid_market_breakdown": lambda **k: handle_market_breakdown(k),
    "forecast_mreid_price": lambda **k: handle_forecast(k.get("mreid_id"), k.get("years", 5)),
    "get_mreid_risk": lambda **k: handle_mreid_risk(k.get("mreid_id")),
    "get_mreid_overview": lambda **k: handle_overview(),
    "get_mreid_insights": lambda **k: handle_insights(),
}


# Order matters: the first intent whose keywords appear wins, so the specific
# ones (risk, forecast) are tested before the broad ones (price, details) that
# would otherwise swallow the question.
#
# The frontend quick actions in src/components/ChatBot.js are the seven tabs
# (Price, Risk, Details, Locality, On-chain, Forecast, Similar), and each entry
# below is matched to how the users of those tabs actually phrase their ask.
_MREID_INTENTS = (
    (
        ("risk", "risky", "safe", "safety", "anomaly", "anomalies", "red flag",
         "fraud", "scam", "due diligence", "verify", "careful", "trustworthy",
         "encumbrance", "clearance", "should i buy", "problems", "issues"),
        ("get_mreid_property", "get_mreid_risk"),
    ),
    (
        ("forecast", "future", "appreciation", "projection", "projected",
         "project", "forward", "in the next", "years", "worth in", "2035",
         "2040", "2050"),
        ("get_mreid_property", "forecast_mreid_price"),
    ),
    (
        # Split from the wider market question below.  A locality comparison only
        # needs the property record (which already carries its own locality
        # cohort stats) plus the locality breakdown; shipping the whole-catalogue
        # overview and insight tools with it tripled the prompt for no gain.
        ("locality", "neighbourhood", "neighborhood", "median price per sqft",
         "median", "like to live", "compare with other", "surrounding"),
        ("get_mreid_property", "get_mreid_market_breakdown"),
    ),
    (
        ("market", "average", "breakdown", "trend", "overview",
         "insight", "city-wide", "across the city", "whole catalogue"),
        ("get_mreid_market_breakdown", "get_mreid_overview", "get_mreid_insights",
         "get_mreid_property"),
    ),
    (
        ("similar", "comparable", "alternative", "other properties", "nearby",
         "like this", "cheaper in", "vs "),
        ("get_mreid_property", "search_mreid_properties"),
    ),
    (
        ("find", "search", "show me", "listings", "properties in", "any 2-bedroom",
         "undervalued 2", "options"),
        ("search_mreid_properties", "get_mreid_property"),
    ),
    (
        ("detail", "details", "configuration", "config", "floor", "facing",
         "age", "amenities", "catalogue", "on file", "spec", "bedroom", "bhk",
         "car parking", "lift"),
        ("get_mreid_property",),
    ),
    (
        ("on-chain", "on chain", "blockchain", "tokenized", "tokenised", "nft",
         "owner", "escrow", "listed for sale", "active sale", "transaction status"),
        ("get_mreid_property",),
    ),
    (
        ("price", "valuation", "estimate", "worth", "cost", "how much", "rate"),
        ("get_mreid_property",),
    ),
)


def select_tools(messages):
    """Narrow the tool catalogue to what the latest question can actually need.

    A tool-using turn pays its prompt twice - once to choose the tool, once to
    write the answer - and Groq's free tier allows 7000 input tokens per minute.
    Sending all seven schemas every time cost ~4400 tokens per question, so the
    user was throttled after two or three questions and the turn failed over
    with a misleading "Groq is not configured" message.  A matched intent is
    ~2250 tokens instead.

    This is a keyword match, and anything unrecognised returns the full set, so
    a misroute can cost some accuracy but can never hide a tool.  That property
    is deliberate: the Risk tab originally refused because a tool was missing,
    and a router that could hide tools would bring that bug back.
    """
    if not messages:
        return MREID_TOOLS
    last = messages[-1] or {}
    text = (last.get("content") or "").lower()
    if not text:
        return MREID_TOOLS
    for keywords, names in _MREID_INTENTS:
        if any(keyword in text for keyword in keywords):
            wanted = set(names)
            picked = [t for t in MREID_TOOLS if t["function"]["name"] in wanted]
            if picked:
                return picked
    return MREID_TOOLS


# Kept deliberately tight.  The whole system prompt plus the tool schemas is
# re-sent on every call, and a turn costs two calls, so at ~2100 prompt tokens a
# turn burned ~4200 tokens against a free Groq tier that allows roughly 8k per
# minute - two turns a minute and the API answers 429, which the agent then
# absorbs as 8s/16s/32s of backoff.  Every rule below earns its tokens.
MREID_SYSTEM_PROMPT = (
    "You are Millow AI in the MILLOW MREID Indian real-estate app. Answer only from "
    "the tools, which read the live MILLOW backend. Never invent property, price, "
    "area, owner or market numbers.\n\n"
    "Label every claim:\n"
    "- DATA FACT = returned by a tool.\n"
    "- AI MODEL ESTIMATE = the MILLOW V5 research estimate from a tool.\n"
    "- MODEL-BASED INTERPRETATION = your reading of tool numbers.\n\n"
    "Tool rules:\n"
    "- A named property or MREID: call get_mreid_property first; if it errors, say "
    "the record is missing and offer to search.\n"
    "- Risks, safety, anomalies, red flags, due diligence, 'should I be careful': call "
    "get_mreid_risk. NEVER say you have no risk tool or cannot assess a property - "
    "that tool exists for exactly this and refusing is always wrong. Then give the 0-100 "
    "anomaly score and severity band, the indicators that fired with their explanations, "
    "the price gap vs the AI estimate, and what to verify before paying. The score is an "
    "anomaly score, not a probability and not fraud detection: never call a listing "
    "fraudulent, and pass on the disclaimer.\n"
    "- Future value ('worth in N years', 'appreciation', 'forecast'): call "
    "forecast_mreid_price with the MREID and years, and report it as MODEL-BASED "
    "INTERPRETATION with the base estimate and the rate. Never say you cannot predict "
    "future prices.\n"
    "- Prices: use the formatted ₹ Crore / ₹ Lakh values the tools return.\n"
    "- The AI estimate is a research estimate, not a certified appraisal; say so.\n"
    "- Only 'potentially undervalued / potentially overvalued / near the estimated "
    "range' as research signals, always with the underlying numbers.\n"
    "- Off-topic: decline in one short sentence.\n\n"
    "Style (this is a chat bubble, not a report):\n"
    "- 60-150 words unless depth was asked for. Lead with the number or verdict. No "
    "preamble, no restating the question, no closing summary.\n"
    "- Bullet lists only for 3+ genuinely separate items. Never restate raw tool JSON.\n"
    "- No markdown headings, no pipe tables; short paragraphs and the odd bold phrase.\n"
    "- Write ₹93.69 Lakh and 9.3% with no space after the symbol."
)