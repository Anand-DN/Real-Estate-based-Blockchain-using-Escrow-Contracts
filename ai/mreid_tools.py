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

import httpx

BACKEND_BASE = os.getenv(
    "MILLOW_PROPERTY_API_URL",
    "http://localhost:8001",
)

_TIMEOUT = httpx.Timeout(30.0)


def _get(path):
    response = httpx.get(f"{BACKEND_BASE}{path}", timeout=_TIMEOUT)
    response.raise_for_status()
    return response.json()


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
    try:
        data = _get("/api/properties/search")
    except Exception:
        data = None
    if data is None:
        return {"error": "Catalogue search failed. Backend unavailable?"}
    if params:
        try:
            qs = "&".join(f"{k}={v}" for k, v in params.items())
            data = _get(f"/api/properties/search?{qs}")
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
                "Search the real MILLOW MREID catalogue with optional filters (city, location, "
                "price/area ranges, bedrooms, AI market signal). Returns real properties with "
                "listed price, AI estimate and signal. Use for 'find me properties in X'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City (Bangalore, Chennai, Delhi, Hyderabad, Kolkata, Mumbai)"},
                    "location": {"type": "string", "description": "Locality substring, e.g. 'Whitefield'"},
                    "min_price": {"type": "number", "description": "Minimum listed price in INR"},
                    "max_price": {"type": "number", "description": "Maximum listed price in INR"},
                    "min_area": {"type": "number", "description": "Minimum area in sqft"},
                    "max_area": {"type": "number", "description": "Maximum area in sqft"},
                    "bedrooms": {"type": "integer", "description": "Exact number of bedrooms"},
                    "ai_signal": {"type": "string", "description": "undervalued | overvalued | in_range"},
                    "sort": {"type": "string", "description": "price_desc, price_asc, ai_difference_desc, ..."},
                    "page_size": {"type": "integer", "description": "Number of results (max 100)"},
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
    "get_mreid_overview": lambda **k: handle_overview(),
    "get_mreid_insights": lambda **k: handle_insights(),
}


MREID_SYSTEM_PROMPT = (
    "You are Millow AI operating inside the MILLOW MREID Indian real-estate platform. "
    "You answer questions about real properties from the MREID catalogue using ONLY "
    "the available tools, whose data comes from the live MILLOW backend — never invent "
    "property, price, area, owner, or market numbers.\n\n"
    "Every claim you make must be labelled with one of these prefixes in the answer:\n"
    "- DATA FACT    -> information returned by a tool from the real dataset/API.\n"
    "- AI MODEL ESTIMATE -> the MILLOW V5 research estimate returned by a tool.\n"
    "- MODEL-BASED INTERPRETATION -> your interpretation of tool-returned "
    "numbers (e.g. 'this looks potentially undervalued relative to the AI estimate').\n\n"
    "Rules:\n"
    "- If the user names a property or MREID, call get_mreid_property first. If it errors, "
    "say you don't have that record and offer to search.\n"
    "- Report prices in Indian Rupees using the formatted values the tool returns "
    "(\u20b9 Crore / \u20b9 Lakh).\n"
    "- The AI estimate is an AI-assisted research estimate, not a certified appraisal; "
    "say so when reporting valuation.\n"
    "- Never describe a property as fraudulent or mispriced; use only 'potentially "
    "undervalued / potentially overvalued / near the estimated market range' as research "
    "signals, always with the underlying numbers.\n"
    "- If the question is unrelated to the MREID catalogue, politely decline in one "
    "short sentence.\n"
    "- Answer only what was asked, in short readable paragraphs. Cite which tool "
    "returned the data (e.g. 'per the catalogue record')."
)