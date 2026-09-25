"""
MILLOW - AI/ML Dashboard router (Phase 6)

Endpoints that back the Dashboard's top cards, market breakdown, AI insights
and charts.  Everything is computed server-side from the real MREID catalogue
(+ V5 predictions, batch) and the offline chain-index snapshot — never by
scanning the chain from the frontend, and never by fabricating numbers.

Routes:
  * GET /api/dashboard/overview
        Top cards: total / tokenized / listed / active sales / finalized,
        plus the AI signal split across the catalogue and model metrics.
  * GET /api/dashboard/market-breakdown
        Average listed / AI / price-per-sqft by city or locality with counts.
  * GET /api/dashboard/insights
        Short, data-grounded market insight sentences (always tied to real
        computed numbers; clearly marked as research observations).

The V5 predictions across the whole catalogue are computed once and cached:
a single batched predict over ~29k rows, reused by every dashboard request.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

import chain_index  # noqa: E402
import properties  # noqa: E402

router = APIRouter()

SIGNAL_UNDERVALUED = properties.SIGNAL_UNDERVALUED
SIGNAL_OVERVALUED = properties.SIGNAL_OVERVALUED
SIGNAL_IN_RANGE = properties.SIGNAL_IN_RANGE

_MODEL_MAE = float(
    json.load(
        open(
            ROOT / "models" / "valuation" / "final" / "metadata.json",
            "r",
            encoding="utf-8",
        )
    ).get("validation", {}).get("mean_MAE")
    or 0.0
)

# ------------------------------------------------------------
# CACHE: full-catalogue V5 predictions + signal labels
# ------------------------------------------------------------

_CACHE = {"rows": None, "stamp": 0.0}
_CACHE_TTL = 300  # seconds


def _full_ai():
    now = time.time()
    if _CACHE["rows"] is not None and now - _CACHE["stamp"] < _CACHE_TTL:
        return _CACHE["rows"]

    df = properties.PROPERTIES.copy()

    prices, ppsfs = properties._batch_ai_predict(df)

    listed = df["price"].to_numpy(dtype=float)
    diff = np.asarray(prices, dtype=float) - listed
    with np.errstate(divide="ignore", invalid="ignore"):
        diff_pct = np.where(listed > 0, diff / np.maximum(listed, 1e-9) * 100.0, 0.0)

    labels = np.empty(len(df), dtype=object)
    labels[diff_pct > properties.SIGNAL_BAND_PCT] = SIGNAL_UNDERVALUED
    labels[diff_pct < -properties.SIGNAL_BAND_PCT] = SIGNAL_OVERVALUED
    labels[
        (diff_pct >= -properties.SIGNAL_BAND_PCT)
        & (diff_pct <= properties.SIGNAL_BAND_PCT)
    ] = SIGNAL_IN_RANGE

    df["_ai_price"] = prices
    df["_ai_ppsf"] = ppsfs
    df["_ai_diff"] = diff
    df["_ai_diff_pct"] = diff_pct
    df["_ai_label"] = labels

    _CACHE["rows"] = df
    _CACHE["stamp"] = now
    return df


def _chain_counts():
    counts = chain_index.counts()
    if counts is None:
        return None
    return {
        "total_properties": int(counts.get("total_properties") or 0),
        "tokenized": int(counts.get("tokenized") or 0),
        "listed": int(counts.get("listed") or 0),
        "active_sales": int(counts.get("active_sale") or 0),
        "finalized": int(counts.get("finalized") or 0),
        "errors": int(counts.get("errors") or 0),
    }


# ============================================================
# ROUTES
# ============================================================


@router.get("/api/dashboard/overview")
def dashboard_overview():
    df = _full_ai()

    label_counts = (
        df.groupby("_ai_label")["_ai_price"].count().to_dict()
    )

    chain = _chain_counts()
    if chain is None:
        chain_state = {
            "available": False,
            "exported_at": None,
            "note": (
                "Chain snapshot not yet available. Run "
                "npx hardhat run scripts/exportChainIndex.js "
                "--network localhost after provisioning."
            ),
        }
    else:
        chain_state = {
            "available": True,
            "exported_at": chain_index.exported_at(),
            **chain,
        }

    return {
        "catalogue": {
            "total": int(len(df)),
            "tokenized": None if chain is None else chain["tokenized"],
            "listed": None if chain is None else chain["listed"],
            "active_sales": None if chain is None else chain["active_sales"],
            "finalized": None if chain is None else chain["finalized"],
            "chain": chain_state,
        },
        "ai": {
            "model": properties.MODEL_NAME,
            "analyzed": int(len(df)),
            "undervalued": int(label_counts.get(SIGNAL_UNDERVALUED, 0)),
            "overvalued": int(label_counts.get(SIGNAL_OVERVALUED, 0)),
            "in_range": int(label_counts.get(SIGNAL_IN_RANGE, 0)),
            "avg_listed": float(round(df["price"].mean(), 2)),
            "avg_ai_estimate": float(round(df["_ai_price"].mean(), 2)),
            "avg_listed_ppsf": float(round(
                (df["price"] / df["area"]).mean(), 2
            )),
            "avg_ai_ppsf": float(round(df["_ai_ppsf"].mean(), 2)),
            "model_mae_inr": round(float(_MODEL_MAE), 2) if _MODEL_MAE else None,
            "note": (
                "Predictions computed once per process over the full MREID "
                "catalogue with the MILLOW V5 model. AI values are research "
                "estimates, not certified appraisals."
            ),
        },
        "disclaimer": properties.DISCLAIMER,
    }


@router.get("/api/dashboard/market-breakdown")
def market_breakdown(
    group: str = Query("city", pattern="^(city|locality)$"),
    city: str | None = None,
):
    df = _full_ai()

    if city:
        city_key = city.strip()
        mask = df["source_city"].str.casefold() == city_key.casefold()
        if not mask.any():
            raise HTTPException(
                status_code=400,
                detail=f"Unknown city '{city}'. "
                       f"Supported cities: {', '.join(properties.KNOWN_CITIES)}.",
            )
        df = df.loc[mask]

    grouped = df.copy()
    grouped["_ppsf"] = grouped["price"] / grouped["area"]

    if group == "city":
        keys = grouped.groupby("source_city")
        label = "city"
    else:
        keys = grouped.groupby(["source_city", "location"])
        label = "location"

    aggs = keys.agg(
        count=("price", "size"),
        avg_listed=("price", "mean"),
        avg_ai=("_ai_price", "mean"),
        avg_listed_ppsf=("_ppsf", "mean"),
        avg_ai_ppsf=("_ai_ppsf", "mean"),
        undervalued=("_ai_label", lambda s: int((s == SIGNAL_UNDERVALUED).sum())),
        overvalued=("_ai_label", lambda s: int((s == SIGNAL_OVERVALUED).sum())),
        in_range=("_ai_label", lambda s: int((s == SIGNAL_IN_RANGE).sum())),
    ).reset_index()

    rows = []
    for _, row in aggs.iterrows():
        entry = {
            "count": int(row["count"]),
            "avg_listed": float(round(row["avg_listed"], 2)),
            "avg_ai": float(round(row["avg_ai"], 2)),
            "avg_listed_ppsf": float(round(row["avg_listed_ppsf"], 2)),
            "avg_ai_ppsf": float(round(row["avg_ai_ppsf"], 2)),
            "undervalued": int(row["undervalued"]),
            "overvalued": int(row["overvalued"]),
            "in_range": int(row["in_range"]),
        }
        if label == "city":
            entry["city"] = row["source_city"]
        else:
            entry["city"] = row["source_city"]
            entry["location"] = row["location"]
        rows.append(entry)

    if label == "city":
        rows.sort(key=lambda r: r["city"])
    else:
        rows.sort(key=lambda r: (r["city"], r["location"]))

    return {
        "group": label,
        "city": city,
        "count": len(rows),
        "rows": rows,
        "note": (
            "Averages are dataset-level research aggregates. They describe "
            "the catalogue, not a specific property, and are not certified "
            "appraisals."
        ),
    }


@router.get("/api/dashboard/insights")
def dashboard_insights():
    df = _full_ai()

    chain = _chain_counts()

    total = int(len(df))
    undervalued = int((df["_ai_label"] == SIGNAL_UNDERVALUED).sum())
    overvalued = int((df["_ai_label"] == SIGNAL_OVERVALUED).sum())
    in_range = int((df["_ai_label"] == SIGNAL_IN_RANGE).sum())

    top_city = (
        df.groupby("source_city")["price"]
        .mean()
        .sort_values(ascending=False)
        .head(1)
    )
    cheapest_city = (
        df.groupby("source_city")["price"]
        .mean()
        .sort_values(ascending=True)
        .head(1)
    )
    largest_city = df["source_city"].value_counts().head(1)

    undervalued_locality = (
        df.loc[df["_ai_label"] == SIGNAL_UNDERVALUED, ["source_city", "location"]]
        .value_counts()
        .head(1)
    )
    overvalued_locality = (
        df.loc[df["_ai_label"] == SIGNAL_OVERVALUED, ["source_city", "location"]]
        .value_counts()
        .head(1)
    )

    insights = [
        {
            "text": (
                f"Across the {total:,} properties in the catalogue, "
                f"{undervalued:,} look potentially undervalued, {overvalued:,} "
                f"potentially overvalued, and {in_range:,} sit near the "
                f"MILLOW AI research estimate."
            ),
            "kind": "MODEL-BASED INTERPRETATION",
        },
    ]

    if not top_city.empty:
        city = top_city.index[0]
        avg = top_city.iloc[0]
        insights.append({
            "text": (
                f"{city} has the highest average listed price at "
                f"{properties.format_indian_price(float(avg))}."
            ),
            "kind": "DATA FACT",
        })
    if not cheapest_city.empty:
        city, avg = cheapest_city.index[0], cheapest_city.iloc[0]
        insights.append({
            "text": (
                f"{city} has the lowest average listed price at "
                f"{properties.format_indian_price(float(avg))}."
            ),
            "kind": "DATA FACT",
        })
    if not largest_city.empty:
        city = largest_city.index[0]
        count = largest_city.iloc[0]
        insights.append({
            "text": (
                f"{city} has the most catalogue entries "
                f"({int(count):,} properties)."
            ),
            "kind": "DATA FACT",
        })
    if not undervalued_locality.empty:
        idx = undervalued_locality.index[0]
        count = undervalued_locality.iloc[0]
        city, location = idx
        insights.append({
            "text": (
                f"{location} ({city}) has the most potentially undervalued "
                f"listings ({int(count)}) relative to the AI research estimate."
            ),
            "kind": "MODEL-BASED INTERPRETATION",
        })
    if not overvalued_locality.empty:
        idx = overvalued_locality.index[0]
        count = overvalued_locality.iloc[0]
        city, location = idx
        insights.append({
            "text": (
                f"{location} ({city}) has the most potentially overvalued "
                f"listings ({int(count)}) relative to the AI research estimate."
            ),
            "kind": "MODEL-BASED INTERPRETATION",
        })

    if chain is not None:
        insights.append({
            "text": (
                f"On-chain state (snapshot {chain_index.exported_at()}): "
                f"{chain['tokenized']:,} of {chain['total_properties']:,} "
                f"properties tokenized, {chain['listed']} listed, "
                f"{chain['active_sales']} active sales, "
                f"{chain['finalized']} finalized."
            ),
            "kind": "DATA FACT",
        })

    return {
        "insights": insights,
        "note": (
            "Insights are deterministic observations computed from the MREID "
            "catalogue, the MILLOW V5 research estimate, and the chain "
            "snapshot. 'MODEL-BASED INTERPRETATION' items are research "
            "signals, not certified valuations or investment advice."
        ),
    }