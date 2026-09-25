"""
MILLOW - Risk & Anomaly Analysis engine (Phase 5)

Two deterministic, explainable analysis layers:

  * Transaction-level : scores Horizon transactions against their comparable
                        cohort (same city/type/quarter) plus the property's
                        own prior-sale history. Inputs are precomputed by
                        scripts/analyze_horizon_transactions.py and are
                        lazy-loaded only when a transaction is scored.
  * Property-level    : scores an MREID listing against comparable MREID
                        listings plus simple listing-quality rules. No model
                        call, no NHB, and the 184MB transaction summary is
                        not loaded. Horizon city aggregates are included only
                        as labelled context.

Scores are anomaly scores (0-100), NOT probabilities and NOT fraud
detection. No calibrated or certified claims are made.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

import properties  # noqa: E402

RISK_DIR = ROOT / "data" / "processed" / "risk"

SUMMARY_PATH = RISK_DIR / "horizon_transaction_summary.pkl"
STATS_PATH = RISK_DIR / "horizon_risk_stats.pkl"
AGG_PATH = RISK_DIR / "horizon_city_aggregates.pkl"

# ------------------------------------------------------------- constants --
Z_MEDIUM = 2.5
Z_HIGH = 4.0
COMPLETENESS_LOW = 0.5
COMPLETENESS_HIGH = 0.25
INDICATOR_COUNT_TRX = 8
INDICATOR_COUNT_PROP = 5
SCORE = "0-100"

# MREID city -> Horizon transaction city (names differ only in these two).
HORIZON_CITY_MAP = {
    "Bangalore": "Bengaluru",
    "Delhi": "New Delhi",
}

SEVERITY_LABEL = {0: "none", 1: "low", 2: "medium", 3: "high"}

SCORE_INTERPRETATION_TXN = (
    "Higher score = more unusual transaction relative to comparable "
    "transactions in the same city/type/quarter and the property's own "
    "prior-sale history. Not a probability."
)
SCORE_INTERPRETATION_PROP = (
    "Higher score = more unusual listing relative to comparable MREID "
    "listings in the same city/localities. Not a probability."
)

METHODOLOGY_TXN = (
    "Indicators compare each transaction to a comparable cohort (same city, "
    "property type and quarter). Robust z-scores use median and median "
    "absolute deviation with mild (2.5) and strong (4.0) thresholds. Rapid "
    "re-sale and price-jump indicators use only prior transactions on the "
    "same property (no future information). The financing-consistency flag "
    "fires when Payment_Mode is Loan but the loan flag differs."
)

METHODOLOGY_PROP = (
    "Indicators are computed from the property listing only, against "
    "comparable MREID listings (same city, or same locality when at least "
    "three comparable listings exist). Robust z-scores use median absolute "
    "deviation with mild (2.5) and strong (4.0) thresholds. No valuation "
    "model, NHB data or Horizon transaction rows feed this score."
)

DISCLAIMER_TXN = (
    "This is a statistical anomaly analysis of the supplied Horizon "
    "transaction dataset, which appears to be procedurally generated "
    "benchmark data. It is not fraud detection, is not a probability, and "
    "is not certified or guaranteed. Treat the result as research context "
    "only."
)

DISCLAIMER_PROP = (
    "This is a statistical anomaly analysis of the MREID property listing "
    "against other MREID listings. It is not fraud detection, not a "
    "probability, and not certified or guaranteed. Treat the result as "
    "research context only."
)

SYNTHETIC_NOTICE = (
    "City-level transaction statistics come from the Horizon "
    "Property_Transactions dataset, which appears to be procedurally "
    "generated benchmark data. They are descriptive context only and are "
    "not fed into the property anomaly score."
)


def _fmt_price(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return properties.format_indian_price(float(value))


def _mad(values):
    med = float(np.median(values))
    return float(np.median(np.abs(values - med))), med


# --------------------------------------------------- property cohort stats --
def _build_property_stats():
    props = properties.PROPERTIES
    ppsf = (props["price"] / props["area"]).to_numpy(dtype=float)
    area_bed = (
        props["area"] / props["no_of_bedrooms"].replace(0, np.nan)
    ).to_numpy(dtype=float)

    city = props["source_city"].to_numpy()
    loc = props["location"].fillna("").to_numpy()

    buckets = {}
    for i in range(len(props)):
        buckets.setdefault((city[i], loc[i]), []).append(i)
    city_buckets = {}
    for i in range(len(props)):
        city_buckets.setdefault(city[i], []).append(i)

    def stats_for(buckets_, values):
        out = {}
        for key, idxs in buckets_.items():
            vals = values[idxs]
            if len(vals) < 3:
                continue
            mad, med = _mad(vals)
            out[key] = {"n": int(len(vals)), "med": med, "mad": mad}
        return out

    return {
        "loc_ppsf": stats_for(buckets, ppsf),
        "city_ppsf": stats_for(city_buckets, ppsf),
        "city_area_bed": stats_for(city_buckets, area_bed),
    }


_prop_stats_cache = None


def _prop_stats():
    global _prop_stats_cache
    if _prop_stats_cache is None:
        _prop_stats_cache = _build_property_stats()
    return _prop_stats_cache


# ----------------------------------------------------------------- loaders --
_aggs_cache = None
_risk_cache = None


def _load_aggs():
    global _aggs_cache
    if _aggs_cache is None:
        _aggs_cache = pd.read_pickle(AGG_PATH)
    return _aggs_cache


def _load_risk():
    global _risk_cache
    if _risk_cache is not None:
        return _risk_cache
    df = pd.read_pickle(SUMMARY_PATH)
    with open(STATS_PATH, "rb") as f:
        stats = pickle.load(f)
    cache = {"df": df, "ids": df["txn_id"].to_numpy(), "stats": stats}
    _risk_cache = cache
    return cache


def risk_available():
    return all(p.exists() for p in (SUMMARY_PATH, STATS_PATH, AGG_PATH))


def _status_for_z(z):
    if z is None or not np.isfinite(z):
        return 0
    a = abs(z)
    if a >= Z_HIGH:
        return 3
    if a >= Z_MEDIUM:
        return 2
    return 0


def _direction_from_z(z):
    if z is None or not np.isfinite(z):
        return "n/a"
    return "above_typical" if z >= 0 else "below_typical"


def _composite(indicators, indicator_count):
    weights = sum(i["severity_points"] for i in indicators)
    return int(round(100.0 * weights / (3 * indicator_count)))


# ------------------------------------------------ transaction indicators ---
def _transaction_indicators(row):
    z_map = {
        "comparable_price": row["z_ppsf"],
        "circle_ratio": row["z_ratio"],
        "negotiation": row["z_neg"],
        "days_on_market": row["z_days"],
        "listed_gap": row["z_listed"],
    }
    s_map = {
        "comparable_price": int(row["s_price"]),
        "circle_ratio": int(row["s_ratio"]),
        "negotiation": int(row["s_neg"]),
        "days_on_market": int(row["s_days"]),
        "listed_gap": int(row["s_listed"]),
    }

    city = str(row["city"])
    qlabel = str(row["qlabel"])
    ttype = str(row["txn_type"])
    cohort_used = str(row["cohort_used"])
    cohort_n = int(row["cohort_n"])

    if cohort_used == "primary":
        comp_txt = f"{cohort_n} comparable {ttype} transactions in {city} ({qlabel})"
    elif cohort_used == "fallback":
        comp_txt = f"{cohort_n} comparable transactions in {city} ({qlabel})"
    else:
        comp_txt = "no usable comparable cohort"

    out = []

    def add_z(id_, title, dimension, z, status, explanation):
        out.append(
            {
                "id": id_,
                "title": title,
                "dimension": dimension,
                "status": SEVERITY_LABEL[status],
                "direction": _direction_from_z(z),
                "z": round(float(z), 2) if np.isfinite(z) else None,
                "explanation": explanation,
                "severity_points": status,
            }
        )

    ppsf = float(row["ppsf"])
    add_z(
        "comparable_price", "Sale price per sqft vs comparables", "pricing",
        z_map["comparable_price"], s_map["comparable_price"],
        f"Sale ₹/sqft {_fmt_price(ppsf)} vs {comp_txt}.",
    )
    ratio = float(row["ratio"])
    add_z(
        "circle_ratio", "Registered price vs circle rate", "registration",
        z_map["circle_ratio"], s_map["circle_ratio"],
        f"Sale/circle-rate ratio {ratio:.3f} vs {comp_txt}.",
    )
    neg = float(row["neg"])
    add_z(
        "negotiation", "Negotiation vs comparables", "negotiation",
        z_map["negotiation"], s_map["negotiation"],
        f"Negotiated discount {neg:.1f}% vs {comp_txt}.",
    )
    days = float(row["days"])
    add_z(
        "days_on_market", "Days on market vs comparables", "turnover",
        z_map["days_on_market"], s_map["days_on_market"],
        f"Time on market {int(days)} days vs {comp_txt}.",
    )
    listed_ratio = float(row["listed_ratio"]) if np.isfinite(row["listed_ratio"]) else None
    add_z(
        "listed_gap", "Transaction price vs property listed price", "listing_gap",
        z_map["listed_gap"], s_map["listed_gap"],
        (
            f"Transaction ₹/sqft is {listed_ratio:.2f}x this property's listed "
            f"₹/sqft (Horizon catalog) relative to other {city} transactions."
            if listed_ratio is not None
            else "No listed price available for this property in the Horizon catalog."
        ),
    )

    prior = int(row["prior_count"])
    gap = int(row["prev_gap_days"])
    churn_status = int(row["s_churn"])
    if churn_status:
        txt = (
            f"Property has {prior} prior sale(s); the closest one was only "
            f"{gap} days before this transaction (rapid repeat sale)."
        )
    else:
        txt = f"Property has {prior} prior sale(s); no rapid repeat pattern detected."
    out.append(
        {
            "id": "rapid_repeat_sale",
            "title": "Rapid repeat sale",
            "dimension": "turnover",
            "status": SEVERITY_LABEL[churn_status],
            "direction": "n/a",
            "z": None,
            "explanation": txt,
            "severity_points": churn_status,
        }
    )

    jump = float(row["jump_mult"]) if np.isfinite(row["jump_mult"]) else None
    jump_status = int(row["s_jump"])
    out.append(
        {
            "id": "price_jump",
            "title": "Price jump vs prior sale",
            "dimension": "pricing",
            "status": SEVERITY_LABEL[jump_status],
            "direction": "n/a",
            "z": None,
            "explanation": (
                f"Sale price is {jump:.2f}x the property's own earlier sale price."
                if jump is not None
                else "No prior sale on this property to compare against."
            ),
            "severity_points": jump_status,
        }
    )

    fin_status = int(row["s_fin"])
    out.append(
        {
            "id": "financing",
            "title": "Financing consistency",
            "dimension": "records",
            "status": SEVERITY_LABEL[fin_status],
            "direction": "n/a",
            "z": None,
            "explanation": (
                "Payment mode is recorded as 'Loan' but the loan flag is not set."
                if fin_status
                else "Payment mode and loan flag are consistent."
            ),
            "severity_points": fin_status,
        }
    )

    return out


# -------------------------------------------------- property indicators -----
def _property_indicators(row, props):
    ppsf = float(row["price"]) / float(row["area"])
    city = str(row["source_city"])
    loc = str(row["location"])

    stats = _prop_stats()
    group = stats["loc_ppsf"].get((city, loc)) or stats["city_ppsf"].get(city)
    z_ppsf = None
    n_txt = "no usable comparables"
    if group:
        n_txt = (
            f"{group['n']} comparable listings in {loc}, {city}"
            if (city, loc) in stats["loc_ppsf"]
            else f"{group['n']} comparable listings in {city}"
        )
        if group["mad"] > 0:
            z_ppsf = 0.6745 * (ppsf - group["med"]) / group["mad"]

    area_bed = None
    z_ab = None
    ab_med = None
    nbed = int(row["no_of_bedrooms"])
    if nbed > 0:
        area_bed = float(row["area"]) / nbed
        cg = stats["city_area_bed"].get(city)
        if cg and cg["mad"] > 0:
            z_ab = 0.6745 * (area_bed - cg["med"]) / cg["mad"]
            ab_med = cg["med"]

    amenity = properties.amenity_cols
    known = int((row[amenity].isin([0, 1])).sum())
    total = int(len(amenity))
    completeness = known / total if total else 1.0

    out = []

    def add(id_, title, dimension, status, direction, z, explanation):
        out.append(
            {
                "id": id_,
                "title": title,
                "dimension": dimension,
                "status": SEVERITY_LABEL[status],
                "direction": direction,
                "z": round(float(z), 2) if z is not None and np.isfinite(z) else None,
                "explanation": explanation,
                "severity_points": status,
            }
        )

    add(
        "listing_price_gap", "Listing price per sqft vs comparables", "pricing",
        _status_for_z(z_ppsf), _direction_from_z(z_ppsf), z_ppsf,
        (
            f"Listed {_fmt_price(ppsf)}/sqft vs median {_fmt_price(group['med'])}/sqft "
            f"of {n_txt}."
            if group
            else "No comparable MREID listings found for this city."
        ),
    )

    add(
        "bedroom_area", "Area per bedroom vs city norms", "structure",
        _status_for_z(z_ab), _direction_from_z(z_ab), z_ab,
        (
            f"{area_bed:.0f} sqft per bedroom across {nbed} bedroom(s) vs a "
            f"typical {ab_med:.0f} sqft per bedroom for {city} listings."
            if area_bed is not None and ab_med is not None
            else f"{area_bed:.0f} sqft per bedroom across {nbed} bedroom(s)."
            if area_bed is not None
            else "Bedroom count not available; area-per-bedroom skipped."
        ),
    )

    add(
        "amenity_disclosure", "Amenity disclosure completeness", "records",
        3 if completeness <= COMPLETENESS_HIGH else (2 if completeness <= COMPLETENESS_LOW else 0),
        "n/a", None,
        f"{known} of {total} amenity attributes are disclosed in this listing.",
    )

    return out


# ------------------------------------------------------------------ payloads --
def transaction_payload(txn_id):
    cache = _load_risk()
    df = cache["df"]
    ids = cache["ids"]

    pos = int(np.searchsorted(ids, txn_id))
    if pos >= len(ids) or ids[pos] != txn_id:
        return None
    row = df.iloc[pos]

    indicators = _transaction_indicators(row)

    return {
        "transaction_id": txn_id,
        "property_id": str(row["prop_id"]),
        "city": str(row["city"]),
        "transaction_type": str(row["txn_type"]),
        "quarter": str(row["qlabel"]),
        "anomaly_score": int(row["score"]),
        "score_scale": SCORE,
        "score_interpretation": SCORE_INTERPRETATION_TXN,
        "indicators": indicators,
        "comparable_group": {
            "level": str(row["cohort_used"]),
            "city": str(row["city"]),
            "quarter": str(row["qlabel"]),
            "n": int(row["cohort_n"]),
        },
        "methodology": METHODOLOGY_TXN,
        "disclaimer": DISCLAIMER_TXN,
    }


def property_payload(mreid_id):
    key = str(mreid_id).strip()
    if key not in properties._id_to_index:
        return None
    idx = properties._id_to_index[key]
    row = properties.PROPERTIES.iloc[idx]

    indicators = _property_indicators(row, properties.PROPERTIES)
    score = _composite(indicators, INDICATOR_COUNT_PROP)

    return {
        "mreid_id": key,
        "city": str(row["source_city"]),
        "analysis_subject": "property_listing",
        "anomaly_score": score,
        "score_scale": SCORE,
        "score_interpretation": SCORE_INTERPRETATION_PROP,
        "listing_context": {
            "price_per_sqft": round(float(row["price"]) / float(row["area"]), 2),
            "area_sqft": int(row["area"]),
            "bedrooms": int(row["no_of_bedrooms"]),
        },
        "indicators": indicators,
        "horizon_context": horizon_context(str(row["source_city"])),
        "methodology": METHODOLOGY_PROP,
        "disclaimer": DISCLAIMER_PROP,
    }


def horizon_context(city):
    """Horizon city aggregates for the mapped city (labelled context)."""
    aggs = _load_aggs()
    mapped = HORIZON_CITY_MAP.get(str(city).strip(), str(city).strip())
    if mapped not in aggs.index:
        return None
    a = aggs.loc[mapped]
    return {
        "city": mapped,
        "n_transactions": int(a["n_txn"]),
        "n_properties": int(a["n_props"]),
        "median_price_per_sqft": round(float(a["med_ppsf"]), 2),
        "median_negotiation_pct": round(float(a["med_neg"]), 2),
        "first_date": str(a["first_date"]),
        "last_date": str(a["last_date"]),
        "median_transactions_per_property": round(float(a["med_txn_per_prop"]), 2),
        "rapid_repeat_share": round(float(a["rapid_share"]), 4),
        "dataset": "Horizon Property_Transactions",
        "synthetic_note": SYNTHETIC_NOTICE,
    }