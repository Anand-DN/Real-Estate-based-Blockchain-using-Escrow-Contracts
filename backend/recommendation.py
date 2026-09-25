"""
MILLOW - Property Recommendation / Similar Properties engine (Phase 6)

Deterministic, explainable similarity ranking over the MREID property
inventory. Two modes share one scoring core:

  * similar_to_property  : rank candidates in the SAME city as a source
                           property (city is a hard eligibility filter, not
                           a scoring dimension).
  * requirements         : hard-filter candidates by user requirements
                           (city/location/bedrooms/price/area/amenities),
                           then relevance-rank the survivors.

City similarity is deliberately NOT a weighted ranking component; in both
modes a city requirement is an eligibility constraint only.

The recommendation score is a similarity/relevance measure (0-100). It is
NOT a quality score, NOT investment advice, NOT a 'best property' ranking,
and NOT derived from AI valuation error.

Ranking dimensions and weights (approved):
    locality          0.30
    bedrooms          0.15
    area              0.20
    price_level       0.25   (price_per_sqft for A-mode, price for B-mode)
    amenities         0.10   (co-known agreement; A-mode only)
The composite is a weighted mean over ACTIVE dimensions. Inactive amenities
are excluded from the denominator (unknown amenities are never treated as
mismatches).

Feature index (normalizers, per-city robust stats, amenity weights) is
derived once at import from MREID_property.csv, the same source that the
properties module already loads. No pairwise similarity matrix is stored.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

import properties  # noqa: E402

# ------------------------------------------------------------- constants --
WEIGHTS = {
    "locality": 0.30,
    "bedrooms": 0.15,
    "area": 0.20,
    "price_level": 0.25,
    "amenities": 0.10,
}

Z_KERNEL = 1.5          # exp(-0.5 * (z/1.5)^2), z = robust z of log value
REQ_KERNEL = 0.35       # absolute log-distance kernel for requirement bands
MIN_KNOWN_AMENITIES = 20
AMENITY_CONFIDENCE_L = 500.0
AMENITY_WEIGHT_CAP = 3.0
RANKING_WEIGHTS_SUM = sum(WEIGHTS.values())  # 1.0

LIMIT_MAX = 12

SCORE_INTERPRETATION = (
    "The recommendation score reflects similarity/relevance to the "
    "requested criteria only. It is not investment advice, not a 'best "
    "property' ranking, and not a prediction of value or return."
)

DISCLAIMER = (
    "Property recommendations are deterministic similarity-based matches "
    "computed from the MREID property dataset. They are not investment "
    "advice, not a 'best property' ranking, and not based on any valuation "
    "model or predicted return."
)

METHODOLOGY = (
    "Candidates are scored on locality exact-match (0.30), bedroom "
    "proximity (0.15), area similarity (0.20), price-level similarity "
    "(0.25) and co-known amenity agreement (0.10). Numerical closeness uses "
    "robust z-scores (median absolute deviation) of log-scaled area and "
    "price level, mapped through a kernel exp(-0.5*(z/1.5)^2). City is an "
    "eligibility constraint, not a ranking dimension. The score is the "
    "weighted mean over active dimensions, scaled to 0-100. It is a "
    "similarity measure, not quality, not a deal score, and not investment "
    "advice."
)


# ============================================================
# FEATURE INDEX (built once per process)
# ============================================================

_index_cache = None


def _logit(p):
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return float(np.log(p / (1.0 - p)))


def _mad_med(values):
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med, mad


def _build_index():
    df = properties.PROPERTIES

    area = df["area"].astype(float).to_numpy()
    price = df["price"].astype(float).to_numpy()
    ppsf = price / np.maximum(area, 1.0)
    log_area = np.log(np.maximum(area, 1.0))
    log_price = np.log(np.maximum(price, 1.0))
    log_ppsf = np.log(np.maximum(ppsf, 1.0))

    city = df["source_city"].astype(str).str.strip().to_numpy()
    loc = df["location"].astype(str).str.strip().to_numpy()
    bed = df["no_of_bedrooms"].astype(int).to_numpy()
    n = len(df)

    # City robustness stats (for A-mode area/price_per_sqft kernels).
    city_stats = {}
    for c in properties.KNOWN_CITIES:
        m = city == c
        la, _ = _mad_med(log_area[m])
        lp, _ = _mad_med(log_ppsf[m])
        city_stats[c] = {
            "n": int(m.sum()),
            "med_log_area": la,
            "mad_log_area": lp,   # reassigned below
        }
    # recompute properly (keep med/mad paired per value)
    city_stats = {}
    for c in properties.KNOWN_CITIES:
        m = city == c
        med_la, mad_la = _mad_med(log_area[m])
        med_lp, mad_lp = _mad_med(log_ppsf[m])
        city_stats[c] = {
            "n": int(m.sum()),
            "med_log_area": med_la,
            "mad_log_area": mad_la,
            "med_log_ppsf": med_lp,
            "mad_log_ppsf": mad_lp,
        }

    # Amenity matrix (values 0/1/9).
    amenity_matrix = df[properties.amenity_cols].to_numpy(dtype=np.int8)

    # Amenity discrimination weights over known (0/1) values only.
    known = (amenity_matrix == 0) | (amenity_matrix == 1)
    ones = amenity_matrix == 1
    known_row_count = known.sum(axis=1)
    overall_known_fraction = float(known.sum()) / max(known.size, 1)
    n_ones = int(ones.sum())
    baseline = _logit(n_ones / max(known.sum(), 1))
    w = np.zeros(len(properties.amenity_cols))
    for j, name in enumerate(properties.amenity_cols):
        kj = known[:, j].sum()
        if kj < MIN_KNOWN_AMENITIES:
            continue
        fj = float(ones[:, j].sum()) / kj
        disc = abs(_logit(fj) - baseline)
        conf = min(1.0, kj / AMENITY_CONFIDENCE_L)
        w[j] = min(disc * conf, AMENITY_WEIGHT_CAP)

    return {
        "n": n,
        "city": city,
        "loc": loc,
        "bed": bed,
        "area": area,
        "price": price,
        "ppsf": ppsf,
        "log_area": log_area,
        "log_price": log_price,
        "log_ppsf": log_ppsf,
        "city_stats": city_stats,
        "amenity": amenity_matrix,
        "amenity_weights": w,
        "amenity_cols": list(properties.amenity_cols),
    }


def index():
    global _index_cache
    if _index_cache is None:
        _index_cache = _build_index()
    return _index_cache


# ============================================================
# SCORING
# ============================================================

def _robust_z(value, med, mad):
    if mad <= 0 or not np.isfinite(mad):
        return np.nan
    return 0.6745 * (value - med) / mad


def _kernel(z):
    z = np.asarray(z, dtype=float)
    out = np.exp(-0.5 * (z / Z_KERNEL) ** 2)
    out[~np.isfinite(out)] = 0.0
    return out


def _bedroom_sim(d):
    d = np.abs(np.asarray(d, dtype=int))
    out = np.zeros(len(d), dtype=float)
    out[d == 0] = 1.0
    out[d == 1] = 0.5
    return out


def _amenity_similarity(a, b):
    """Co-known weighted agreement between two amenity rows.

    Unknown (9) never counts as a mismatch. Returns (similarity, n_matching,
    n_known_both) or (None, 0, 0) when there is no co-known amenity weight.
    """
    both_known = (a != 9) & (b != 9)
    w = index()["amenity_weights"]
    wsum = float(w[both_known].sum())
    if wsum <= 0:
        return None, 0, 0
    agree = (a == b) & both_known
    n_match = int(((a == 1) & (b == 1) & both_known).sum())
    sim = float(w[agree].sum() / wsum)
    return sim, n_match, int(both_known.sum())


def _dimension_scores(idx, candidates, src_pos=None, req=None):
    """Vectorized per-dimension similarity for candidate positions."""
    xi = index()
    out = {}
    if src_pos is None:
        src_pos = -1

    # locality
    if src_pos != -1:
        same_loc = (
            (xi["city"][candidates] == xi["city"][src_pos])
            & (xi["loc"][candidates] == xi["loc"][src_pos])
        )
        out["locality"] = same_loc.astype(float)
    elif req.get("location"):
        req_loc = str(req["location"]).strip().casefold()
        out["locality"] = np.where(
            np.char.lower(np.asarray(xi["loc"][candidates], dtype=str))
            == req_loc,
            1.0,
            0.0,
        )

    # bedrooms
    if src_pos != -1:
        d = xi["bed"][candidates] - xi["bed"][src_pos]
    elif req.get("bedrooms") is not None:
        d = xi["bed"][candidates] - int(req["bedrooms"])
    else:
        skip = True
        d = np.zeros(len(candidates), dtype=int)
    if src_pos != -1 or req.get("bedrooms") is not None:
        out["bedrooms"] = _bedroom_sim(d)

    # area (robust z difference of log area, kernel-mapped)
    if src_pos != -1:
        cs = xi["city_stats"][xi["city"][src_pos]]
        med, mad = cs["med_log_area"], cs["mad_log_area"]
        z = _robust_z(xi["log_area"][candidates], med, mad) - _robust_z(
            xi["log_area"][src_pos], med, mad
        )
        out["area"] = _kernel(z)
    elif req.get("area_log_mid") is not None:
        d = np.abs(xi["log_area"][candidates] - req["area_log_mid"])
        out["area"] = np.exp(-0.5 * (d / REQ_KERNEL) ** 2)

    # price level (ppsf for similar-to-property, price for requirements)
    if src_pos != -1:
        cs = xi["city_stats"][xi["city"][src_pos]]
        med, mad = cs["med_log_ppsf"], cs["mad_log_ppsf"]
        z = _robust_z(xi["log_ppsf"][candidates], med, mad) - _robust_z(
            xi["log_ppsf"][src_pos], med, mad
        )
        out["price_level"] = _kernel(z)
    elif req.get("price_log_mid") is not None:
        d = np.abs(xi["log_price"][candidates] - req["price_log_mid"])
        out["price_level"] = np.exp(-0.5 * (d / REQ_KERNEL) ** 2)

    # amenities (similar-to-property mode only; co-known agreement)
    if src_pos != -1:
        w = xi["amenity_weights"]
        a = xi["amenity"][src_pos]
        b = xi["amenity"][candidates]
        both_known = (a != 9) & (b != 9)
        wsum_by = np.where(both_known, w, 0.0).sum(axis=1)
        agree = (a == b) & both_known
        n_match = ((a == 1) & (b == 1) & both_known)
        with np.errstate(invalid="ignore", divide="ignore"):
            sim = np.where(
                wsum_by > 0,
                np.where(agree, w, 0.0).sum(axis=1) / wsum_by,
                0.0,
            )
        out["amenities"] = sim
        out["_amenity_match_count"] = n_match.sum(axis=1)

    return out


def _composite(dim_scores):
    total_w = 0.0
    total = 0.0
    for k, w in WEIGHTS.items():
        if k not in dim_scores:
            continue
        v = dim_scores[k]
        if v is None:
            continue
        total_w += w
        total += w * float(v)
    if total_w <= 0:
        return 0.0
    return total / total_w


def _reason_lines(mode, dim_scores, src_row=None, req=None, cand_row=None, n_match=0):
    reasons = []
    if mode == "similar_to_property":
        if dim_scores.get("locality") == 1.0:
            reasons.append(f"Same locality: {src_row['location']}")
        else:
            reasons.append(f"Compared against {src_row['source_city']} listings")
        if dim_scores.get("bedrooms") == 1.0:
            reasons.append(f"Similar {src_row['no_of_bedrooms']} bedrooms")
        elif 0 < dim_scores.get("bedrooms", 0) < 1.0:
            reasons.append(f"Close to {src_row['no_of_bedrooms']} bedrooms")
        if dim_scores.get("area") is not None and dim_scores["area"] >= 0.6:
            reasons.append(
                f"Similar area (~{int(cand_row['area']):,} sqft)"
            )
        if dim_scores.get("price_level") is not None and dim_scores["price_level"] >= 0.6:
            reasons.append(
                "Similar price level (~"
                f"{properties.format_indian_price(float(cand_row['price']) / max(float(cand_row['area']), 1.0))}"
                "/sqft)"
            )
        if n_match >= 1:
            reasons.append(f"{n_match} matching amenit{'y' if n_match == 1 else 'ies'}")
    else:
        if req.get("city"):
            reasons.append(f"Matches {req['city']}")
        if req.get("location") and dim_scores.get("locality") == 1.0:
            reasons.append(f"Same locality: {req['location']}")
        if req.get("location") and dim_scores.get("locality") != 1.0:
            reasons.append(f"Different locality ({req.get('city') or 'any city'})")
        if req.get("bedrooms") is not None:
            if dim_scores.get("bedrooms") == 1.0:
                reasons.append(f"{req['bedrooms']}-bedroom match")
            elif 0 < dim_scores.get("bedrooms", 0) < 1.0:
                reasons.append(f"Close to {req['bedrooms']} bedrooms")
        if req.get("area_log_mid") is not None:
            reasons.append("Within requested area range")
        if req.get("price_log_mid") is not None:
            reasons.append("Within requested price range")
        if req.get("amenities"):
            reasons.append(f"Required amenities confirmed: {', '.join(req['amenities'])}")
    return reasons[:5]


# ============================================================
# QUERY BUILDERS
# ============================================================

def similar_property_query(mreid_id):
    """Build (src_pos, candidates, req) for A-mode."""
    key = str(mreid_id).strip()
    if key not in properties._id_to_index:
        return None
    src_pos = properties._id_to_index[key]
    xi = index()
    city = xi["city"][src_pos]
    candidates = np.flatnonzero(xi["city"] == city)
    candidates = candidates[candidates != src_pos]
    return src_pos, candidates, {"city": city}


def requirements_query(req):
    """Hard-filter mask for B-mode. Raises ValueError with a message on
    invalid requirements."""
    xi = index()
    mask = np.ones(xi["n"], dtype=bool)

    city = (req.get("city") or "").strip()
    location = (req.get("location") or "").strip()
    bedrooms = req.get("bedrooms")
    min_price = req.get("min_price")
    max_price = req.get("max_price")
    min_area = req.get("min_area")
    max_area = req.get("max_area")
    amenities = [a.strip() for a in (req.get("amenities") or []) if a.strip()]

    if city:
        city_keys = {c.casefold(): c for c in properties.KNOWN_CITIES}
        if city.casefold() not in city_keys:
            raise ValueError(
                f"Unknown city '{city}'. Supported cities: "
                f"{', '.join(properties.KNOWN_CITIES)}."
            )
        canonical = city_keys[city.casefold()]
        mask &= np.char.lower(np.asarray(xi["city"], dtype=str)) == city.casefold()
        req["city"] = canonical

    if location:
        mask &= (
            np.char.find(
                np.char.lower(np.asarray(xi["loc"], dtype=str)),
                location.casefold(),
            )
            >= 0
        )

    if min_price is not None or max_price is not None:
        if min_price is not None and max_price is not None and min_price > max_price:
            raise ValueError("min_price cannot be greater than max_price.")
        if min_price is not None:
            mask &= xi["price"] >= min_price
        if max_price is not None:
            mask &= xi["price"] <= max_price
        if min_price is not None and max_price is not None:
            mid = float(np.log(np.maximum(min_price, 1.0))) / 1.0 + float(
                np.log(np.maximum(max_price, 1.0))
            )
            req["price_log_mid"] = mid / 2.0
        elif min_price is not None:
            req["price_log_mid"] = float(np.log(np.maximum(min_price, 1.0)))
        else:
            req["price_log_mid"] = float(np.log(np.maximum(max_price, 1.0)))

    if min_area is not None or max_area is not None:
        if min_area is not None and max_area is not None and min_area > max_area:
            raise ValueError("min_area cannot be greater than max_area.")
        if min_area is not None:
            mask &= xi["area"] >= min_area
        if max_area is not None:
            mask &= xi["area"] <= max_area
        if min_area is not None and max_area is not None:
            req["area_log_mid"] = 0.5 * (
                float(np.log(np.maximum(min_area, 1.0)))
                + float(np.log(np.maximum(max_area, 1.0)))
            )
        elif min_area is not None:
            req["area_log_mid"] = float(np.log(np.maximum(min_area, 1.0)))
        else:
            req["area_log_mid"] = float(np.log(np.maximum(max_area, 1.0)))

    if bedrooms is not None:
        mask &= xi["bed"] == int(bedrooms)

    if amenities:
        known = {c.casefold(): c for c in properties.amenity_cols}
        unknown = [a for a in amenities if a.casefold() not in known]
        if unknown:
            raise ValueError(f"Unknown amenity: {', '.join(unknown)}.")
        for a in amenities:
            col = known[a.casefold()]
            j = properties.amenity_cols.index(col)
            mask &= xi["amenity"][:, j] == 1

    if not any([
        city, location, bedrooms is not None,
        min_price is not None, max_price is not None,
        min_area is not None, max_area is not None,
        amenities,
    ]):
        raise ValueError(
            "Provide at least one requirement (city, location, bedrooms, "
            "price range, area range or amenities)."
        )

    candidates = np.flatnonzero(mask)
    return candidates, req


def _query_description(mode, req, src_pos=None, query_echo=None):
    if mode == "similar_to_property":
        row = properties.PROPERTIES.iloc[src_pos]
        return {
            "mreid_id": str(row["mreid_id"]),
            "city": str(row["source_city"]),
            "location": str(row["location"]),
            "bedrooms": int(row["no_of_bedrooms"]),
            "area": int(row["area"]),
            "price_per_sqft": round(float(row["price_per_sqft"]), 2),
        }
    if query_echo is not None:
        return dict(query_echo)
    return {k: v for k, v in req.items() if not k.endswith("_log_mid")}


_RECOMMENDATION_META = {
    "score_scale": "0-100",
    "score_interpretation": SCORE_INTERPRETATION,
    "methodology": METHODOLOGY,
    "disclaimer": DISCLAIMER,
}


def _item_payload(pos, dim_scores, reasons, ai_price, ai_ppsf):
    row = properties.PROPERTIES.iloc[pos]
    from properties import _summary

    base = _summary(row, ai_price, ai_ppsf)
    return {
        "mreid_id": base["mreid_id"],
        "recommendation_score": round(100.0 * _composite(dim_scores)),
        "dimension_scores": {
            k: round(float(v), 3)
            for k, v in dim_scores.items()
            if not k.startswith("_")
        },
        "reasons": reasons,
        "property": base,
    }


def recommend_similar(mreid_id, limit):
    q = similar_property_query(mreid_id)
    if q is None:
        return None
    src_pos, candidates, req = q
    return _rank(candidates, "similar_to_property", req, src_pos=src_pos, limit=limit, query_echo=None)


def recommend_requirements(req, limit):
    _PUBLIC_FILTER_KEYS = (
        "city", "location", "bedrooms",
        "min_price", "max_price", "min_area", "max_area", "amenities",
    )
    echo = {
        k: (list(v) if k == "amenities" else v)
        for k, v in req.items()
        if k in _PUBLIC_FILTER_KEYS and v is not None
    }
    candidates, working = requirements_query(dict(req))
    return _rank(
        candidates, "requirements", working, src_pos=None, limit=limit, query_echo=echo
    )


def _rank(candidates, mode, req, src_pos, limit, query_echo=None):
    xi = index()
    src_pos = src_pos if src_pos is not None else np.intp(-1)
    ds = _dimension_scores(idx=xi, candidates=candidates, src_pos=src_pos, req=req)

    # composite per candidate = weighted mean over present dimensions
    total_w = np.zeros(len(candidates))
    total = np.zeros(len(candidates))
    for k, w in WEIGHTS.items():
        if k not in ds:
            continue
        v = ds[k]
        total_w += w
        total += w * v
    comp = np.where(total_w > 0, total / np.maximum(total_w, 1e-12), 0.0)

    order = np.argsort(-comp, kind="stable")
    order = order[:int(limit)]

    ai_prices, ai_ppsfs = properties._batch_ai_predict(
        properties.PROPERTIES.iloc[candidates[order]]
    )

    items = []
    for i, rank_pos in enumerate(order):
        pos = candidates[rank_pos]
        n_match = int(ds["_amenity_match_count"][rank_pos]) if "_amenity_match_count" in ds else 0
        dims = {k: float(v[rank_pos]) for k, v in ds.items() if not k.startswith("_")}
        reasons = _reason_lines(
            mode,
            dims,
            src_row=properties.PROPERTIES.iloc[src_pos] if src_pos != -1 else None,
            req=req,
            cand_row=properties.PROPERTIES.iloc[pos],
            n_match=n_match,
        )
        items.append(_item_payload(pos, dims, reasons, ai_prices[i], ai_ppsfs[i]))

    return {
        "mode": mode,
        "query": _query_description(mode, req, src_pos if src_pos != -1 else None, query_echo),
        "count": len(items),
        "limit": int(limit),
        "candidate_count": int(len(candidates)),
        "items": items,
        **_RECOMMENDATION_META,
    }