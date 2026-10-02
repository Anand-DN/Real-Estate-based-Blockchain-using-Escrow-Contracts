"""Experiment 2 conformal prediction core: scores, quantiles, intervals, metrics.

PHASE 1 SCOPE. This module is PURE:

  - no filesystem reads or writes
  - no model training
  - no imports from backend, .chain, or any Experiment 1 write path
  - no global mutable state

Every function takes arrays as arguments and returns arrays or plain dicts, so the
whole module is unit-testable without touching the repository.

Design decisions fixed in Phase 0 (§0 of docs/research/EXPERIMENT_2_CONFORMAL_PROTOCOL_DRAFT.md):

  cal_frac = 0.20                                    (not used here; Phase 2 concern)
  methods  = A, B1, B3, C1, C2, C3                    (B2 and C4 EXCLUDED)
  levels   = 0.80, 0.90, 0.95
  F3 gate  = median_relative_width <= 2.00 at nominal 0.90

METHOD SET (closed; no method may be added after results are observed):

  A   split conformal, absolute residual
      score      s_i = |y_i - yhat_i|
      interval   [yhat - q, yhat + q]
      width      constant in log space

  B1  normalized conformal, sigma_hat(x) = yhat(x)
      score      s_i = |y_i - yhat_i| / yhat_i
      interval   [yhat - q*sigma, yhat + q*sigma]
      width      varies with the prediction

  B3  normalized conformal with sigma_hat(x) = 1  -- IMPLEMENTATION CONTROL
      Must reproduce method A to within floating-point tolerance. If it does not,
      the normalization code is wrong. B3 is a self-test, not a method.

  C1  Mondrian conformal by source_city           (per-category quantiles)
  C2  Mondrian conformal by predicted-price tertile (per-band quantiles)
  C3  grouped conformal by source_city__location  -- SECONDARY ONLY

All scores and intervals are computed in log1p(price) space, the space the frozen
backbone predicts and the space MAE_log / R2_log are defined in. INR quantities are
derived at presentation time only.

NO CONFORMAL VALIDITY IS CLAIMED BY THIS MODULE. It computes quantities. Whether
those quantities indicate calibrated uncertainty is an empirical question that only
Phase 3-5 can answer.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Phase 0 pre-registered constants. These are CONTRACT, not defaults.
# ---------------------------------------------------------------------------

#: The only calibration fraction. Not to be varied after results are observed.
CAL_FRAC = 0.20

#: Pre-registered nominal coverage levels.
NOMINAL_LEVELS: Tuple[float, ...] = (0.80, 0.90, 0.95)

#: Pre-registered method set. B2 (kNN-normalized) and C4 (weighted) are EXCLUDED.
METHODS: Tuple[str, ...] = ("A", "B1", "B3", "C1", "C2", "C3")

#: Methods that produce a constant interval width in log space.
HOMOGENEOUS_METHODS: Tuple[str, ...] = ("A", "B3", "C1", "C2")

#: F3 usability gate: median row-level relative width at nominal 0.90.
F3_GATE_LEVEL = 0.90
F3_GATE_MAX_MEDIAN_RELATIVE_WIDTH = 2.00

#: Pre-declared subgroup suppression rule (§12.1).
MIN_SUBGROUP_N = 100

#: Target-space constants, matching scripts/valuation_v2_1/protocol.py.
TARGET_COLUMN = "price"
TRAINING_SPACE = "log1p(price)"
GROUP_COLUMN = "group"

#: Guard floor for B1's divisor. log1p(price) is >= 14.51 on this dataset, so this
#: guard should never activate. It exists so that a silent clip cannot occur, and
#: any activation is counted and reported rather than hidden (§7).
SCALE_FLOOR = 1e-9

#: Absolute tolerance for the B3-equals-A self-test (F0 gate).
B3_MATCH_TOLERANCE = 1e-12


def log_target(price: Sequence[float] | np.ndarray) -> np.ndarray:
    """log1p(price), matching protocol.log_target exactly."""
    return np.log1p(np.asarray(price, dtype=float))


def to_rupees(log_prediction: Sequence[float] | np.ndarray) -> np.ndarray:
    """Invert to INR with a floor at 0, matching protocol.to_rupees exactly."""
    return np.maximum(np.expm1(np.asarray(log_prediction, dtype=float)), 0.0)


def _as_float_1d(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    """Validate and return a finite 1-D float64 array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite; found {int((~np.isfinite(arr)).sum())} non-finite value(s)")
    return arr


def _as_float_2d(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    """Validate and return a finite 2-D float64 array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


# ---------------------------------------------------------------------------
# 7. Nonconformity scores
# ---------------------------------------------------------------------------


def absolute_residual_score(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """s_i = |y_i - yhat_i| in log space. Underlies A, B3, C1, C2, C3."""
    yt = _as_float_1d(y_true, "y_true")
    yp = _as_float_1d(y_pred, "y_pred")
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    return np.abs(yt - yp)


def normalized_residual_score(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    scale: Sequence[float] | np.ndarray | None = None,
) -> Tuple[np.ndarray, int]:
    """s_i = |y_i - yhat_i| / sigma_hat(x_i).

    scale=None implements B3 (sigma_hat == 1), the implementation control.
    scale=y_pred implements B1 (sigma_hat(x) = yhat(x)).

    Returns (scores, n_guard_activations). The guard count is reported, never
    swallowed: a non-positive or non-finite divisor would silently distort the
    score, so it is floored at SCALE_FLOOR and counted (§7).
    """
    yt = _as_float_1d(y_true, "y_true")
    yp = _as_float_1d(y_pred, "y_pred")
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")

    numerator = np.abs(yt - yp)
    if scale is None:
        return numerator, 0

    sig = _as_float_1d(scale, "scale")
    if sig.shape != yt.shape:
        raise ValueError(f"scale shape {sig.shape} does not match y_true {yt.shape}")

    bad = ~np.isfinite(sig) | (sig <= 0.0)
    n_guard = int(bad.sum())
    sig = np.where(bad, SCALE_FLOOR, sig)
    return numerator / sig, n_guard


def conformal_scores(
    method: str,
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    *,
    y_pred_cal: Sequence[float] | np.ndarray | None = None,
    strata: Sequence[Any] | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Nonconformity scores for one method.

    method
      "A"   absolute residual
      "B1"  residual / yhat
      "B3"  residual / 1  (control; identical to A)
      "C1"  absolute residual, tagged with its source_city stratum
      "C2"  absolute residual, tagged with its predicted-price tertile stratum
      "C3"  absolute residual (grouping happens in build_intervals)

    Returns (scores, info). info carries n_guard_activations, width_is_constant,
    and, for the Mondrian methods, the stratum labels aligned to scores.
    """
    if method not in METHODS:
        raise ValueError(
            f"method {method!r} is not in the pre-registered set {METHODS}. "
            "B2 and C4 are EXCLUDED by Phase 0 decision and cannot be added."
        )

    if method == "B1":
        if y_pred_cal is None:
            raise ValueError("B1 requires y_pred_cal to supply sigma_hat(x) = yhat(x)")
        scores, n_guard = normalized_residual_score(y_true, y_pred, scale=y_pred_cal)
        return scores, {
            "method": method,
            "n_guard_activations": n_guard,
            "width_is_constant": False,
            "scale": "y_hat",
        }

    if method == "B3":
        scores, n_guard = normalized_residual_score(y_true, y_pred, scale=None)
        return scores, {
            "method": method,
            "n_guard_activations": n_guard,
            "width_is_constant": True,
            "scale": "one",
        }

    scores = absolute_residual_score(y_true, y_pred)
    info: Dict[str, Any] = {
        "method": method,
        "n_guard_activations": 0,
        "width_is_constant": method in HOMOGENEOUS_METHODS,
        "scale": "one",
    }

    if method in ("C1", "C2"):
        if strata is None:
            raise ValueError(f"{method} requires strata labels")
        labels = np.asarray(strata)
        if labels.shape != scores.shape:
            raise ValueError(
                f"strata shape {labels.shape} does not match scores {scores.shape}"
            )
        info["strata"] = labels
    return scores, info


# ---------------------------------------------------------------------------
# 8. Quantile calculation
# ---------------------------------------------------------------------------


def conformal_quantile(
    scores: Sequence[float] | np.ndarray,
    alpha: float,
) -> Dict[str, Any]:
    """Exact finite-sample conformal quantile.

        k    = ceil( (n + 1) * (1 - alpha) )
        qhat = s_(k)   if k <= n
              = +inf   if k > n

    numpy.quantile interpolation is deliberately NOT used: the guarantee is
    stated for the k-th order statistic specifically, and interpolating between
    order statistics silently breaks it (§8).

    The k > n branch is unreachable at the planned calibration sizes but is
    implemented and unit-tested regardless, because an unreachable-but-untested
    guard is how a protocol acquires a silent failure mode.

    Returns a dict so callers cannot ignore the diagnostics:
      qhat, alpha, nominal, n, k, unbounded, method.
    """
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {a}")

    s = _as_float_1d(scores, "scores")
    n = int(s.size)
    k = int(np.ceil((n + 1) * (1.0 - a)))
    unbounded = k > n

    ordered = np.sort(s, kind="mergesort")
    qhat = float("inf") if unbounded else float(ordered[k - 1])

    return {
        "qhat": qhat,
        "alpha": a,
        "nominal": 1.0 - a,
        "n": n,
        "k": k,
        "unbounded": unbounded,
        "method": "exact_order_statistic",
    }


def conformal_quantile_by_stratum(
    scores: Sequence[float] | np.ndarray,
    strata: Sequence[Any],
    alpha: float,
    *,
    min_n: int = 1,
) -> Dict[str, Any]:
    """Mondrian quantiles: one exact quantile per stratum, computed on that
    stratum's calibration scores only.

    A stratum with fewer than min_n calibration rows cannot support an exact
    quantile. Such a stratum is recorded with fallback_used=True and a null
    quantile rather than being silently dropped or silently pooled.

    Returns {stratum_key: {"qhat", "n", "k", "unbounded", "fallback_used"}}.
    """
    s = _as_float_1d(scores, "scores")
    labels = np.asarray(strata)
    if labels.shape != s.shape:
        raise ValueError(
            f"strata shape {labels.shape} does not match scores {s.shape}"
        )

    out: Dict[Any, Dict[str, Any]] = {}
    for key in _unique_stable(labels):
        mask = labels == key
        n_stratum = int(mask.sum())
        if n_stratum < min_n:
            out[key] = {
                "qhat": None,
                "n": n_stratum,
                "k": None,
                "unbounded": None,
                "fallback_used": True,
            }
            continue
        q = conformal_quantile(s[mask], alpha)
        out[key] = {
            "qhat": q["qhat"],
            "n": q["n"],
            "k": q["k"],
            "unbounded": q["unbounded"],
            "fallback_used": False,
        }
    return out


def _unique_stable(labels: np.ndarray) -> list:
    """Unique labels in first-appearance order, so output ordering is deterministic."""
    seen: Dict[Any, None] = {}
    for v in labels.tolist():
        if v not in seen:
            seen[v] = None
    return list(seen.keys())


# ---------------------------------------------------------------------------
# Interval construction
# ---------------------------------------------------------------------------


def build_intervals(
    method: str,
    y_pred: Sequence[float] | np.ndarray,
    qhat: float | Mapping[Any, Dict[str, Any]],
    *,
    scale: Sequence[float] | np.ndarray | None = None,
    strata: Sequence[Any] | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Build (lower, upper) in log space.

    Pooled methods (A, B1, B3)
        qhat is a float; lower = yhat - qhat*sigma, upper = yhat + qhat*sigma.
        sigma is 1 except for B1, where scale must be supplied.

    Mondrian methods (C1, C2)
        qhat is a per-stratum mapping keyed by the stratum label. Rows whose
        stratum has fallback_used=True receive an unbounded interval
        (-inf, +inf), which is honest: no interval was calibrated for them, so
        no interval is claimed. The count is reported.

    C3 is handled by the caller, which supplies one qhat per group.

    Returns (lower, upper, info).
    """
    if method not in METHODS:
        raise ValueError(f"method {method!r} not in pre-registered set {METHODS}")

    yp = _as_float_1d(y_pred, "y_pred")

    if method in ("A", "B3", "B1"):
        if isinstance(qhat, Mapping):
            raise ValueError(f"method {method} expects a scalar qhat, got a mapping")
        q = float(qhat)
        if q == float("inf"):
            return (
                np.full(yp.shape, -np.inf),
                np.full(yp.shape, np.inf),
                {"method": method, "unbounded": True, "n_unbounded": int(yp.size),
                 "n_fallback": 0, "width_is_constant": method in HOMOGENEOUS_METHODS},
            )
        if method == "B1":
            if scale is None:
                raise ValueError("B1 requires scale = yhat(x)")
            sig = _as_float_1d(scale, "scale")
            if sig.shape != yp.shape:
                raise ValueError(f"scale shape {sig.shape} != y_pred {yp.shape}")
        else:
            sig = np.ones_like(yp)
        half = q * sig
        return yp - half, yp + half, {
            "method": method,
            "unbounded": False,
            "n_unbounded": 0,
            "n_fallback": 0,
            "width_is_constant": method in HOMOGENEOUS_METHODS,
        }

    # C1 / C2 : per-stratum quantiles
    if not isinstance(qhat, Mapping):
        raise ValueError(f"{method} expects a per-stratum qhat mapping, got {type(qhat)}")
    if strata is None:
        raise ValueError(f"{method} requires strata labels")

    labels = np.asarray(strata)
    if labels.shape != yp.shape:
        raise ValueError(f"strata shape {labels.shape} != y_pred {yp.shape}")

    lower = np.empty(yp.shape, dtype=float)
    upper = np.empty(yp.shape, dtype=float)
    n_fallback = 0
    n_unbounded = 0

    for key in _unique_stable(labels):
        mask = labels == key
        entry = qhat.get(key)
        if entry is None or entry.get("fallback_used"):
            lower[mask] = -np.inf
            upper[mask] = np.inf
            n_fallback += int(mask.sum())
            continue
        q = entry["qhat"]
        if q is None or q == float("inf"):
            lower[mask] = -np.inf
            upper[mask] = np.inf
            n_unbounded += int(mask.sum())
            continue
        half = float(q)
        lower[mask] = yp[mask] - half
        upper[mask] = yp[mask] + half

    return lower, upper, {
        "method": method,
        "unbounded": n_unbounded > 0,
        "n_unbounded": n_unbounded,
        "n_fallback": n_fallback,
        "width_is_constant": True,
    }


# ---------------------------------------------------------------------------
# 9. Coverage metrics
# ---------------------------------------------------------------------------


def coverage_metrics(
    y_true: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
    nominal: float,
    *,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Coverage for one (method, regime, fold, level, stratum) cell.

    Clopper-Pearson bounds are used rather than Wald intervals: coverage sits
    near 0.80-0.95 with n in the thousands, which is exactly where the normal
    approximation is least trustworthy (§9). This serves RQ10 -- separating real
    deviation from nominal from Monte-Carlo noise.

    Miscoverage is split into low and high tails rather than pooled, because an
    absolute-residual score on a right-skewed target can under-cover from one
    side only and pooling would hide it.
    """
    yt = _as_float_1d(y_true, "y_true")
    lo = _as_float_1d(lower, "lower")
    up = _as_float_1d(upper, "upper")
    if not (yt.shape == lo.shape == up.shape):
        raise ValueError(f"shape mismatch: y {yt.shape} lo {lo.shape} up {up.shape}")

    nom = float(nominal)
    if not 0.0 < nom < 1.0:
        raise ValueError(f"nominal must be in (0, 1), got {nom}")

    inside = (yt >= lo) & (yt <= up)
    n = int(yt.size)
    hits = int(inside.sum())
    empirical = hits / n

    below = int((yt < lo).sum())
    above = int((yt > up).sum())

    cp_low, cp_high = _clopper_pearson(hits, n, confidence)

    coverage_error = empirical - nom
    return {
        "n_evaluated": n,
        "n_covered": hits,
        "empirical_coverage": float(empirical),
        "nominal": nom,
        "alpha": 1.0 - nom,
        "coverage_error": float(coverage_error),
        "abs_coverage_error": float(abs(coverage_error)),
        "miscoverage_low": float(below / n),
        "miscoverage_high": float(above / n),
        "n_miscoverage_low": below,
        "n_miscoverage_high": above,
        "coverage_cp_lower": float(cp_low),
        "coverage_cp_upper": float(cp_high),
        "coverage_confidence": float(confidence),
        "coverage_significant": bool(cp_low > nom or cp_high < nom),
    }


def _clopper_pearson(successes: int, n: int, confidence: float) -> Tuple[float, float]:
    """Exact binomial confidence interval via the Beta distribution."""
    if n <= 0:
        return 0.0, 1.0
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(stats.beta.ppf(alpha / 2, successes, n - successes + 1))
    upper = 1.0 if successes == n else float(stats.beta.ppf(1 - alpha / 2, successes + 1, n - successes))
    return lower, upper


def coverage_sequence_by_level(
    coverages: Sequence[float],
    levels: Sequence[float] = NOMINAL_LEVELS,
) -> Dict[str, Any]:
    """Ordered empirical-coverage sequence across nominal levels.

    Implements the CORRECTED F5 (§0.5, §16): report whether the observed
    sequence is directionally monotone. This is DESCRIPTIVE. No particular
    direction is required, and non-monotonicity is not a falsifier.
    """
    cov = np.asarray(coverages, dtype=float)
    lev = np.asarray(levels, dtype=float)
    if cov.shape != lev.shape:
        raise ValueError(f"levels shape {lev.shape} != coverages shape {cov.shape}")

    order = np.argsort(lev)
    lev_sorted = lev[order]
    cov_sorted = cov[order]

    diffs = np.diff(cov_sorted)
    if diffs.size == 0:
        direction = "undefined"
        monotone = False
    elif np.all(diffs >= 0):
        direction = "increasing"
        monotone = True
    elif np.all(diffs <= 0):
        direction = "decreasing"
        monotone = True
    else:
        direction = "non_monotone"
        monotone = False

    return {
        "levels": lev_sorted.tolist(),
        "coverages": cov_sorted.tolist(),
        "coverage_errors": (cov_sorted - lev_sorted).tolist(),
        "direction": direction,
        "is_directionally_monotone": bool(monotone),
        "descriptive_only": True,
    }


# ---------------------------------------------------------------------------
# 10. Sharpness metrics
# ---------------------------------------------------------------------------


def interval_score(
    y_true: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Gneiting & Raftery (2007) interval score, per observation.

        IS_a = (u - l) + (2/a)(l - y) 1{y < l} + (2/a)(y - u) 1{y > u}

    Rewards coverage and penalises width in one proper scoring rule, so a method
    cannot hit nominal coverage with uselessly wide intervals without the score
    noticing.
    """
    yt = _as_float_1d(y_true, "y_true")
    lo = _as_float_1d(lower, "lower")
    up = _as_float_1d(upper, "upper")
    if not (yt.shape == lo.shape == up.shape):
        raise ValueError(f"shape mismatch: y {yt.shape} lo {lo.shape} up {up.shape}")

    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {a}")

    width = up - lo
    below = np.maximum(lo - yt, 0.0)
    above = np.maximum(yt - up, 0.0)
    return width + (2.0 / a) * (below + above)


def sharpness_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
    alpha: float,
    *,
    method: str = "A",
) -> Dict[str, Any]:
    """Width, interval score and F3 usability for one cell.

    Space handling: widths are reported primarily in log1p(price), the space the
    backbone predicts. INR widths are secondary and business-facing.

    The relative_width caveat of §10.1 is enforced structurally. For the
    homogeneous methods the log-space width is the constant 2*qhat, so
    relative_width = (upper-lower)/y_pred = 2*qhat*exp(-y_pred)... which is NOT
    constant. It is the INR relative width (upper_inr - lower_inr)/point_inr that
    equals 2*sinh(qhat) exactly, because expm1 is applied after the subtraction.
    Both are reported, and width_is_constant is recorded so a reader cannot
    misread the relative-width column.
    """
    yt = _as_float_1d(y_true, "y_true")
    yp = _as_float_1d(y_pred, "y_pred")
    lo = _as_float_1d(lower, "lower")
    up = _as_float_1d(upper, "upper")
    if not (yt.shape == yp.shape == lo.shape == up.shape):
        raise ValueError("y_true, y_pred, lower, upper must share a shape")

    a = float(alpha)
    width_log = up - lo
    iscore = interval_score(yt, lo, up, a)

    y_inr = to_rupees(yt)
    lo_inr = to_rupees(lo)
    up_inr = to_rupees(up)
    point_inr = to_rupees(yp)
    width_inr = up_inr - lo_inr

    # Relative width. For homogeneous methods this is a deterministic function of
    # qhat alone (§10.1) and carries no per-row information; the caller is told so.
    denom_log = np.where(np.abs(yp) > 0, np.abs(yp), np.nan)
    rel_width_log = width_log / denom_log
    denom_inr = np.where(point_inr > 0, point_inr, np.nan)
    rel_width_inr = width_inr / denom_inr

    width_is_constant = method in HOMOGENEOUS_METHODS
    median_rel_inr = float(np.nanmedian(rel_width_inr)) if np.any(np.isfinite(rel_width_inr)) else float("nan")

    return {
        "method": method,
        "alpha": a,
        "nominal": 1.0 - a,
        "n": int(yt.size),
        "width_is_constant": bool(width_is_constant),
        "mean_width_log": float(np.mean(width_log)),
        "median_width_log": float(np.median(width_log)),
        "sd_width_log": float(np.std(width_log, ddof=1)) if yt.size > 1 else 0.0,
        "min_width_log": float(np.min(width_log)),
        "max_width_log": float(np.max(width_log)),
        "mean_width_inr": float(np.mean(width_inr)),
        "median_width_inr": float(np.median(width_inr)),
        "mean_relative_width_log": float(np.nanmean(rel_width_log)) if np.any(np.isfinite(rel_width_log)) else float("nan"),
        "median_relative_width_log": float(np.nanmedian(rel_width_log)) if np.any(np.isfinite(rel_width_log)) else float("nan"),
        "mean_relative_width_inr": float(np.nanmean(rel_width_inr)) if np.any(np.isfinite(rel_width_inr)) else float("nan"),
        "median_relative_width_inr": median_rel_inr,
        "interval_score_log": float(np.mean(iscore)),
        "median_interval_score_log": float(np.median(iscore)),
        "winkler_score_log": float(np.mean(iscore)),
    }


def evaluate_usability_gate(
    metrics_by_level: Mapping[float, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Apply the pre-registered F3 gate at nominal 0.90.

    The threshold is fixed at Phase 0 and is Experiment-specific. It is not a
    universal real-estate standard and must never be presented as one.
    Calibration and usability are independent: a method may be calibrated and
    still fail this gate, and both facts are reported separately.
    """
    key = F3_GATE_LEVEL
    if key not in metrics_by_level:
        raise ValueError(
            f"F3 gate requires metrics at nominal {key}; got {sorted(metrics_by_level)}"
        )
    median_rel = metrics_by_level[key]["median_relative_width_inr"]
    usable = bool(median_rel <= F3_GATE_MAX_MEDIAN_RELATIVE_WIDTH)
    return {
        "gate_level_nominal": key,
        "median_relative_width_inr": float(median_rel),
        "threshold": F3_GATE_MAX_MEDIAN_RELATIVE_WIDTH,
        "usable": usable,
        "threshold_provenance": "Experiment-specific operational threshold; "
                                 "NOT a universal real-estate industry standard",
        "reported_independently_of_calibration": True,
    }


# ---------------------------------------------------------------------------
# 12. Subgroup coverage
# ---------------------------------------------------------------------------


def subgroup_coverage(
    y_true: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
    subgroup_labels: Sequence[Any],
    nominal: float,
    *,
    min_n: int = MIN_SUBGROUP_N,
    confidence: float = 0.95,
) -> list:
    """Coverage per subgroup, with the pre-declared suppression rule.

    A cell with n < min_n keeps its coverage value but is flagged
    coverage_reliable=False and is excluded from any summary statement. Without
    this, a 12-row cell could become a headline claim.

    Note the smallest city x true-price-band cell on this dataset is 1 row
    (Appendix A13), so the suppression rule is load-bearing, not decorative.
    """
    yt = _as_float_1d(y_true, "y_true")
    lo = _as_float_1d(lower, "lower")
    up = _as_float_1d(upper, "upper")
    labels = np.asarray(subgroup_labels)
    if not (yt.shape == lo.shape == up.shape == labels.shape):
        raise ValueError("all inputs must share a shape")

    rows = []
    for key in _unique_stable(labels):
        mask = labels == key
        m = coverage_metrics(yt[mask], lo[mask], up[mask], nominal, confidence=confidence)
        m["subgroup"] = key
        m["n_subgroup_rows_total"] = int(mask.sum())
        m["min_n_required"] = int(min_n)
        m["coverage_reliable"] = bool(m["n_evaluated"] >= min_n)
        rows.append(m)
    return rows


def price_band_labels(
    price: Sequence[float] | np.ndarray,
    edges: Sequence[float] = (0.0, 5e6, 1e7, 2e7, 4e7, 1e8, float("inf")),
    labels: Sequence[str] = ("<0.5cr", "0.5-1cr", "1-2cr", "2-4cr", "4-10cr", ">=10cr"),
) -> np.ndarray:
    """Fixed absolute INR price bands.

    Edges are absolute constants, not dataset quantiles, so the banding stays
    meaningful if the dataset changes. This banding depends on the TRUE price,
    which is why it is used for DIAGNOSTIC reporting only and never to assign a
    calibration stratum (§12.2a).
    """
    p = _as_float_1d(price, "price")
    if len(labels) != len(edges) - 1:
        raise ValueError("labels must have len(edges) - 1 entries")
    out = np.full(p.shape, "", dtype=object)
    for i, lab in enumerate(labels):
        lo_e, hi_e = edges[i], edges[i + 1]
        out[(p >= lo_e) & (p < hi_e)] = lab
    if np.any(out == ""):
        raise ValueError("price fell outside all declared edges")
    return out


def predicted_tertile_labels(
    y_pred: Sequence[float] | np.ndarray,
    cutpoints: Sequence[float] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predicted-price tertile labels for C2, plus the cutpoints used.

    cutpoints must be computed on T'_k predictions and applied unchanged to C_k
    and test_k (§12.2b). If cutpoints is None they are derived from y_pred,
    which is only correct when y_pred IS the T'_k prediction array.

    This is the ASSIGNABLE banding: it depends on the prediction, not the label,
    so unlike price_band_labels it is usable at prediction time.
    """
    yp = _as_float_1d(y_pred, "y_pred")
    if cutpoints is None:
        cuts = np.quantile(yp, [1.0 / 3.0, 2.0 / 3.0], method="linear")
    else:
        cuts = _as_float_1d(cutpoints, "cutpoints")
        if cuts.size != 2:
            raise ValueError(f"expected 2 tertile cutpoints, got {cuts.size}")
        if not cuts[0] < cuts[1]:
            raise ValueError(f"cutpoints must be strictly increasing, got {cuts.tolist()}")

    lab = np.full(yp.shape, "high", dtype=object)
    lab[yp < cuts[1]] = "mid"
    lab[yp < cuts[0]] = "low"
    return lab, cuts


# ---------------------------------------------------------------------------
# F0 implementation-validity gates
# ---------------------------------------------------------------------------


def b3_matches_a(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    nominal: float,
    *,
    tolerance: float = B3_MATCH_TOLERANCE,
) -> Dict[str, Any]:
    """F0 gate: sigma_hat == 1 must reproduce method A exactly.

    B3 exists to catch a broken normalization code path. If B3 and A disagree,
    the implementation is wrong and no other result is interpretable.
    """
    yt = _as_float_1d(y_true, "y_true")
    yp = _as_float_1d(y_pred, "y_pred")

    a_scores, _ = conformal_scores("A", yt, yp)
    b3_scores, _ = conformal_scores("B3", yt, yp)
    a_q = conformal_quantile(a_scores, 1.0 - nominal)["qhat"]
    b3_q = conformal_quantile(b3_scores, 1.0 - nominal)["qhat"]

    a_lo, a_up, _ = build_intervals("A", yp, a_q)
    b3_lo, b3_up, _ = build_intervals("B3", yp, b3_q)

    finite = np.isfinite(a_lo) & np.isfinite(a_up)
    max_diff = 0.0
    if np.any(finite):
        max_diff = max(
            float(np.max(np.abs(a_lo[finite] - b3_lo[finite]))),
            float(np.max(np.abs(a_up[finite] - b3_up[finite]))),
        )
    return {
        "nominal": float(nominal),
        "qhat_A": a_q,
        "qhat_B3": b3_q,
        "max_abs_interval_difference": max_diff,
        "tolerance": tolerance,
        "passes": bool(max_diff <= tolerance),
    }
