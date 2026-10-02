"""Experiment 2 Phase 4: conformal scoring and interval generation.

This phase is a PURE CONSUMER of the frozen upstream artifacts:

    Phase 1  scripts/experiment_2/conformal.py          (pure score/quantile code)
    Phase 2  artifacts/valuation/v2_2/conformal/calibration_folds_v2_2.json
    Phase 3  artifacts/valuation/v2_2/conformal/nested_models_manifest.json
             models/valuation/v2_2/conformal/{random,location_grouped}/

It computes, for every (regime, outer_fold, method, nominal_level), conformal
scores calibrated on C_k and intervals on test_k, entirely in log1p(price)
space. Interval endpoints are transformed to INR at presentation time with
expm1 and a floor at 0.

METHODS (closed set, exactly as resolved in the protocol addendum 8.1):

    A   |y - yhat|
    B1  |y - yhat| / yhat          (sigma_hat(x) = yhat)
    B3  |y - yhat| / 1             (control; must equal A)
    C1  |y - yhat| per source_city (Mondrian)
    C2  |y - yhat| per predicted-price tertile (Mondrian; cutpoints from T'_k)
    C3  one score per calibration group, s_g = mean(|y - yhat|), then one
        finite-sample quantile across the calibration-group scores

C1/C2 fallback (binding resolution 8.1.2):
    unseen or insufficient stratum  ->  (-inf, +inf) with fallback_used=True
    insufficient is defined ONLY as  k > n_stratum
    a pooled C_k quantile is NEVER substituted.

ISOLATION. Calibration quantiles, strata, bands and cutpoints are computed from
C_k and T'_k only. No test observation participates in any of them. The
per-fold scoring function `compute_fold_conformal` does not even receive the
test labels y_test, so test-driven calibration is structurally impossible.

SCOPE. Scoring only. No coverage, interval-width, subgroup, monotonicity or
method-superiority quantity is computed or claimed here. All interval rows are
persisted so a later phase can evaluate them.

SAFETY. Writes only under:

    artifacts/valuation/v2_2/conformal/scores/
    artifacts/valuation/v2_2/conformal/intervals/
    artifacts/valuation/v2_2/conformal/conformal_scoring_manifest.json

No model is retrained. Nothing under Experiment 1, V2.1, blockchain, the
dataset, the token map, .gitignore or .gitattributes is touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_2 import build_calibration_folds as B  # noqa: E402
from scripts.experiment_2 import conformal as C  # noqa: E402
from scripts.experiment_2 import train_nested_models as T  # noqa: E402
from scripts.valuation_v2_1 import protocol as P  # noqa: E402


# ============================================================
# CONTRACT
# ============================================================

#: SHA-256 of the Phase 3 nested-model manifest. If it changes, Phase 4 refuses
#: to score rather than consume an unknown model set.
NESTED_MANIFEST_SHA256 = (
    "b2072d0b53f7475437ddeffc50ff3fe1eaf592261c289aa48c40e37d435b3475"
)

#: Pre-registered levels, inherited from the frozen Phase 1 module.
NOMINAL_LEVELS: Tuple[float, ...] = C.NOMINAL_LEVELS

#: Closed method set, inherited from the frozen Phase 1 module.
METHODS: Tuple[str, ...] = C.METHODS

REGIMES: Tuple[str, ...] = B.REGIMES

SCORES_DIRNAME = "scores"
INTERVALS_DIRNAME = "intervals"
MANIFEST_NAME = "conformal_scoring_manifest.json"
PROTECTED_BEFORE_NAME = "protected_state_before_phase4.json"
PROTECTED_AFTER_NAME = "protected_state_after_phase4.json"

#: Fixed column order of the per-observation interval table.
INTERVAL_COLUMNS: Tuple[str, ...] = (
    "regime",
    "fold",
    "row_index",
    "mreid_id",
    "source_city",
    "location",
    "group",
    "method",
    "nominal",
    "alpha",
    "stratum",
    "point_pred_log",
    "point_pred_inr",
    "actual_log",
    "actual_price",
    "quantile",
    "lower_log",
    "upper_log",
    "lower_inr",
    "upper_inr",
    "fallback_used",
)

_SCORE_DEFINITION: Dict[str, str] = {
    "A": "|y - yhat|",
    "B1": "|y - yhat| / yhat",
    "B3": "|y - yhat| / 1",
    "C1": "|y - yhat|, one quantile per source_city",
    "C2": "|y - yhat|, one quantile per predicted-price tertile",
    "C3": "mean(|y - yhat|) per calibration group, one quantile across groups",
}


# ============================================================
# SMALL HELPERS
# ============================================================


def level_tag(nominal: float) -> str:
    """Stable short tag for a nominal level, e.g. 0.90 -> '90'."""
    return f"{int(round(float(nominal) * 100))}"


def _json_float(value: Any) -> Any:
    """JSON-safe float: None for None or non-finite, else float."""
    if value is None:
        return None
    v = float(value)
    return v if np.isfinite(v) else None


def group_mean_scores(
    scores: Sequence[float] | np.ndarray,
    groups: Sequence[Any],
) -> np.ndarray:
    """One score per calibration group: s_g = mean(score_i for i in group g).

    Groups are visited in first-appearance order so the output is deterministic.
    This is the C3 aggregation fixed by resolution 8.1.2.
    """
    s = np.asarray(scores, dtype=float)
    g = np.asarray(groups)
    if s.ndim != 1 or g.ndim != 1:
        raise ValueError("scores and groups must be 1-D")
    if s.shape != g.shape:
        raise ValueError(f"scores shape {s.shape} != groups shape {g.shape}")
    if s.size == 0:
        raise ValueError("scores must be non-empty")
    order = C._unique_stable(g)
    return np.array([float(np.mean(s[g == key])) for key in order], dtype=float)


def prepare_qmap(
    stratum_q: Mapping[Any, Mapping[str, Any]],
    test_labels: Sequence[Any],
) -> Dict[Any, Dict[str, Any]]:
    """Build the per-stratum quantile mapping used by build_intervals.

    Any stratum that is unseen in C_k, or whose exact quantile is unbounded
    (k > n_stratum), is mapped to an explicit fallback entry: qhat=None,
    fallback_used=True. build_intervals turns that into (-inf, +inf). A pooled
    quantile is never substituted (resolution 8.1.2).
    """
    labels = np.asarray(test_labels)
    qmap: Dict[Any, Dict[str, Any]] = {}
    for key in C._unique_stable(labels):
        if key not in stratum_q:
            qmap[key] = {
                "qhat": None, "n": 0, "k": None,
                "unbounded": None, "fallback_used": True,
            }
            continue
        e = stratum_q[key]
        unbounded = e.get("unbounded")
        if e.get("fallback_used") or e.get("qhat") is None or unbounded:
            qmap[key] = {
                "qhat": None,
                "n": int(e.get("n", 0)),
                "k": (None if e.get("k") is None else int(e["k"])),
                "unbounded": (None if unbounded is None else bool(unbounded)),
                "fallback_used": True,
            }
        else:
            qmap[key] = {
                "qhat": float(e["qhat"]),
                "n": int(e["n"]),
                "k": int(e["k"]),
                "unbounded": False,
                "fallback_used": False,
            }
    return qmap


def per_row_q(
    qmap: Mapping[Any, Mapping[str, Any]],
    test_labels: Sequence[Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-observation quantile (NaN when fallback) and fallback flag."""
    labels = np.asarray(test_labels)
    q = np.full(labels.shape, np.nan, dtype=float)
    fallback = np.zeros(labels.shape, dtype=bool)
    for key in C._unique_stable(labels):
        mask = labels == key
        entry = qmap.get(key)
        if entry is None or entry.get("fallback_used") or entry.get("qhat") is None:
            fallback[mask] = True
        else:
            q[mask] = float(entry["qhat"])
    return q, fallback


def _pooled_cell(
    method: str,
    nominal: float,
    alpha: float,
    q: Mapping[str, Any],
    info: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "method": method,
        "nominal": float(nominal),
        "alpha": float(alpha),
        "score_definition": _SCORE_DEFINITION[method],
        "scale": "one",
        "quantile": None if q["unbounded"] else float(q["qhat"]),
        "n": int(q["n"]),
        "k": int(q["k"]),
        "unbounded": bool(q["unbounded"]),
        "fallback_used": False,
        "n_unbounded": int(info.get("n_unbounded", 0)),
        "n_fallback": int(info.get("n_fallback", 0)),
        "width_is_constant": bool(info.get("width_is_constant", True)),
    }


def _mondrian_cell(
    method: str,
    nominal: float,
    alpha: float,
    stratum_q: Mapping[Any, Mapping[str, Any]],
    qmap: Mapping[Any, Mapping[str, Any]],
    test_labels: Sequence[Any],
    info: Mapping[str, Any],
    partition: str,
) -> Dict[str, Any]:
    labels = np.asarray(test_labels)
    strata: List[Dict[str, Any]] = []
    for key in C._unique_stable(labels):
        mask = labels == key
        src = stratum_q.get(key)
        entry = qmap.get(key, {})
        strata.append(
            {
                "stratum": str(key),
                "n_calibration": int(src["n"]) if src else 0,
                "k": (None if not src or src.get("k") is None else int(src["k"])),
                "qhat": (None if not src else _json_float(src.get("qhat"))),
                "unbounded": (None if not src else src.get("unbounded")),
                "fallback_used": bool(entry.get("fallback_used", True)),
                "n_test_rows": int(mask.sum()),
            }
        )
    return {
        "method": method,
        "nominal": float(nominal),
        "alpha": float(alpha),
        "score_definition": _SCORE_DEFINITION[method],
        "partition": partition,
        "quantile": None,
        "n_calibration_strata": int(len(stratum_q)),
        "n_test_strata": int(len(strata)),
        "fallback_used": bool(int(info.get("n_fallback", 0)) > 0),
        "n_fallback": int(info.get("n_fallback", 0)),
        "n_unbounded": int(info.get("n_unbounded", 0)),
        "width_is_constant": True,
        "strata": strata,
    }


# ============================================================
# PURE PER-FOLD CONFORMAL COMPUTATION
# ============================================================


def compute_fold_conformal(
    *,
    regime: str,
    fold_id: int,
    y_cal: Sequence[float] | np.ndarray,
    yhat_cal: Sequence[float] | np.ndarray,
    yhat_test: Sequence[float] | np.ndarray,
    yhat_fit: Sequence[float] | np.ndarray,
    city_cal: Sequence[Any],
    city_test: Sequence[Any],
    group_cal: Sequence[Any],
    group_test: Sequence[Any],
    levels: Sequence[float] = NOMINAL_LEVELS,
) -> Dict[str, Any]:
    """Compute every (method, level) score/quantile/interval for one fold.

    Deliberately receives NO test labels: calibration quantities cannot depend
    on them. Bands for C2 use cutpoints computed on the T'_k predictions
    (yhat_fit) and applied unchanged to C_k and test_k.
    """
    y_cal = np.asarray(y_cal, dtype=float)
    yhat_cal = np.asarray(yhat_cal, dtype=float)
    yhat_test = np.asarray(yhat_test, dtype=float)
    yhat_fit = np.asarray(yhat_fit, dtype=float)
    city_cal = np.asarray(city_cal)
    city_test = np.asarray(city_test)
    group_cal = np.asarray(group_cal)
    group_test = np.asarray(group_test)

    if not (y_cal.shape == yhat_cal.shape == city_cal.shape == group_cal.shape):
        raise ValueError("calibration arrays must share one shape")
    if not (yhat_test.shape == city_test.shape == group_test.shape):
        raise ValueError("test arrays must share one shape")
    if np.asarray(levels).tolist() != list(NOMINAL_LEVELS):
        raise ValueError(
            f"levels are pre-registered as {NOMINAL_LEVELS}; refusing {list(levels)}"
        )

    # C2 cutpoints: T'_k predictions only (resolution 8.1 / protocol 12.2b).
    cuts = np.asarray(
        np.quantile(yhat_fit, [1.0 / 3.0, 2.0 / 3.0], method="linear"), dtype=float
    )
    band_cal, _ = C.predicted_tertile_labels(yhat_cal, cutpoints=cuts)
    band_test, _ = C.predicted_tertile_labels(yhat_test, cutpoints=cuts)

    s_abs, _ = C.conformal_scores("A", y_cal, yhat_cal)
    s_b1, info_b1 = C.conformal_scores("B1", y_cal, yhat_cal, y_pred_cal=yhat_cal)

    group_keys = C._unique_stable(group_cal)
    group_scores = group_mean_scores(s_abs, group_cal)

    n_test = int(yhat_test.size)
    cells: List[Dict[str, Any]] = []
    intervals: Dict[Tuple[str, float], Dict[str, np.ndarray]] = {}

    for method in METHODS:
        for nominal in levels:
            nominal = float(nominal)
            alpha = 1.0 - nominal
            key = (method, nominal)

            if method in ("A", "B3"):
                q = C.conformal_quantile(s_abs, alpha)
                lower, upper, info = C.build_intervals(method, yhat_test, q["qhat"])
                qrow = np.full(n_test, q["qhat"], dtype=float)
                frow = np.zeros(n_test, dtype=bool)
                cell = _pooled_cell(method, nominal, alpha, q, info)

            elif method == "B1":
                q = C.conformal_quantile(s_b1, alpha)
                lower, upper, info = C.build_intervals(
                    "B1", yhat_test, q["qhat"], scale=yhat_test
                )
                qrow = np.full(n_test, q["qhat"], dtype=float)
                frow = np.zeros(n_test, dtype=bool)
                cell = _pooled_cell("B1", nominal, alpha, q, info)
                cell["scale"] = "y_hat"
                cell["n_guard_activations"] = int(info_b1["n_guard_activations"])

            elif method == "C1":
                q = C.conformal_quantile_by_stratum(s_abs, city_cal, alpha, min_n=1)
                qmap = prepare_qmap(q, city_test)
                lower, upper, info = C.build_intervals(
                    "C1", yhat_test, qmap, strata=city_test
                )
                qrow, frow = per_row_q(qmap, city_test)
                cell = _mondrian_cell(
                    "C1", nominal, alpha, q, qmap, city_test, info, "source_city"
                )

            elif method == "C2":
                q = C.conformal_quantile_by_stratum(s_abs, band_cal, alpha, min_n=1)
                qmap = prepare_qmap(q, band_test)
                lower, upper, info = C.build_intervals(
                    "C2", yhat_test, qmap, strata=band_test
                )
                qrow, frow = per_row_q(qmap, band_test)
                cell = _mondrian_cell(
                    "C2", nominal, alpha, q, qmap, band_test, info,
                    "predicted_price_tertile",
                )
                cell["cutpoints"] = [float(c) for c in cuts]

            else:  # C3
                q = C.conformal_quantile(group_scores, alpha)
                lower, upper, info = C.build_intervals("A", yhat_test, q["qhat"])
                qrow = np.full(n_test, q["qhat"], dtype=float)
                frow = np.zeros(n_test, dtype=bool)
                cell = _pooled_cell("C3", nominal, alpha, q, info)
                cell["n_groups"] = int(len(group_keys))
                cell["group_key"] = B.GROUP_COLUMN
                cell["quantile_source"] = "calibration group mean absolute residuals"

            cells.append(cell)
            intervals[key] = {
                "lower": lower,
                "upper": upper,
                "q": qrow,
                "fallback": frow,
            }

    return {
        "regime": regime,
        "fold": int(fold_id),
        "cutpoints": [float(c) for c in cuts],
        "n_calibration_rows": int(y_cal.size),
        "n_test_rows": n_test,
        "n_calibration_groups": int(len(group_keys)),
        "group_score_definition": _SCORE_DEFINITION["C3"],
        "band_test": [str(b) for b in band_test.tolist()],
        "b1_n_guard_activations": int(info_b1["n_guard_activations"]),
        "cells": cells,
        "intervals": intervals,
    }


# ============================================================
# PERSISTENCE
# ============================================================


def scores_dir(artifact_root: Path) -> Path:
    return artifact_root / SCORES_DIRNAME


def intervals_dir(artifact_root: Path) -> Path:
    return artifact_root / INTERVALS_DIRNAME


def quantiles_json_path(artifact_root: Path, regime: str, fold_id: int) -> Path:
    return scores_dir(artifact_root) / f"quantiles_{regime}_fold{fold_id}.json"


def intervals_csv_path(artifact_root: Path, regime: str, fold_id: int) -> Path:
    return intervals_dir(artifact_root) / f"intervals_{regime}_fold{fold_id}.csv"


def _predict(pipe: Any, x_all: pd.DataFrame, idx: Sequence[int]) -> np.ndarray:
    return np.asarray(pipe.predict(x_all.iloc[list(idx)]), dtype=float)


def build_interval_records(
    *,
    regime: str,
    fold_id: int,
    entry: Mapping[str, Any],
    raw: pd.DataFrame,
    groups: pd.Series,
    y_all: np.ndarray,
    yhat_test: np.ndarray,
    band_test: Sequence[str],
    result: Mapping[str, Any],
) -> pd.DataFrame:
    """Long-format per-observation interval table for one fold."""
    test_idx = list(entry["test"])
    y_test = np.asarray(y_all, dtype=float)[test_idx]
    price_test = raw[P.TARGET_COLUMN].to_numpy(dtype=float)[test_idx]
    mreid = raw["mreid_id"].astype(str).to_numpy()[test_idx]
    city = raw["source_city"].astype(str).to_numpy()[test_idx]
    loc = raw["location"].astype(str).to_numpy()[test_idx]
    grp = np.asarray(groups).astype(str)[test_idx]
    band = np.asarray(band_test)

    point_log = yhat_test.astype(float)
    point_inr = P.to_rupees(point_log)
    n = point_log.size

    frames: List[pd.DataFrame] = []
    for method_order, method in enumerate(METHODS):
        if method == "C1":
            stratum = city
        elif method == "C2":
            stratum = band
        elif method == "C3":
            stratum = grp
        else:
            stratum = np.full(n, "", dtype=object)
        for nominal in NOMINAL_LEVELS:
            ip = result["intervals"][(method, float(nominal))]
            lower_log = np.asarray(ip["lower"], dtype=float)
            upper_log = np.asarray(ip["upper"], dtype=float)
            qrow = np.asarray(ip["q"], dtype=float)
            frow = np.asarray(ip["fallback"], dtype=bool)
            frames.append(
                pd.DataFrame(
                    {
                        "regime": regime,
                        "fold": int(fold_id),
                        "row_index": np.asarray(test_idx, dtype=np.int64),
                        "mreid_id": mreid,
                        "source_city": city,
                        "location": loc,
                        "group": grp,
                        "method": method,
                        "nominal": float(nominal),
                        "alpha": 1.0 - float(nominal),
                        "stratum": stratum,
                        "point_pred_log": point_log,
                        "point_pred_inr": point_inr,
                        "actual_log": y_test,
                        "actual_price": price_test,
                        "quantile": qrow,
                        "lower_log": lower_log,
                        "upper_log": upper_log,
                        "lower_inr": P.to_rupees(lower_log),
                        "upper_inr": P.to_rupees(upper_log),
                        "fallback_used": frow,
                        "_method_order": method_order,
                    }
                )
            )
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(
        ["_method_order", "nominal", "row_index"], kind="mergesort"
    ).reset_index(drop=True)
    df = df.drop(columns=["_method_order"])
    return df[list(INTERVAL_COLUMNS)]


def score_fold(
    *,
    regime: str,
    entry: Mapping[str, Any],
    record: Mapping[str, Any],
    raw: pd.DataFrame,
    groups: pd.Series,
    artifact_root: Path,
) -> Dict[str, Any]:
    """Score one fold and persist its quantile and interval artifacts."""
    fold_id = int(entry["outer_fold"])
    if int(record["fold"]) != fold_id or record["regime"] != regime:
        raise RuntimeError(f"manifest record does not match {regime} fold {fold_id}")

    model_path = T.ROOT / record["model_file"]
    actual_sha = T.sha256_of(model_path)
    if actual_sha != record["model_sha256"]:
        raise RuntimeError(
            f"{regime} fold {fold_id}: model hash mismatch; refusing to score"
        )
    pipe = joblib.load(model_path)

    x_all = raw[P.INPUT_FEATURES]
    y_all = np.asarray(P.log_target(raw[P.TARGET_COLUMN]), dtype=float)

    fit_idx = list(entry["fit"])
    cal_idx = list(entry["calibrate"])
    test_idx = list(entry["test"])

    yhat_fit = _predict(pipe, x_all, fit_idx)
    yhat_cal = _predict(pipe, x_all, cal_idx)
    yhat_test = _predict(pipe, x_all, test_idx)

    group_all = groups.astype(str)

    result = compute_fold_conformal(
        regime=regime,
        fold_id=fold_id,
        y_cal=y_all[cal_idx],
        yhat_cal=yhat_cal,
        yhat_test=yhat_test,
        yhat_fit=yhat_fit,
        city_cal=raw["source_city"].astype(str).to_numpy()[cal_idx],
        city_test=raw["source_city"].astype(str).to_numpy()[test_idx],
        group_cal=group_all.to_numpy()[cal_idx],
        group_test=group_all.to_numpy()[test_idx],
    )

    df = build_interval_records(
        regime=regime,
        fold_id=fold_id,
        entry=entry,
        raw=raw,
        groups=groups,
        y_all=y_all,
        yhat_test=yhat_test,
        band_test=result["band_test"],
        result=result,
    )

    qj = quantiles_json_path(artifact_root, regime, fold_id)
    ic = intervals_csv_path(artifact_root, regime, fold_id)
    T.assert_safe_write_target(qj)
    T.assert_safe_write_target(ic)
    qj.parent.mkdir(parents=True, exist_ok=True)
    ic.parent.mkdir(parents=True, exist_ok=True)

    qdoc: Dict[str, Any] = {
        "experiment": "MILLOW Experiment 2 - conformal prediction",
        "phase": "Phase 4 - conformal scoring and interval generation",
        "regime": regime,
        "fold": fold_id,
        "model_file": record["model_file"],
        "model_sha256": record["model_sha256"],
        "training_space": P.TRAINING_SPACE,
        "nominal_levels": [float(x) for x in NOMINAL_LEVELS],
        "methods": list(METHODS),
        "n_calibration_rows": result["n_calibration_rows"],
        "n_test_rows": result["n_test_rows"],
        "n_calibration_groups": result["n_calibration_groups"],
        "group_key": B.GROUP_COLUMN,
        "group_score_definition": result["group_score_definition"],
        "c2_cutpoints_from_T_prime": result["cutpoints"],
        "band_test": result["band_test"],
        "b1_n_guard_activations": result["b1_n_guard_activations"],
        "fold_digests": dict(entry["digests"]),
        "cells": result["cells"],
    }
    with open(qj, "w", encoding="utf-8") as fh:
        json.dump(qdoc, fh, indent=2, sort_keys=False)
        fh.write("\n")

    df.to_csv(
        ic,
        index=False,
        na_rep="",
        float_format="%.10g",
        lineterminator="\n",
        columns=list(INTERVAL_COLUMNS),
    )

    fallback_counts: Dict[str, Dict[str, int]] = {"C1": {}, "C2": {}}
    for method in ("C1", "C2"):
        for nominal in NOMINAL_LEVELS:
            mask = (df["method"] == method) & (df["nominal"] == float(nominal))
            fallback_counts[method][level_tag(nominal)] = int(
                df.loc[mask, "fallback_used"].sum()
            )

    c3_groups: Dict[str, Dict[str, int]] = {}
    for cell in result["cells"]:
        if cell["method"] == "C3":
            c3_groups[level_tag(cell["nominal"])] = {
                "n_groups": int(cell["n_groups"]),
                "k": int(cell["k"]),
                "unbounded": bool(cell["unbounded"]),
            }

    return {
        "regime": regime,
        "fold": fold_id,
        "model_file": record["model_file"],
        "model_sha256": record["model_sha256"],
        "n_fit": len(fit_idx),
        "n_calibrate": len(cal_idx),
        "n_test": len(test_idx),
        "n_calibration_rows": result["n_calibration_rows"],
        "n_calibration_groups": result["n_calibration_groups"],
        "n_intervals": int(len(df)),
        "n_intervals_per_method": {
            m: int((df["method"] == m).sum()) for m in METHODS
        },
        "fallback_counts": fallback_counts,
        "c3_groups": c3_groups,
        "b1_n_guard_activations": result["b1_n_guard_activations"],
        "quantiles_file": T.rel_to_root(qj),
        "quantiles_sha256": T.sha256_of(qj),
        "intervals_file": T.rel_to_root(ic),
        "intervals_sha256": T.sha256_of(ic),
        "fold_digests": dict(entry["digests"]),
    }


# ============================================================
# DETERMINISM DIGEST
# ============================================================


def deterministic_digest(paths: Sequence[Path], artifact_root: Path) -> str:
    """SHA-256 over sorted '<path relative to artifact_root>:<file sha>' lines.

    Paths are expressed relative to the output root so the digest is invariant
    to where the artifacts are written (real tree vs a temporary reproduction
    directory). No wall-clock value enters the digest.
    """
    lines = []
    for p in sorted(paths, key=lambda q: q.relative_to(artifact_root).as_posix()):
        rel = p.relative_to(artifact_root).as_posix()
        lines.append(f"{rel}:{T.sha256_of(p)}")
    return T.sha256_of_bytes(("\n".join(lines) + "\n").encode("utf-8"))


# ============================================================
# RUN
# ============================================================


def load_nested_manifest(
    input_root: Path = T.ARTIFACT_ROOT,
) -> Dict[str, Any]:
    """Load and verify the Phase 3 manifest before any scoring occurs."""
    path = input_root / T.MANIFEST_JSON.name
    actual = T.sha256_of(path)
    if actual != NESTED_MANIFEST_SHA256:
        raise RuntimeError(
            "Phase 3 nested-model manifest has changed. Refusing to score.\n"
            f"  expected {NESTED_MANIFEST_SHA256}\n  actual   {actual}"
        )
    with open(path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("protected_state_clean") is not True:
        raise RuntimeError("Phase 3 manifest reports a protected-state violation")
    if len(manifest.get("models", [])) != T.EXPECTED_MODEL_COUNT:
        raise RuntimeError("Phase 3 manifest does not contain 10 models")
    return manifest


def run_phase4(
    *,
    artifact_root: Path = T.ARTIFACT_ROOT,
    input_root: Path = T.ARTIFACT_ROOT,
    model_root: Path = T.MODEL_ROOT,
) -> Dict[str, Any]:
    """Score every fold, persist artifacts, and return the Phase 4 manifest."""
    del model_root  # models are addressed through the manifest's recorded paths
    T.assert_safe_write_target(scores_dir(artifact_root) / "probe.json")
    T.assert_safe_write_target(intervals_dir(artifact_root) / "probe.csv")
    artifact_root.mkdir(parents=True, exist_ok=True)

    before = T.snapshot_protected_state()
    with open(artifact_root / PROTECTED_BEFORE_NAME, "w", encoding="utf-8") as fh:
        json.dump(before, fh, indent=2)
        fh.write("\n")

    doc = T.load_calibration_folds()
    manifest_in = load_nested_manifest(input_root)
    raw, _dup, _inv = P.build_dataset()
    groups = P.add_group_labels(raw)

    records_by_cell: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for rec in manifest_in["models"]:
        records_by_cell[(rec["regime"], int(rec["fold"]))] = rec

    fold_summaries: List[Dict[str, Any]] = []
    for regime in REGIMES:
        for entry in doc["regimes"][regime]["folds"]:
            fold_id = int(entry["outer_fold"])
            record = records_by_cell.get((regime, fold_id))
            if record is None:
                raise RuntimeError(f"no Phase 3 model for {regime} fold {fold_id}")
            summary = score_fold(
                regime=regime,
                entry=entry,
                record=record,
                raw=raw,
                groups=groups,
                artifact_root=artifact_root,
            )
            fold_summaries.append(summary)
            print(
                f"  [{len(fold_summaries):>2}/10] {regime:<17} fold {fold_id}  "
                f"cal={summary['n_calibrate']:>5} test={summary['n_test']:>5} "
                f"intervals={summary['n_intervals']:>7} "
                f"C1_fb={sum(summary['fallback_counts']['C1'].values()):>5} "
                f"C2_fb={sum(summary['fallback_counts']['C2'].values()):>5}"
            )

    artifact_files: List[Path] = []
    for summary in fold_summaries:
        artifact_files.append(ROOT / summary["quantiles_file"])
        artifact_files.append(ROOT / summary["intervals_file"])
    digest = deterministic_digest(artifact_files, artifact_root)

    total_intervals = sum(s["n_intervals"] for s in fold_summaries)
    per_method: Dict[str, int] = {}
    for m in METHODS:
        per_method[m] = sum(s["n_intervals_per_method"][m] for s in fold_summaries)
    fallback_totals: Dict[str, Dict[str, int]] = {"C1": {}, "C2": {}}
    for m in ("C1", "C2"):
        for nominal in NOMINAL_LEVELS:
            tag = level_tag(nominal)
            fallback_totals[m][tag] = sum(
                s["fallback_counts"][m][tag] for s in fold_summaries
            )

    after = T.snapshot_protected_state()
    with open(artifact_root / PROTECTED_AFTER_NAME, "w", encoding="utf-8") as fh:
        json.dump(after, fh, indent=2)
        fh.write("\n")
    diff = T.compare_protected_states(before, after)

    manifest: Dict[str, Any] = {
        "experiment": "MILLOW Experiment 2 - conformal prediction",
        "phase": "Phase 4 - conformal scoring and interval generation",
        "protocol": "V2.1 (frozen, read-only)",
        "status": "Phase 4 scoring complete",
        "methods": list(METHODS),
        "nominal_levels": [float(x) for x in NOMINAL_LEVELS],
        "regimes": list(REGIMES),
        "training_space": P.TRAINING_SPACE,
        "interval_endpoint_space": "log1p(price); INR via expm1, floor 0",
        "exact_finite_sample_order_statistic": True,
        "interpolation_used": False,
        "calibration_inputs": {
            "calibration_folds": {
                "path": str(B.CALIBRATION_FOLDS_JSON.relative_to(ROOT)).replace("\\", "/"),
                "sha256": T.CALIBRATION_FOLDS_SHA256,
                "parent_folds_sha256": B.FROZEN_FOLDS_SHA256,
            },
            "nested_models_manifest": {
                "path": str((input_root / T.MANIFEST_JSON.name).relative_to(ROOT)).replace("\\", "/"),
                "sha256": NESTED_MANIFEST_SHA256,
            },
        },
        "scope_note": (
            "Scoring only. No coverage, width, subgroup, monotonicity or "
            "method-superiority quantity is computed or claimed."
        ),
        "n_folds": len(fold_summaries),
        "total_intervals": int(total_intervals),
        "intervals_per_method": per_method,
        "c1_fallback_totals": fallback_totals["C1"],
        "c2_fallback_totals": fallback_totals["C2"],
        "deterministic_content_digest": digest,
        "protected_state": diff,
        "protected_state_clean": bool(diff["clean"]),
        "environment": T.environment_record(),
        "folds": fold_summaries,
    }

    manifest_path = artifact_root / MANIFEST_NAME
    T.assert_safe_write_target(manifest_path)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")

    if not diff["clean"]:
        raise RuntimeError(
            "Protected state changed during Phase 4 scoring: "
            f"{json.dumps(diff, indent=2)}"
        )
    return manifest


# ============================================================
# VERIFICATION
# ============================================================


def scan_interval_files(artifact_root: Path) -> List[Path]:
    base = intervals_dir(artifact_root)
    return sorted(base.glob("intervals_*_fold*.csv"))


def verify_phase4(
    *,
    artifact_root: Path = T.ARTIFACT_ROOT,
) -> Dict[str, Any]:
    """Reload and independently verify the Phase 4 artifacts."""
    manifest_path = artifact_root / MANIFEST_NAME
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    if manifest.get("protected_state_clean") is not True:
        raise RuntimeError("Phase 4 manifest reports a protected-state violation")

    doc = T.load_calibration_folds()
    load_nested_manifest()
    expected_cells: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for regime in REGIMES:
        for entry in doc["regimes"][regime]["folds"]:
            expected_cells[(regime, int(entry["outer_fold"]))] = entry

    seen: List[Tuple[str, int]] = []
    results: List[Dict[str, Any]] = []
    n_intervals_total = 0
    for summary in manifest["folds"]:
        regime = summary["regime"]
        fold_id = int(summary["fold"])
        seen.append((regime, fold_id))
        entry = expected_cells.get((regime, fold_id))
        if entry is None:
            raise RuntimeError(f"unexpected Phase 4 fold {regime} {fold_id}")

        qj = ROOT / summary["quantiles_file"]
        ic = ROOT / summary["intervals_file"]
        if T.sha256_of(qj) != summary["quantiles_sha256"]:
            raise RuntimeError(f"{regime} fold {fold_id}: quantile file hash mismatch")
        if T.sha256_of(ic) != summary["intervals_sha256"]:
            raise RuntimeError(f"{regime} fold {fold_id}: interval file hash mismatch")

        with open(qj, "r", encoding="utf-8") as fh:
            qdoc = json.load(fh)
        if qdoc["n_calibration_rows"] != len(entry["calibrate"]):
            raise RuntimeError(f"{regime} fold {fold_id}: calibration row count drift")
        if qdoc["n_test_rows"] != len(entry["test"]):
            raise RuntimeError(f"{regime} fold {fold_id}: test row count drift")
        if qdoc["fold_digests"] != dict(entry["digests"]):
            raise RuntimeError(f"{regime} fold {fold_id}: fold digest drift")

        df = pd.read_csv(ic, low_memory=False)
        if list(df.columns) != list(INTERVAL_COLUMNS):
            raise RuntimeError(f"{regime} fold {fold_id}: interval column drift")
        if set(df["method"]) != set(METHODS):
            raise RuntimeError(f"{regime} fold {fold_id}: method set drift")
        if set(np.round(df["nominal"].unique(), 6)) != set(NOMINAL_LEVELS):
            raise RuntimeError(f"{regime} fold {fold_id}: nominal level drift")
        if len(df) != summary["n_intervals"]:
            raise RuntimeError(f"{regime} fold {fold_id}: interval row count drift")

        # Interval construction: bounded rows must satisfy lower <= point <= upper.
        finite = np.isfinite(df["lower_log"]) & np.isfinite(df["upper_log"])
        if not bool(((df.loc[finite, "lower_log"] <= df.loc[finite, "point_pred_log"]) &
                     (df.loc[finite, "point_pred_log"] <= df.loc[finite, "upper_log"])).all()):
            raise RuntimeError(f"{regime} fold {fold_id}: interval ordering violated")
        # Unbounded rows must be exactly (-inf, +inf), never partially infinite.
        lo_bad = df["lower_log"].isna() | (df["lower_log"] == np.inf)
        up_bad = df["upper_log"].isna() | (df["upper_log"] == -np.inf)
        if bool(lo_bad.any() or up_bad.any()):
            raise RuntimeError(f"{regime} fold {fold_id}: malformed unbounded endpoint")

        # C1/C2 fallback counts.
        for method in ("C1", "C2"):
            for nominal in NOMINAL_LEVELS:
                tag = level_tag(nominal)
                mask = (df["method"] == method) & (df["nominal"] == float(nominal))
                observed = int(df.loc[mask, "fallback_used"].sum())
                if observed != summary["fallback_counts"][method][tag]:
                    raise RuntimeError(
                        f"{regime} fold {fold_id} {method}@{tag}: fallback count drift"
                    )
        # C3 group aggregation and n/k.
        c3cells = {level_tag(c["nominal"]): c for c in qdoc["cells"] if c["method"] == "C3"}
        for nominal in NOMINAL_LEVELS:
            tag = level_tag(nominal)
            cell = c3cells[tag]
            if int(cell["n_groups"]) != int(qdoc["n_calibration_groups"]):
                raise RuntimeError(f"{regime} fold {fold_id} C3@{tag}: group count drift")
            expected_k = int(np.ceil((cell["n_groups"] + 1) * nominal))
            if int(cell["k"]) != expected_k:
                raise RuntimeError(f"{regime} fold {fold_id} C3@{tag}: k drift")

        n_intervals_total += len(df)
        results.append(
            {
                "regime": regime,
                "fold": fold_id,
                "n_intervals": int(len(df)),
                "verified": True,
            }
        )

    if set(seen) != set(expected_cells):
        raise RuntimeError("Phase 4 folds do not match the Phase 2 contract")
    if n_intervals_total != manifest["total_intervals"]:
        raise RuntimeError("Phase 4 total interval count drift")

    files: List[Path] = []
    for s in manifest["folds"]:
        files.append(ROOT / s["quantiles_file"])
        files.append(ROOT / s["intervals_file"])
    if deterministic_digest(files, artifact_root) != manifest["deterministic_content_digest"]:
        raise RuntimeError("Phase 4 deterministic content digest drift")

    return {
        "verified": True,
        "n_folds": len(results),
        "n_intervals": int(n_intervals_total),
        "deterministic_content_digest": manifest["deterministic_content_digest"],
        "protected_state_clean": manifest["protected_state_clean"],
    }


# ============================================================
# CLI
# ============================================================


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Experiment 2 Phase 4 scorer")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="reload and verify existing Phase 4 artifacts without scoring",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="override the artifact output root (used for determinism checks)",
    )
    args = parser.parse_args(argv)

    artifact_root = Path(args.out_dir).resolve() if args.out_dir else T.ARTIFACT_ROOT

    if args.verify_only:
        report = verify_phase4(artifact_root=artifact_root)
        print(json.dumps(report, indent=2))
        return 0

    manifest = run_phase4(artifact_root=artifact_root)
    report = verify_phase4(artifact_root=artifact_root)
    payload = {
        "status": manifest["status"],
        "total_intervals": manifest["total_intervals"],
        "deterministic_content_digest": manifest["deterministic_content_digest"],
        "protected_state_clean": manifest["protected_state_clean"],
        "verification": report,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
