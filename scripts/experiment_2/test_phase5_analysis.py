"""Phase 5 tests: conformal coverage, sharpness, subgroup and decision analysis.

Two groups of tests:

  PURE      - exercise the analysis logic on synthetic arrays and tables. No
              filesystem access, no models, no dataset. These prove the
              infinite-aware coverage, fold aggregation, subgroup suppression,
              decision thresholds and digest stability structurally.

  ARTIFACT  - reload and verify the persisted Phase 5 analysis artifacts.
              Skipped when the analysis has not been generated, so the pure
              suite always runs.

The tests make no statistical claim about coverage, calibration or usability.
They verify construction and the pre-registered thresholds only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.experiment_2 import analyze_conformal as A
from scripts.experiment_2 import conformal as C
from scripts.experiment_2 import run_conformal as R
from scripts.experiment_2 import train_nested_models as T


# ===========================================================================
# PURE: finite-aware coverage
# ===========================================================================


def test_coverage_counts_full_infinite_interval_as_covered():
    y = np.array([1.0, 2.0, 3.0])
    lo = np.array([-np.inf, 0.0, 2.0])
    up = np.array([np.inf, 4.0, 2.0])
    m = A.coverage_finite_aware(y, lo, up, 0.90)
    assert m["n_evaluated"] == 3
    assert m["n_infinite"] == 1
    assert m["n_covered"] == 2
    assert m["empirical_coverage"] == pytest.approx(2.0 / 3.0)
    assert m["coverage_error"] == pytest.approx(2.0 / 3.0 - 0.90)


def test_coverage_reports_lower_and_upper_miscoverage():
    y = np.array([0.0, 5.0, 5.0, 5.0])
    lo = np.array([1.0, 1.0, 1.0, 1.0])
    up = np.array([2.0, 2.0, 2.0, 2.0])
    m = A.coverage_finite_aware(y, lo, up, 0.80)
    assert m["n_miscoverage_low"] == 1
    assert m["n_miscoverage_high"] == 3
    assert m["miscoverage_low"] == pytest.approx(0.25)
    assert m["miscoverage_high"] == pytest.approx(0.75)


def test_coverage_all_infinite_is_fully_covered():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    lo = np.full(4, -np.inf)
    up = np.full(4, np.inf)
    m = A.coverage_finite_aware(y, lo, up, 0.95)
    assert m["n_infinite"] == 4
    assert m["n_bounded"] == 0
    assert m["empirical_coverage"] == 1.0


def test_coverage_requires_at_least_one_row():
    with pytest.raises(ValueError):
        A.coverage_finite_aware(np.array([]), np.array([]), np.array([]), 0.90)


# ===========================================================================
# PURE: cell metrics (coverage + sharpness)
# ===========================================================================


def test_cell_metrics_relative_width_is_width_over_fitted():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([2.0, 2.0, 2.0])
    lo = p - 0.5
    up = p + 0.5
    m = A.compute_cell_metrics(y, p, lo, up, 0.80, "A")
    assert m["mean_width_log"] == pytest.approx(1.0)
    assert m["mean_relative_width_log"] == pytest.approx(0.5)
    assert m["width_metrics_scope"] == "all_rows"


def test_cell_metrics_scopes_width_to_bounded_rows_when_infinite():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([2.0, 2.0, 2.0])
    lo = np.array([-np.inf, 1.5, 2.5])
    up = np.array([np.inf, 2.5, 3.5])
    m = A.compute_cell_metrics(y, p, lo, up, 0.90, "C1")
    assert m["n_infinite"] == 1
    assert m["width_metrics_scope"] == "bounded_rows_only"
    assert m["mean_width_log"] == pytest.approx(1.0)


def test_cell_metrics_homogeneous_sd_width_is_zero():
    p = np.array([1.0, 2.0, 3.0, 4.0])
    lo = p - 0.5
    up = p + 0.5
    m = A.compute_cell_metrics(p, p, lo, up, 0.80, "A")
    assert m["sd_width_log"] == pytest.approx(0.0)
    assert m["width_is_constant"] is True


def test_cell_interval_score_matches_frozen_function():
    y = np.array([1.0, 2.0, 3.0, 10.0])
    p = np.array([2.0, 2.0, 2.0, 2.0])
    lo = p - 0.5
    up = p + 0.5
    m = A.compute_cell_metrics(y, p, lo, up, 0.80, "A")
    expected = float(np.mean(C.interval_score(y, lo, up, 0.20)))
    assert m["interval_score_log"] == pytest.approx(expected)


# ===========================================================================
# PURE: fold aggregation
# ===========================================================================


def test_summary_uses_sample_sd_ddof_one():
    vals = [0.80, 0.90, 1.00, 0.90, 0.90]
    rows = []
    for fold, v in enumerate(vals, 1):
        row = {
            "method": "A",
            "regime": "random",
            "fold": fold,
            "nominal": 0.90,
        }
        for key in A.CELL_METRIC_KEYS:
            row.setdefault(key, 0.0)
        row["empirical_coverage"] = v
        rows.append(row)
    summary = A.summary_rows(pd.DataFrame(rows))
    r = summary[
        (summary["method"] == "A")
        & (summary["regime"] == "random")
        & (summary["metric"] == "empirical_coverage")
    ].iloc[0]
    assert r["std"] == pytest.approx(float(np.std(vals, ddof=1)))
    assert r["n_folds"] == 5
    assert r["std_ddof"] == 1


# ===========================================================================
# PURE: subgroup construction and suppression
# ===========================================================================


def _interval_table(n_small=3, n_large=120):
    rows = []
    for i in range(n_small):
        rows.append(
            {
                "method": "A",
                "nominal": 0.90,
                "actual_log": float(i),
                "point_pred_log": float(i),
                "lower_log": float(i) - 0.5,
                "upper_log": float(i) + 0.5,
                "actual_price": 100.0 + i,
                "source_city": "Small",
                "fallback_used": False,
            }
        )
    for i in range(n_large):
        rows.append(
            {
                "method": "A",
                "nominal": 0.90,
                "actual_log": float(i),
                "point_pred_log": float(i),
                "lower_log": float(i) - 0.5,
                "upper_log": float(i) + 0.5,
                "actual_price": 1000.0 + i,
                "source_city": "Large",
                "fallback_used": False,
            }
        )
    return pd.DataFrame(rows)


def test_subgroup_reliable_only_at_min_n():
    df = _interval_table()
    rows = A.build_subgroup_rows(
        regime="random", fold_id=1, df=df, cutpoints=[1.0, 2.0]
    )
    by_city = {
        r["subgroup"]: r
        for r in rows
        if r["subgroup_type"] == "source_city" and r["method"] == "A"
        and np.isclose(r["nominal"], 0.90)
    }
    assert by_city["Small"]["coverage_reliable"] is False
    assert by_city["Small"]["n_evaluated"] == 3
    assert by_city["Large"]["coverage_reliable"] is True
    assert by_city["Large"]["n_evaluated"] == 120
    assert by_city["Large"]["min_n_required"] == A.MIN_SUBGROUP_N


def test_subgroup_price_bands_use_true_price_labels():
    df = _interval_table()
    rows = A.build_subgroup_rows(
        regime="random", fold_id=1, df=df, cutpoints=[1.0, 2.0]
    )
    types = {r["subgroup_type"] for r in rows}
    assert types == {"source_city", "true_price_band", "predicted_price_band"}


def _strata_frame(widths_by_stratum):
    rows = []
    for stratum, widths in widths_by_stratum.items():
        for w in widths:
            rows.append(
                {
                    "method": "C1",
                    "nominal": 0.90,
                    "stratum": stratum,
                    "lower_log": 0.0,
                    "upper_log": float(w),
                }
            )
    return pd.DataFrame(rows)


def test_within_stratum_invariance_passes_for_constant_within_stratum():
    df = _strata_frame({"a": [1.0, 1.0, 1.0], "b": [2.0, 2.0, 2.0]})
    r = A.within_stratum_invariance(df)
    assert r["n_strata"] == 2
    assert r["n_violations"] == 0
    assert r["max_relative_sd"] <= A.F0_RELATIVE_TOLERANCE


def test_within_stratum_invariance_flags_varying_within_stratum():
    df = _strata_frame({"a": [1.0, 2.0, 3.0], "b": [2.0, 2.0, 2.0]})
    r = A.within_stratum_invariance(df)
    assert r["n_violations"] == 1


# ===========================================================================
# PURE: calibration coverage reconstructed from quantile metadata
# ===========================================================================


def _qdoc_for_calibration():
    cells = []
    for method in C.METHODS:
        for nominal in C.NOMINAL_LEVELS:
            if method in ("C1", "C2"):
                cells.append(
                    {
                        "method": method,
                        "nominal": float(nominal),
                        "strata": [
                            {
                                "n_calibration": 10,
                                "k": 9,
                                "fallback_used": False,
                                "qhat": 1.0,
                                "n_test_rows": 3,
                            },
                            {
                                "n_calibration": 0,
                                "k": None,
                                "fallback_used": True,
                                "qhat": None,
                                "n_test_rows": 1,
                            },
                        ],
                    }
                )
            else:
                cells.append(
                    {
                        "method": method,
                        "nominal": float(nominal),
                        "n": 10,
                        "k": 9,
                        "unbounded": False,
                    }
                )
    return {"cells": cells}


def test_calibration_coverage_pooled_is_k_over_n():
    q = _qdoc_for_calibration()
    rows = A.calibration_coverage_from_quantiles(qdoc=q, df=pd.DataFrame())
    pooled = {r["method"]: r for r in rows if np.isclose(r["nominal"], 0.90)}
    assert pooled["A"]["calibration_coverage"] == pytest.approx(0.9)
    assert pooled["B1"]["calibration_coverage"] == pytest.approx(0.9)


def test_calibration_coverage_mondrian_weighted_with_fallback():
    q = _qdoc_for_calibration()
    rows = A.calibration_coverage_from_quantiles(qdoc=q, df=pd.DataFrame())
    c1 = [r for r in rows if r["method"] == "C1" and np.isclose(r["nominal"], 0.90)][0]
    # weights 3 (k/n=0.9) and 1 (fallback -> 1.0): (3*0.9 + 1*1.0)/4
    assert c1["calibration_coverage"] == pytest.approx((3 * 0.9 + 1.0) / 4.0)


# ===========================================================================
# PURE: decision table logic
# ===========================================================================


def _synth_fold_metrics(*, a_grouped_coverage=0.90, a_width_090=1.0,
                        bad_homogeneous_width=False):
    rows = []
    for method in C.METHODS:
        for regime in A.REGIMES:
            for fold in range(1, 6):
                for nominal in C.NOMINAL_LEVELS:
                    cov = 0.90
                    sig = False
                    sd_width = 0.0
                    width = 1.0
                    if method == "A" and regime == "location_grouped":
                        cov = a_grouped_coverage
                    if method == "A" and np.isclose(nominal, 0.90):
                        width = a_width_090
                    if bad_homogeneous_width and method in A.GLOBAL_HOMOGENEOUS_METHODS:
                        sd_width = 0.5
                    rows.append(
                        {
                            "method": method,
                            "regime": regime,
                            "fold": fold,
                            "nominal": float(nominal),
                            "empirical_coverage": float(cov),
                            "coverage_error": float(cov) - float(nominal),
                            "abs_coverage_error": abs(float(cov) - float(nominal)),
                            "coverage_significant": sig,
                            "median_relative_width_inr": width,
                            "mean_width_log": width,
                            "sd_width_log": sd_width,
                        }
                    )
    return pd.DataFrame(rows)


def _empty_subgroup():
    return pd.DataFrame(
        columns=[
            "method", "regime", "nominal", "coverage_reliable",
            "coverage_significant", "subgroup_type", "subgroup",
        ]
    )


def _monotonicity(direction="increasing"):
    return {
        (method, regime, fold): {"direction": direction}
        for method in C.METHODS
        for regime in A.REGIMES
        for fold in range(1, 6)
    }


def _outcome(rows, criterion, scope):
    hits = [r for r in rows if r["criterion"] == criterion and r["scope"] == scope]
    assert len(hits) == 1, f"{criterion}|{scope} matched {len(hits)}"
    return hits[0]["outcome"]


def _build_decisions(*, fold_metrics, subgroup, monotonicity,
                     b3_max_interval_difference=0.0, within_stratum=None):
    if within_stratum is None:
        within_stratum = {
            "max_relative_sd": 0.0, "n_strata": 0, "n_violations": 0
        }
    return A.build_decision_rows(
        fold_metrics=fold_metrics,
        subgroup=subgroup,
        monotonicity=monotonicity,
        b3_max_interval_difference=b3_max_interval_difference,
        within_stratum=within_stratum,
    )


def test_f0_passes_on_clean_homogeneous_widths_and_b3_match():
    rows = _build_decisions(
        fold_metrics=_synth_fold_metrics(),
        subgroup=_empty_subgroup(),
        monotonicity=_monotonicity(),
        b3_max_interval_difference=0.0,
    )
    assert _outcome(rows, "F0", "global|A_B3_constant_width") == "pass"
    assert _outcome(rows, "F0", "global|B3_reproduces_A") == "pass"
    assert (
        _outcome(rows, "F0", "C1_C2|within_stratum_width_invariance") == "pass"
    )


def test_f0_fails_when_b3_drifts_or_A_width_varies():
    rows = _build_decisions(
        fold_metrics=_synth_fold_metrics(bad_homogeneous_width=True),
        subgroup=_empty_subgroup(),
        monotonicity=_monotonicity(),
        b3_max_interval_difference=1e-6,
    )
    assert _outcome(rows, "F0", "global|A_B3_constant_width") == "fail"
    assert _outcome(rows, "F0", "global|B3_reproduces_A") == "fail"


def test_f0_flags_within_stratum_violation_for_c1_c2():
    rows = _build_decisions(
        fold_metrics=_synth_fold_metrics(),
        subgroup=_empty_subgroup(),
        monotonicity=_monotonicity(),
        within_stratum={
            "max_relative_sd": 0.01, "n_strata": 60, "n_violations": 3
        },
    )
    assert (
        _outcome(rows, "F0", "C1_C2|within_stratum_width_invariance") == "fail"
    )


def test_f2_not_rejected_when_grouped_coverage_is_nominal():
    rows = _build_decisions(
        fold_metrics=_synth_fold_metrics(a_grouped_coverage=0.90),
        subgroup=_empty_subgroup(),
        monotonicity=_monotonicity(),
        b3_max_interval_difference=0.0,
    )
    assert _outcome(rows, "F2", "A|location_grouped") == "not_rejected"


def test_f2_rejected_only_when_all_three_levels_exceed_threshold():
    rows = _build_decisions(
        fold_metrics=_synth_fold_metrics(a_grouped_coverage=0.70),
        subgroup=_empty_subgroup(),
        monotonicity=_monotonicity(),
        b3_max_interval_difference=0.0,
    )
    assert _outcome(rows, "F2", "A|location_grouped") == "rejected"


def test_f3_gate_uses_median_relative_width_at_090():
    usable = _build_decisions(
        fold_metrics=_synth_fold_metrics(a_width_090=1.0),
        subgroup=_empty_subgroup(),
        monotonicity=_monotonicity(),
        b3_max_interval_difference=0.0,
    )
    assert _outcome(usable, "F3", "A|random") == "usable"
    not_usable = _build_decisions(
        fold_metrics=_synth_fold_metrics(a_width_090=3.0),
        subgroup=_empty_subgroup(),
        monotonicity=_monotonicity(),
        b3_max_interval_difference=0.0,
    )
    assert _outcome(not_usable, "F3", "A|random") == "not_usable"


def test_f5_is_descriptive_regardless_of_direction():
    for direction in ("increasing", "decreasing", "non_monotone", "undefined"):
        rows = _build_decisions(
            fold_metrics=_synth_fold_metrics(),
            subgroup=_empty_subgroup(),
            monotonicity=_monotonicity(direction),
            b3_max_interval_difference=0.0,
        )
        assert _outcome(rows, "F5", "A|random") == "descriptive"


def test_decision_table_covers_all_criteria():
    rows = _build_decisions(
        fold_metrics=_synth_fold_metrics(),
        subgroup=_empty_subgroup(),
        monotonicity=_monotonicity(),
        b3_max_interval_difference=0.0,
    )
    criteria = {r["criterion"] for r in rows}
    assert criteria == {"F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7"}


# ===========================================================================
# PURE: deterministic digest
# ===========================================================================


def test_deterministic_digest_is_order_independent(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("x\n1\n", encoding="utf-8")
    b.write_text("y\n2\n", encoding="utf-8")
    d1 = A.deterministic_digest([a, b], tmp_path)
    d2 = A.deterministic_digest([b, a], tmp_path)
    assert d1 == d2


def test_deterministic_digest_changes_with_content(tmp_path):
    a = tmp_path / "a.csv"
    a.write_text("x\n1\n", encoding="utf-8")
    d1 = A.deterministic_digest([a], tmp_path)
    a.write_text("x\n2\n", encoding="utf-8")
    d2 = A.deterministic_digest([a], tmp_path)
    assert d1 != d2


# ===========================================================================
# ARTIFACT: reload, determinism, isolation, protected state
# ===========================================================================

_ANALYSIS = A.analysis_dir(T.ARTIFACT_ROOT)
_MANIFEST = _ANALYSIS / A.ANALYSIS_MANIFEST_JSON
_HAVE_ARTIFACTS = _MANIFEST.exists()
requires_artifacts = pytest.mark.skipif(
    not _HAVE_ARTIFACTS, reason="Phase 5 analysis not generated"
)


@pytest.fixture(scope="module")
def phase5_manifest():
    if not _HAVE_ARTIFACTS:
        pytest.skip("Phase 5 analysis not generated")
    with open(_MANIFEST, "r", encoding="utf-8") as fh:
        return json.load(fh)


@requires_artifacts
def test_manifest_is_analysis_only_and_protected_state_clean(phase5_manifest):
    assert phase5_manifest["analysis_only"] is True
    assert phase5_manifest["models_trained"] == 0
    assert phase5_manifest["models_loaded_for_prediction"] == 0
    assert phase5_manifest["folds_regenerated"] is False
    assert phase5_manifest["protected_state_clean"] is True
    assert phase5_manifest["protected_state"]["clean"] is True


@requires_artifacts
def test_expected_row_counts(phase5_manifest):
    counts = phase5_manifest["row_counts"]
    assert counts["fold_metrics"] == 180
    assert counts["summary_metrics"] == 288
    assert counts["regime_shift_coverage"] == 72
    assert counts["subgroup_metrics"] == 2700
    assert counts["decision_table"] == 59
    assert counts["n_figures"] == 12
    assert counts["n_figure_files"] == 24


@requires_artifacts
def test_all_tables_and_figures_present():
    for name in A.RESULT_TABLES:
        assert (_ANALYSIS / name).exists(), name
    for stem in A.FIGURE_STEMS:
        assert (A.figures_dir(_ANALYSIS) / f"{stem}.png").exists(), stem
        assert (A.figures_dir(_ANALYSIS) / f"{stem}.pdf").exists(), stem


@requires_artifacts
def test_recorded_output_hashes_match_reloaded_files(phase5_manifest):
    for rel, sha in phase5_manifest["outputs"].items():
        p = A.ROOT / rel
        assert p.exists(), rel
        assert T.sha256_of(p) == sha, rel


@requires_artifacts
def test_deterministic_content_digest_recomputes(phase5_manifest):
    files = [_ANALYSIS / name for name in A.RESULT_TABLES]
    for stem in A.FIGURE_STEMS:
        files.append(A.figures_dir(_ANALYSIS) / f"{stem}.png")
        files.append(A.figures_dir(_ANALYSIS) / f"{stem}.pdf")
    assert (
        A.deterministic_digest(files, _ANALYSIS)
        == phase5_manifest["deterministic_content_digest"]
    )


@requires_artifacts
def test_fold_metrics_cover_six_methods_three_levels_ten_folds():
    fm = pd.read_csv(_ANALYSIS / A.FOLD_METRICS_CSV)
    assert set(fm["method"]) == set(C.METHODS)
    assert set(fm["nominal"].round(2).tolist()) == {0.80, 0.90, 0.95}
    assert fm["fold"].nunique() == 5
    assert set(fm["regime"]) == set(A.REGIMES)


@requires_artifacts
def test_full_phase5_verification_passes():
    report = A.verify_phase5()
    assert report["verified"] is True
    assert report["protected_state_clean"] is True
    assert report["n_figure_files"] == 24
