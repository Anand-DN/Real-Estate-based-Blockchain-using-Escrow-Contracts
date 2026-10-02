"""Phase 1 unit tests for the Experiment 2 conformal core.

PURE TESTS. No filesystem access, no model training, no dataset loading. Every
test constructs its own arrays, so the suite runs in milliseconds and cannot
mutate Experiment 1 state.

Coverage of the mandated Phase 1 functions:
  conformal_scores, conformal_quantile, build_intervals,
  coverage_metrics, sharpness_metrics, subgroup_coverage

Plus the F0 implementation-validity gates and the exact-quantile contract.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.experiment_2 import conformal as C


# ---------------------------------------------------------------------------
# Phase 0 pre-registration constants are contract, not defaults
# ---------------------------------------------------------------------------


def test_cal_frac_is_the_preregistered_value():
    assert C.CAL_FRAC == 0.20


def test_nominal_levels_are_exactly_the_preregistered_three():
    assert C.NOMINAL_LEVELS == (0.80, 0.90, 0.95)


def test_method_set_is_closed_and_excludes_b2_and_c4():
    assert C.METHODS == ("A", "B1", "B3", "C1", "C2", "C3")
    assert "B2" not in C.METHODS
    assert "C4" not in C.METHODS


def test_f3_gate_threshold_is_preregistered():
    assert C.F3_GATE_LEVEL == 0.90
    assert C.F3_GATE_MAX_MEDIAN_RELATIVE_WIDTH == 2.00


def test_min_subgroup_n_is_the_declared_suppression_rule():
    assert C.MIN_SUBGROUP_N == 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_price(inr):
    return np.log1p(np.asarray(inr, dtype=float))


# ---------------------------------------------------------------------------
# Target-space helpers must match protocol.py exactly
# ---------------------------------------------------------------------------


def test_log_target_matches_protocol():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.valuation_v2_1 import protocol as P

    prices = np.array([2e6, 6.86e6, 8.5e8])
    np.testing.assert_allclose(C.log_target(prices), P.log_target(prices), rtol=0, atol=0)


def test_to_rupees_floors_at_zero_and_inverts_log1p():
    prices = np.array([2e6, 6.86e6, 8.5e8])
    np.testing.assert_allclose(C.to_rupees(C.log_target(prices)), prices, rtol=1e-12)
    assert C.to_rupees(np.array([-5.0]))[0] == 0.0


# ---------------------------------------------------------------------------
# 7. Nonconformity scores
# ---------------------------------------------------------------------------


def test_absolute_residual_score_is_the_plain_difference():
    s = C.absolute_residual_score([1.0, 2.0, 3.0], [1.5, 2.0, 1.0])
    np.testing.assert_allclose(s, [0.5, 0.0, 2.0])


def test_absolute_residual_score_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        C.absolute_residual_score([1.0, 2.0], [1.0])


def test_absolute_residual_score_rejects_non_finite():
    with pytest.raises(ValueError, match="finite"):
        C.absolute_residual_score([1.0, np.nan], [1.0, 2.0])


def test_conformal_scores_a_is_absolute_residual():
    yt = np.array([1.0, 2.0, 3.0])
    yp = np.array([1.2, 1.8, 3.4])
    s, info = C.conformal_scores("A", yt, yp)
    np.testing.assert_allclose(s, [0.2, 0.2, 0.4])
    assert info["width_is_constant"] is True


def test_conformal_scores_b1_divides_by_prediction():
    yt = np.array([2.0, 4.0])
    yp = np.array([1.0, 2.0])
    s, info = C.conformal_scores("B1", yt, yp, y_pred_cal=yp)
    np.testing.assert_allclose(s, [1.0, 1.0])
    assert info["width_is_constant"] is False
    assert info["scale"] == "y_hat"


def test_conformal_scores_b3_is_a_control_and_equals_a():
    yt = np.array([2.0, 4.0, 6.0])
    yp = np.array([1.0, 2.0, 3.0])
    a, _ = C.conformal_scores("A", yt, yp)
    b3, info = C.conformal_scores("B3", yt, yp)
    np.testing.assert_array_equal(a, b3)
    assert info["width_is_constant"] is True
    assert info["scale"] == "one"


def test_b1_guard_floor_is_counted_not_swallowed():
    # A zero divisor would silently distort the score; it must be counted.
    yt = np.array([1.0, 1.0])
    yp = np.array([0.0, 2.0])
    s, info = C.conformal_scores("B1", yt, yp, y_pred_cal=yp)
    assert info["n_guard_activations"] == 1
    assert np.all(np.isfinite(s))


def test_conformal_scores_rejects_excluded_method():
    for excluded in ("B2", "C4"):
        with pytest.raises(ValueError, match="EXCLUDED"):
            C.conformal_scores(excluded, [1.0, 2.0], [1.0, 2.0])


def test_conformal_scores_c1_requires_strata():
    with pytest.raises(ValueError, match="requires strata"):
        C.conformal_scores("C1", [1.0, 2.0], [1.0, 2.0])


def test_conformal_scores_c1_rejects_misaligned_strata():
    with pytest.raises(ValueError, match="does not match"):
        C.conformal_scores("C1", [1.0, 2.0], [1.0, 2.0], strata=["a"])


def test_conformal_scores_c1_tags_strata():
    s, info = C.conformal_scores(
        "C1", [1.0, 2.0, 3.0], [1.1, 2.1, 2.6], strata=["mumbai", "delhi", "mumbai"]
    )
    np.testing.assert_allclose(s, [0.1, 0.1, 0.4])
    assert list(info["strata"]) == ["mumbai", "delhi", "mumbai"]


# ---------------------------------------------------------------------------
# 8. Quantile calculation -- the exact finite-sample contract
# ---------------------------------------------------------------------------


def test_conformal_quantile_matches_hand_computed_order_statistic():
    scores = np.arange(1.0, 11.0)          # 1..10, n = 10
    q = C.conformal_quantile(scores, 0.10)
    # k = ceil(11 * 0.9) = ceil(9.9) = 10 -> s_(10) = 10
    assert q["k"] == 10
    assert q["qhat"] == 10.0
    assert q["n"] == 10
    assert q["unbounded"] is False


@pytest.mark.parametrize("alpha", [0.20, 0.10, 0.05])
def test_conformal_quantile_k_formula_is_exact(alpha):
    scores = np.arange(1.0, 101.0)         # n = 100
    q = C.conformal_quantile(scores, alpha)
    assert q["k"] == math.ceil(101 * (1.0 - alpha))
    assert q["qhat"] == float(q["k"])
    assert q["nominal"] == pytest.approx(1.0 - alpha)


def test_conformal_quantile_is_order_invariant():
    scores = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
    q1 = C.conformal_quantile(scores, 0.10)
    q2 = C.conformal_quantile(scores[::-1].copy(), 0.10)
    assert q1["qhat"] == q2["qhat"]


def test_conformal_quantile_does_not_interpolate():
    # Interpolation would return 5.5 here; the exact k-th order statistic is 5.0.
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    q = C.conformal_quantile(scores, 0.10)   # k = 10 -> 10.0
    assert q["qhat"] == 10.0


def test_conformal_quantile_unbounded_branch_when_k_exceeds_n():
    # n = 5, alpha = 0.5 -> k = ceil(6*0.5) = 3 <= 5. Need k > n:
    # n = 5, alpha = 0.01 -> k = ceil(6*0.99) = 6 > 5 -> unbounded.
    q = C.conformal_quantile(np.arange(1.0, 6.0), 0.01)
    assert q["k"] == 6
    assert q["unbounded"] is True
    assert q["qhat"] == float("inf")


def test_conformal_quantile_unbounded_branch_reachable_and_flagged():
    # The guard is unreachable at planned sizes but must still work (§8).
    q = C.conformal_quantile(np.array([1.0, 2.0]), 0.001)
    assert q["unbounded"] is True


def test_conformal_quantile_rejects_invalid_alpha():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            C.conformal_quantile(np.arange(1.0, 5.0), bad)


def test_conformal_quantile_rejects_non_finite_scores():
    with pytest.raises(ValueError, match="finite"):
        C.conformal_quantile(np.array([1.0, np.inf]), 0.1)


def test_conformal_quantile_by_stratum_computes_per_stratum():
    scores = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    strata = np.array(["a", "a", "a", "b", "b", "b"])
    out = C.conformal_quantile_by_stratum(scores, strata, 0.5)
    assert out["a"]["n"] == 3
    assert out["b"]["n"] == 3
    assert out["a"]["qhat"] < out["b"]["qhat"]
    assert out["a"]["fallback_used"] is False


def test_conformal_quantile_by_stratum_flags_undersized_stratum():
    scores = np.array([1.0, 2.0, 3.0, 10.0])
    strata = np.array(["a", "a", "a", "b"])
    # min_n=3: stratum "a" has exactly 3 rows so it qualifies; "b" has 1 and does not.
    out = C.conformal_quantile_by_stratum(scores, strata, 0.5, min_n=3)
    assert out["b"]["fallback_used"] is True
    assert out["b"]["qhat"] is None
    assert out["a"]["fallback_used"] is False
    assert out["a"]["n"] == 3


# ---------------------------------------------------------------------------
# build_intervals
# ---------------------------------------------------------------------------


def test_build_intervals_a_is_symmetric_in_log_space():
    yp = np.array([10.0, 12.0])
    lo, up, info = C.build_intervals("A", yp, 0.5)
    np.testing.assert_allclose(lo, [9.5, 11.5])
    np.testing.assert_allclose(up, [10.5, 12.5])
    assert info["width_is_constant"] is True


def test_homogeneous_interval_width_is_exactly_two_qhat():
    yp = np.array([10.0, 15.0, 20.0, 25.0])
    q = 1.25
    lo, up, _ = C.build_intervals("A", yp, q)
    np.testing.assert_allclose(up - lo, np.full(yp.shape, 2 * q))
    # §10.1: sd_width must be exactly 0 for a homogeneous method.
    assert np.std(up - lo, ddof=1) == 0.0


def test_relative_width_identity_for_homogeneous_method():
    # §10.1: INR relative width ~= 2*sinh(qhat) for every row, independent of yhat.

    # The identity is exact when the denominator is exp(yhat). The protocol
    # divides by expm1(yhat), which introduces the correction factor
    # exp(yhat)/(exp(yhat)-1). At yhat = 15 that factor is 1 + 3.1e-7, so the
    # identity holds to ~7 significant figures, NOT to machine epsilon.
    # The tolerance below reflects that measured precision.
    q = 1.0534
    exact = 2 * math.sinh(q)
    yp = np.array([14.0, 16.0, 20.0])
    lo, up, _ = C.build_intervals("A", yp, q)
    rel = (C.to_rupees(up) - C.to_rupees(lo)) / C.to_rupees(yp)
    np.testing.assert_allclose(rel, np.full(yp.shape, exact), rtol=1e-6)


def test_relative_width_identity_is_exact_against_exp_not_expm1():
    # Against exp(yhat) the identity is exact to machine precision. This pins
    # down WHERE the small deviation in the previous test comes from, so a
    # future change to to_rupees cannot silently invalidate §10.1.
    q = 1.0534
    yp = np.array([14.0, 16.0, 20.0])
    lo, up, _ = C.build_intervals("A", yp, q)
    rel_exp = (np.exp(up) - np.exp(lo)) / np.exp(yp)
    np.testing.assert_allclose(
        rel_exp, np.full(yp.shape, 2 * math.sinh(q)), rtol=1e-13
    )


def test_relative_width_is_almost_constant_across_rows_for_method_a():
    # The operational consequence of §10.1: for method A, relative width carries
    # essentially no per-row information. It must not be used to rank methods.
    q = 1.0534
    yp = np.array([14.0, 15.0, 16.0, 18.0, 20.0])
    lo, up, _ = C.build_intervals("A", yp, q)
    rel = (C.to_rupees(up) - C.to_rupees(lo)) / C.to_rupees(yp)
    assert float(np.ptp(rel)) / float(np.mean(rel)) < 1e-5


def test_build_intervals_b1_width_varies_with_prediction():
    yp = np.array([10.0, 20.0])
    q = 0.5
    lo, up, info = C.build_intervals("B1", yp, q, scale=yp)
    width = up - lo
    assert width[1] > width[0]
    assert info["width_is_constant"] is False


def test_build_intervals_b1_requires_scale():
    with pytest.raises(ValueError, match="requires scale"):
        C.build_intervals("B1", np.array([10.0]), 0.5)


def test_build_intervals_rejects_mapping_for_pooled_method():
    with pytest.raises(ValueError, match="scalar qhat"):
        C.build_intervals("A", np.array([10.0]), {"mumbai": {"qhat": 1.0}})


def test_build_intervals_rejects_scalar_for_mondrian_method():
    with pytest.raises(ValueError, match="per-stratum"):
        C.build_intervals("C1", np.array([10.0]), 1.0, strata=["mumbai"])


def test_build_intervals_unbounded_qhat_gives_infinite_interval():
    lo, up, info = C.build_intervals("A", np.array([10.0, 11.0]), float("inf"))
    assert np.all(np.isneginf(lo))
    assert np.all(np.isposinf(up))
    assert info["unbounded"] is True


def test_build_intervals_c1_applies_per_stratum_quantiles():
    yp = np.array([10.0, 10.0, 20.0])
    strata = np.array(["mumbai", "delhi", "mumbai"])
    qhat = {
        "mumbai": {"qhat": 1.0, "n": 2, "k": 2, "unbounded": False, "fallback_used": False},
        "delhi": {"qhat": 3.0, "n": 1, "k": 1, "unbounded": False, "fallback_used": False},
    }
    lo, up, info = C.build_intervals("C1", yp, qhat, strata=strata)
    np.testing.assert_allclose(lo, [9.0, 7.0, 19.0])
    np.testing.assert_allclose(up, [11.0, 13.0, 21.0])
    assert info["n_fallback"] == 0


def test_build_intervals_c1_fallback_stratum_is_unbounded_not_silently_pooled():
    yp = np.array([10.0, 10.0])
    strata = np.array(["mumbai", "hyderabad"])
    qhat = {
        "mumbai": {"qhat": 1.0, "n": 2, "k": 2, "unbounded": False, "fallback_used": False},
        "hyderabad": {"qhat": None, "n": 1, "k": None, "unbounded": None, "fallback_used": True},
    }
    lo, up, info = C.build_intervals("C1", yp, qhat, strata=strata)
    assert info["n_fallback"] == 1
    assert up[1] == float("inf") and lo[1] == float("-inf")


def test_build_intervals_rejects_excluded_method():
    with pytest.raises(ValueError, match="not in pre-registered"):
        C.build_intervals("B2", np.array([10.0]), 1.0)


# ---------------------------------------------------------------------------
# 9. Coverage metrics
# ---------------------------------------------------------------------------


def test_coverage_metrics_on_a_perfect_interval():
    yt = np.array([1.0, 2.0, 3.0])
    lo = np.array([0.5, 1.5, 2.5])
    up = np.array([1.5, 2.5, 3.5])
    m = C.coverage_metrics(yt, lo, up, 0.90)
    assert m["empirical_coverage"] == 1.0
    assert m["n_covered"] == 3
    assert m["coverage_error"] == pytest.approx(0.10)


def test_coverage_significant_is_false_for_tiny_n_even_at_100_percent():
    # With n=3 the Clopper-Pearson interval is [0.292, 1.000], which CONTAINS
    # 0.90. Perfect coverage on 3 observations is not evidence of
    # miscalibration, and the gate must say so.
    m = C.coverage_metrics(np.array([1.0, 2.0, 3.0]),
                           np.array([0.5, 1.5, 2.5]),
                           np.array([1.5, 2.5, 3.5]), 0.90)
    assert m["coverage_cp_lower"] < 0.90 < m["coverage_cp_upper"]
    assert m["coverage_significant"] is False


def test_coverage_significant_is_true_at_100_percent_with_large_n():
    m = C.coverage_metrics(np.zeros(5000), -np.ones(5000), np.ones(5000), 0.90)
    assert m["empirical_coverage"] == 1.0
    assert m["coverage_cp_lower"] > 0.90
    assert m["coverage_significant"] is True


def test_coverage_metrics_counts_miscoverage_tails_separately():
    yt = np.array([0.0, 1.0, 2.0, 3.0])
    lo = np.full(4, 0.5)
    up = np.full(4, 2.5)
    m = C.coverage_metrics(yt, lo, up, 0.90)
    assert m["n_miscoverage_low"] == 1
    assert m["n_miscoverage_high"] == 1
    assert m["miscoverage_low"] == 0.25
    assert m["miscoverage_high"] == 0.25
    assert m["empirical_coverage"] == 0.5


def test_coverage_metrics_interval_is_inclusive_at_the_boundaries():
    yt = np.array([1.0, 2.0])
    lo = np.array([1.0, 1.0])
    up = np.array([1.0, 3.0])
    m = C.coverage_metrics(yt, lo, up, 0.50)
    assert m["n_covered"] == 2


def test_coverage_metrics_clopper_pearson_brackets_the_point_estimate():
    # Construct an interval that misses a known fraction so the point estimate
    # is strictly interior to the CP bounds. An all-covered interval puts the
    # point estimate AT the upper bound of 1.0 by construction.
    rng = np.random.default_rng(0)
    n = 5000
    yt = rng.normal(size=n)
    lo = yt.copy()
    up = yt.copy()
    miss = rng.choice(n, size=500, replace=False)   # 10% miss from above
    up[miss] = yt[miss] - 1.0
    m = C.coverage_metrics(yt, lo, up, 0.90)
    assert m["empirical_coverage"] == pytest.approx(0.90, abs=0.01)
    assert m["coverage_cp_lower"] < m["empirical_coverage"] < m["coverage_cp_upper"]


def test_clopper_pearson_handles_degenerate_success_counts():
    lo, hi = C._clopper_pearson(0, 10, 0.95)
    assert lo == 0.0 and 0.0 < hi < 1.0
    lo, hi = C._clopper_pearson(10, 10, 0.95)
    assert hi == 1.0 and 0.0 < lo < 1.0
    lo, hi = C._clopper_pearson(0, 0, 0.95)
    assert (lo, hi) == (0.0, 1.0)


def test_coverage_metrics_nominal_inside_interval_is_not_significant():
    # Construct a case whose empirical coverage is very close to nominal.
    n = 10000
    rng = np.random.default_rng(42)
    yt = rng.normal(size=n)
    lo = yt.copy()
    up = yt.copy()
    hits = int(round(0.90 * n))
    idx = rng.choice(n, size=n - hits, replace=False)
    up[idx] = yt[idx] - 1.0     # force exactly (n - hits) misses from above
    m = C.coverage_metrics(yt, lo, up, 0.90)
    assert m["empirical_coverage"] == pytest.approx(0.90, abs=1e-9)
    assert m["coverage_significant"] is False


def test_coverage_metrics_rejects_invalid_nominal():
    with pytest.raises(ValueError, match="nominal"):
        C.coverage_metrics(np.array([1.0]), np.array([0.0]), np.array([2.0]), 1.5)


def test_coverage_metrics_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        C.coverage_metrics(np.array([1.0, 2.0]), np.array([0.0]), np.array([3.0]), 0.90)


def test_coverage_sequence_reports_monotone_increasing():
    out = C.coverage_sequence_by_level([0.81, 0.91, 0.96])
    assert out["direction"] == "increasing"
    assert out["is_directionally_monotone"] is True
    assert out["descriptive_only"] is True


def test_coverage_sequence_reports_monotone_decreasing():
    out = C.coverage_sequence_by_level([0.96, 0.91, 0.81])
    assert out["direction"] == "decreasing"
    assert out["is_directionally_monotone"] is True


def test_coverage_sequence_reports_non_monotone_without_failing():
    # Corrected F5: non-monotonicity is reported, never treated as a failure.
    out = C.coverage_sequence_by_level([0.95, 0.90, 0.92])
    assert out["direction"] == "non_monotone"
    assert out["is_directionally_monotone"] is False
    assert out["descriptive_only"] is True
    assert len(out["coverage_errors"]) == 3


def test_coverage_sequence_orders_levels_before_judging():
    out = C.coverage_sequence_by_level([0.96, 0.81, 0.91], levels=[0.95, 0.80, 0.90])
    assert out["levels"] == [0.80, 0.90, 0.95]
    assert out["coverages"] == [0.81, 0.91, 0.96]


# ---------------------------------------------------------------------------
# 10. Sharpness metrics
# ---------------------------------------------------------------------------


def test_interval_score_is_width_when_all_covered():
    yt = np.array([1.0, 2.0])
    lo = np.array([0.5, 1.5])
    up = np.array([1.5, 2.5])
    np.testing.assert_allclose(C.interval_score(yt, lo, up, 0.10), [1.0, 1.0])


def test_interval_score_penalises_miscoverage_linearly():
    yt = np.array([0.0])
    lo = np.array([1.0])
    up = np.array([2.0])
    alpha = 0.10
    expected = (2.0 - 1.0) + (2.0 / alpha) * (1.0 - 0.0)
    np.testing.assert_allclose(C.interval_score(yt, lo, up, alpha), [expected])


def test_interval_score_rejects_invalid_alpha():
    with pytest.raises(ValueError, match="alpha"):
        C.interval_score(np.array([1.0]), np.array([0.0]), np.array([2.0]), 0.0)


def test_sharpness_sd_width_is_zero_for_homogeneous_methods():
    yp = np.array([10.0, 12.0, 14.0])
    lo, up, _ = C.build_intervals("A", yp, 1.0)
    m = C.sharpness_metrics(np.array([10.5, 12.5, 13.5]), yp, lo, up, 0.10, method="A")
    assert m["sd_width_log"] == 0.0
    assert m["width_is_constant"] is True


def test_sharpness_sd_width_is_positive_for_b1():
    yp = np.array([10.0, 20.0, 30.0])
    lo, up, _ = C.build_intervals("B1", yp, 0.5, scale=yp)
    m = C.sharpness_metrics(np.array([10.5, 20.5, 30.5]), yp, lo, up, 0.10, method="B1")
    assert m["sd_width_log"] > 0.0
    assert m["width_is_constant"] is False


def test_sharpness_mean_equals_median_for_constant_width():
    yp = np.linspace(14.0, 20.0, 101)
    lo, up, _ = C.build_intervals("A", yp, 0.9)
    m = C.sharpness_metrics(np.full(101, 15.0), yp, lo, up, 0.10, method="A")
    assert m["mean_width_log"] == pytest.approx(m["median_width_log"])


def test_sharpness_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="share a shape"):
        C.sharpness_metrics(
            np.array([1.0, 2.0]), np.array([1.0]), np.array([0.0]), np.array([2.0]), 0.1
        )


# ---------------------------------------------------------------------------
# F3 usability gate
# ---------------------------------------------------------------------------


def test_f3_gate_passes_for_a_narrow_interval():
    yp = np.array([15.0, 15.0, 15.0])
    lo, up, _ = C.build_intervals("A", yp, 0.4)
    m = C.sharpness_metrics(np.array([15.0] * 3), yp, lo, up, 0.10, method="A")
    gate = C.evaluate_usability_gate({0.90: m})
    assert gate["usable"] is True
    assert gate["gate_level_nominal"] == 0.90
    assert gate["threshold"] == 2.00


def test_f3_gate_fails_for_a_wide_interval():
    # q = 1.0534 -> relative width ~2.52 > 2.00, the Section 10.3 expectation.
    yp = np.array([15.0, 15.0, 15.0])
    lo, up, _ = C.build_intervals("A", yp, 1.0534)
    m = C.sharpness_metrics(np.array([15.0] * 3), yp, lo, up, 0.10, method="A")
    gate = C.evaluate_usability_gate({0.90: m})
    assert gate["usable"] is False
    assert gate["median_relative_width_inr"] == pytest.approx(
        2 * math.sinh(1.0534), rel=1e-6
    )


def test_f3_gate_boundary_sits_at_asinh_one():
    # The gate boundary in q is q = asinh(1.0) = 0.88137359, derived from
    # 2*sinh(q) = 2.00. Because the protocol divides by expm1(yhat), the measured
    # relative width at exactly asinh(1) lands a hair ABOVE 2.00 (by ~6e-7),
    # so the boundary is tested with a tolerance matched to the identity's real
    # precision rather than to machine epsilon.
    q_boundary = math.asinh(1.0)
    yp = np.array([15.0, 15.0, 15.0])
    lo, up, _ = C.build_intervals("A", yp, q_boundary)
    m = C.sharpness_metrics(np.array([15.0] * 3), yp, lo, up, 0.10, method="A")
    gate = C.evaluate_usability_gate({0.90: m})
    assert gate["median_relative_width_inr"] == pytest.approx(2.00, rel=1e-5)


def test_f3_gate_boundary_transitions_at_asinh_one():
    # Just inside the boundary passes, just outside fails. This pins the
    # threshold as a behavioural boundary rather than an arithmetic coincidence.
    yp = np.array([15.0])
    inside = math.asinh(1.0) - 1e-4
    outside = math.asinh(1.0) + 1e-4

    lo, up, _ = C.build_intervals("A", yp, inside)
    m_in = C.sharpness_metrics(np.array([15.0]), yp, lo, up, 0.10, method="A")
    assert C.evaluate_usability_gate({0.90: m_in})["usable"] is True

    lo, up, _ = C.build_intervals("A", yp, outside)
    m_out = C.sharpness_metrics(np.array([15.0]), yp, lo, up, 0.10, method="A")
    assert C.evaluate_usability_gate({0.90: m_out})["usable"] is False


def test_f3_gate_requires_the_090_level():
    yp = np.array([15.0, 15.0])
    lo, up, _ = C.build_intervals("A", yp, 0.5)
    m = C.sharpness_metrics(np.array([15.0] * 2), yp, lo, up, 0.20, method="A")
    with pytest.raises(ValueError, match="requires metrics at nominal 0.9"):
        C.evaluate_usability_gate({0.80: m})


def test_f3_gate_records_its_provenance_and_independence():
    yp = np.array([15.0])
    lo, up, _ = C.build_intervals("A", yp, 0.5)
    m = C.sharpness_metrics(np.array([15.0]), yp, lo, up, 0.10, method="A")
    gate = C.evaluate_usability_gate({0.90: m})
    assert "NOT a universal" in gate["threshold_provenance"]
    assert gate["reported_independently_of_calibration"] is True


# ---------------------------------------------------------------------------
# 12. Subgroup coverage
# ---------------------------------------------------------------------------


def test_subgroup_coverage_splits_by_label():
    yt = np.array([1.0, 2.0, 3.0, 4.0])
    lo = np.array([0.0, 1.0, 2.0, 3.0])
    up = np.array([2.0, 3.0, 4.0, 5.0])
    rows = C.subgroup_coverage(yt, lo, up, ["a", "a", "b", "b"], 0.90, min_n=1)
    assert [r["subgroup"] for r in rows] == ["a", "b"]
    assert rows[0]["n_evaluated"] == 2
    assert rows[1]["n_evaluated"] == 2


def test_subgroup_coverage_suppresses_undersized_cells():
    yt = np.array([1.0, 2.0, 3.0])
    lo = np.array([0.0, 1.0, 2.0])
    up = np.array([2.0, 3.0, 4.0])
    rows = C.subgroup_coverage(yt, lo, up, ["a", "a", "tiny"], 0.90, min_n=2)
    by = {r["subgroup"]: r for r in rows}
    assert by["a"]["coverage_reliable"] is True
    assert by["tiny"]["coverage_reliable"] is False
    assert by["tiny"]["n_evaluated"] == 1
    # The value is still present, just flagged, so it cannot silently vanish.
    assert "empirical_coverage" in by["tiny"]


def test_subgroup_coverage_default_min_n_is_100():
    yt = np.array([1.0, 2.0])
    lo = np.array([0.0, 1.0])
    up = np.array([2.0, 3.0])
    rows = C.subgroup_coverage(yt, lo, up, ["a", "b"], 0.90)
    assert all(r["coverage_reliable"] is False for r in rows)
    assert rows[0]["min_n_required"] == 100


def test_subgroup_coverage_preserves_first_appearance_order():
    yt = np.array([1.0, 2.0, 3.0])
    lo = np.zeros(3)
    up = np.full(3, 10.0)
    rows = C.subgroup_coverage(yt, lo, up, ["z", "a", "z"], 0.90, min_n=1)
    assert [r["subgroup"] for r in rows] == ["z", "a"]


# ---------------------------------------------------------------------------
# Price banding
# ---------------------------------------------------------------------------


def test_price_band_labels_use_fixed_absolute_edges():
    p = np.array([2.0e6, 7.5e6, 1.5e7, 3.0e7, 6.0e7, 2.0e8])
    lab = C.price_band_labels(p)
    assert list(lab) == ["<0.5cr", "0.5-1cr", "1-2cr", "2-4cr", "4-10cr", ">=10cr"]


def test_price_band_labels_reject_values_outside_the_edges():
    with pytest.raises(ValueError, match="outside all declared edges"):
        C.price_band_labels(np.array([-5.0]))


def test_price_band_labels_reject_mismatched_label_count():
    with pytest.raises(ValueError, match="len"):
        C.price_band_labels(np.array([1e6]), edges=(0.0, 1e7), labels=("a", "b"))


def test_predicted_tertile_labels_are_assignable_and_balanced():
    yp = np.arange(100.0)
    lab, cuts = C.predicted_tertile_labels(yp)
    assert list(lab[:33]) == ["low"] * 33
    assert list(lab[-1:]) == ["high"]
    assert cuts[0] < cuts[1]


def test_predicted_tertile_labels_apply_fixed_cutpoints():
    yp = np.array([1.0, 2.0, 3.0])
    lab, cuts = C.predicted_tertile_labels(yp, cutpoints=[1.5, 2.5])
    assert list(lab) == ["low", "mid", "high"]
    np.testing.assert_allclose(cuts, [1.5, 2.5])


def test_predicted_tertile_labels_reject_non_increasing_cutpoints():
    with pytest.raises(ValueError, match="strictly increasing"):
        C.predicted_tertile_labels(np.array([1.0, 2.0, 3.0]), cutpoints=[2.5, 1.5])


def test_predicted_tertile_and_true_price_bands_are_different_objects():
    # Guards the §12.2 distinction: one is label-based (diagnostic), one is not.
    assert "true" not in C.predicted_tertile_labels.__doc__.lower().replace("truly", "")


# ---------------------------------------------------------------------------
# F0 implementation-validity gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nominal", C.NOMINAL_LEVELS)
def test_b3_matches_a_at_every_preregistered_level(nominal):
    rng = np.random.default_rng(7)
    yt = rng.normal(loc=15.0, scale=0.5, size=4000)
    yp = rng.normal(loc=15.0, scale=0.5, size=4000)
    out = C.b3_matches_a(yt, yp, nominal)
    assert out["passes"] is True
    assert out["max_abs_interval_difference"] <= C.B3_MATCH_TOLERANCE


def test_b3_matches_a_on_a_realistic_log_price_scale():
    yt = _log_price(np.array([2e6, 6.86e6, 3.35e7, 8.5e8]))
    yp = yt + np.array([0.05, -0.2, 0.3, -0.4])
    out = C.b3_matches_a(yt, yp, 0.90)
    assert out["passes"] is True


def test_b3_gate_would_catch_a_broken_normalisation():
    # If the "sigma == 1" path were implemented with a scale of 2 instead of 1,
    # the F0 gate must fail. n must be large enough that k <= n, otherwise both
    # quantiles are unbounded and the comparison degenerates to inf - inf = nan.
    n = 500
    rng = np.random.default_rng(3)
    yt = rng.normal(15.0, 0.5, size=n)
    yp = yt + rng.normal(0.0, 0.2, size=n)

    a_scores, _ = C.conformal_scores("A", yt, yp)
    broken, _ = C.normalized_residual_score(yt, yp, scale=np.full(n, 2.0))

    a_q = C.conformal_quantile(a_scores, 0.10)
    broken_q = C.conformal_quantile(broken, 0.10)
    assert a_q["unbounded"] is False and broken_q["unbounded"] is False
    assert abs(a_q["qhat"] - broken_q["qhat"]) > C.B3_MATCH_TOLERANCE


def test_b3_gate_is_undefined_rather_than_nan_when_both_are_unbounded():
    # Guards the inf - inf = nan hazard. b3_matches_a only differences the
    # finite mask, so it must report 0.0 rather than nan when both are unbounded.
    out = C.b3_matches_a(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 0.999)
    assert out["max_abs_interval_difference"] == 0.0
    assert out["passes"] is True


# ---------------------------------------------------------------------------
# End-to-end on synthetic data: split conformal must attain nominal coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nominal", [0.80, 0.90])
def test_split_conformal_attains_nominal_coverage_on_exchangeable_data(nominal):
    """Positive control on the machinery itself.

    Under exchangeable data, split conformal should cover at approximately the
    nominal rate. This is a test of the implementation, not an Experiment 2
    result, and uses synthetic data only.
    """
    rng = np.random.default_rng(20240101)
    y_cal = rng.normal(15.0, 1.0, size=5000)
    pred_cal = y_cal + rng.normal(0.0, 0.5, size=5000)
    scores, _ = C.conformal_scores("A", y_cal, pred_cal)
    q = C.conformal_quantile(scores, 1.0 - nominal)["qhat"]

    y_test = rng.normal(15.0, 1.0, size=20000)
    pred_test = y_test + rng.normal(0.0, 0.5, size=20000)
    lo, up, _ = C.build_intervals("A", pred_test, q)
    m = C.coverage_metrics(y_test, lo, up, nominal)

    assert m["coverage_significant"] is False
    assert abs(m["coverage_error"]) < 0.02


def test_normalized_conformal_covers_on_heteroscedastic_data():
    """B1 must also cover when the residual scale tracks the prediction."""
    rng = np.random.default_rng(11)
    y_cal = rng.uniform(14.0, 20.0, size=5000)
    sig = y_cal.copy()                       # heteroscedastic by construction
    pred_cal = y_cal + rng.normal(0.0, 0.02, size=5000) * sig
    scores, _ = C.conformal_scores("B1", y_cal, pred_cal, y_pred_cal=pred_cal)
    q = C.conformal_quantile(scores, 0.10)["qhat"]

    y_test = rng.uniform(14.0, 20.0, size=20000)
    pred_test = y_test + rng.normal(0.0, 0.02, size=20000) * y_test
    lo, up, _ = C.build_intervals("B1", pred_test, q, scale=pred_test)
    m = C.coverage_metrics(y_test, lo, up, 0.90)

    assert m["coverage_significant"] is False


def test_shifted_data_must_be_allowed_to_undercover():
    """The experiment must be able to show failure.

    This is the location-shift scenario: the predictor degrades on the shifted
    distribution, so residual scale grows and a quantile calibrated on the
    in-distribution calibration set is too narrow.

    The earlier version of this test shifted y AND set pred = y + noise, which
    leaves the residual scale unchanged and therefore covers perfectly. A shift
    in the target alone does not break calibration when the predictor is defined
    relative to the target. The predictor must actually get worse.
    """
    rng = np.random.default_rng(5)
    y_cal = rng.normal(15.0, 0.3, size=5000)
    pred_cal = y_cal + rng.normal(0.0, 0.3, size=5000)
    scores, _ = C.conformal_scores("A", y_cal, pred_cal)
    q = C.conformal_quantile(scores, 0.10)["qhat"]

    # Test distribution has the same scale, but the predictor is anchored to the
    # calibration mean, so residuals on the shifted target are much larger.
    y_test = rng.normal(16.5, 0.3, size=20000)
    pred_test = np.full(20000, 15.0) + rng.normal(0.0, 0.3, size=20000)
    lo, up, _ = C.build_intervals("A", pred_test, q)
    m = C.coverage_metrics(y_test, lo, up, 0.90)

    assert m["empirical_coverage"] < 0.90
    assert m["coverage_significant"] is True


def test_pure_target_shift_with_a_relative_predictor_does_not_undercover():
    """Documents the boundary of the case above.

    When the predictor tracks the target (pred = y + noise) a shift in y alone
    leaves the residual scale unchanged, so split conformal still covers. This is
    the correct behaviour, and it is why the undercoverage test must degrade the
    predictor rather than merely move the target.
    """
    rng = np.random.default_rng(5)
    y_cal = rng.normal(15.0, 0.3, size=5000)
    pred_cal = y_cal + rng.normal(0.0, 0.3, size=5000)
    scores, _ = C.conformal_scores("A", y_cal, pred_cal)
    q = C.conformal_quantile(scores, 0.10)["qhat"]

    y_test = rng.normal(16.5, 0.3, size=20000)
    pred_test = y_test + rng.normal(0.0, 0.3, size=20000)
    lo, up, _ = C.build_intervals("A", pred_test, q)
    m = C.coverage_metrics(y_test, lo, up, 0.90)

    assert m["coverage_significant"] is False


def test_overcoverage_is_also_detectable():
    """Overcoverage must be reportable as its own outcome (F4)."""
    rng = np.random.default_rng(6)
    y_cal = rng.normal(15.0, 1.0, size=5000)
    pred_cal = y_cal + rng.normal(0.0, 0.5, size=5000)
    scores, _ = C.conformal_scores("A", y_cal, pred_cal)
    q = C.conformal_quantile(scores, 0.05)["qhat"]       # deliberately over-wide

    y_test = rng.normal(15.0, 1.0, size=20000)
    pred_test = y_test + rng.normal(0.0, 0.5, size=20000)
    lo, up, _ = C.build_intervals("A", pred_test, q)
    m = C.coverage_metrics(y_test, lo, up, 0.90)

    assert m["empirical_coverage"] > 0.95
    assert m["coverage_error"] > 0.05
