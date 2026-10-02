"""Phase 4 tests: conformal scoring and interval generation.

Two groups of tests:

  PURE      - exercise the scoring logic on synthetic arrays. No filesystem
              access, no models, no dataset. These prove the calibration/test
              isolation and the fallback semantics structurally.

  ARTIFACT  - reload and verify the persisted Phase 4 artifacts. Skipped when
              the artifacts have not been generated, so the pure suite always
              runs.

The tests make no statistical claim about coverage, calibration or usability.
They verify construction and isolation only.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.experiment_2 import build_calibration_folds as B
from scripts.experiment_2 import conformal as C
from scripts.experiment_2 import run_conformal as R
from scripts.experiment_2 import train_nested_models as T
from scripts.valuation_v2_1 import protocol as P


# ===========================================================================
# PURE: pre-registered contract
# ===========================================================================


def test_methods_are_exactly_the_six_registered():
    assert R.METHODS == ("A", "B1", "B3", "C1", "C2", "C3")


def test_nominal_levels_are_the_three_registered():
    assert R.NOMINAL_LEVELS == (0.80, 0.90, 0.95)


def test_level_tag_mapping_is_stable():
    assert R.level_tag(0.80) == "80"
    assert R.level_tag(0.90) == "90"
    assert R.level_tag(0.95) == "95"


def test_interval_column_order_is_fixed():
    assert R.INTERVAL_COLUMNS[0] == "regime"
    assert "lower_log" in R.INTERVAL_COLUMNS
    assert R.INTERVAL_COLUMNS[-1] == "fallback_used"


# ===========================================================================
# PURE: exact finite-sample order statistic, no interpolation
# ===========================================================================


def test_exact_quantile_is_kth_order_statistic_not_interpolated():
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    q = C.conformal_quantile(scores, alpha=0.5)
    assert q["k"] == 3
    assert q["qhat"] == 3.0  # the 3rd order statistic
    assert q["qhat"] != float(np.quantile(scores, 0.5))


@pytest.mark.parametrize("alpha", [0.20, 0.10, 0.05])
def test_quantile_k_formula_is_ceil_n_plus_one(alpha):
    n = 400
    q = C.conformal_quantile(np.arange(1, n + 1, dtype=float), alpha)
    assert q["k"] == int(np.ceil((n + 1) * (1.0 - alpha)))
    assert q["unbounded"] is False


def test_quantile_unbounded_when_k_exceeds_n_reachable():
    q = C.conformal_quantile(np.array([1.0, 2.0]), alpha=0.05)
    assert q["k"] == 3
    assert q["k"] > q["n"]
    assert q["unbounded"] is True


# ===========================================================================
# PURE: calibration/test isolation
# ===========================================================================


def _synthetic(n_cal=400, n_test=120, n_fit=800, seed=0, zero_scale=False):
    rng = np.random.default_rng(seed)
    y_cal = 14.5 + rng.normal(0, 0.3, n_cal)
    yhat_cal = y_cal + rng.normal(0, 0.2, n_cal)
    if zero_scale:
        yhat_cal = yhat_cal.copy()
        yhat_cal[0] = 0.0
    yhat_fit = 14.5 + rng.normal(0, 0.3, n_fit)
    yhat_test = 14.5 + rng.normal(0, 0.3, n_test)
    city_cal = rng.choice(["a", "b"], n_cal)
    city_test = rng.choice(["a", "b"], n_test)
    group_cal = rng.choice(["g1", "g2", "g3"], n_cal)
    group_test = rng.choice(["g1", "g2", "g3"], n_test)
    return dict(
        y_cal=y_cal, yhat_cal=yhat_cal, yhat_test=yhat_test, yhat_fit=yhat_fit,
        city_cal=city_cal, city_test=city_test,
        group_cal=group_cal, group_test=group_test,
    )


def _cells_by_key(result):
    return {(c["method"], float(c["nominal"])): c for c in result["cells"]}


def _synthetic_many_groups(n_cal=400, n_test=120, n_fit=800, seed=0, n_groups=200):
    """Synthetic fold with many calibration groups, so C3 k <= n at all levels."""
    d = _synthetic(n_cal=n_cal, n_test=n_test, n_fit=n_fit, seed=seed)
    rng = np.random.default_rng(seed + 1)
    d["group_cal"] = np.array([f"g{i}" for i in rng.integers(0, n_groups, n_cal)])
    d["group_test"] = np.array([f"g{i}" for i in rng.integers(0, n_groups, n_test)])
    return d


def test_scoring_function_does_not_receive_test_labels():
    params = set(inspect.signature(R.compute_fold_conformal).parameters)
    for forbidden in ("y_test", "actual", "actual_price", "price_test"):
        assert forbidden not in params


def test_calibration_quantiles_do_not_depend_on_test_predictions():
    data = _synthetic()
    other = dict(data)
    other["yhat_test"] = data["yhat_test"] + 100.0  # completely different test set
    r1 = _cells_by_key(R.compute_fold_conformal(regime="random", fold_id=1, **data))
    r2 = _cells_by_key(R.compute_fold_conformal(regime="random", fold_id=1, **other))
    for method in ("A", "B1", "B3", "C3"):
        assert r1[(method, 0.9)]["quantile"] == r2[(method, 0.9)]["quantile"]
    # Mondrian strata quantiles also unchanged (compare shared strata: the
    # deliberately shifted test predictions can empty out a C2 band).
    for method in ("C1", "C2"):
        s1 = {s["stratum"]: s["qhat"] for s in r1[(method, 0.9)]["strata"]}
        s2 = {s["stratum"]: s["qhat"] for s in r2[(method, 0.9)]["strata"]}
        common = set(s1) & set(s2)
        assert common
        assert all(s1[k] == s2[k] for k in common)


# ===========================================================================
# PURE: C1 city isolation
# ===========================================================================


def test_c1_uses_only_scores_from_its_own_city():
    # City 'a' has tiny residuals, city 'b' has larger ones.
    y_cal = np.ones(8)
    yhat_cal = np.array([0.9, 0.8, 0.7, 0.6, 3.0, 2.0, 1.0, 0.0])
    # |r| : a = 0.1, 0.2, 0.3, 0.4 ; b = 2, 1, 0, 1
    city_cal = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    yhat_fit = np.linspace(0.0, 10.0, 30)
    yhat_test = np.array([2.0, 2.0])
    city_test = np.array(["a", "b"])
    grp_cal = np.array(["g1", "g1", "g1", "g1", "g2", "g2", "g2", "g2"])
    grp_test = np.array(["g1", "g2"])
    res = R.compute_fold_conformal(
        regime="random", fold_id=1,
        y_cal=y_cal, yhat_cal=yhat_cal, yhat_test=yhat_test, yhat_fit=yhat_fit,
        city_cal=city_cal, city_test=city_test, group_cal=grp_cal, group_test=grp_test,
    )
    c1 = _cells_by_key(res)[("C1", 0.80)]
    strata = {s["stratum"]: s for s in c1["strata"]}
    # n = 4 per city, k = ceil(5*0.8) = 4 -> the largest score in each city.
    assert strata["a"]["qhat"] == pytest.approx(0.4)
    assert strata["b"]["qhat"] == pytest.approx(2.0)
    assert strata["a"]["unbounded"] is False
    # city 'a' uses only its own small scores, not city b's larger ones
    assert strata["a"]["qhat"] < strata["b"]["qhat"]


# ===========================================================================
# PURE: C2 cutpoints come from T'_k, bands are assignable
# ===========================================================================


def test_c2_cutpoints_come_from_fit_predictions_not_test():
    data = _synthetic()
    r1 = R.compute_fold_conformal(regime="random", fold_id=1, **data)
    changed_test = dict(data)
    changed_test["yhat_test"] = data["yhat_test"] * 10.0
    r2 = R.compute_fold_conformal(regime="random", fold_id=1, **changed_test)
    assert r1["cutpoints"] == r2["cutpoints"]

    changed_fit = dict(data)
    changed_fit["yhat_fit"] = data["yhat_fit"] + 5.0
    r3 = R.compute_fold_conformal(regime="random", fold_id=1, **changed_fit)
    assert r3["cutpoints"] != r1["cutpoints"]


# ===========================================================================
# PURE: C3 group aggregation
# ===========================================================================


def test_group_mean_scores_is_mean_within_group_in_stable_order():
    scores = np.array([1.0, 3.0, 10.0, 20.0, 2.0])
    groups = np.array(["b", "b", "a", "a", "c"])
    got = R.group_mean_scores(scores, groups)
    assert got.tolist() == [2.0, 15.0, 2.0]  # first-appearance order: b, a, c


def test_c3_takes_one_quantile_over_calibration_groups():
    data = _synthetic_many_groups()
    res = R.compute_fold_conformal(regime="random", fold_id=1, **data)
    c3 = _cells_by_key(res)[("C3", 0.90)]
    n_groups = res["n_calibration_groups"]
    assert n_groups > 100
    assert c3["n"] == n_groups
    assert c3["k"] == int(np.ceil((n_groups + 1) * 0.90))
    assert c3["unbounded"] is False
    # interval is symmetric around the point prediction
    ip = res["intervals"][("C3", 0.90)]
    assert np.allclose(ip["lower"], data["yhat_test"] - c3["quantile"])
    assert np.allclose(ip["upper"], data["yhat_test"] + c3["quantile"])


def test_c3_does_not_use_test_group_identities_in_the_quantile():
    data = _synthetic_many_groups()
    changed = dict(data)
    changed["group_test"] = np.array(["zzz"] * len(data["group_test"]))
    r1 = _cells_by_key(R.compute_fold_conformal(regime="random", fold_id=1, **data))
    r2 = _cells_by_key(R.compute_fold_conformal(regime="random", fold_id=1, **changed))
    assert r1[("C3", 0.95)]["quantile"] == r2[("C3", 0.95)]["quantile"]
    assert r1[("C3", 0.95)]["quantile"] is not None


# ===========================================================================
# PURE: C1/C2 fallback semantics (unseen / insufficient)
# ===========================================================================


def _mondrian_data(city_cal, city_test):
    n = len(city_cal)
    return dict(
        y_cal=np.full(n, 1.0),
        yhat_cal=np.zeros(n),
        yhat_test=np.array([0.5, 0.5, 0.5]),
        yhat_fit=np.linspace(0.0, 1.0, 30),
        city_cal=np.asarray(city_cal),
        city_test=np.asarray(city_test),
        group_cal=np.full(n, "g"),
        group_test=np.array(["g", "g", "g"]),
    )


def test_unseen_c1_stratum_is_unbounded_and_flagged():
    data = _mondrian_data(["a"] * 50, ["a", "a", "z"])
    res = R.compute_fold_conformal(regime="random", fold_id=1, **data)
    c1 = _cells_by_key(res)[("C1", 0.90)]
    ip = res["intervals"][("C1", 0.90)]
    assert c1["n_fallback"] == 1
    assert ip["fallback"][2] is np.True_ or bool(ip["fallback"][2])
    assert ip["lower"][2] == -np.inf and ip["upper"][2] == np.inf
    # bounded strata are untouched
    assert np.isfinite(ip["lower"][0]) and np.isfinite(ip["upper"][0])


def test_insufficient_c1_stratum_k_gt_n_falls_back_not_pooled():
    # 'a' has 1 calibration row -> k = ceil(2*0.95) = 2 > 1 at 95%.
    data = _mondrian_data(["a"] + ["b"] * 60, ["a", "b", "b"])
    res = R.compute_fold_conformal(regime="random", fold_id=1, **data)
    c1 = _cells_by_key(res)[("C1", 0.95)]
    ip = res["intervals"][("C1", 0.95)]
    strata = {s["stratum"]: s for s in c1["strata"]}
    assert strata["a"]["n_calibration"] == 1
    assert strata["a"]["fallback_used"] is True
    assert ip["lower"][0] == -np.inf and ip["upper"][0] == np.inf
    # The fallback must not equal a pooled interval.
    pooled_lower = data["yhat_test"][0] - res["intervals"][("A", 0.95)]["q"][0]
    assert not np.isclose(ip["lower"][0], pooled_lower)


def test_pooled_methods_are_unbounded_when_k_exceeds_n():
    data = dict(
        y_cal=np.array([1.0, 2.0]),
        yhat_cal=np.array([0.0, 0.0]),
        yhat_test=np.array([1.0, 1.0]),
        yhat_fit=np.array([0.0, 1.0, 2.0]),
        city_cal=np.array(["a", "a"]),
        city_test=np.array(["a", "a"]),
        group_cal=np.array(["g", "g"]),
        group_test=np.array(["g", "g"]),
    )
    res = R.compute_fold_conformal(regime="random", fold_id=1, **data)
    a = _cells_by_key(res)[("A", 0.95)]
    assert a["unbounded"] is True
    assert a["quantile"] is None
    assert np.all(res["intervals"][("A", 0.95)]["lower"] == -np.inf)
    assert np.all(res["intervals"][("A", 0.95)]["upper"] == np.inf)


# ===========================================================================
# PURE: interval construction, space transformation, B1/B3
# ===========================================================================


def test_b3_reproduces_a_and_b1_normalises():
    data = _synthetic()
    cells = _cells_by_key(R.compute_fold_conformal(regime="random", fold_id=1, **data))
    assert cells[("A", 0.90)]["quantile"] == cells[("B3", 0.90)]["quantile"]
    ip_a = R.compute_fold_conformal(regime="random", fold_id=1, **data)["intervals"][("A", 0.9)]
    ip_b1 = R.compute_fold_conformal(regime="random", fold_id=1, **data)["intervals"][("B1", 0.9)]
    width_a = ip_a["upper"] - ip_a["lower"]
    width_b1 = ip_b1["upper"] - ip_b1["lower"]
    assert np.all(width_a > 0)
    assert not np.allclose(width_a, width_b1)  # B1 width tracks the prediction
    assert cells[("B1", 0.90)]["scale"] == "y_hat"


def test_b1_guard_activations_are_reported():
    data = _synthetic(zero_scale=True)
    res = R.compute_fold_conformal(regime="random", fold_id=1, **data)
    assert res["b1_n_guard_activations"] >= 1
    assert _cells_by_key(res)[("B1", 0.90)]["n_guard_activations"] >= 1


def test_interval_endpoints_are_log_space_then_inr():
    data = _synthetic()
    res = R.compute_fold_conformal(regime="random", fold_id=1, **data)
    ip = res["intervals"][("A", 0.90)]
    # endpoints are in log space; INR conversion is expm1 with a floor at 0
    np.testing.assert_allclose(P.to_rupees(ip["lower"]), np.maximum(np.expm1(ip["lower"]), 0.0))
    np.testing.assert_allclose(P.to_rupees(ip["upper"]), np.maximum(np.expm1(ip["upper"]), 0.0))


def test_bounded_intervals_satisfy_ordering():
    data = _synthetic()
    res = R.compute_fold_conformal(regime="random", fold_id=1, **data)
    for method in ("A", "B1", "B3"):
        ip = res["intervals"][(method, 0.90)]
        assert np.all(ip["lower"] <= ip["upper"])
        assert np.all(ip["lower"] <= data["yhat_test"])
        assert np.all(data["yhat_test"] <= ip["upper"])


# ===========================================================================
# ARTIFACT: reload, determinism, isolation, protected state
# ===========================================================================

_MANIFEST = T.ARTIFACT_ROOT / R.MANIFEST_NAME
_HAVE_ARTIFACTS = _MANIFEST.exists()
requires_artifacts = pytest.mark.skipif(
    not _HAVE_ARTIFACTS, reason="Phase 4 artifacts not generated"
)


@pytest.fixture(scope="module")
def phase4_manifest():
    if not _HAVE_ARTIFACTS:
        pytest.skip("Phase 4 artifacts not generated")
    with open(_MANIFEST, "r", encoding="utf-8") as fh:
        return json.load(fh)


@requires_artifacts
def test_manifest_reports_clean_protected_state(phase4_manifest):
    assert phase4_manifest["protected_state_clean"] is True
    assert phase4_manifest["protected_state"]["clean"] is True


@requires_artifacts
def test_all_six_methods_and_three_levels_are_present(phase4_manifest):
    assert tuple(phase4_manifest["methods"]) == ("A", "B1", "B3", "C1", "C2", "C3")
    assert phase4_manifest["nominal_levels"] == [0.80, 0.90, 0.95]


@requires_artifacts
def test_ten_folds_and_expected_total_intervals(phase4_manifest):
    assert phase4_manifest["n_folds"] == 10
    assert len(phase4_manifest["folds"]) == 10
    total = sum(s["n_intervals"] for s in phase4_manifest["folds"])
    assert total == phase4_manifest["total_intervals"]
    # each test row contributes one interval per (method, level) = 18
    for s in phase4_manifest["folds"]:
        assert s["n_intervals"] == s["n_test"] * 18


@requires_artifacts
def test_artifact_hashes_match_reloaded_files(phase4_manifest):
    for s in phase4_manifest["folds"]:
        assert T.sha256_of(T.ROOT / s["quantiles_file"]) == s["quantiles_sha256"]
        assert T.sha256_of(T.ROOT / s["intervals_file"]) == s["intervals_sha256"]


@requires_artifacts
def test_interval_endpoints_and_ordering_on_reload(phase4_manifest):
    for s in phase4_manifest["folds"]:
        df = pd.read_csv(T.ROOT / s["intervals_file"], low_memory=False)
        finite = np.isfinite(df["lower_log"]) & np.isfinite(df["upper_log"])
        assert bool(
            (
                (df.loc[finite, "lower_log"] <= df.loc[finite, "point_pred_log"])
                & (df.loc[finite, "point_pred_log"] <= df.loc[finite, "upper_log"])
            ).all()
        )
        # INR columns are exactly the inverted log endpoints
        np.testing.assert_allclose(
            df.loc[finite, "lower_inr"].to_numpy(),
            np.maximum(np.expm1(df.loc[finite, "lower_log"].to_numpy()), 0.0),
        )


@requires_artifacts
def test_c3_group_counts_and_k_on_reload(phase4_manifest):
    for s in phase4_manifest["folds"]:
        with open(T.ROOT / s["quantiles_file"], "r", encoding="utf-8") as fh:
            q = json.load(fh)
        c3 = {R.level_tag(c["nominal"]): c for c in q["cells"] if c["method"] == "C3"}
        for nominal in R.NOMINAL_LEVELS:
            cell = c3[R.level_tag(nominal)]
            assert cell["n"] == q["n_calibration_groups"]
            assert cell["k"] == int(np.ceil((q["n_calibration_groups"] + 1) * nominal))


@requires_artifacts
def test_c1_c2_fallback_counts_match_manifest(phase4_manifest):
    for s in phase4_manifest["folds"]:
        df = pd.read_csv(T.ROOT / s["intervals_file"], low_memory=False)
        for method in ("C1", "C2"):
            for nominal in R.NOMINAL_LEVELS:
                tag = R.level_tag(nominal)
                mask = (df["method"] == method) & (df["nominal"] == float(nominal))
                assert int(df.loc[mask, "fallback_used"].sum()) == s["fallback_counts"][method][tag]


@requires_artifacts
def test_calibration_and_test_isolation_from_phase2(phase4_manifest):
    doc = T.load_calibration_folds()
    by_cell = {
        (regime, int(e["outer_fold"])): e
        for regime in B.REGIMES
        for e in doc["regimes"][regime]["folds"]
    }
    for s in phase4_manifest["folds"]:
        entry = by_cell[(s["regime"], int(s["fold"]))]
        assert s["n_calibrate"] == len(entry["calibrate"])
        assert s["n_test"] == len(entry["test"])
        assert s["fold_digests"] == entry["digests"]
        # no test row may appear in the calibration set
        assert not (set(entry["calibrate"]) & set(entry["test"]))


@requires_artifacts
def test_phase4_does_not_touch_frozen_parent_folds():
    assert B.verify_frozen_folds_untouched() == B.FROZEN_FOLDS_SHA256


@requires_artifacts
def test_protected_state_matches_phase3_snapshot():
    after = T.ARTIFACT_ROOT / T.PROTECTED_AFTER_JSON.name
    p4 = T.ARTIFACT_ROOT / R.PROTECTED_AFTER_NAME
    with open(after, "r", encoding="utf-8") as fh:
        before_files = json.load(fh)["files"]
    with open(p4, "r", encoding="utf-8") as fh:
        after_files = json.load(fh)["files"]
    assert before_files == after_files


@requires_artifacts
def test_deterministic_content_digest_recomputes(phase4_manifest):
    files = []
    for s in phase4_manifest["folds"]:
        files.append(T.ROOT / s["quantiles_file"])
        files.append(T.ROOT / s["intervals_file"])
    got = R.deterministic_digest(files, T.ARTIFACT_ROOT)
    assert got == phase4_manifest["deterministic_content_digest"]


@requires_artifacts
def test_repeated_fold_scoring_is_byte_identical(phase4_manifest, tmp_path):
    doc = T.load_calibration_folds()
    manifest_in = R.load_nested_manifest()
    raw, _dup, _inv = P.build_dataset()
    groups = P.add_group_labels(raw)

    regime = "random"
    entry = doc["regimes"][regime]["folds"][0]
    fold_id = int(entry["outer_fold"])
    record = next(
        r for r in manifest_in["models"]
        if r["regime"] == regime and int(r["fold"]) == fold_id
    )
    summary = R.score_fold(
        regime=regime, entry=entry, record=record, raw=raw, groups=groups,
        artifact_root=tmp_path,
    )
    real = next(
        s for s in phase4_manifest["folds"]
        if s["regime"] == regime and int(s["fold"]) == fold_id
    )
    assert summary["intervals_sha256"] == real["intervals_sha256"]
    assert summary["quantiles_sha256"] == real["quantiles_sha256"]


@requires_artifacts
def test_full_phase4_verification_passes():
    report = R.verify_phase4()
    assert report["verified"] is True
    assert report["n_folds"] == 10
    assert report["protected_state_clean"] is True
