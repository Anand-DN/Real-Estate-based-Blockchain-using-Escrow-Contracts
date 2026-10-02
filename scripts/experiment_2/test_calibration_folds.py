"""Phase 2 tests for the nested calibration fold derivation.

These tests DO read the frozen V2.1 fold contract, read-only, and DO build the
real dataset, because the invariants are properties of the real fold structure.
They write NOTHING to disk: write_calibration_folds is exercised against a
pytest tmp_path, and the real output path is verified only for refusal.

No model is trained anywhere in this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.experiment_2 import build_calibration_folds as B
from scripts.experiment_2 import conformal as C
from scripts.valuation_v2_1 import protocol as P


@pytest.fixture(scope="module")
def dataset():
    raw, _dup, _inv = P.build_dataset()
    return raw, P.add_group_labels(raw)


@pytest.fixture(scope="module")
def derived(dataset):
    raw, groups = dataset
    return B.build_calibration_folds(), groups


# ---------------------------------------------------------------------------
# Phase 0 constants are contract
# ---------------------------------------------------------------------------


def test_cal_frac_is_the_preregistered_value():
    assert B.CAL_FRAC == 0.20


def test_random_state_is_the_preregistered_value():
    assert B.RANDOM_STATE == 42


def test_frozen_folds_hash_matches_the_recorded_baseline():
    assert B.sha256_of(P.FOLDS_PATH) == B.FROZEN_FOLDS_SHA256


def test_verify_frozen_folds_untouched_passes_on_the_real_file():
    assert B.verify_frozen_folds_untouched() == B.FROZEN_FOLDS_SHA256


def test_build_refuses_a_different_cal_frac():
    for bad in (0.10, 0.25, 0.15):
        with pytest.raises(ValueError, match="pre-registered"):
            B.build_calibration_folds(cal_frac=bad)


def test_build_refuses_a_different_random_state():
    with pytest.raises(ValueError, match="random_state"):
        B.build_calibration_folds(random_state=7)


# ---------------------------------------------------------------------------
# Protected paths must be refused (L15)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "candidate",
    [
        "artifacts/valuation/v2_1/cv_folds_v2_1.json",
        "artifacts/valuation/v2_1/model_comparison/per_fold_metrics.json",
        "models/valuation/v2_1/model_comparison/CatBoost/random_fold1.joblib",
        "models/valuation/valuation_v2_xgboost_random.joblib",
        "artifacts/valuation/valuation_v2_metrics.csv",
        "scripts/valuation_v2_1/protocol.py",
        "backend/main.py",
        ".chain/state.json",
    ],
)
def test_write_guard_refuses_every_protected_path(candidate):
    with pytest.raises(RuntimeError, match="Refusing to write"):
        B.assert_safe_write_target(P.ROOT / candidate)


def test_write_guard_allows_the_experiment_2_namespace():
    B.assert_safe_write_target(P.ROOT / "artifacts" / "valuation" / "v2_2" / "conformal" / "x.json")


def test_write_guard_allows_the_experiment_2_model_namespace():
    B.assert_safe_write_target(
        P.ROOT / "models" / "valuation" / "v2_2" / "conformal" / "CatBoost" / "random_fold1.joblib"
    )


# ---------------------------------------------------------------------------
# Inner split semantics
# ---------------------------------------------------------------------------


def test_derive_inner_split_is_deterministic(dataset):
    raw, groups = dataset
    frozen = P.load_folds()
    train = frozen["random"].train[0]
    a = B.derive_inner_split(train, regime="random", groups=groups)
    b = B.derive_inner_split(train, regime="random", groups=groups)
    assert a == b


def test_derive_inner_split_is_a_partition_of_train(dataset):
    raw, groups = dataset
    frozen = P.load_folds()
    train = frozen["random"].train[0]
    fit, cal = B.derive_inner_split(train, regime="random", groups=groups)
    assert set(fit).isdisjoint(cal)
    assert set(fit) | set(cal) == set(train)
    assert len(fit) + len(cal) == len(train)


def test_derive_inner_split_realises_roughly_the_preregistered_fraction(dataset):
    raw, groups = dataset
    frozen = P.load_folds()
    for regime in B.REGIMES:
        for train in frozen[regime].train:
            fit, cal = B.derive_inner_split(train, regime=regime, groups=groups)
            frac = len(cal) / len(train)
            # GroupKFold balances by group count, not row count, so allow slack.
            assert 0.14 <= frac <= 0.26, f"{regime}: realised {frac:.4f}"


def test_grouped_inner_split_has_zero_group_overlap(dataset):
    raw, groups = dataset
    frozen = P.load_folds()
    train = frozen["location_grouped"].train[0]
    fit, cal = B.derive_inner_split(train, regime="location_grouped", groups=groups)
    g_fit = set(groups.iloc[fit])
    g_cal = set(groups.iloc[cal])
    assert not (g_fit & g_cal)


def test_grouped_inner_split_matches_the_outer_group_semantics(dataset):
    raw, groups = dataset
    frozen = P.load_folds()
    train = frozen["location_grouped"].train[0]
    fit, cal = B.derive_inner_split(train, regime="location_grouped", groups=groups)
    # Calibration rows must be from localities the fit set never saw, which is
    # what makes the grouped result interpretable as shift, not regime mismatch.
    g_fit = set(groups.iloc[fit])
    assert all(str(loc) not in g_fit for loc in groups.iloc[cal])


def test_derive_inner_split_rejects_unknown_regime(dataset):
    raw, groups = dataset
    with pytest.raises(ValueError, match="unknown regime"):
        B.derive_inner_split([0, 1, 2, 3, 4], regime="nonsense")


def test_derive_inner_split_requires_groups_for_grouped_regime(dataset):
    raw, groups = dataset
    with pytest.raises(ValueError, match="requires the group Series"):
        B.derive_inner_split([0, 1, 2, 3, 4], regime="location_grouped")


def test_derive_inner_split_rejects_unsorted_train_indices():
    with pytest.raises(ValueError, match="sorted ascending"):
        B.derive_inner_split([3, 1, 2, 4, 5], regime="random")


def test_derive_inner_split_rejects_empty_train():
    with pytest.raises(ValueError, match="non-empty"):
        B.derive_inner_split([], regime="random")


# ---------------------------------------------------------------------------
# Full derived structure: the invariants from the Phase 0 approval
# ---------------------------------------------------------------------------


def test_derived_structure_covers_all_ten_cells(derived):
    doc, _groups = derived
    assert set(doc["regimes"]) == {"random", "location_grouped"}
    for regime in B.REGIMES:
        assert len(doc["regimes"][regime]["folds"]) == 5


def test_derived_does_not_modify_the_frozen_contract(derived):
    doc, _groups = derived
    assert doc["folds_regenerated"] is False
    assert doc["frozen_fold_contract_modified"] is False
    assert doc["parent_folds_sha256"] == B.FROZEN_FOLDS_SHA256


@pytest.mark.parametrize("regime", B.REGIMES)
def test_every_fold_satisfies_the_row_invariants(derived, regime):
    """The five row-level requirements from the Phase 0 approval §7."""
    doc, _groups = derived
    frozen = P.load_folds()
    for entry, (tr, va) in zip(
        doc["regimes"][regime]["folds"], zip(frozen[regime].train, frozen[regime].valid)
    ):
        fit, cal, test = set(entry["fit"]), set(entry["calibrate"]), set(entry["test"])
        assert fit.isdisjoint(cal), "T' ∩ C must be empty"
        assert fit.isdisjoint(test), "T' ∩ test must be empty"
        assert cal.isdisjoint(test), "C ∩ test must be empty"
        assert fit | cal == set(tr), "T' ∪ C must equal frozen train_k"
        assert entry["test"] == list(va), "test must equal frozen valid_k exactly"


@pytest.mark.parametrize("fold", [1, 2, 3, 4, 5])
def test_every_grouped_fold_satisfies_the_group_invariants(derived, fold):
    """The three group-level requirements from the Phase 0 approval §7."""
    doc, groups = derived
    entry = doc["regimes"]["location_grouped"]["folds"][fold - 1]
    g_fit = set(groups.iloc[entry["fit"]].astype(str))
    g_cal = set(groups.iloc[entry["calibrate"]].astype(str))
    g_test = set(groups.iloc[entry["test"]].astype(str))
    assert g_fit.isdisjoint(g_cal), "groups(T') ∩ groups(C) must be empty"
    assert g_fit.isdisjoint(g_test), "groups(T') ∩ groups(test) must be empty"
    assert g_cal.isdisjoint(g_test), "groups(C) ∩ groups(test) must be empty"


def test_grouped_test_localities_are_entirely_unseen_by_fit_and_calibration(derived):
    """Every grouped test row must be from a locality absent from BOTH T' and C.

    This is the property that makes the location-grouped regime a genuine
    unseen-locality test rather than a partial one.
    """
    doc, groups = derived
    for entry in doc["regimes"]["location_grouped"]["folds"]:
        seen = set(groups.iloc[entry["fit"]].astype(str)) | set(
            groups.iloc[entry["calibrate"]].astype(str)
        )
        test_locs = set(groups.iloc[entry["test"]].astype(str))
        assert test_locs.isdisjoint(seen)


def test_random_regime_test_rows_mostly_come_from_seen_localities(dataset, derived):
    """Contrast check: the random regime is NOT a locality test.

    This guards against the two regimes silently becoming the same experiment.
    """
    doc, _groups = derived
    raw, groups = dataset
    for entry in doc["regimes"]["random"]["folds"]:
        seen = set(groups.iloc[entry["fit"]].astype(str)) | set(
            groups.iloc[entry["calibrate"]].astype(str)
        )
        frac = float(groups.iloc[entry["test"]].astype(str).isin(seen).mean())
        assert frac > 0.9, f"random regime test seen-fraction only {frac:.3f}"


def test_derived_calibration_sizes_are_substantial(derived):
    """Calibration sets must be large enough for all three nominal levels."""
    doc, _groups = derived
    est = B.quantile_estimability(doc)
    assert est["k_le_n"].all(), "some cell cannot support an exact quantile"
    assert est["n_calibrate"].min() > 1000


def test_every_nominal_level_is_estimable_in_every_cell(derived):
    doc, _groups = derived
    est = B.quantile_estimability(doc, levels=C.NOMINAL_LEVELS)
    assert len(est) == 2 * 5 * 3
    assert est["k_le_n"].all()


def test_grouped_calibration_group_counts_are_reported(dataset, derived):
    """§15.5: group-level effective n must be visible, not assumed."""
    raw, groups = dataset
    doc, _ = derived
    summary = B.summarize(doc, groups)
    grouped = summary[summary["regime"] == "location_grouped"]
    assert (grouped["groups_calibrate"] > 0).all()


def test_grouped_inner_split_balances_group_count_not_row_count(dataset, derived):
    """Documents a real property of GroupKFold that affects interpretation.

    GroupKFold balances the NUMBER of groups across inner folds, not the number
    of rows. Because group sizes span 1..685 (global mean 15.87, median 3),
    the realised row fraction differs across outer folds (0.16..0.24) even
    though cal_frac is 0.20 everywhere, and each calibration set contains the
    same 287 groups.

    Consequence: in the grouped regime, cal_frac is a TARGET for group count,
    not an achieved row fraction. This is inherent to group-disjoint
    calibration, not a bug, and it is asserted here so it cannot drift.
    """
    raw, groups = dataset
    doc, _ = derived
    grouped = B.summarize(doc, groups)
    grouped = grouped[grouped["regime"] == "location_grouped"]

    assert set(grouped["groups_calibrate"]) == {287.0}
    frac = grouped["cal_frac_realised"]
    assert frac.min() >= 0.15 and frac.max() <= 0.25, (
        f"realised cal_frac drifted outside [0.15, 0.25]: {frac.tolist()}"
    )

    # The deviation is a consequence of group-size imbalance, not of randomness.
    global_mean_group = groups.value_counts().mean()
    assert global_mean_group > 10.0
    ratio = float((grouped["n_calibrate"] / grouped["groups_calibrate"]).mean())
    assert ratio > 8.0, (
        "rows-per-group in calibration should exceed the global mean group size "
        "because whole large groups land in single folds"
    )


def test_group_level_quantile_is_estimable_but_thin(dataset, derived):
    """C3 power check: ~287 calibration groups, k95 = 274."""
    raw, groups = dataset
    doc, _ = derived
    for entry in doc["regimes"]["location_grouped"]["folds"]:
        n_groups = int(groups.iloc[entry["calibrate"]].nunique())
        for nominal in C.NOMINAL_LEVELS:
            k = int(np.ceil((n_groups + 1) * nominal))
            assert k <= n_groups, "group-level quantile not estimable"
        # Secondary by construction: the group count is ~6x smaller than rows.
        assert n_groups * 5 < entry["n_calibrate"]


def _corrupt(doc, regime, fold, mutate):
    """Deep-copy the derived doc and apply a mutation to one fold entry."""
    clone = json.loads(json.dumps(doc))
    entry = clone["regimes"][regime]["folds"][fold - 1]
    mutate(entry)
    return clone


def test_verifier_detects_a_calibration_row_moved_from_the_test_fold(derived):
    """Replacing a calibration row with a test row breaks C ∩ test = {} (L1).

    L1 is checked before L4, so L1 is what fires. Both invariants are violated;
    the verifier reports the earliest.
    """
    doc, groups = derived

    def mutate(entry):
        entry["calibrate"] = entry["calibrate"][:-1] + [entry["test"][0]]

    corrupted = _corrupt(doc, "random", 1, mutate)
    frozen = P.load_folds()
    with pytest.raises(RuntimeError, match="L1"):
        B.verify_all_invariants(corrupted, frozen, groups)


def test_verifier_detects_a_test_row_replaced_by_a_fit_row(derived):
    """A test row swapped for a fit row breaks both L2 and L5.

    L2 (fit ∩ test) is checked first, so L2 is what fires. Both are violations;
    the verifier reports the earliest.
    """
    doc, groups = derived

    def mutate(entry):
        entry["test"] = list(entry["test"])[:-1] + [entry["fit"][0]]

    corrupted = _corrupt(doc, "random", 1, mutate)
    frozen = P.load_folds()
    with pytest.raises(RuntimeError, match="L2"):
        B.verify_all_invariants(corrupted, frozen, groups)


def test_verifier_detects_a_dropped_fit_row(derived):
    """Silently discarding a fit row must fail the completeness check L4."""
    doc, groups = derived

    def mutate(entry):
        entry["fit"] = entry["fit"][:-1]

    corrupted = _corrupt(doc, "random", 1, mutate)
    frozen = P.load_folds()
    with pytest.raises(RuntimeError, match="L4"):
        B.verify_all_invariants(corrupted, frozen, groups)


def test_verifier_detects_a_test_reordering_that_breaks_bit_identity(derived):
    """Reordering test indices keeps the SET valid but breaks bit-identity (L5)."""
    doc, groups = derived

    def mutate(entry):
        entry["test"] = list(reversed(entry["test"]))

    corrupted = _corrupt(doc, "random", 1, mutate)
    frozen = P.load_folds()
    with pytest.raises(RuntimeError, match="L5"):
        B.verify_all_invariants(corrupted, frozen, groups)


def test_verifier_detects_a_calibration_row_moved_out_of_the_fit_set(derived):
    """Duplicating a fit row into calibrate breaks T' ∩ C = {} (L3)."""
    doc, groups = derived

    def mutate(entry):
        entry["calibrate"] = entry["calibrate"] + [entry["fit"][0]]

    corrupted = _corrupt(doc, "location_grouped", 1, mutate)
    frozen = P.load_folds()
    with pytest.raises(RuntimeError, match="L3"):
        B.verify_all_invariants(corrupted, frozen, groups)


def test_verifier_detects_a_group_leak_that_preserves_the_row_partition(derived):
    """The dangerous case: rows stay disjoint, but a locality spans fit and cal.

    This is what a naive row-wise split of the grouped regime would produce. A
    single SWAP of one fit row with one calibration row keeps every row-level
    invariant true: the sets stay disjoint, the union still equals train_k, and
    the test set is untouched. Only the group checks can catch it, which is
    exactly why L9 exists and why it is tested separately from L3.
    """
    doc, groups = derived
    mutated = _corrupt(doc, "location_grouped", 1, lambda e: None)
    me = mutated["regimes"]["location_grouped"]["folds"][0]

    victim = me["fit"][0]  # a locality that remains present in fit after removal
    donor = me["calibrate"][0]
    assert str(groups.iloc[victim]) in set(groups.iloc[me["fit"][1:]].astype(str))

    me["fit"] = sorted([i for i in me["fit"] if i != victim] + [donor])
    me["calibrate"] = sorted([i for i in me["calibrate"] if i != donor] + [victim])

    frozen = P.load_folds()
    with pytest.raises(RuntimeError, match="L9"):
        B.verify_all_invariants(mutated, frozen, groups)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_derived_document_is_json_serialisable(derived):
    doc, _groups = derived
    text = json.dumps(doc)
    assert len(text) > 1000
    json.loads(text)  # round-trips


def test_derived_document_records_provenance(derived):
    doc, _groups = derived
    assert doc["cal_frac"] == 0.20
    assert doc["random_state"] == 42
    assert doc["protocol"] == "V2.1 (frozen, read-only)"
    assert doc["parent_folds"].endswith("cv_folds_v2_1.json")
    for regime in B.REGIMES:
        for entry in doc["regimes"][regime]["folds"]:
            for key in ("frozen_train", "fit", "calibrate", "test"):
                assert len(entry["digests"][key]) == 64


def test_write_to_tmp_path_works_and_refuses_protected_paths(derived, tmp_path):
    doc, groups = derived
    target = tmp_path / "nested" / "calibration_folds.json"
    written = B.write_calibration_folds(doc, target)
    assert written.exists()
    loaded = json.loads(written.read_text(encoding="utf-8"))
    frozen = P.load_folds()
    B.verify_all_invariants(loaded, frozen, groups)

    with pytest.raises(RuntimeError, match="Refusing to write"):
        B.write_calibration_folds(doc, P.FOLDS_PATH)


def test_frozen_folds_untouched_after_derivation(derived):
    assert B.sha256_of(P.FOLDS_PATH) == B.FROZEN_FOLDS_SHA256


# ---------------------------------------------------------------------------
# The derived structure must be usable by the Phase 1 library
# ---------------------------------------------------------------------------


def test_derived_calibration_scores_can_be_split_conformal_scored(derived):
    """Interface check between Phase 2 output and the Phase 1 scorer.

    Uses synthetic scores; no model is trained. Confirms the index sets feed
    conformal_scores and conformal_quantile without a shape or contract error.
    """
    doc, _groups = derived
    entry = doc["regimes"]["random"]["folds"][0]
    rng = np.random.default_rng(0)
    y_cal = rng.normal(15.0, 0.4, size=len(entry["calibrate"]))
    p_cal = y_cal + rng.normal(0.0, 0.2, size=len(entry["calibrate"]))
    scores, info = C.conformal_scores("A", y_cal, p_cal)
    q = C.conformal_quantile(scores, 1.0 - 0.90)
    assert q["unbounded"] is False
    assert info["width_is_constant"] is True

    y_test = rng.normal(15.0, 0.4, size=len(entry["test"]))
    p_test = y_test + rng.normal(0.0, 0.2, size=len(entry["test"]))
    lo, up, _ = C.build_intervals("A", p_test, q["qhat"])
    m = C.coverage_metrics(y_test, lo, up, 0.90)
    assert m["n_evaluated"] == len(entry["test"])