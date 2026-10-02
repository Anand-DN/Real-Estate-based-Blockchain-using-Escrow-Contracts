"""Phase 3 tests for the nested CatBoost training module.

These tests DO build the real dataset and DO read the frozen V2.1 fold contract
and the Phase 2 calibration-fold artifact, all read-only. The one training test
fits on a small subset written to a pytest tmp_path, so no model is ever
written into the real Experiment 2 namespace by the test suite.

The full 10-model training run is exercised by
``python -m scripts.experiment_2.train_nested_models`` followed by
``--verify-only``; here we test the mechanics and the frozen contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from scripts.experiment_2 import train_nested_models as T
from scripts.valuation_v2_1 import protocol as P


@pytest.fixture(scope="module")
def dataset():
    raw, _dup, _inv = P.build_dataset()
    return raw, P.add_group_labels(raw)


@pytest.fixture(scope="module")
def small():
    raw, _dup, _inv = P.build_dataset()
    df = raw.iloc[:600].reset_index(drop=True)
    return df, P.add_group_labels(df)


def _entry(fit, cal, test, fold=1):
    return {
        "outer_fold": fold,
        "fit": sorted(int(i) for i in fit),
        "calibrate": sorted(int(i) for i in cal),
        "test": sorted(int(i) for i in test),
        "digests": {"fit": "f", "calibrate": "c", "test": "t"},
    }


# ---------------------------------------------------------------------------
# Frozen configuration
# ---------------------------------------------------------------------------


def test_catboost_config_is_the_frozen_experiment_1_configuration():
    assert T.CATBOOST_CONFIG == {
        "iterations": 700,
        "learning_rate": 0.04,
        "depth": 7,
        "l2_leaf_reg": 2.0,
        "random_seed": 42,
        "loss_function": "RMSE",
        "verbose": False,
        "allow_writing_files": False,
    }


def test_build_estimator_matches_the_frozen_configuration():
    model = T.build_estimator()
    params = model.get_params()
    for key, value in T.CATBOOST_CONFIG.items():
        assert params[key] == value


def test_expected_model_count_is_ten():
    assert T.EXPECTED_MODEL_COUNT == 10
    assert T.REGIMES == ("random", "location_grouped")


def test_feature_contract_records_v2_1():
    contract = T.feature_contract()
    assert contract["protocol"] == "V2.1 (frozen, read-only)"
    assert contract["final_feature_count"] == P.FINAL_FEATURE_COUNT
    assert contract["input_features"] == P.INPUT_FEATURES
    assert contract["training_space"] == "log1p(price)"


def test_environment_record_includes_catboost():
    record = T.environment_record()
    assert "catboost" in record
    assert "packages" in record


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------


def test_calibration_fold_hash_constant_matches_the_artifact():
    assert T.sha256_of(T.B.CALIBRATION_FOLDS_JSON) == T.CALIBRATION_FOLDS_SHA256


def test_load_calibration_folds_returns_five_folds_per_regime():
    doc = T.load_calibration_folds()
    for regime in T.REGIMES:
        assert len(doc["regimes"][regime]["folds"]) == 5


def test_load_calibration_folds_rejects_a_changed_artifact(tmp_path):
    fake = tmp_path / "calibration_folds_v2_2.json"
    fake.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="has changed"):
        T.load_calibration_folds(fake)


# ---------------------------------------------------------------------------
# Disjointness enforcement
# ---------------------------------------------------------------------------


def test_assert_fold_disjoint_accepts_real_folds(dataset):
    raw, groups = dataset
    doc = T.load_calibration_folds()
    for regime in T.REGIMES:
        for entry in doc["regimes"][regime]["folds"]:
            T.assert_fold_disjoint(entry, regime, groups)


def test_assert_fold_disjoint_rejects_row_overlap(small):
    df, groups = small
    entry = _entry(range(0, 400), range(350, 500), range(500, 600))
    with pytest.raises(RuntimeError, match="T'_k intersect C_k"):
        T.assert_fold_disjoint(entry, "random", groups)


def test_assert_fold_disjoint_rejects_test_overlap(small):
    df, groups = small
    entry = _entry(range(0, 400), range(400, 500), range(450, 600))
    with pytest.raises(RuntimeError, match="C_k intersect test_k"):
        T.assert_fold_disjoint(entry, "random", groups)


def test_assert_fold_disjoint_rejects_group_leak_without_row_overlap():
    groups = pd.Series(
        ["g0"] * 150 + ["g1"] * 150 + ["g2"] * 150 + ["g3"] * 150
    )
    fit = list(range(0, 100)) + list(range(150, 250))  # g0, g1
    cal = list(range(100, 150)) + list(range(300, 350))  # g0, g2
    test = list(range(450, 550))  # g3
    entry = _entry(fit, cal, test)
    with pytest.raises(RuntimeError, match="localities cross"):
        T.assert_fold_disjoint(entry, "location_grouped", groups)


# ---------------------------------------------------------------------------
# Single-fold training mechanics
# ---------------------------------------------------------------------------


def test_train_fold_persists_a_reloadable_model(small, tmp_path):
    df, groups = small
    entry = _entry(range(0, 400), range(400, 500), range(500, 600))
    record = T.train_fold("random", entry, df, groups, model_root=tmp_path)

    model_path = Path(record["model_file"])
    assert model_path.exists()
    assert T.sha256_of(model_path) == record["model_sha256"]
    assert record["nested_training_rows"] == 400
    assert record["calibration_rows"] == 100
    assert record["test_rows"] == 100

    pipe = joblib.load(model_path)
    x = df[P.INPUT_FEATURES].iloc[entry["test"]]
    preds = np.asarray(pipe.predict(x), dtype=float)
    assert preds.shape == (100,)
    assert np.all(np.isfinite(preds))
    assert record["reload_prediction_probe"]["exact_match"] is True


def test_train_fold_records_the_full_provenance(small, tmp_path):
    df, groups = small
    entry = _entry(range(0, 400), range(400, 500), range(500, 600), fold=2)
    record = T.train_fold("random", entry, df, groups, model_root=tmp_path)
    for key in (
        "regime",
        "fold",
        "nested_training_rows",
        "calibration_rows",
        "test_rows",
        "model_config",
        "feature_contract",
        "calibration_folds",
        "model_sha256",
    ):
        assert key in record
    assert record["regime"] == "random"
    assert record["fold"] == 2
    assert record["trained_from_T_prime_only"] is True
    assert record["no_calibration_rows_in_fit"] is True
    assert record["no_test_rows_in_fit"] is True
    assert record["fit_identity_proof"]["location_counts_match_T_prime"] is True
    assert record["calibration_folds"]["parent_folds_sha256"] == T.B.FROZEN_FOLDS_SHA256


def test_train_fold_proves_fitted_statistics_belong_to_T_prime(small, tmp_path):
    df, groups = small
    entry = _entry(range(0, 400), range(400, 500), range(500, 600))
    record = T.train_fold("random", entry, df, groups, model_root=tmp_path)
    model_path = Path(record["model_file"])
    pipe = joblib.load(model_path)
    freq = pipe.named_steps["frequency"]
    expected = df.iloc[entry["fit"]]["location"].value_counts()
    assert int(freq.n_fit_rows_) == 400
    assert (freq.location_counts_.sort_index() == expected.sort_index()).all()


# ---------------------------------------------------------------------------
# Write guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "candidate",
    [
        "models/valuation/v2_1/model_comparison/CatBoost/random_fold1.joblib",
        "artifacts/valuation/v2_1/cv_folds_v2_1.json",
        "backend/main.py",
        ".chain/state.json",
    ],
)
def test_write_guard_refuses_protected_paths(candidate):
    with pytest.raises(RuntimeError, match="Refusing to write"):
        T.assert_safe_write_target(P.ROOT / candidate)


def test_write_guard_allows_the_experiment_2_model_namespace():
    T.assert_safe_write_target(
        P.ROOT / "models" / "valuation" / "v2_2" / "conformal" / "random" / "x.joblib"
    )
