"""
Non-live tests for the Experiment 1 visualisation and model-selection
analysis module.

These tests verify that the analysis is faithful to the persisted Experiment 1
record, that its statistics are computed correctly, and that it structurally
cannot train a model. They never train, load a model, regenerate a fold,
touch the chain state or start any node.

    python -m pytest scripts/valuation_v2_1/test_model_comparison_analysis.py -q
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.valuation_v2_1 import model_comparison_analysis as A  # noqa: E402

COMPARISON = ROOT / "artifacts" / "valuation" / "v2_1" / "model_comparison"
FIGURE_DIR = COMPARISON / "figures"

pytestmark = pytest.mark.skipif(
    not A.INPUT_COMPARISON.exists() or not A.INPUT_PER_FOLD.exists(),
    reason="Persisted Experiment 1 comparison artifacts are required.",
)


@pytest.fixture(scope="module")
def tables():
    return A.load_tables()


# ============================================================
# SCOPE: THE MODULE MUST NOT BE ABLE TO TRAIN
# ============================================================

FORBIDDEN_TRAINING_TOKENS = (
    "XGBRegressor",
    "LGBMRegressor",
    "CatBoostRegressor",
    "RandomForestRegressor",
    "sklearn.linear_model",
    "sklearn.ensemble",
    "joblib.load",
    "joblib.dump",
    ".fit(",
    "make_pipeline",
    "generate_folds",
    "KFold",
    "GroupKFold",
)


def test_analysis_module_contains_no_training_capability():
    """The analysis must be structurally incapable of training a model."""
    source = inspect.getsource(A)
    offenders = [
        token for token in FORBIDDEN_TRAINING_TOKENS if token in source
    ]
    assert not offenders, (
        "model_comparison_analysis.py must not reference training or "
        f"fold-generation machinery, found: {offenders}"
    )


def test_analysis_module_does_not_import_protocol_constants():
    """Metric values must come from artifacts, not from protocol constants."""
    source = inspect.getsource(A)
    assert "from scripts.valuation_v2_1 import protocol" not in source
    assert "import protocol" not in source


def test_analysis_module_uses_non_interactive_backend():
    assert A.matplotlib.get_backend().lower() == "agg"


# ============================================================
# INPUT ARTIFACTS
# ============================================================

def test_persisted_inputs_present():
    for path in (
        A.INPUT_COMPARISON,
        A.INPUT_PER_FOLD,
        A.INPUT_PER_FOLD_JSON,
        A.INPUT_SUMMARY,
    ):
        assert path.exists(), f"Missing persisted input: {path}"


def test_table_shapes(tables):
    comparison, per_fold, summary = tables
    assert len(comparison) == len(A.MODELS) * len(A.REGIMES)
    assert len(per_fold) == len(A.MODELS) * len(A.REGIMES) * 5
    assert len(summary) == len(A.MODELS) * len(A.REGIMES) * len(A.METRIC_INFO)
    assert set(comparison["model"]) == set(A.MODELS)
    assert set(comparison["regime"]) == set(A.REGIMES)
    assert set(per_fold["model"]) == set(A.MODELS)
    assert set(per_fold["regime"]) == set(A.REGIMES)


def test_per_fold_has_exactly_five_numbered_folds(tables):
    _, per_fold, _ = tables
    for model in A.MODELS:
        for regime in A.REGIMES:
            folds = sorted(
                per_fold[
                    (per_fold["model"] == model)
                    & (per_fold["regime"] == regime)
                ]["fold"].tolist()
            )
            assert folds == [1, 2, 3, 4, 5]


# ============================================================
# ARTIFACT SELF-CONSISTENCY
# ============================================================

def test_summary_agrees_with_per_fold(tables):
    """The figures cannot disagree with the Experiment 1 record."""
    comparison, per_fold, summary = tables
    agreement = A.verify_artifact_agreement(comparison, per_fold, summary)
    assert agreement["all_agree"] is True
    assert agreement["checks_run"] == (
        len(A.MODELS) * len(A.REGIMES) * len(A.METRIC_INFO)
    )
    assert agreement["max_relative_difference"] < 1e-9


def test_persisted_n_folds_and_ddof_are_five_and_one(tables):
    _, _, summary = tables
    assert set(summary["n_folds"].tolist()) == {5}
    assert set(summary["std_ddof"].tolist()) == {1}


def test_metric_definitions_cover_the_protocol_contract(tables):
    _, per_fold, _ = tables
    for metric in A.METRIC_INFO:
        assert metric in per_fold.columns
    assert len(A.METRIC_INFO) == 8


# ============================================================
# STATISTICS
# ============================================================

def test_welch_difference_matches_closed_form():
    """Two identical samples: delta 0, SE = sqrt(2) * sd/sqrt(n)."""
    result = A.welch_difference(10.0, 2.0, 5, 10.0, 2.0, 5)
    assert result["delta"] == pytest.approx(0.0, abs=1e-12)
    expected_se = np.sqrt(2.0 * (2.0 ** 2) / 5.0)
    assert result["welch_se"] == pytest.approx(expected_se, rel=1e-12)
    # Equal variances with n1 = n2 = 5 give df = n1 + n2 - 2 = 8, which
    # coincides with the pooled t-test df only because the variances match.
    assert result["welch_df"] == pytest.approx(8.0, rel=1e-12)
    assert result["ci95_low"] < 0.0 < result["ci95_high"]


def test_welch_df_falls_below_the_pooled_value_when_variances_differ():
    """Unequal variances must reduce the degrees of freedom."""
    equal = A.welch_difference(1.0, 1.0, 5, 2.0, 1.0, 5)
    unequal = A.welch_difference(1.0, 0.1, 5, 2.0, 1.0, 5)
    assert equal["welch_df"] == pytest.approx(8.0, rel=1e-12)
    assert unequal["welch_df"] < 8.0


def test_welch_difference_intervals_bracket_the_delta():
    result = A.welch_difference(0.33, 0.01, 5, 0.24, 0.05, 5)
    assert result["delta"] == pytest.approx(-0.09, abs=1e-9)
    assert result["ci95_low"] < result["delta"] < result["ci95_high"]
    assert result["welch_df"] > 0


def test_welch_difference_never_uses_paired_semantics():
    """Guard against a paired t-difference being reintroduced.

    The two regimes use different splits, so a paired standard error would
    be the smaller of std/sqrt(n) and would ignore between-regime variance.
    """
    independent = A.welch_difference(1.0, 1.0, 5, 2.0, 1.0, 5)
    paired_like_se = 1.0 / np.sqrt(5.0)
    assert independent["welch_se"] > paired_like_se


def test_relative_change_is_refused_for_non_ratio_scales():
    assert np.isnan(A.relative_change_pct(0.3, 0.2, ratio_scale=False))
    assert A.relative_change_pct(
        100.0, 110.0, ratio_scale=True
    ) == pytest.approx(10.0)
    assert A.relative_change_pct(
        100.0, 90.0, ratio_scale=True
    ) == pytest.approx(-10.0)
    assert np.isnan(A.relative_change_pct(0.0, 1.0, ratio_scale=True))


# ============================================================
# NO COMPOSITE SCORE, NO RANKING
# ============================================================

def test_no_composite_score_or_ranking_is_produced(tables):
    comparison, _, _ = tables
    shift = A.build_shift_all_metrics(comparison)
    assert "composite" not in " ".join(shift.columns).lower()
    assert "score" not in " ".join(shift.columns).lower()
    assert "rank" not in " ".join(shift.columns).lower()
    dominance = A.build_regime_dominance(comparison)
    for column in dominance.columns:
        assert "rank" not in column.lower()
        assert "score" not in column.lower()
        assert "weight" not in column.lower()


def test_ordering_table_reports_positions_not_a_ranking(tables):
    comparison, _, _ = tables
    dominance = A.build_regime_dominance(comparison)
    assert len(dominance) == len(A.REGIMES) * len(A.METRIC_INFO)
    position_columns = [f"position_{i}" for i in range(1, len(A.MODELS) + 1)]
    for column in position_columns:
        assert column in dominance.columns
        assert set(dominance[column]) <= set(
            A.MODEL_LABELS[m] for m in A.MODELS
        )


def test_manifest_declares_no_ranking():
    path = COMPARISON / "analysis_manifest.json"
    if not path.exists():
        pytest.skip("analysis_manifest.json not generated yet")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["composite_score_computed"] is False
    assert manifest["models_ranked"] is False
    assert manifest["models_trained"] == 0
    assert manifest["folds_regenerated"] is False
    assert manifest["protocol_modified"] is False
    assert manifest["dataset_modified"] is False
    assert manifest["anchor_modified"] is False
    assert manifest["chain_touched"] is False
    assert manifest["live_tests_run"] is False


# ============================================================
# TABLES
# ============================================================

def test_thesis_summary_table_shape_and_convention(tables):
    comparison, _, _ = tables
    df, rows = A.build_thesis_summary(comparison)
    assert len(df) == len(A.MODELS)
    assert len(rows) == len(A.MODELS)
    assert len(df.columns) == 11
    assert list(df["Model"]) == [A.MODEL_LABELS[m] for m in A.MODELS]
    for cell in df.iloc[:, 1:].to_numpy().ravel():
        assert "±" in cell, f"mean ± std convention missing in {cell!r}"


def test_generalization_table_shape_and_columns(tables):
    comparison, _, _ = tables
    df, records = A.build_generalization_table(comparison)
    assert len(df) == len(A.MODELS)
    assert len(records) == len(A.MODELS)
    for column in (
        "Random R² (log)",
        "Grouped R² (log)",
        "R² (log) absolute change",
        "Random MedAPE (%)",
        "Grouped MedAPE (%)",
        "MedAPE absolute change",
    ):
        assert column in df.columns
    for record in records:
        # R2 relative change must be refused, not emitted as NaN text.
        assert record["R² (log) relative change"].startswith(
            "not applicable"
        )


def test_generalization_deltas_match_persisted_means(tables):
    """Every reported change must equal grouped mean minus random mean."""
    comparison, _, _ = tables
    r2 = A.metric_block(comparison, "R2_log")
    _, records = A.build_generalization_table(comparison)
    by_label = {A.MODEL_LABELS[m]: m for m in A.MODELS}
    for record in records:
        model = by_label[record["Model"]]
        expected = (
            r2[model]["location_grouped"]["mean"] - r2[model]["random"]["mean"]
        )
        assert record["_r2_delta"] == pytest.approx(expected, rel=1e-12)


def test_shift_table_covers_all_metrics_and_models(tables):
    comparison, _, _ = tables
    shift = A.build_shift_all_metrics(comparison)
    assert len(shift) == len(A.MODELS) * len(A.METRIC_INFO)
    assert set(shift["Metric key"]) == set(A.METRIC_INFO)
    assert set(shift["Model"]) == {A.MODEL_LABELS[m] for m in A.MODELS}
    assert shift["Better direction"].isin(
        ["lower_is_better", "higher_is_better"]
    ).all()
    # R2 relative change must be flagged as not meaningful.
    r2_rows = shift[shift["Metric key"].isin(["R2_INR", "R2_log"])]
    assert not r2_rows["Relative change is meaningful"].any()


def test_discrimination_ratio_matches_its_definition(tables):
    comparison, _, _ = tables
    table = A.build_discrimination_table(comparison)
    assert len(table) == len(A.REGIMES) * len(A.METRIC_INFO)
    for _, row in table.iterrows():
        expected = (
            row["Between-model spread"] / row["Mean within-model fold SD"]
        )
        assert row["Discrimination ratio"] == pytest.approx(
            expected, rel=1e-12
        )
        assert row["Separates models at n=5"] == bool(expected >= 1.0)


def test_shift_absolute_change_equals_difference_of_persisted_means(tables):
    comparison, _, _ = tables
    block = A.metric_block(comparison, "MAE_INR")
    shift = A.build_shift_all_metrics(comparison)
    labels = {A.MODEL_LABELS[m]: m for m in A.MODELS}
    for _, row in shift[shift["Metric key"] == "MAE_INR"].iterrows():
        model = labels[row["Model"]]
        expected = (
            block[model]["location_grouped"]["mean"]
            - block[model]["random"]["mean"]
        )
        assert row["Absolute change"] == pytest.approx(expected, rel=1e-12)


# ============================================================
# FIGURES
# ============================================================

EXPECTED_FIGURES = (
    "01_mae_comparison",
    "02_rmse_comparison",
    "03_r2_comparison",
    "04_medape_comparison",
    "05_r2_log_comparison",
    "06_location_generalization",
    "07_fold_variability",
    "08_training_time",
)


@pytest.mark.parametrize("stem", EXPECTED_FIGURES)
def test_figure_exists_in_png_and_vector(stem):
    png = FIGURE_DIR / f"{stem}.png"
    pdf = FIGURE_DIR / f"{stem}.pdf"
    if not png.exists():
        pytest.skip(f"{png.name} not generated yet")
    assert png.exists(), f"PNG missing for {stem}"
    assert png.stat().st_size > 10_000, f"{png.name} suspiciously small"
    assert pdf.exists(), f"vector PDF missing for {stem}"
    assert pdf.stat().st_size > 5_000


def test_all_figures_written_into_the_declared_directory():
    if not FIGURE_DIR.exists():
        pytest.skip("figure directory not generated yet")
    for path in FIGURE_DIR.iterdir():
        assert path.parent == FIGURE_DIR
        assert path.suffix in {".png", ".pdf"}


def test_training_time_figure_omits_the_untimed_anchor():
    """The anchor was not retrained, so it must not appear in timing."""
    if not A.INPUT_TIMING.exists():
        pytest.skip("no persisted timing artifact")
    timing = __import__("pandas").read_csv(A.INPUT_TIMING)
    assert "XGBoost" not in set(timing["model"])
    for _, row in timing.iterrows():
        assert float(row["total_fit_seconds"]) > 0.0
        assert float(row["total_predict_seconds"]) >= 0.0


def test_stability_table_extrema_are_true_persisted_fold_extrema(tables):
    """Minima and maxima must come from per_fold_metrics.csv, not from a
    multiple of the SD. A mean +/- 2 SD band is a distributional claim, not
    the observed range, and with n=5 it can fall outside the real fold values.
    """
    comparison, per_fold, _ = tables
    df = A.build_stability_table(comparison, per_fold)
    assert len(df) == len(A.MODELS)
    for model in A.MODELS:
        row = df[df["Model"] == A.MODEL_LABELS[model]].iloc[0]
        for regime, regime_label in (
            ("random", "Random"),
            ("location_grouped", "Grouped"),
        ):
            observed = A.per_fold_metric(
                per_fold, model, regime, A.PRIMARY_METRIC
            )
            assert len(observed) == 5
            assert float(row[f"{regime_label} fold min (R² log)"]) == pytest.approx(
                observed.min(), rel=1e-12
            )
            assert float(row[f"{regime_label} fold max (R² log)"]) == pytest.approx(
                observed.max(), rel=1e-12
            )
            assert float(row[f"{regime_label} fold min (R² log)"]) < float(
                row[f"{regime_label} fold max (R² log)"]
            )


def test_stability_table_sd_matches_persisted_summary(tables):
    comparison, _, _ = tables
    df = A.build_stability_table(comparison, A.load_tables()[1])
    block = A.metric_block(comparison, A.PRIMARY_METRIC)
    for model in A.MODELS:
        row = df[df["Model"] == A.MODEL_LABELS[model]].iloc[0]
        assert float(row["Random fold SD (R² log)"]) == pytest.approx(
            block[model]["random"]["std"], rel=1e-12
        )
        assert float(row["Grouped fold SD (R² log)"]) == pytest.approx(
            block[model]["location_grouped"]["std"], rel=1e-12
        )
        assert float(row["SD inflation under location hold-out"]) == pytest.approx(
            block[model]["location_grouped"]["std"]
            / block[model]["random"]["std"],
            rel=1e-12,
        )


def test_stability_table_reports_every_model_with_no_missing_values(tables):
    comparison, per_fold, _ = tables
    df = A.build_stability_table(comparison, per_fold)
    assert list(df["Model"]) == [A.MODEL_LABELS[m] for m in A.MODELS]
    numeric = df.select_dtypes(include="number")
    assert not numeric.isna().any().any()
    assert (df["SD inflation under location hold-out"] > 1.0).all()


# ============================================================
# SAFETY
# ============================================================

def test_figure_and_table_writes_stay_inside_the_comparison_namespace():
    assert FIGURE_DIR == A.BASE / "figures"
    assert A.BASE == (
        ROOT / "artifacts" / "valuation" / "v2_1" / "model_comparison"
    )
    for path in (
        A.OUT_MASTERS,
        A.OUT_GENERALIZATION,
        A.OUT_SHIFT_ALL,
        A.OUT_RESULTS_JSON,
        A.OUT_MANIFEST,
    ):
        assert path.parent == A.BASE


def test_anchor_and_historical_artifacts_are_not_write_targets():
    source = inspect.getsource(A)
    for forbidden in (
        "xgb_anchor_metrics_v2_1.json",
        "cv_folds_v2_1.json",
        "valuation_v2_metrics.csv",
        ".chain",
    ):
        assert forbidden not in source, (
            f"{forbidden} must not be referenced by the analysis script"
        )
