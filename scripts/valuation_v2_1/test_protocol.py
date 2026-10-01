"""
MILLOW RESEARCH EXPERIMENT 1
Non-live tests for the V2.1 baseline protocol.

These tests are offline. They read the processed CSV, build synthetic
frames in memory, and never touch the blockchain, never train a
persisted model, and never write to models/ or artifacts/valuation/.

Run from the project root:

    python -m pytest scripts/valuation_v2_1/test_protocol.py -v

Test coverage maps one-to-one onto the nine V2.1 requirements:

     1 train-only frequency calculation
     2 unseen location fallback
     3 duplicate removal
     4 amenity feature construction
     5 preprocessing fit isolation
     6 fold reproducibility
     7 grouped-fold zero group overlap
     8 metric calculations
     9 prediction-space conversion
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.valuation_v2_1 import protocol as P  # noqa: E402


# ============================================================
# FIXTURES
# ============================================================

def _synthetic_frame(
    n_rows: int = 600,
    seed: int = 7,
    with_duplicates: bool = True,
) -> pd.DataFrame:
    """A small in-memory frame with the real schema.

    Used instead of the full 29,135-row dataset so the tests stay fast and
    deterministic. Duplicates are injected on purpose, because the duplicate
    policy needs a known ground truth to assert against.
    """
    rng = np.random.default_rng(seed)

    cities = ["mumbai", "delhi", "bangalore"]
    locations = [
        "andheri west", "bandra west", "connaught place",
        "koramangala", "whitefield", "saket",
    ]

    records = []
    for i in range(n_rows):
        records.append(
            {
                "mreid_id": f"MREID_{i:07d}",
                "price": float(rng.integers(2_000_000, 40_000_000)),
                "area": float(rng.integers(200, 3000)),
                "location": locations[i % len(locations)],
                "no_of_bedrooms": int(rng.integers(1, 5)),
                "resale": int(rng.integers(0, 2)),
                "source_city": cities[i % len(cities)],
                "source_file": f"{cities[i % len(cities)].title()}.csv",
            }
        )

    frame = pd.DataFrame(records)

    for c in P.AMENITIES:
        frame[c] = rng.choice([0, 1, 9], size=n_rows, p=[0.3, 0.2, 0.5])

    if with_duplicates:
        # Inject exact duplicates on the protocol key.
        dupe_source = frame.iloc[:20].copy()
        frame = pd.concat([frame, dupe_source], ignore_index=True)

    return frame


@pytest.fixture(scope="module")
def raw_frame() -> pd.DataFrame:
    return _synthetic_frame()


@pytest.fixture(scope="module")
def real_dataset():
    df, report, invalid = P.build_dataset(P.load_raw())
    return df, report, invalid


# ============================================================
# 1. TRAIN-ONLY FREQUENCY CALCULATION
# ============================================================

def test_frequency_counts_use_training_rows_only(raw_frame):
    """A validation-only location must not inflate the count it receives.

    The location 'ghost town' appears 40 times in the validation partition
    and zero times in training. If fit() saw validation rows, its mapped
    frequency would be 40. It must be the documented fallback instead.
    """
    train = raw_frame.iloc[:400].copy()
    valid = raw_frame.iloc[400:].copy()

    valid.loc[:, "location"] = "ghost town"

    transformer = P.TrainOnlyFrequencyFeatures()
    transformer.fit(train)

    # The transformer only ever saw training rows.
    assert transformer.n_fit_rows_ == 400

    mapped = transformer.transform(valid)

    assert (mapped["location_frequency"] == P.FALLBACK_FREQUENCY).all()
    assert (mapped["log_location_frequency"] == 0.0).all()

    # A location genuinely present in training maps to its TRAIN count, not
    # to the combined train+validation count.
    present = valid.copy()
    present.loc[:, "location"] = "andheri west"

    mapped_present = transformer.transform(present)
    train_count = int((train["location"] == "andheri west").sum())
    combined_count = int((raw_frame["location"] == "andheri west").sum())

    assert (mapped_present["location_frequency"] == train_count).all()
    assert train_count < combined_count, "fixture must actually differ"


def test_frequency_counts_are_immune_to_validation_row_count(raw_frame):
    """Doubling validation rows must not change any mapped frequency.

    This is the direct regression test for the V2 defect, where the count
    grew with the size of the evaluation partition.
    """
    train = raw_frame.iloc[:400]
    transformer = P.TrainOnlyFrequencyFeatures().fit(train)

    probe = train.head(50).copy()

    small = transformer.transform(probe)["location_frequency"].to_numpy()
    large = transformer.transform(
        pd.concat([probe] * 4, ignore_index=True)
    )["location_frequency"].to_numpy()

    # pd.concat tiles the block, so the expected layout is a tile, not an
    # interleave.
    assert np.array_equal(np.tile(small, 4), large)


def test_city_frequency_is_frozen_out(raw_frame):
    """Frozen decision: city_frequency is dropped as redundant.

    It is a deterministic 1:1 restatement of source_city, which is already a
    one-hot categorical feature. The transformer must not produce it, and it
    must not appear anywhere in the frozen feature set.
    """
    assert "city_frequency" not in P.FREQUENCY_FEATURES
    assert "city_frequency" not in P.FINAL_FEATURES
    assert "city_frequency" not in P.NUMERIC_FEATURES
    assert "city_frequency" not in P.INPUT_FEATURES

    transformer = P.TrainOnlyFrequencyFeatures().fit(raw_frame.iloc[:400])
    mapped = transformer.transform(raw_frame.head(10))

    assert "city_frequency" not in mapped.columns

    # The removal is recorded with its justification.
    assert "city_frequency" in P.REDUNDANT_FEATURES_REMOVED


def test_frequency_counts_are_train_only(raw_frame):
    """Fitted on training rows only: validation rows cannot inflate a count."""
    train = raw_frame.iloc[:400]
    valid = raw_frame.iloc[400:].copy()

    # A location that appears only in the validation partition.
    valid.loc[:, "location"] = "ghost town"
    n_valid = len(valid)

    transformer = P.TrainOnlyFrequencyFeatures().fit(train)
    ghost_count = transformer.transform(valid)["location_frequency"].iloc[0]

    # A count computed on train+valid would return n_valid; train-only must
    # return the fallback instead.
    assert ghost_count == P.FALLBACK_FREQUENCY
    assert ghost_count != float(n_valid)


def test_frequency_transformer_requires_dataframe():
    transformer = P.TrainOnlyFrequencyFeatures()
    with pytest.raises(TypeError):
        transformer.fit(np.zeros((4, 3)))


def test_frequency_transformer_requires_fit_before_transform():
    transformer = P.TrainOnlyFrequencyFeatures()
    with pytest.raises(RuntimeError):
        transformer.transform(_synthetic_frame(10))


# ============================================================
# 2. UNSEEN LOCATION FALLBACK
# ============================================================

def test_unseen_location_gets_documented_fallback(raw_frame):
    """Fallback is exactly 0.0, and log1p of it is exactly 0.0."""
    train = raw_frame.iloc[:400]
    transformer = P.TrainOnlyFrequencyFeatures().fit(train)

    unseen = train.head(5).copy()
    unseen.loc[:, "location"] = "a place that never appears in training"

    mapped = transformer.transform(unseen)

    assert (mapped["location_frequency"] == 0.0).all()
    assert (mapped["log_location_frequency"] == 0.0).all()
    assert P.FALLBACK_FREQUENCY == 0.0


def test_unseen_location_fallback_does_not_depend_on_city(raw_frame):
    """The fallback is keyed on location alone.

    A row whose city is unseen in training but whose location IS seen still
    receives the real location count. The frequency feature is per-category
    and location-driven, so an unrelated city string cannot zero it out.
    """
    train = raw_frame.iloc[:400]
    transformer = P.TrainOnlyFrequencyFeatures().fit(train)

    seen_location = train.head(5).copy()
    seen_location.loc[:, "source_city"] = "a city never seen in training"

    mapped = transformer.transform(seen_location)

    assert (mapped["location_frequency"] > 0.0).all()
    assert (mapped["log_location_frequency"] > 0.0).all()


def test_seen_location_is_not_given_the_fallback(raw_frame):
    train = raw_frame.iloc[:400]
    transformer = P.TrainOnlyFrequencyFeatures().fit(train)

    mapped = transformer.transform(train.head(20))

    assert (mapped["location_frequency"] > 0.0).all()
    assert (mapped["log_location_frequency"] > 0.0).all()


def test_fallback_distinguishes_unseen_from_singleton(raw_frame):
    """0.0 for unseen, and a real singleton count of 1.0, stay distinct.

    This is why the protocol does not smooth unseen to 1.0.
    """
    train = raw_frame.iloc[:400].copy()

    transformer = P.TrainOnlyFrequencyFeatures().fit(train)

    singleton_train = train.iloc[:1].copy()
    singleton_train.loc[:, "location"] = "lonely place"

    singleton_freq = P.TrainOnlyFrequencyFeatures().fit(
        singleton_train
    ).transform(singleton_train)["location_frequency"].iloc[0]

    unseen = train.head(3).copy()
    unseen.loc[:, "location"] = "never seen anywhere"
    unseen_freq = transformer.transform(unseen)["location_frequency"].iloc[0]

    assert unseen_freq == 0.0
    assert singleton_freq == 1.0
    assert unseen_freq != singleton_freq


# ============================================================
# 3. DUPLICATE REMOVAL
# ============================================================

def test_exact_duplicates_removed_on_protocol_key(raw_frame):
    deduped, report = P.remove_duplicates(raw_frame)

    assert report.original_rows == len(raw_frame)
    assert report.duplicate_rows_removed == 20
    assert report.final_rows == len(raw_frame) - 20
    assert len(deduped) == report.final_rows

    # Row accounting is internally consistent.
    assert (
        report.original_rows
        - report.duplicate_rows_removed
        == report.final_rows
    )

    # No duplicate survives.
    assert not deduped.duplicated(subset=P.DUPLICATE_KEY).any()

    # The key is exactly the documented five fields.
    assert list(report.key) == [
        "source_city",
        "location",
        "area",
        "no_of_bedrooms",
        "price",
    ]
    assert "mreid_id" not in report.key


def test_dedup_keeps_first_occurrence(raw_frame):
    deduped, _ = P.remove_duplicates(raw_frame)

    first = raw_frame.iloc[0]
    row = deduped.iloc[0]

    assert row["mreid_id"] == first["mreid_id"]
    assert row["price"] == first["price"]


def test_dedup_does_not_remove_distinct_properties_sharing_a_location():
    """Same location and area, different price: both must survive.

    This guards against over-aggressive de-duplication, which would silently
    discard legitimate inventory.
    """
    frame = _synthetic_frame(20, with_duplicates=False)
    twin = frame.iloc[[0]].copy()
    twin["price"] = twin["price"] + 1_000_000
    frame = pd.concat([frame, twin], ignore_index=True)

    deduped, report = P.remove_duplicates(frame)

    assert report.duplicate_rows_removed == 0
    assert len(deduped) == len(frame)


def test_dedup_keeps_properties_sharing_location_and_price_but_differing_area():
    frame = _synthetic_frame(20, with_duplicates=False)
    near = frame.iloc[[0]].copy()
    near["area"] = near["area"] + 50
    frame = pd.concat([frame, near], ignore_index=True)

    deduped, report = P.remove_duplicates(frame)

    assert report.duplicate_rows_removed == 0


def test_no_exact_duplicate_spans_any_random_fold(real_dataset):
    """The end-to-end guarantee: zero duplicates across every CV boundary."""
    df, _, _ = real_dataset
    folds = P.generate_folds(df, regime="random")

    for tr, va in zip(folds.train, folds.valid):
        train_keys = {
            tuple(row)
            for row in df.iloc[list(tr)][P.DUPLICATE_KEY]
            .to_numpy()
            .tolist()
        }
        valid_keys = [
            tuple(row)
            for row in df.iloc[list(va)][P.DUPLICATE_KEY]
            .to_numpy()
            .tolist()
        ]
        assert not any(k in train_keys for k in valid_keys)


# ============================================================
# 4. AMENITY FEATURE CONSTRUCTION
# ============================================================

def test_amenities_fully_specified_flag_definitions(raw_frame):
    frame = raw_frame.copy()

    # Force one row fully specified, one fully unknown, one mixed.
    frame.loc[frame.index[0], P.AMENITIES] = 1
    frame.loc[frame.index[1], P.AMENITIES] = 0
    mixed = frame.loc[frame.index[2]].copy()
    mixed[P.AMENITIES] = 1
    mixed[P.AMENITIES[0]] = 9
    frame.loc[frame.index[2]] = mixed

    cleaned, _ = P._clean(frame)
    built = P._add_rowwise_features(cleaned)

    flag = built["amenities_fully_specified"].to_numpy()
    yes_count = built["amenity_yes_count"].to_numpy()

    assert flag[0] == 1
    assert flag[1] == 1
    assert flag[2] == 0

    # amenity_yes_count still counts only definite 1s.
    assert yes_count[0] == len(P.AMENITIES)
    assert yes_count[1] == 0
    assert yes_count[2] == len(P.AMENITIES) - 1


def test_amenity_unknown_is_never_reinterpreted(real_dataset):
    """Raw amenity columns must still contain 9, and one-hot must expose it."""
    df, _, _ = real_dataset

    values = set(df[P.AMENITIES].to_numpy().ravel().tolist())
    assert values == {0, 1, 9}
    assert 9 in values

    # 9 is not collapsed to 0 or 1 anywhere in the feature frame.
    for column in P.AMENITIES:
        column_values = set(df[column].dropna().unique().tolist())
        assert column_values.issubset({0, 1, 9})

    # Fit on a bounded sample; the full 28k rows are not needed to observe
    # that each amenity keeps three levels.
    sample = df.sample(5000, random_state=11)

    # The preprocessor is the second pipeline step, so the frequency columns
    # must exist before it can be fitted. Fit through the pipeline to respect
    # that ordering.
    pipe = P.make_pipeline(estimator=None)
    pipe.fit(
        sample[P.INPUT_FEATURES],
        P.log_target(sample[P.TARGET_COLUMN]),
    )
    encoder = pipe.named_steps["preprocessor"].named_transformers_[
        "categorical"
    ]

    for index, column in enumerate(P.ONEHOT_FEATURES):
        if column in P.AMENITIES:
            categories = encoder.categories_[index]
            assert len(categories) == 3, (
                f"{column} lost a level: {categories}"
            )
            # sklearn 1.9 keeps the original numeric dtype for an
            # all-numeric column, so the levels are 0, 1, 9 as numbers.
            assert sorted(int(v) for v in categories) == [0, 1, 9]


def test_amenity_known_count_redundancy_is_removed():
    """The V2 degenerate pair must not reappear in V2.1."""
    for forbidden in ("amenity_known_count", "amenity_unknown_count"):
        assert forbidden not in P.FINAL_FEATURES
        assert forbidden not in P.NUMERIC_FEATURES

    assert "amenities_fully_specified" in P.FINAL_FEATURES
    assert "amenity_yes_count" in P.FINAL_FEATURES


def test_amenity_yes_count_and_flag_are_not_redundant(real_dataset):
    """The two retained amenity features must carry independent signal."""
    df, _, _ = real_dataset

    yes = df["amenity_yes_count"].to_numpy(dtype=float)
    flag = df["amenities_fully_specified"].to_numpy(dtype=float)

    # A row can be fully specified yet have zero amenities present.
    assert (flag == 1).sum() > 0
    assert (yes == 0).sum() > 0
    assert not np.array_equal(yes, flag)

    # Flag is strictly binary.
    assert set(np.unique(flag).tolist()).issubset({0.0, 1.0})


def test_amenity_flag_is_row_local(real_dataset):
    """Computing it before the split cannot leak, because it is row-local."""
    df, _, _ = real_dataset

    shuffled = df.sample(frac=1.0, random_state=3).sort_index()
    assert np.array_equal(
        df["amenities_fully_specified"].to_numpy(),
        shuffled["amenities_fully_specified"].to_numpy(),
    )


# ============================================================
# 5. PREPROCESSING FIT ISOLATION
# ============================================================

def test_pipeline_has_expected_step_order():
    pipe = P.make_pipeline(estimator=None)
    names = [name for name, _ in pipe.steps]
    assert names == ["frequency", "preprocessor", "model"]


def test_standard_scaler_is_fitted_on_training_rows_only(raw_frame):
    """The fitted scaler mean must equal the train-only mean, and must
    differ from the full-dataset mean."""
    cleaned, _ = P._clean(raw_frame)
    df = P._add_rowwise_features(cleaned).reset_index(drop=True)

    train_idx = np.arange(0, 400)
    valid_idx = np.arange(400, len(df))

    pipe = P.make_pipeline(estimator=None)
    pipe.fit(
        df.iloc[train_idx][P.INPUT_FEATURES],
        P.log_target(df.iloc[train_idx][P.TARGET_COLUMN]),
    )

    scaler = pipe.named_steps["preprocessor"].named_transformers_["numeric"]

    frequency = P.TrainOnlyFrequencyFeatures().fit(df.iloc[train_idx])
    train_view = frequency.transform(df.iloc[train_idx])
    full_view = frequency.transform(df)

    train_mean = train_view[P.NUMERIC_FEATURES].mean().to_numpy()
    full_mean = full_view[P.NUMERIC_FEATURES].mean().to_numpy()

    assert np.allclose(scaler.mean_, train_mean)
    assert not np.allclose(scaler.mean_, full_mean)


def test_scaler_refits_per_fold_and_is_not_shared(real_dataset):
    df, _, _ = real_dataset
    folds = P.generate_folds(df, regime="random")

    means = []
    for tr, _ in zip(folds.train, folds.valid):
        pipe = P.make_pipeline(estimator=None)
        pipe.fit(
            df.iloc[list(tr)][P.INPUT_FEATURES],
            P.log_target(df.iloc[list(tr)][P.TARGET_COLUMN]),
        )
        scaler = pipe.named_steps["preprocessor"].named_transformers_[
            "numeric"
        ]
        means.append(scaler.mean_.copy())

    assert len({m.tobytes() for m in means}) == len(means), (
        "scaler statistics must differ per fold"
    )


def test_unseen_location_does_not_change_onehot_width(raw_frame):
    """handle_unknown='ignore' must map unseen levels to an all-zero row
    rather than adding columns or raising."""
    cleaned, _ = P._clean(raw_frame)
    df = P._add_rowwise_features(cleaned).reset_index(drop=True)

    train_idx = np.arange(0, 400)
    valid_idx = np.arange(400, len(df))

    pipe = P.make_pipeline(estimator=None)
    pipe.fit(
        df.iloc[train_idx][P.INPUT_FEATURES],
        P.log_target(df.iloc[train_idx][P.TARGET_COLUMN]),
    )

    baseline = pipe.transform(df.iloc[valid_idx][P.INPUT_FEATURES])

    novel = df.iloc[valid_idx].copy()
    novel.loc[:, "location"] = "brand new locality"
    novel_out = pipe.transform(novel[P.INPUT_FEATURES])

    assert baseline.shape == novel_out.shape

    encoder = pipe.named_steps["preprocessor"].named_transformers_[
        "categorical"
    ]
    assert encoder.handle_unknown == "ignore"


def test_preprocessor_configures_scaler_and_encoder(raw_frame):
    """The numeric branch is a StandardScaler and the categorical branch
    keeps handle_unknown='ignore'. Read from the fitted transformer, since
    ColumnTransformer only exposes its fitted sub-transformers after fit."""
    cleaned, _ = P._clean(raw_frame)
    df = P._add_rowwise_features(cleaned).reset_index(drop=True)

    # Fit through the pipeline so the frequency columns exist when the
    # preprocessor is fitted.
    pipe = P.make_pipeline(estimator=None)
    pipe.fit(
        df[P.INPUT_FEATURES],
        P.log_target(df[P.TARGET_COLUMN]),
    )
    transformer = pipe.named_steps["preprocessor"]

    numeric_step = transformer.named_transformers_["numeric"]
    categorical_step = transformer.named_transformers_["categorical"]

    assert isinstance(numeric_step, StandardScaler)
    assert categorical_step.handle_unknown == "ignore"
    assert list(numeric_step.feature_names_in_) == P.NUMERIC_FEATURES
    assert list(categorical_step.feature_names_in_) == P.ONEHOT_FEATURES


def test_pipeline_input_excludes_frequency_columns():
    """Callers must not be able to pre-supply the frequency features, which
    is what allowed the V2 leak."""
    for column in P.FREQUENCY_FEATURES:
        assert column not in P.INPUT_FEATURES
        assert column in P.FINAL_FEATURES

    assert len(P.INPUT_FEATURES) == len(P.FINAL_FEATURES) - len(
        P.FREQUENCY_FEATURES
    )


def test_frozen_feature_count_is_48():
    """Frozen decision: 48 features, derived, not asserted by hand.

    3 basic + 6 row-wise + 2 train-only frequency + 2 categorical
    + 35 raw amenities = 48.
    """
    assert P.FINAL_FEATURE_COUNT == 48
    assert len(P.FINAL_FEATURES) == 48
    assert len(P.INPUT_FEATURES) == 46
    assert len(P.NUMERIC_FEATURES) == 11
    assert len(P.ONEHOT_FEATURES) == 37

    # The count is computed, and the parts sum to it.
    assert len(P.FINAL_FEATURES) == (
        len(P.BASIC_FEATURES)
        + len(P.ROWWISE_ENGINEERED_FEATURES)
        + len(P.FREQUENCY_FEATURES)
        + len(P.CATEGORICAL_FEATURES)
        + len(P.AMENITIES)
    )
    assert len(P.NUMERIC_FEATURES) + len(P.ONEHOT_FEATURES) == 48


def test_amenity_completeness_flag_is_labelled_as_provenance():
    """The completeness indicator is a data-recording/provenance indicator,
    not an ordinary property amenity."""
    summary = P.protocol_summary(
        P.DuplicateReport(
            original_rows=2,
            duplicate_rows_removed=0,
            final_rows=2,
        ),
        invalid_rows_removed=0,
    )
    note = summary["amenity_note"].lower()
    assert "provenance" in note
    assert "not an ordinary property amenity" in note


# ============================================================
# 6. FOLD REPRODUCIBILITY
# ============================================================

def test_fold_generation_is_reproducible(real_dataset):
    df, _, _ = real_dataset

    a = P.generate_folds(df, regime="random")
    b = P.generate_folds(df, regime="random")

    assert a.digest() == b.digest()
    assert a.train == b.train
    assert a.valid == b.valid


def test_grouped_fold_generation_is_reproducible(real_dataset):
    df, _, _ = real_dataset

    a = P.generate_folds(df, regime="location_grouped")
    b = P.generate_folds(df, regime="location_grouped")

    assert a.digest() == b.digest()


def test_folds_use_the_frozen_random_state(real_dataset):
    df, _, _ = real_dataset

    folds = P.generate_folds(df, regime="random")
    assert folds.random_state == 42
    assert P.RANDOM_STATE == 42

    other = P.generate_folds(df, regime="random", random_state=7)
    assert other.digest() != folds.digest()


def test_persisted_folds_round_trip(real_dataset, tmp_path):
    df, _, _ = real_dataset

    folds = P.generate_all_folds(df)
    path = P.save_folds(folds, path=tmp_path / "folds.json")
    reloaded = P.load_folds(path)

    for regime, original in folds.items():
        assert reloaded[regime].digest() == original.digest()
        assert reloaded[regime].train == original.train
        assert reloaded[regime].valid == original.valid


def test_every_row_is_validated_exactly_once(real_dataset):
    df, _, _ = real_dataset

    for regime in ("random", "location_grouped"):
        folds = P.generate_folds(df, regime=regime)
        seen = np.concatenate(
            [np.asarray(v, dtype=np.int64) for v in folds.valid]
        )
        assert len(seen) == len(df)
        assert set(seen.tolist()) == set(range(len(df)))


def test_no_train_valid_row_overlap(real_dataset):
    df, _, _ = real_dataset

    for regime in ("random", "location_grouped"):
        report = P.validate_folds(P.generate_folds(df, regime=regime), df)
        assert report["valid"], report["problems"]
        assert report["problems"] == []


def test_fold_partitions_are_balanced_enough(real_dataset):
    """K-fold random regime should give near-equal validation sizes."""
    df, _, _ = real_dataset
    folds = P.generate_folds(df, regime="random")

    sizes = [len(v) for v in folds.valid]
    assert max(sizes) - min(sizes) <= 2


# ============================================================
# 7. GROUPED FOLDS HAVE ZERO GROUP OVERLAP
# ============================================================

def test_grouped_folds_have_zero_group_overlap(real_dataset):
    df, _, _ = real_dataset
    folds = P.generate_folds(df, regime="location_grouped")

    overlap = folds.group_overlap(df["group"].to_numpy())
    assert overlap == [0] * folds.n_folds()


def test_grouped_split_holds_out_whole_localities(real_dataset):
    df, _, _ = real_dataset
    folds = P.generate_folds(df, regime="location_grouped")

    train_groups = set(df["group"].iloc[list(folds.train[0])])
    valid_groups = set(df["group"].iloc[list(folds.valid[0])])

    assert train_groups.isdisjoint(valid_groups)


def test_group_column_definition_is_city_plus_location(real_dataset):
    df, _, _ = real_dataset

    assert df["group"].iloc[0] == (
        f"{df['source_city'].iloc[0]}__{df['location'].iloc[0]}"
    )
    assert "__" in df["group"].iloc[0]


def test_grouped_regime_differs_from_random_regime(real_dataset):
    """A genuine grouped split must not coincide with a plain random one."""
    df, _, _ = real_dataset

    random_folds = P.generate_folds(df, regime="random")
    grouped_folds = P.generate_folds(df, regime="location_grouped")

    assert random_folds.digest() != grouped_folds.digest()


def test_unknown_regime_is_rejected(real_dataset):
    df, _, _ = real_dataset

    with pytest.raises(ValueError):
        P.generate_folds(df, regime="not_a_regime")


# ============================================================
# 8. METRIC CALCULATIONS
# ============================================================

def test_metrics_are_exact_on_a_perfect_prediction():
    prices = np.array([2_000_000.0, 5_000_000.0, 10_000_000.0])
    perfect_log = P.log_target(prices)

    m = P.compute_metrics(prices, perfect_log)

    assert m["MAE_INR"] == pytest.approx(0.0, abs=1e-6)
    assert m["RMSE_INR"] == pytest.approx(0.0, abs=1e-6)
    assert m["R2_INR"] == pytest.approx(1.0, abs=1e-9)
    assert m["MAPE_percent"] == pytest.approx(0.0, abs=1e-9)
    assert m["MedAPE_percent"] == pytest.approx(0.0, abs=1e-9)
    assert m["MAE_log"] == pytest.approx(0.0, abs=1e-12)
    assert m["RMSE_log"] == pytest.approx(0.0, abs=1e-12)
    assert m["R2_log"] == pytest.approx(1.0, abs=1e-9)


def test_metrics_match_hand_computed_values():
    prices = np.array([1_000_000.0, 2_000_000.0, 4_000_000.0])
    predictions = np.array([1_100_000.0, 1_800_000.0, 5_000_000.0])

    log_pred = P.log_target(predictions)
    m = P.compute_metrics(prices, log_pred)

    residuals = np.abs(prices - predictions)
    assert m["MAE_INR"] == pytest.approx(residuals.mean())
    assert m["RMSE_INR"] == pytest.approx(np.sqrt((residuals ** 2).mean()))

    pct = residuals / prices
    assert m["MAPE_percent"] == pytest.approx(pct.mean() * 100)
    assert m["MedAPE_percent"] == pytest.approx(np.median(pct) * 100)


def test_medape_is_robust_to_outliers_where_mape_is_not():
    """One catastrophic relative error moves MAPE but not MedAPE.

    MAPE is the mean of relative errors, so a single 0.1% prediction on a
    1M property contributes 100%. MedAPE uses the median.
    """
    prices = np.array([1_000_000.0] * 11)
    predictions = np.full(11, 1_000_000.0)
    predictions[0] = 1_000.0

    m = P.compute_metrics(prices, P.log_target(predictions))

    assert m["MedAPE_percent"] == pytest.approx(0.0)
    assert m["MAPE_percent"] > 9.0
    assert m["MAPE_percent"] > m["MedAPE_percent"] * 100


def test_metric_keys_are_exactly_the_protocol_contract():
    assert set(P.METRIC_KEYS) == {
        "MAE_INR",
        "RMSE_INR",
        "R2_INR",
        "MAPE_percent",
        "MedAPE_percent",
        "MAE_log",
        "RMSE_log",
        "R2_log",
    }

    prices = [1_000_000.0, 2_000_000.0, 3_000_000.0]
    assert len(P.compute_metrics(prices, P.log_target(prices))) == len(
        P.METRIC_KEYS
    )


def test_percentage_metrics_are_finite_on_real_dataset(real_dataset):
    df, _, _ = real_dataset

    sample = df.sample(2000, random_state=1)
    noisy = P.log_target(
        sample[P.TARGET_COLUMN].to_numpy() * 1.25
    )

    m = P.compute_metrics(sample[P.TARGET_COLUMN].to_numpy(), noisy)

    for key in P.METRIC_KEYS:
        value = m[key]
        assert np.isfinite(value), f"{key} was not finite"


def test_log_and_rupee_r2_can_diverge(real_dataset):
    """Confirms the two spaces are computed independently, not aliased."""
    df, _, _ = real_dataset

    sample = df.sample(3000, random_state=2).sort_index()
    prices = sample[P.TARGET_COLUMN].to_numpy()

    # A model biased only on the luxury tail.
    pred = np.where(prices > 3e7, prices * 1.9, prices * 0.95)

    m = P.compute_metrics(prices, P.log_target(pred))

    assert m["R2_INR"] != m["R2_log"]


# ============================================================
# 9. PREDICTION-SPACE CONVERSION
# ============================================================

def test_log_and_rupee_round_trip():
    prices = np.array([1.0, 1_000_000.0, 50_000_000.0, 854_599_999.0])

    log_space = P.log_target(prices)
    assert np.allclose(P.to_rupees(log_space), prices)


def test_negative_log_prediction_floors_at_zero():
    """expm1 of a negative log value is negative; the protocol floors it."""
    log_pred = np.array([-5.0, 0.0, 14.5])

    rupees = P.to_rupees(log_pred)

    assert rupees[0] == 0.0
    assert rupees[1] == 0.0
    assert (rupees >= 0.0).all()
    assert rupees[2] > 0.0


def test_predictions_are_reported_in_log_space():
    """A fitted pipeline emits log-space output, not rupees.

    This is why the evaluation step must invert explicitly. The test fits a
    throwaway Ridge in memory; nothing is saved.
    """
    from sklearn.linear_model import Ridge

    cleaned, _ = P._clean(_synthetic_frame(800))
    df = P._add_rowwise_features(cleaned).reset_index(drop=True)

    pipe = P.make_pipeline(Ridge(alpha=1.0))
    pipe.fit(
        df[P.INPUT_FEATURES],
        P.log_target(df[P.TARGET_COLUMN]),
    )

    raw_prediction = pipe.predict(df[P.INPUT_FEATURES])

    # Log-space magnitudes are around log1p(price), roughly 14 to 21.
    assert raw_prediction.min() > 5.0
    assert raw_prediction.max() < 30.0

    in_rupees = P.to_rupees(raw_prediction)
    assert in_rupees.min() > 100_000.0


def test_metric_spaces_are_documented():
    assert P.TRAINING_SPACE == "log1p(price)"
    assert P.RUPEE_SPACE == "INR"
    assert P.METRIC_SPACE["MAE_INR"] == "INR"
    assert P.METRIC_SPACE["MAE_log"] == "log1p(price)"
    assert set(P.METRIC_SPACE) == set(P.METRIC_KEYS)


def test_real_dataset_amenity_flag_counts(real_dataset):
    """Sanity check on the real dataset, guarding the documented figure."""
    df, report, _ = real_dataset

    assert len(df) == report.final_rows
    assert set(
        df["amenities_fully_specified"].unique().tolist()
    ).issubset({0, 1})
    assert df[P.TARGET_COLUMN].min() > 0, "no zero targets, MAPE is safe"
    assert set(df[P.AMENITIES].to_numpy().ravel().tolist()) == {0, 1, 9}
