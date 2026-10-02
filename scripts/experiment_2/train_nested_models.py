"""Experiment 2 Phase 3: train the 10 nested CatBoost models.

This module trains exactly the ten nested CatBoost models required by the
Experiment 2 conformal protocol:

    random regime            : 5 models
    location-grouped regime  : 5 models
    total                    : 10 models

The models are the leakage-controlled nested models, NOT the Experiment 1
models. Each model is fitted on T'_k (the nested fit subset) only. It is never
fitted on C_k (the calibration subset) or on test_k (the frozen Experiment 1
validation fold). Preprocessing statistics (location frequency, standard
scaler, one-hot vocabulary) are therefore derived from T'_k alone, because the
whole sklearn Pipeline is fitted inside this module.

FROZEN MODEL CONFIGURATION (Experiment 1 CatBoost backbone, unchanged):

    iterations     = 700
    learning_rate  = 0.04
    depth          = 7
    l2_leaf_reg    = 2.0
    random_seed    = 42
    loss_function  = RMSE
    verbose        = False
    allow_writing_files = False

No tuning, no early stopping, no feature engineering, no feature selection, no
target transformation, no additional models, no second seed.

SCOPE. This phase is TRAINING ONLY. No conformal score, quantile, interval,
coverage, interval width, subgroup coverage or monotonicity quantity is
computed here. The only prediction performed is the basic reload /
reproducibility probe required by the Phase 3 verification contract.

SAFETY. Writes only under:

    models/valuation/v2_2/conformal/{random,location_grouped}/
    artifacts/valuation/v2_2/conformal/

Nothing under any Experiment 1 path is written. No blockchain interaction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_2 import build_calibration_folds as B  # noqa: E402
from scripts.valuation_v2_1 import protocol as P  # noqa: E402

import catboost  # noqa: E402


# ============================================================
# FROZEN CONFIGURATION
# ============================================================

#: The Experiment 1 CatBoost configuration, frozen. Do not edit.
CATBOOST_CONFIG: Dict[str, Any] = {
    "iterations": 700,
    "learning_rate": 0.04,
    "depth": 7,
    "l2_leaf_reg": 2.0,
    "random_seed": 42,
    "loss_function": "RMSE",
    "verbose": False,
    "allow_writing_files": False,
}

#: SHA-256 of the Phase 2 calibration-fold artifact. If the artifact changes,
#: Phase 3 refuses to run rather than train against an unknown split.
CALIBRATION_FOLDS_SHA256 = (
    "47b89d8b41e98301129860cbf253f8c30255e272ec79f8277da54bb73acc6ea9"
)

REGIMES: Tuple[str, ...] = ("random", "location_grouped")

#: The number of models the protocol requires.
EXPECTED_MODEL_COUNT = 10

# ============================================================
# NAMESPACING
# ============================================================

MODEL_ROOT = ROOT / "models" / "valuation" / "v2_2" / "conformal"
ARTIFACT_ROOT = ROOT / "artifacts" / "valuation" / "v2_2" / "conformal"

MANIFEST_JSON = ARTIFACT_ROOT / "nested_models_manifest.json"
PROTECTED_BEFORE_JSON = ARTIFACT_ROOT / "protected_state_before.json"
PROTECTED_AFTER_JSON = ARTIFACT_ROOT / "protected_state_after.json"

#: Files whose bytes must not change during Phase 3. Directories are walked.
PROTECTED_TREES: Tuple[str, ...] = (
    "models/valuation/v2_1",
    "artifacts/valuation/v2_1",
)

PROTECTED_FILES: Tuple[str, ...] = (
    "data/processed/MREID_property.csv",
    "data/processed/millow_token_map.csv",
    "chain-manifest.json",
    ".chain/state.json",
    ".chain/millow-anvil-state-29135.json.gz",
    ".chain/millow-anvil-state-29135.json.gz.sha256",
    ".chain/millow-anvil-state-29135.json.gz.meta.json",
)


# ============================================================
# LOW-LEVEL HELPERS
# ============================================================


def sha256_of(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def rel_to_root(path: Path) -> str:
    """Repository-relative POSIX path when possible, else absolute POSIX path."""
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def assert_safe_write_target(path: Path) -> None:
    """Refuse to write anywhere that could collide with a protected path."""
    B.assert_safe_write_target(path)


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def environment_record() -> Dict[str, Any]:
    record = P.environment_record()
    record["catboost"] = catboost.__version__
    record["platform"] = platform.platform()
    return record


# ============================================================
# PROTECTED-STATE SNAPSHOT
# ============================================================


def _iter_protected_files() -> List[Path]:
    files: List[Path] = []
    for rel in PROTECTED_TREES:
        base = ROOT / rel
        if not base.exists():
            raise FileNotFoundError(f"protected tree missing: {rel}")
        for path in sorted(base.rglob("*")):
            if path.is_file():
                files.append(path)
    for rel in PROTECTED_FILES:
        path = ROOT / rel
        if not path.exists():
            raise FileNotFoundError(f"protected file missing: {rel}")
        files.append(path)
    return files


def snapshot_protected_state() -> Dict[str, Any]:
    """Hash every protected file and record git HEAD and tags."""
    files: Dict[str, str] = {}
    for path in _iter_protected_files():
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        files[rel] = sha256_of(path)
    payload = json.dumps(files, sort_keys=True).encode("utf-8")
    tags = [t for t in _git("tag", "--list").splitlines() if t]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git("rev-parse", "HEAD"),
        "git_tags": tags,
        "n_files": len(files),
        "files": files,
        "tree_digest": sha256_of_bytes(payload),
    }


def compare_protected_states(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return a structured diff of two protected-state snapshots."""
    b = before["files"]
    a = after["files"]
    added = sorted(set(a) - set(b))
    removed = sorted(set(b) - set(a))
    changed = sorted(k for k in (set(a) & set(b)) if a[k] != b[k])
    return {
        "clean": not (added or removed or changed),
        "added": added,
        "removed": removed,
        "changed": changed,
        "git_head_before": before["git_head"],
        "git_head_after": after["git_head"],
        "git_tags_before": before["git_tags"],
        "git_tags_after": after["git_tags"],
    }


# ============================================================
# INPUT CONTRACT
# ============================================================


def load_calibration_folds(
    path: Path = B.CALIBRATION_FOLDS_JSON,
) -> Dict[str, Any]:
    """Load and verify the Phase 2 artifact before any training occurs.

    Verification:
      1. The artifact's SHA-256 matches the frozen Phase 3 constant.
      2. The artifact records the frozen V2.1 parent-fold digest.
      3. Every row/group invariant from Phase 2 is re-asserted against the
         frozen V2.1 fold contract and the real dataset.
    """
    actual = sha256_of(path)
    if actual != CALIBRATION_FOLDS_SHA256:
        raise RuntimeError(
            "Calibration-fold artifact has changed. Refusing to train.\n"
            f"  expected {CALIBRATION_FOLDS_SHA256}\n  actual   {actual}"
        )
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)

    # The artifact itself must record the frozen parent hash.
    if doc.get("parent_folds_sha256") != B.FROZEN_FOLDS_SHA256:
        raise RuntimeError(
            "Calibration-fold artifact does not record the frozen V2.1 parent "
            f"hash.\n  expected {B.FROZEN_FOLDS_SHA256}\n"
            f"  recorded {doc.get('parent_folds_sha256')}"
        )

    # Re-derive the frozen contract and check the artifact consumed it intact.
    B.verify_frozen_folds_untouched()
    raw, _dup, _inv = P.build_dataset()
    groups = P.add_group_labels(raw)
    frozen = P.load_folds()
    B.verify_all_invariants(doc, frozen, groups)

    for regime in REGIMES:
        folds = doc["regimes"][regime]["folds"]
        if len(folds) != len(frozen[regime].train):
            raise RuntimeError(f"{regime}: fold count mismatch")
    return doc


def assert_fold_disjoint(
    entry: Mapping[str, Any],
    regime: str,
    groups: pd.Series,
) -> None:
    """Assert the row-wise (and, grouped, group-wise) invariances for one fold."""
    fit = set(entry["fit"])
    cal = set(entry["calibrate"])
    test = set(entry["test"])
    k = entry["outer_fold"]

    if fit & cal:
        raise RuntimeError(f"{regime} fold {k}: T'_k intersect C_k is not empty")
    if fit & test:
        raise RuntimeError(f"{regime} fold {k}: T'_k intersect test_k is not empty")
    if cal & test:
        raise RuntimeError(f"{regime} fold {k}: C_k intersect test_k is not empty")

    if entry["fit"] != sorted(entry["fit"]):
        raise RuntimeError(f"{regime} fold {k}: fit indices are not sorted")
    if entry["calibrate"] != sorted(entry["calibrate"]):
        raise RuntimeError(f"{regime} fold {k}: calibrate indices are not sorted")

    if regime == "location_grouped":
        g_fit = set(groups.iloc[entry["fit"]].astype(str))
        g_cal = set(groups.iloc[entry["calibrate"]].astype(str))
        g_test = set(groups.iloc[entry["test"]].astype(str))
        if g_fit & g_cal:
            raise RuntimeError(
                f"{regime} fold {k}: localities cross T'_k <-> C_k"
            )
        if g_fit & g_test:
            raise RuntimeError(
                f"{regime} fold {k}: localities cross T'_k <-> test_k"
            )
        if g_cal & g_test:
            raise RuntimeError(
                f"{regime} fold {k}: localities cross C_k <-> test_k"
            )


# ============================================================
# MODEL CONSTRUCTION
# ============================================================


def build_estimator() -> CatBoostRegressor:
    """Instantiate the single frozen CatBoost estimator. No search, no seed 2."""
    return CatBoostRegressor(**CATBOOST_CONFIG)


def build_pipeline() -> Any:
    """Assemble the frozen V2.1 pipeline around CatBoost.

    dense_output=True is CatBoost's documented input-format accommodation
    (CatBoost rejects scipy sparse); it changes no fitted statistic.
    """
    return P.make_pipeline(build_estimator(), dense_output=True)


def feature_contract() -> Dict[str, Any]:
    """Record the V2.1 feature contract this model was trained against."""
    return {
        "protocol": "V2.1 (frozen, read-only)",
        "input_features": list(P.INPUT_FEATURES),
        "final_features": list(P.FINAL_FEATURES),
        "final_feature_count": int(P.FINAL_FEATURE_COUNT),
        "numeric_features": list(P.NUMERIC_FEATURES),
        "onehot_features": list(P.ONEHOT_FEATURES),
        "frequency_features": list(P.FREQUENCY_FEATURES),
        "categorical_features": list(P.CATEGORICAL_FEATURES),
        "target_column": P.TARGET_COLUMN,
        "training_space": P.TRAINING_SPACE,
        "random_state": int(P.RANDOM_STATE),
    }


# ============================================================
# TRAINING ONE FOLD
# ============================================================


def _fit_identity_proof(
    pipe: Any,
    df: pd.DataFrame,
    fit_idx: Sequence[int],
) -> Dict[str, Any]:
    """Prove the fitted pipeline saw exactly the T'_k rows and nothing else.

    TrainOnlyFrequencyFeatures is the FIRST pipeline step. Its fitted
    n_fit_rows_ must equal len(T'_k), and its fitted location counts must equal
    the counts computed on T'_k alone. Any C_k or test_k row in fitting would
    change at least one of those.
    """
    freq = pipe.named_steps["frequency"]
    expected_counts = df.iloc[list(fit_idx)]["location"].value_counts()
    actual = freq.location_counts_
    expected = expected_counts
    match = (
        int(freq.n_fit_rows_) == len(fit_idx)
        and actual.shape == expected.shape
        and bool((actual.sort_index() == expected.sort_index()).all())
    )
    return {
        "frequency_n_fit_rows": int(freq.n_fit_rows_),
        "expected_n_fit_rows": int(len(fit_idx)),
        "location_counts_match_T_prime": bool(match),
    }


def train_fold(
    regime: str,
    entry: Mapping[str, Any],
    df: pd.DataFrame,
    groups: pd.Series,
    *,
    model_root: Path = MODEL_ROOT,
) -> Dict[str, Any]:
    """Fit one nested CatBoost model on T'_k and persist it plus metadata."""
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    if CATBOOST_CONFIG["random_seed"] != 42:
        raise RuntimeError("random_seed must be 42")
    if abs(CATBOOST_CONFIG["learning_rate"] - 0.04) > 1e-12:
        raise RuntimeError("learning_rate must be 0.04")
    if CATBOOST_CONFIG["iterations"] != 700:
        raise RuntimeError("iterations must be 700")
    if CATBOOST_CONFIG["depth"] != 7:
        raise RuntimeError("depth must be 7")
    if abs(CATBOOST_CONFIG["l2_leaf_reg"] - 2.0) > 1e-12:
        raise RuntimeError("l2_leaf_reg must be 2.0")
    if CATBOOST_CONFIG["loss_function"] != "RMSE":
        raise RuntimeError("loss_function must be RMSE")

    assert_fold_disjoint(entry, regime, groups)

    fold_id = int(entry["outer_fold"])
    fit_idx = list(entry["fit"])
    cal_idx = list(entry["calibrate"])
    test_idx = list(entry["test"])

    x_all = df[P.INPUT_FEATURES]
    y_all = P.log_target(df[P.TARGET_COLUMN])

    pipe = build_pipeline()

    fit_started = datetime.now(timezone.utc)
    pipe.fit(x_all.iloc[fit_idx], y_all[fit_idx])
    fit_seconds = (datetime.now(timezone.utc) - fit_started).total_seconds()

    proof = _fit_identity_proof(pipe, df, fit_idx)
    if not proof["location_counts_match_T_prime"]:
        raise RuntimeError(
            f"{regime} fold {fold_id}: fitted frequency statistics do not match "
            "T'_k; a non-T'_k row may have entered fitting."
        )

    out_dir = model_root / regime
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"catboost_{regime}_fold{fold_id}.joblib"
    model_path = out_dir / model_name
    meta_path = out_dir / f"catboost_{regime}_fold{fold_id}.meta.json"
    assert_safe_write_target(model_path)

    joblib.dump(pipe, model_path)
    model_sha = sha256_of(model_path)

    # Reload/reproducibility probe: a deterministic slice of the test rows.
    probe_idx = test_idx[:64] if len(test_idx) >= 64 else list(test_idx)
    in_memory = np.asarray(pipe.predict(x_all.iloc[probe_idx]), dtype=float)
    reloaded = joblib.load(model_path)
    from_disk = np.asarray(reloaded.predict(x_all.iloc[probe_idx]), dtype=float)
    max_abs_diff = float(np.max(np.abs(in_memory - from_disk))) if len(probe_idx) else 0.0

    record: Dict[str, Any] = {
        "experiment": "MILLOW Experiment 2 - conformal prediction",
        "phase": "Phase 3 - nested model training",
        "regime": regime,
        "fold": fold_id,
        "model": "CatBoost",
        "model_file": rel_to_root(model_path),
        "model_sha256": model_sha,
        "nested_training_rows": int(len(fit_idx)),
        "calibration_rows": int(len(cal_idx)),
        "test_rows": int(len(test_idx)),
        "train_target": "log1p(price)",
        "model_config": dict(CATBOOST_CONFIG),
        "feature_contract": feature_contract(),
        "calibration_folds": {
            "path": str(B.CALIBRATION_FOLDS_JSON.relative_to(ROOT)).replace("\\", "/"),
            "sha256": CALIBRATION_FOLDS_SHA256,
            "parent_folds_sha256": B.FROZEN_FOLDS_SHA256,
            "fit_digest": entry["digests"]["fit"],
            "calibrate_digest": entry["digests"]["calibrate"],
            "test_digest": entry["digests"]["test"],
        },
        "trained_from_T_prime_only": True,
        "no_calibration_rows_in_fit": True,
        "no_test_rows_in_fit": True,
        "group_disjoint": regime == "location_grouped",
        "fit_identity_proof": proof,
        "fit_seconds": round(fit_seconds, 3),
        "reload_prediction_probe": {
            "n_rows": int(len(probe_idx)),
            "exact_match": bool(max_abs_diff == 0.0),
            "max_abs_diff": max_abs_diff,
        },
        "environment": environment_record(),
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return record


# ============================================================
# TRAIN ALL
# ============================================================


def train_all(
    *,
    model_root: Path = MODEL_ROOT,
    artifact_root: Path = ARTIFACT_ROOT,
) -> Dict[str, Any]:
    """Train all 10 nested models, recording protected state around the run."""
    assert_safe_write_target(model_root / "random" / "probe.joblib")
    assert_safe_write_target(artifact_root / "probe.json")
    artifact_root.mkdir(parents=True, exist_ok=True)

    before = snapshot_protected_state()
    with open(artifact_root / PROTECTED_BEFORE_JSON.name, "w", encoding="utf-8") as fh:
        json.dump(before, fh, indent=2)
        fh.write("\n")

    doc = load_calibration_folds()
    raw, _dup, _inv = P.build_dataset()
    groups = P.add_group_labels(raw)

    models: List[Dict[str, Any]] = []
    for regime in REGIMES:
        for entry in doc["regimes"][regime]["folds"]:
            record = train_fold(
                regime, entry, raw, groups, model_root=model_root
            )
            models.append(record)
            print(
                f"  [{len(models):>2}/{EXPECTED_MODEL_COUNT}] "
                f"{regime:<17} fold {record['fold']}  "
                f"fit={record['nested_training_rows']:>6}  "
                f"cal={record['calibration_rows']:>5}  "
                f"test={record['test_rows']:>5}  "
                f"sha={record['model_sha256'][:12]}"
            )

    if len(models) != EXPECTED_MODEL_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_MODEL_COUNT} models, trained {len(models)}"
        )

    manifest: Dict[str, Any] = {
        "experiment": "MILLOW Experiment 2 - conformal prediction",
        "phase": "Phase 3 - nested model training",
        "protocol": "V2.1 (frozen, read-only)",
        "status": "10/10 models trained",
        "n_models": len(models),
        "regimes": list(REGIMES),
        "model_config": dict(CATBOOST_CONFIG),
        "feature_contract": feature_contract(),
        "calibration_folds": {
            "path": str(B.CALIBRATION_FOLDS_JSON.relative_to(ROOT)).replace("\\", "/"),
            "sha256": CALIBRATION_FOLDS_SHA256,
            "parent_folds_sha256": B.FROZEN_FOLDS_SHA256,
        },
        "scope_note": (
            "Training only. No conformal score, quantile, interval, coverage, "
            "width, subgroup, monotonicity or figure quantity is computed."
        ),
        "no_tuning": True,
        "no_early_stopping": True,
        "no_second_seed": True,
        "models": models,
    }
    with open(artifact_root / MANIFEST_JSON.name, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")

    after = snapshot_protected_state()
    with open(artifact_root / PROTECTED_AFTER_JSON.name, "w", encoding="utf-8") as fh:
        json.dump(after, fh, indent=2)
        fh.write("\n")

    diff = compare_protected_states(before, after)
    manifest["protected_state"] = diff
    manifest["protected_state_clean"] = diff["clean"]
    with open(artifact_root / MANIFEST_JSON.name, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")

    if not diff["clean"]:
        raise RuntimeError(
            "Protected state changed during training: "
            f"{json.dumps(diff, indent=2)}"
        )
    return manifest


# ============================================================
# INDEPENDENT VERIFICATION
# ============================================================


def verify_artifacts(
    *,
    model_root: Path = MODEL_ROOT,
    artifact_root: Path = ARTIFACT_ROOT,
) -> Dict[str, Any]:
    """Reload every persisted model and independently verify the manifest."""
    manifest_path = artifact_root / MANIFEST_JSON.name
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    doc = load_calibration_folds()
    raw, _dup, _inv = P.build_dataset()
    groups = P.add_group_labels(raw)
    x_all = raw[P.INPUT_FEATURES]

    expected_by_cell: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for regime in REGIMES:
        for entry in doc["regimes"][regime]["folds"]:
            expected_by_cell[(regime, int(entry["outer_fold"]))] = entry

    records = manifest["models"]
    seen: List[Tuple[str, int]] = []
    results: List[Dict[str, Any]] = []

    for record in records:
        regime = record["regime"]
        fold_id = int(record["fold"])
        seen.append((regime, fold_id))

        entry = expected_by_cell.get((regime, fold_id))
        if entry is None:
            raise RuntimeError(f"unexpected model cell {regime} fold {fold_id}")

        assert_fold_disjoint(entry, regime, groups)

        # Row counts must match the Phase 2 artifact exactly.
        if record["nested_training_rows"] != len(entry["fit"]):
            raise RuntimeError(f"{regime} fold {fold_id}: fit row count mismatch")
        if record["calibration_rows"] != len(entry["calibrate"]):
            raise RuntimeError(f"{regime} fold {fold_id}: calibration row count mismatch")
        if record["test_rows"] != len(entry["test"]):
            raise RuntimeError(f"{regime} fold {fold_id}: test row count mismatch")

        # Model config must be the frozen configuration.
        if record["model_config"] != CATBOOST_CONFIG:
            raise RuntimeError(f"{regime} fold {fold_id}: model config drift")

        # Artifact identity must be recorded.
        cf = record["calibration_folds"]
        if cf["sha256"] != CALIBRATION_FOLDS_SHA256:
            raise RuntimeError(f"{regime} fold {fold_id}: calibration digest mismatch")
        if cf["parent_folds_sha256"] != B.FROZEN_FOLDS_SHA256:
            raise RuntimeError(f"{regime} fold {fold_id}: parent fold digest mismatch")
        if cf["fit_digest"] != entry["digests"]["fit"]:
            raise RuntimeError(f"{regime} fold {fold_id}: fit digest mismatch")

        # On-disk bytes must match the recorded hash.
        model_path = ROOT / record["model_file"]
        actual_sha = sha256_of(model_path)
        if actual_sha != record["model_sha256"]:
            raise RuntimeError(f"{regime} fold {fold_id}: model file hash mismatch")

        # Reload and reproduce predictions on the test rows, and prove the
        # fitted frequency statistics belong to T'_k.
        pipe = joblib.load(model_path)
        probe_idx = entry["test"][:64] if len(entry["test"]) >= 64 else list(entry["test"])
        preds = np.asarray(pipe.predict(x_all.iloc[probe_idx]), dtype=float)
        if not np.all(np.isfinite(preds)):
            raise RuntimeError(f"{regime} fold {fold_id}: non-finite predictions")

        freq = pipe.named_steps["frequency"]
        if int(freq.n_fit_rows_) != len(entry["fit"]):
            raise RuntimeError(
                f"{regime} fold {fold_id}: fitted frequency saw "
                f"{int(freq.n_fit_rows_)} rows, expected {len(entry['fit'])}"
            )
        expected_counts = raw.iloc[list(entry["fit"])]["location"].value_counts()
        actual_counts = freq.location_counts_
        if not (
            actual_counts.shape == expected_counts.shape
            and bool((actual_counts.sort_index() == expected_counts.sort_index()).all())
        ):
            raise RuntimeError(
                f"{regime} fold {fold_id}: fitted frequency statistics are not T'_k"
            )

        results.append(
            {
                "regime": regime,
                "fold": fold_id,
                "model": record["model_file"],
                "model_sha256": actual_sha,
                "nested_training_rows": record["nested_training_rows"],
                "calibration_rows": record["calibration_rows"],
                "test_rows": record["test_rows"],
                "reload_ok": True,
                "trained_from_T_prime_only": True,
            }
        )

    expected_cells = set(expected_by_cell)
    if set(seen) != expected_cells:
        raise RuntimeError(
            f"model cells do not match the Phase 2 artifact: "
            f"missing {sorted(expected_cells - set(seen))}, "
            f"extra {sorted(set(seen) - expected_cells)}"
        )
    if len(records) != EXPECTED_MODEL_COUNT:
        raise RuntimeError(f"expected 10 models, manifest has {len(records)}")

    # Protected state must have been clean across the training run.
    if manifest.get("protected_state_clean") is not True:
        raise RuntimeError("manifest reports a protected-state violation")

    return {
        "verified": True,
        "n_models": len(results),
        "reload_verified": all(r["reload_ok"] for r in results),
        "models": results,
    }


# ============================================================
# CLI
# ============================================================


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Experiment 2 Phase 3 trainer")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="reload and verify existing artifacts without training",
    )
    args = parser.parse_args(argv)

    if args.verify_only:
        report = verify_artifacts()
        print(json.dumps(report, indent=2))
        return 0

    manifest = train_all()
    report = verify_artifacts()
    payload = {
        "status": manifest["status"],
        "n_models": manifest["n_models"],
        "protected_state_clean": manifest["protected_state_clean"],
        "verification": report,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
