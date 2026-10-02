"""Experiment 2 Phase 2: deterministic nested calibration fold derivation.

This module derives the conformal nested split from the FROZEN V2.1 fold
contract. It NEVER writes to, regenerates, or edits cv_folds_v2_1.json. The
frozen file is opened read-only and its SHA-256 is verified before and after
derivation (protocol L12).

Design (protocol §5.1, §5.6), with cal_frac = 0.20 fixed at Phase 0:

  For each (regime, outer_fold):
      train_k  = frozen train_k from cv_folds_v2_1.json
      T'_k     = fit subset of train_k
      C_k      = calibration subset of train_k
      test_k   = frozen valid_k, unchanged

  Hard invariants, all asserted before anything is written:
      T'_k ∩ C_k    = {}
      T'_k ∩ test_k  = {}
      C_k   ∩ test_k = {}
      T'_k ∪ C_k     = train_k      (exactly; no row dropped)
      test_k         = frozen valid_k (bit-identical)

  For location_grouped, additionally, on groups:
      groups(T'_k) ∩ groups(C_k)   = {}
      groups(T'_k) ∩ groups(test_k) = {}
      groups(C_k)   ∩ groups(test_k) = {}

NO MODEL IS TRAINED HERE. This module only computes index sets.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from scripts.valuation_v2_1 import protocol as P

# ---------------------------------------------------------------------------
# Phase 0 pre-registered constants
# ---------------------------------------------------------------------------

#: The single pre-registered calibration fraction (§0.1).
CAL_FRAC = 0.20

#: Pre-registered random state, inherited from protocol.RANDOM_STATE (§0.6).
RANDOM_STATE = 42

REGIMES: Tuple[str, ...] = ("random", "location_grouped")

#: The frozen V2.1 location-grouped split key, matching P.add_group_labels.
GROUP_COLUMN = "source_city__location"

#: SHA-256 of the frozen fold contract. Verified before and after derivation (L12).
FROZEN_FOLDS_SHA256 = (
    "989b79656b051b3b6c8c583b4170a5aaa1c31676260772247bbc98d1eb095156"
)

#: Paths this module must never write to. Mirrors the V2.1 protection list and
#: extends it with the Experiment 1 model and artifact trees (§18.5).
PROTECTED_PREFIXES: Tuple[str, ...] = (
    "models/valuation/valuation_v2",
    "artifacts/valuation/valuation_v2",
    "models/valuation/v2_1",
    "artifacts/valuation/v2_1",
    "scripts/valuation_v2_1",
    "backend",
    ".chain",
)

#: Experiment 2 output. New namespace; nothing under v2_1 is touched.
OUT_DIR = P.ROOT / "artifacts" / "valuation" / "v2_2" / "conformal"
CALIBRATION_FOLDS_JSON = OUT_DIR / "calibration_folds_v2_2.json"


def sha256_of(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_safe_write_target(path: Path) -> None:
    """Refuse to write anywhere that could collide with protected paths (§18.5).

    Re-implemented here rather than by editing the frozen V2.1 module, so that
    scripts/valuation_v2_1 stays byte-identical.
    """
    resolved = str(path.resolve()).replace("\\", "/")
    root = str(P.ROOT).replace("\\", "/")
    rel = resolved[len(root) + 1 :] if resolved.startswith(root) else resolved
    for bad in PROTECTED_PREFIXES:
        bad_norm = bad.replace("\\", "/")
        if rel == bad_norm or rel.startswith(bad_norm + "/") or resolved.startswith(
            str(P.ROOT / bad).replace("\\", "/")
        ):
            raise RuntimeError(
                f"Refusing to write {rel}: collides with protected path {bad_norm}."
            )


def verify_frozen_folds_untouched(path: Path = P.FOLDS_PATH) -> str:
    """L12: the frozen fold contract must be byte-identical to the recorded hash."""
    actual = sha256_of(path)
    if actual != FROZEN_FOLDS_SHA256:
        raise RuntimeError(
            "Frozen V2.1 fold contract has changed. Refusing to derive Experiment 2 "
            f"folds.\n  expected {FROZEN_FOLDS_SHA256}\n  actual   {actual}"
        )
    return actual


def _digest_indices(indices: List[int]) -> str:
    """Stable digest of an index list, mirroring the frozen file's convention."""
    payload = ",".join(str(i) for i in indices)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derive_inner_split(
    train_indices: List[int],
    *,
    regime: str,
    cal_frac: float = CAL_FRAC,
    random_state: int = RANDOM_STATE,
    groups: pd.Series | None = None,
    n_splits: int = 5,
) -> Tuple[List[int], List[int]]:
    """Split train_k into a fit set T' and a calibration set C.

    Semantics mirror the outer splitter of the same regime (§5.6):

      regime == "random"
          Row-wise KFold(shuffle=True, random_state=42). Matches the outer
          random splitter's semantics, so a locality may appear in T', C and
          test together. That is correct: the random regime does NOT claim to
          test locality generalisation.

      regime == "location_grouped"
          GroupKFold(shuffle=True, random_state=42) on source_city__location,
          which guarantees groups(T') ∩ groups(C) = {}. Calibration
          localities are therefore unseen by the fit set, exactly mirroring the
          test condition. This is what makes the location-grouped result
          interpretable as shift-induced miscalibration rather than a train/test
          regime mismatch inside the calibration set itself.

    The split is deterministic: same inputs and random_state give identical
    output. Uses fold 0 of the inner KFold, so C is always one inner fold.
    """
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")

    train_arr = np.asarray(train_indices, dtype=np.int64)
    if train_arr.size == 0:
        raise ValueError("train_indices must be non-empty")
    if not np.array_equal(np.sort(train_arr), train_arr):
        raise ValueError("train_indices must be sorted ascending")

    if regime == "random":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fit_rel, cal_rel = next(splitter.split(train_arr))
    else:
        if groups is None:
            raise ValueError("location_grouped requires the group Series")
        g = groups.iloc[train_arr]
        if len(g) != len(train_arr):
            raise ValueError("groups length does not match train_indices length")
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fit_rel, cal_rel = next(
            splitter.split(train_arr, y=np.zeros(len(train_arr)), groups=g.to_numpy())
        )

    fit_idx = sorted(int(i) for i in train_arr[fit_rel])
    cal_idx = sorted(int(i) for i in train_arr[cal_rel])
    return fit_idx, cal_idx


def build_calibration_folds(
    *,
    cal_frac: float = CAL_FRAC,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Derive the full nested fold structure from the frozen V2.1 contract.

    Returns a JSON-serialisable dict. Does NOT write anything: the caller
    decides whether to persist it, so the derivation can be unit-tested without
    filesystem side effects.
    """
    if abs(cal_frac - CAL_FRAC) > 1e-12:
        raise ValueError(
            f"cal_frac is pre-registered as {CAL_FRAC}; refusing {cal_frac}. "
            "Phase 0 fixed a single value and forbade comparing alternatives."
        )
    if random_state != RANDOM_STATE:
        raise ValueError(
            f"random_state is pre-registered as {RANDOM_STATE}; refusing {random_state}."
        )

    frozen_hash = verify_frozen_folds_untouched()
    raw, _dup, _inv = P.build_dataset()
    groups = P.add_group_labels(raw)
    frozen = P.load_folds()

    doc: Dict[str, Any] = {
        "experiment": "MILLOW Experiment 2 - conformal prediction",
        "phase": "Phase 2 - nested calibration fold derivation",
        "protocol": "V2.1 (frozen, read-only)",
        "parent_folds": str(P.FOLDS_PATH.relative_to(P.ROOT)).replace("\\", "/"),
        "parent_folds_sha256": frozen_hash,
        "folds_regenerated": False,
        "frozen_fold_contract_modified": False,
        "cal_frac": cal_frac,
        "random_state": random_state,
        "inner_splitter": {
            "random": "KFold(shuffle=True)",
            "location_grouped": "GroupKFold(shuffle=True)",
        },
        # P.add_group_labels() returns an unnamed Series, so its .name is None.
        # Record the frozen V2.1 group key explicitly instead.
        "group_column": GROUP_COLUMN,
        "group_key_expression": "source_city.astype(str) + '__' + location.astype(str)",
        "dataset": {
            "path": str(P.DATA_PATH.relative_to(P.ROOT)).replace("\\", "/"),
            "rows": int(len(raw)),
        },
        "regimes": {},
    }

    for regime in REGIMES:
        fs: P.FoldSet = frozen[regime]
        regime_doc: Dict[str, Any] = {
            "splitter": "derived from frozen " + str(fs.splitter),
            "n_outer_splits": len(fs.train),
            # FoldSet.digest is a METHOD, not an attribute, so it must be called.
            # It reproduces the digest recorded in cv_folds_v2_1.json, which lets
            # this document prove it consumed the frozen structure unmodified.
            "outer_fold_digest": fs.digest(),
            "folds": [],
        }
        for k, (tr, va) in enumerate(zip(fs.train, fs.valid), start=1):
            fit_idx, cal_idx = derive_inner_split(
                tr,
                regime=regime,
                cal_frac=cal_frac,
                random_state=random_state,
                groups=groups,
            )
            entry = {
                "outer_fold": k,
                "n_frozen_train": len(tr),
                "n_frozen_valid": len(va),
                "n_fit": len(fit_idx),
                "n_calibrate": len(cal_idx),
                "n_test": len(va),
                "fit": fit_idx,
                "calibrate": cal_idx,
                "test": list(va),
                "digests": {
                    "frozen_train": _digest_indices(list(tr)),
                    "fit": _digest_indices(fit_idx),
                    "calibrate": _digest_indices(cal_idx),
                    "test": _digest_indices(list(va)),
                },
            }
            regime_doc["folds"].append(entry)
        doc["regimes"][regime] = regime_doc

    # All invariants are verified on the finished structure before returning.
    verify_all_invariants(doc, frozen, groups)
    return doc


def verify_all_invariants(
    doc: Dict[str, Any],
    frozen: Dict[str, P.FoldSet],
    groups: pd.Series,
) -> None:
    """Assert every fold invariant from §5.1, §7 and the Phase 0 approval.

    Raises on the first violation. Called before any write, and independently
    callable as a standalone verification of a written file.
    """
    for regime in REGIMES:
        folds = doc["regimes"][regime]["folds"]
        fs = frozen[regime]

        if len(folds) != len(fs.train):
            raise RuntimeError(
                f"{regime}: expected {len(fs.train)} folds, derived {len(folds)}"
            )

        for entry, (tr, va) in zip(folds, zip(fs.train, fs.valid)):
            k = entry["outer_fold"]
            fit = set(entry["fit"])
            cal = set(entry["calibrate"])
            test = set(entry["test"])
            tr_set = set(tr)

            # L1 / L2 / L3 / L4 / L5
            if fit & cal:
                raise RuntimeError(f"{regime} fold {k}: fit ∩ calibrate is not empty (L3)")
            if fit & test:
                raise RuntimeError(f"{regime} fold {k}: fit ∩ test is not empty (L2)")
            if cal & test:
                raise RuntimeError(f"{regime} fold {k}: calibrate ∩ test is not empty (L1)")
            if fit | cal != tr_set:
                raise RuntimeError(
                    f"{regime} fold {k}: fit ∪ calibrate != frozen train (L4)"
                )
            if entry["test"] != list(va):
                raise RuntimeError(
                    f"{regime} fold {k}: test is not bit-identical to frozen valid (L5)"
                )
            if entry["fit"] != sorted(entry["fit"]):
                raise RuntimeError(f"{regime} fold {k}: fit indices not sorted")
            if entry["calibrate"] != sorted(entry["calibrate"]):
                raise RuntimeError(f"{regime} fold {k}: calibrate indices not sorted")

            if regime == "location_grouped":
                g_fit = set(groups.iloc[sorted(fit)].astype(str))
                g_cal = set(groups.iloc[sorted(cal)].astype(str))
                g_test = set(groups.iloc[sorted(test)].astype(str))
                # L9
                if g_fit & g_cal:
                    raise RuntimeError(
                        f"{regime} fold {k}: groups(fit) ∩ groups(calibrate) "
                        f"is not empty, {len(g_fit & g_cal)} shared (L9)"
                    )
                # L10
                if g_fit & g_test:
                    raise RuntimeError(
                        f"{regime} fold {k}: groups(fit) ∩ groups(test) "
                        f"is not empty (L10)"
                    )
                # additional: calibration must mirror the test condition
                if g_cal & g_test:
                    raise RuntimeError(
                        f"{regime} fold {k}: groups(calibrate) ∩ groups(test) "
                        f"is not empty (required by §7)"
                    )


def write_calibration_folds(
    doc: Dict[str, Any],
    path: Path = CALIBRATION_FOLDS_JSON,
) -> Path:
    """Persist the derived folds, after re-verifying the frozen contract (L12)."""
    verify_frozen_folds_untouched()
    assert_safe_write_target(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=False)
        fh.write("\n")
    # Final check: derivation must not have disturbed the frozen file.
    verify_frozen_folds_untouched()
    return path


def summarize(doc: Dict[str, Any], groups: pd.Series) -> pd.DataFrame:
    """Human-readable summary of the derived structure, for the audit report."""
    rows: List[Dict[str, Any]] = []
    for regime in REGIMES:
        for entry in doc["regimes"][regime]["folds"]:
            k = entry["outer_fold"]
            row: Dict[str, Any] = {
                "regime": regime,
                "outer_fold": k,
                "n_frozen_train": entry["n_frozen_train"],
                "n_frozen_valid": entry["n_frozen_valid"],
                "n_fit": entry["n_fit"],
                "n_calibrate": entry["n_calibrate"],
                "n_test": entry["n_test"],
                "cal_frac_realised": entry["n_calibrate"] / entry["n_frozen_train"],
            }
            if regime == "location_grouped":
                row["groups_fit"] = int(groups.iloc[entry["fit"]].nunique())
                row["groups_calibrate"] = int(groups.iloc[entry["calibrate"]].nunique())
                row["groups_test"] = int(groups.iloc[entry["test"]].nunique())
            else:
                row["groups_fit"] = None
                row["groups_calibrate"] = None
                row["groups_test"] = None
            rows.append(row)
    return pd.DataFrame(rows)


def quantile_estimability(doc: Dict[str, Any], levels=(0.80, 0.90, 0.95)) -> pd.DataFrame:
    """Confirm every cell can support an exact conformal quantile.

    The exact rule needs k = ceil((n+1)(1-alpha)) <= n. Reported rather than
    assumed, so an unestimable cell could never pass silently.
    """
    rows: List[Dict[str, Any]] = []
    for regime in REGIMES:
        for entry in doc["regimes"][regime]["folds"]:
            n_cal = entry["n_calibrate"]
            for nominal in levels:
                k = int(np.ceil((n_cal + 1) * nominal))
                rows.append(
                    {
                        "regime": regime,
                        "outer_fold": entry["outer_fold"],
                        "nominal": nominal,
                        "n_calibrate": n_cal,
                        "k": k,
                        "k_le_n": bool(k <= n_cal),
                        "quantile_resolution_1_over_n": 1.0 / n_cal,
                    }
                )
    return pd.DataFrame(rows)