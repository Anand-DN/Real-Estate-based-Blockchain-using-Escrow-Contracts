"""Frozen constants and integrity identifiers for Experiment 3.

Every value here is transcribed directly from the frozen protocol
(``v1.5.2-experiment3-protocol-final``) and implementation specification
(``v1.5.3-experiment3-implementation-spec``). This module introduces no
scientific choice and performs no training.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List

from scripts.valuation_v2_1 import protocol as v21

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

E3_DIR = ROOT / "artifacts" / "valuation" / "v2_3"

DATASET_PATH = v21.DATA_PATH
FOLDS_PATH = v21.FOLDS_PATH
PROTOCOL_V2_1_JSON = v21.V2_1_DIR / "protocol_v2_1.json"
PER_FOLD_METRICS_CSV = (
    v21.V2_1_DIR / "model_comparison" / "per_fold_metrics.csv"
)

# ---------------------------------------------------------------------------
# Frozen integrity identifiers (spec Section 12)
# ---------------------------------------------------------------------------

DATASET_SHA256 = (
    "1be10465221668ffde0fe7392efbc183ef77a654549a2724b26439978fad8877"
)
# The spec's protected "V2.1 dataset and folds" digest identifies the persisted
# fold file (Experiment 2 protocol line 365; Experiment 3 protocol line 757).
FOLDS_JSON_SHA256 = (
    "989b79656b051b3b6c8c583b4170a5aaa1c31676260772247bbc98d1eb095156"
)
CONFIG_JSON_SHA256 = (
    "e0b6c7d3c1eb56cad5d4d43d2e958b724cf7c74018e6c693139bec9d82610c33"
)

FOLD_DIGESTS: Dict[str, str] = {
    "random": (
        "1a9f896345492d60b4a8cc677a090b27c8affd745ef302f9fbbf4ea55afc415f"
    ),
    "location_grouped": (
        "4c1c331cc993ddedb58b0a177e118912eb657613e3c9e4f8b5ae2bc7963ec611"
    ),
}

# Frozen states that must be byte-identical before and after any run.
PROTECTED_STATE_FILES = [
    ROOT / ".chain" / "state.json",
    ROOT / "data" / "processed" / "millow_token_map.csv",
    ROOT / "src" / "config.json",
    DATASET_PATH,
    FOLDS_PATH,
    PROTOCOL_V2_1_JSON,
]

# ---------------------------------------------------------------------------
# Dataset / fold invariants
# ---------------------------------------------------------------------------

EXPECTED_ORIGINAL_ROWS = 29135
EXPECTED_DUPLICATE_ROWS_REMOVED = 737
EXPECTED_ROWS = 28398
EXPECTED_FOLDS = 5
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------

INPUT_FEATURES: List[str] = list(v21.INPUT_FEATURES)
FINAL_FEATURES: List[str] = list(v21.FINAL_FEATURES)
NUMERIC_FEATURES: List[str] = list(v21.NUMERIC_FEATURES)
ONEHOT_FEATURES: List[str] = list(v21.ONEHOT_FEATURES)
CATEGORICAL_FEATURES: List[str] = list(v21.CATEGORICAL_FEATURES)
AMENITIES: List[str] = list(v21.AMENITIES)

EXPECTED_INPUT_FEATURE_COUNT = 46
EXPECTED_FINAL_FEATURE_COUNT = 48
EXPECTED_NUMERIC_COUNT = 11
EXPECTED_ONEHOT_COUNT = 37

PROHIBITED_TARGET_DERIVED = [
    "price",
    "log1p(price)",
    "derived_price_per_sqft",
    "group",
]
EXCLUDED_RAW_COLUMNS = ["mreid_id", "source_file", "derived_price_per_sqft"]

# ---------------------------------------------------------------------------
# E4 similarity space (spec Section 4.2)
# ---------------------------------------------------------------------------

# Every frozen V2.1 feature except location and source_city (target-derived
# quantities are not features at all and are already absent).
E4_COORDS: List[str] = [
    c for c in FINAL_FEATURES if c not in CATEGORICAL_FEATURES
]

E4_NUMERIC: List[str] = [c for c in E4_COORDS if c in NUMERIC_FEATURES]
E4_AMENITIES: List[str] = [c for c in E4_COORDS if c in AMENITIES]

EXPECTED_E4_COORD_COUNT = 46
EXPECTED_E4_NUMERIC_COUNT = 11
EXPECTED_E4_AMENITY_COUNT = 35

# ---------------------------------------------------------------------------
# Graph contract (spec Sections 3-4)
# ---------------------------------------------------------------------------

K_NEIGHBORS = 5
DIRECTED = True
DISTANCE = "euclidean"
TIE_BREAK = "ascending_positional_node_id"
EDGE_WEIGHT = 1
ZERO_DISTANCE_RETAINED = True
PRIMARY_CONDITION = "A3"

# ---------------------------------------------------------------------------
# Frozen model architecture (spec Sections 7 and 14, B1/B2)
# ---------------------------------------------------------------------------

ARCHITECTURE: Dict[str, Any] = {
    "model_family": "GraphSAGE",
    "num_layers": 2,
    "hidden_dim": 64,
    "aggregation": "mean",
    "activation": "ReLU",
    "dropout": 0.20,
    "residual_skip": False,
    "output": "scalar",
    "output_space": "log1p(price)",
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "max_epochs": 200,
    "early_stopping": "train-side-validation-loss",
    "patience": 20,
    "loss": "MSE",
    "seed": RANDOM_STATE,
    "gcn_symmetric_normalisation": False,
    "architecture_search": False,
}

# Inner train'-validation split used ONLY for early stopping. It is a
# deterministic partition of train_k; the specification fixes the architecture,
# so this split performs no model selection.
INNER_VALIDATION_FRACTION = 0.10
INNER_VALIDATION_SEED = RANDOM_STATE

# ---------------------------------------------------------------------------
# Conditions (spec Section 9)
# ---------------------------------------------------------------------------

CONDITIONS = ["A1", "A2", "A3"]

CONDITION_DESCRIPTIONS = {
    "A0": "Frozen Experiment 1 CatBoost baseline (read-only, not retrained)",
    "A1": "Relationship-free GraphSAGE control (self/root only, no edges)",
    "A2": "Locality/city GraphSAGE (E1/E2/E5 via O(N) hubs)",
    "A3": "E4 property-similarity GraphSAGE (PRIMARY, directed k=5)",
}

REGIMES = ["location_grouped", "random"]
PRIMARY_REGIME = "location_grouped"

# ---------------------------------------------------------------------------
# Computational budget (spec Section 11) — analytical gate already satisfied
# ---------------------------------------------------------------------------

ANALYTICAL_BUDGET_COMPLETE = True
ANALYTICAL_BUDGET = {
    "graph_nodes_full": EXPECTED_ROWS,
    "e4_directed_edges_full_estimate": 5 * EXPECTED_ROWS,
    "model_fits_primary": 5,
    "model_fits_a1_a3": 15,
    "model_fits_incl_random_secondary": 30,
    "a0_fits": 0,
    "architecture_search": False,
    "cpu_feasible": True,
}

# Frozen reference metrics for A0 (spec Section 8), re-read from Exp-1 files
# at evaluation time; recorded here only for the preflight consistency check.
A0_REFERENCE = {
    "random": {
        "R2_log": 0.363993,
        "MAE_log": 0.465792,
        "MedAPE_percent": 32.524296,
        "R2_log_sd": 0.010551,
        "MAE_log_sd": 0.007833,
        "MedAPE_percent_sd": 0.743129,
    },
    "location_grouped": {
        "R2_log": 0.267686,
        "MAE_log": 0.512773,
        "MedAPE_percent": 37.293596,
        "R2_log_sd": 0.037828,
        "MAE_log_sd": 0.024097,
        "MedAPE_percent_sd": 2.312190,
    },
}


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def protected_state_hashes() -> Dict[str, str]:
    """Hash every protected-state file (missing files recorded as '')."""
    out: Dict[str, str] = {}
    for path in PROTECTED_STATE_FILES:
        rel = str(path.relative_to(ROOT)) if path.exists() else str(path)
        out[rel] = sha256_file(path) if path.exists() else ""
    return out
