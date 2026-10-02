"""Phase 1 preflight: static checks against the frozen Experiment 3 contract.

Every check is a pure assertion over frozen inputs; no graph is built and no
model is trained. If any check fails the experiment must STOP (spec Section 11
gate; protocol fail-fast). The report is written to the Experiment 3 artifact
namespace and printed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np

from scripts.valuation_v2_1 import protocol as v21
from scripts.experiment_3 import frozen


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class PreflightReport:
    checks: List[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def n_checks(self) -> int:
        return len(self.checks)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def add(self, name: str, condition: bool, detail: str = "") -> None:
        self.checks.append(Check(name=name, passed=bool(condition), detail=detail))

    def as_dict(self) -> Dict[str, object]:
        return {
            "passed": self.passed,
            "n_checks": self.n_checks,
            "n_failed": self.n_failed,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in self.checks
            ],
        }


def _safe(fn: Callable[[], None], report: PreflightReport, name: str) -> None:
    try:
        fn()
    except AssertionError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        report.add(name, False, f"exception: {exc!r}")


def run_preflight(dataset=None, folds=None) -> PreflightReport:
    report = PreflightReport()

    # --- dataset -----------------------------------------------------------
    report.add(
        "dataset.exists",
        frozen.DATASET_PATH.exists(),
        str(frozen.DATASET_PATH),
    )
    if frozen.DATASET_PATH.exists():
        report.add(
            "dataset.sha256",
            frozen.sha256_file(frozen.DATASET_PATH) == frozen.DATASET_SHA256,
            frozen.DATASET_SHA256,
        )

    if dataset is None:
        dataset, dup_report, invalid_removed = v21.build_dataset()
    else:
        dataset, dup_report, invalid_removed = dataset

    report.add("dataset.rows", len(dataset) == frozen.EXPECTED_ROWS, f"{len(dataset)}")
    report.add(
        "dataset.dedup_original",
        dup_report.original_rows == frozen.EXPECTED_ORIGINAL_ROWS,
        f"{dup_report.original_rows}",
    )
    report.add(
        "dataset.dedup_removed",
        dup_report.duplicate_rows_removed == frozen.EXPECTED_DUPLICATE_ROWS_REMOVED,
        f"{dup_report.duplicate_rows_removed}",
    )
    report.add("dataset.invalid_removed", invalid_removed == 0, f"{invalid_removed}")

    # --- feature contract --------------------------------------------------
    report.add(
        "features.input_count",
        len(frozen.INPUT_FEATURES) == frozen.EXPECTED_INPUT_FEATURE_COUNT,
        f"{len(frozen.INPUT_FEATURES)}",
    )
    report.add(
        "features.final_count",
        len(frozen.FINAL_FEATURES) == frozen.EXPECTED_FINAL_FEATURE_COUNT,
        f"{len(frozen.FINAL_FEATURES)}",
    )
    report.add(
        "features.numeric_count",
        len(frozen.NUMERIC_FEATURES) == frozen.EXPECTED_NUMERIC_COUNT,
        f"{len(frozen.NUMERIC_FEATURES)}",
    )
    report.add(
        "features.onehot_count",
        len(frozen.ONEHOT_FEATURES) == frozen.EXPECTED_ONEHOT_COUNT,
        f"{len(frozen.ONEHOT_FEATURES)}",
    )
    report.add(
        "features.input_order_matches",
        frozen.INPUT_FEATURES == list(v21.INPUT_FEATURES),
    )
    report.add(
        "features.no_target_derived",
        not (set(frozen.INPUT_FEATURES) & set(frozen.PROHIBITED_TARGET_DERIVED)),
    )
    report.add(
        "features.no_excluded_raw",
        not (set(frozen.INPUT_FEATURES) & set(frozen.EXCLUDED_RAW_COLUMNS)),
    )

    # --- E4 coordinate space ----------------------------------------------
    report.add(
        "e4.coord_count",
        len(frozen.E4_COORDS) == frozen.EXPECTED_E4_COORD_COUNT,
        f"{len(frozen.E4_COORDS)}",
    )
    report.add(
        "e4.numeric_count",
        len(frozen.E4_NUMERIC) == frozen.EXPECTED_E4_NUMERIC_COUNT,
        f"{len(frozen.E4_NUMERIC)}",
    )
    report.add(
        "e4.amenity_count",
        len(frozen.E4_AMENITIES) == frozen.EXPECTED_E4_AMENITY_COUNT,
        f"{len(frozen.E4_AMENITIES)}",
    )
    report.add(
        "e4.excludes_location_city",
        not (set(frozen.E4_COORDS) & set(frozen.CATEGORICAL_FEATURES)),
    )
    report.add(
        "e4.no_target_derived",
        not (set(frozen.E4_COORDS) & set(frozen.PROHIBITED_TARGET_DERIVED)),
    )
    report.add(
        "e4.covers_all_nontarget_noncategorical",
        set(frozen.E4_COORDS)
        == (set(frozen.FINAL_FEATURES) - set(frozen.CATEGORICAL_FEATURES)),
    )

    # --- folds -------------------------------------------------------------
    report.add("folds.exists", frozen.FOLDS_PATH.exists(), str(frozen.FOLDS_PATH))
    if folds is None:
        folds = v21.load_folds()
    report.add(
        "folds.n_regimes",
        set(folds.keys()) == set(frozen.REGIMES),
        str(sorted(folds.keys())),
    )
    for regime in frozen.REGIMES:
        fs = folds.get(regime)
        if fs is None:
            report.add(f"folds.{regime}.present", False)
            continue
        report.add(
            f"folds.{regime}.digest",
            fs.digest() == frozen.FOLD_DIGESTS[regime],
            fs.digest(),
        )
        report.add(
            f"folds.{regime}.n_folds",
            fs.n_folds() == frozen.EXPECTED_FOLDS,
            f"{fs.n_folds()}",
        )
        disjoint = all(
            not (set(tr) & set(va)) for tr, va in zip(fs.train, fs.valid)
        )
        report.add(f"folds.{regime}.disjoint", disjoint)
        coverage = all(
            len(set(tr) | set(va)) == len(dataset)
            for tr, va in zip(fs.train, fs.valid)
        )
        report.add(f"folds.{regime}.coverage", coverage)
        if fs.group_column is not None:
            overlap = fs.group_overlap(dataset[fs.group_column].to_numpy())
            report.add(
                f"folds.{regime}.group_isolation",
                max(overlap) == 0,
                str(overlap),
            )

    # --- frozen V2.1 protocol json ----------------------------------------
    report.add(
        "folds.json_sha256",
        frozen.FOLDS_PATH.exists()
        and frozen.sha256_file(frozen.FOLDS_PATH) == frozen.FOLDS_JSON_SHA256,
        frozen.FOLDS_JSON_SHA256,
    )
    report.add(
        "protocol_v2_1.exists",
        frozen.PROTOCOL_V2_1_JSON.exists(),
        str(frozen.PROTOCOL_V2_1_JSON),
    )
    if frozen.PROTOCOL_V2_1_JSON.exists():
        with open(frozen.PROTOCOL_V2_1_JSON, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        report.add(
            "protocol_v2_1.input_columns_match",
            list(payload["features"]["pipeline_input_columns"])
            == list(frozen.INPUT_FEATURES),
        )
        report.add(
            "protocol_v2_1.numeric_match",
            list(payload["features"]["numeric_scaled"]) == list(frozen.NUMERIC_FEATURES),
        )
        report.add(
            "protocol_v2_1.onehot_match",
            list(payload["features"]["one_hot"]) == list(frozen.ONEHOT_FEATURES),
        )

    # --- architecture / budget gate ---------------------------------------
    arch = frozen.ARCHITECTURE
    report.add(
        "arch.matches_spec",
        (
            arch["model_family"] == "GraphSAGE"
            and arch["num_layers"] == 2
            and arch["hidden_dim"] == 64
            and arch["aggregation"] == "mean"
            and arch["activation"] == "ReLU"
            and abs(arch["dropout"] - 0.20) < 1e-12
            and arch["residual_skip"] is False
            and arch["output"] == "scalar"
            and arch["optimizer"] == "Adam"
            and abs(arch["learning_rate"] - 0.001) < 1e-12
            and abs(arch["weight_decay"] - 1e-4) < 1e-15
            and arch["max_epochs"] == 200
            and arch["patience"] == 20
            and arch["loss"] == "MSE"
            and arch["seed"] == 42
            and arch["gcn_symmetric_normalisation"] is False
            and arch["architecture_search"] is False
        ),
    )
    report.add(
        "budget.analytical_complete",
        frozen.ANALYTICAL_BUDGET_COMPLETE is True,
    )
    report.add(
        "budget.no_architecture_search",
        arch["architecture_search"] is False
        and frozen.ANALYTICAL_BUDGET["architecture_search"] is False,
    )

    # --- graph contract ----------------------------------------------------
    report.add(
        "graph.contract",
        frozen.K_NEIGHBORS == 5
        and frozen.DIRECTED is True
        and frozen.DISTANCE == "euclidean"
        and frozen.TIE_BREAK == "ascending_positional_node_id"
        and frozen.EDGE_WEIGHT == 1
        and frozen.ZERO_DISTANCE_RETAINED is True,
    )
    report.add(
        "ablation.a1_neq_a3",
        frozen.CONDITION_DESCRIPTIONS["A1"] != frozen.CONDITION_DESCRIPTIONS["A3"]
        and set(frozen.CONDITIONS) == {"A1", "A2", "A3"},
    )

    # --- protected state ---------------------------------------------------
    if (frozen.ROOT / "src" / "config.json").exists():
        report.add(
            "protected.config_sha256",
            frozen.sha256_file(frozen.ROOT / "src" / "config.json")
            == frozen.CONFIG_JSON_SHA256,
        )
    else:
        report.add("protected.config_sha256", False, "src/config.json missing")
    report.add(
        "protected.token_map_present",
        (frozen.ROOT / "data" / "processed" / "millow_token_map.csv").exists(),
    )
    report.add(
        "protected.chain_state_present",
        (frozen.ROOT / ".chain" / "state.json").exists(),
    )

    return report


def main() -> int:
    report = run_preflight()
    out_dir = frozen.E3_DIR / "preflight"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "preflight_report.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report.as_dict(), fh, indent=2)

    for c in report.checks:
        status = "PASS" if c.passed else "FAIL"
        print(f"[{status}] {c.name}" + (f"  ({c.detail})" if c.detail else ""))
    print(
        f"\nPREFLIGHT: {'PASS' if report.passed else 'FAIL'} "
        f"({report.n_checks - report.n_failed}/{report.n_checks})"
    )
    print(f"report: {out_path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
