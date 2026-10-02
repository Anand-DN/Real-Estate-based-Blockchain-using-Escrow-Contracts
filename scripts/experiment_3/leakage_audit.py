"""Phase 6 leakage audit (spec Section 13).

Combines the frozen discipline checklist with code-level structural assertions
over freshly rebuilt graphs. No training occurs here.
"""

from __future__ import annotations

import json
from typing import Dict, List

import numpy as np

from scripts.valuation_v2_1 import protocol as v21
from scripts.experiment_3 import data, frozen, graphs


def _structural_checks() -> List[Dict[str, object]]:
    dataset, _, _ = data.load_dataset()
    folds = v21.load_folds()
    fs = folds[frozen.PRIMARY_REGIME]
    fold = 0
    tr = np.asarray(fs.train[fold], dtype=np.int64)
    te = np.asarray(fs.valid[fold], dtype=np.int64)
    enc = data.fit_fold(dataset, tr, te)

    checks: List[Dict[str, object]] = []
    tr_set = set(tr.tolist())
    te_set = set(te.tolist())

    a3 = graphs.build_a3(enc)
    train_ei = a3.train_edge_index
    infer_ei = a3.infer_edge_index
    train_only = all(set(e.tolist()) <= tr_set for e in train_ei)
    test_src = set(infer_ei[0].tolist()) & te_set
    # every edge whose target is a test node must source from a training node
    test_to_train = all(
        (infer_ei[0][i] in tr_set)
        for i in range(infer_ei.shape[1])
        if infer_ei[1][i] in te_set
    )
    test_test = any(
        (infer_ei[0][i] in te_set) and (infer_ei[1][i] in te_set)
        for i in range(infer_ei.shape[1])
    )
    checks.append(
        {
            "check": "A3_train_edges_train_only",
            "passed": train_only,
            "detail": f"{train_ei.shape[1]} train edges",
        }
    )
    checks.append(
        {
            "check": "A3_no_test_test_edges",
            "passed": not test_test,
        }
    )
    checks.append(
        {
            "check": "A3_no_test_source_edges",
            "passed": len(test_src) == 0,
            "detail": f"{len(test_src)} test-source edges",
        }
    )
    checks.append(
        {
            "check": "A3_test_targets_sourced_from_training",
            "passed": bool(test_to_train),
        }
    )
    self_loop = np.any(train_ei[0] == train_ei[1])
    checks.append({"check": "A3_no_self_edges", "passed": not bool(self_loop)})
    dup = len(
        set(
            zip(
                a3.train_edge_index[0].tolist(),
                a3.train_edge_index[1].tolist(),
            )
        )
    ) != a3.train_edge_index.shape[1]
    checks.append({"check": "A3_no_duplicate_edges", "passed": not dup})

    a2 = graphs.build_a2(enc, enc.node_dim)
    a2_src_test = set(a2.train_edge_index[0].tolist()) & te_set
    checks.append(
        {
            "check": "A2_no_test_source_edges_in_training_graph",
            "passed": len(a2_src_test) == 0,
            "detail": f"{len(a2_src_test)} test-source edges",
        }
    )
    # inference edges with a test source must be hub -> test only (no test -> hub)
    bad = 0
    infer_src_test = set(a2.infer_edge_index[0].tolist()) & te_set
    bad += len(infer_src_test)
    checks.append(
        {
            "check": "A2_no_test_source_edges_at_inference",
            "passed": bad == 0,
            "detail": f"{bad}",
        }
    )

    a1 = graphs.build_a1(enc)
    checks.append(
        {"check": "A1_no_edges", "passed": a1.train_edge_index.shape[1] == 0}
    )

    checks.append(
        {
            "check": "E4_excludes_target",
            "passed": not (
                set(frozen.E4_COORDS) & set(frozen.PROHIBITED_TARGET_DERIVED)
            ),
        }
    )
    return checks


FROZEN_CHECKLIST = [
    ("Load dataset; V2.1 clean + dedup", "TRAIN-INDEPENDENT (deterministic, no target)"),
    ("Row-wise feature engineering", "TRAIN-INDEPENDENT (pure row function)"),
    ("location_frequency / log_location_frequency", "TRAIN-ONLY (fitted on train_k)"),
    ("Numeric scaler (V2.1 StandardScaler)", "TRAIN-ONLY"),
    ("One-hot encoder vocabulary", "TRAIN-ONLY"),
    ("E4 kNN index", "TRAIN-ONLY"),
    ("E4 training edges", "TRAIN-ONLY graph"),
    ("E4 test edges", "PREDICTION-TIME (test->training only)"),
    ("E1 same-city edges (A2)", "PREDICTION-TIME AVAILABLE (hub -> test)"),
    ("E2 same-locality edges (A2)", "PREDICTION-TIME AVAILABLE (hub -> test)"),
    ("E5 locality aggregates (A2)", "TRAIN-ONLY (hub member means)"),
    ("Any target/price-based similarity", "PROHIBITED (not present)"),
    ("derived_price_per_sqft", "PROHIBITED (not present)"),
    ("Test labels in graph", "PROHIBITED (not present)"),
    ("Test->test edges (primary)", "PROHIBITED (none constructed)"),
    ("Fitting on test rows", "PROHIBITED (fit on train_k only)"),
    ("Graph defined using test results", "PROHIBITED"),
    ("Selection using location_grouped outer test", "PROHIBITED"),
]


def run_audit() -> Dict[str, object]:
    structural = _structural_checks()
    return {
        "discipline_checklist": [
            {"step": s, "discipline": d} for s, d in FROZEN_CHECKLIST
        ],
        "structural_checks": structural,
        "all_structural_passed": all(c["passed"] for c in structural),
        "n_structural": len(structural),
        "n_failed": sum(1 for c in structural if not c["passed"]),
    }


def main() -> int:
    report = run_audit()
    path = frozen.E3_DIR / "leakage_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    for c in report["structural_checks"]:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['check']}")
    print(
        f"\nLEAKAGE AUDIT: "
        f"{'PASS' if report['all_structural_passed'] else 'FAIL'} "
        f"({report['n_structural'] - report['n_failed']}/{report['n_structural']})"
    )
    print(f"report: {path}")
    return 0 if report["all_structural_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
