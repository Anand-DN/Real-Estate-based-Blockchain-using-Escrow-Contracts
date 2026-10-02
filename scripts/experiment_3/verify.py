"""Phase 7 reproducibility + protected-state verification.

Recomputes every fold metric from the persisted predictions, checks the
aggregate, hashes every Experiment 3 artifact, and confirms the protected
state is unchanged. No training occurs here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from scripts.valuation_v2_1 import protocol as v21
from scripts.experiment_3 import frozen, metrics


def _hash_tree(root: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not root.exists():
        return out
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_hashes.json":
            out[str(path.relative_to(frozen.ROOT))] = frozen.sha256_file(path)
    return out


def verify() -> Dict[str, object]:
    report: Dict[str, object] = {}

    per_fold = pd.read_csv(frozen.E3_DIR / "per_fold_metrics.csv")
    recomputed: List[Dict] = []
    max_delta = {k: 0.0 for k in metrics.METRIC_KEYS}
    for _, row in per_fold.iterrows():
        regime = row["regime"]
        fold = int(row["fold"])
        condition = row["condition"]
        pred_path = (
            frozen.E3_DIR
            / "predictions"
            / regime
            / f"fold{fold}_{condition}.csv"
        )
        pred = pd.read_csv(pred_path)
        got = v21.compute_metrics(
            pred["price_true"].to_numpy(), pred["prediction_log"].to_numpy()
        )
        for k in metrics.METRIC_KEYS:
            delta = abs(got[k] - float(row[k]))
            max_delta[k] = max(max_delta[k], delta)
        recomputed.append(
            {
                "regime": regime,
                "fold": fold,
                "condition": condition,
                "match": all(
                    np.isclose(
                        got[k], float(row[k]), rtol=1e-9, atol=1e-6
                    )
                    for k in metrics.METRIC_KEYS
                ),
            }
        )
    report["predictions_recompute"] = {
        "n_folds": len(recomputed),
        "all_match": all(r["match"] for r in recomputed),
        "max_abs_delta": max_delta,
    }

    # Aggregate consistency
    agg = json.load(open(frozen.E3_DIR / "aggregate_metrics.json", encoding="utf-8"))
    agg_ok = True
    details = []
    for regime, conds in agg.items():
        for condition, values in conds.items():
            if condition == "A0":
                continue
            block = per_fold[
                (per_fold["regime"] == regime)
                & (per_fold["condition"] == condition)
            ]
            for idx, key in enumerate(metrics.METRIC_KEYS):
                expected = float(block[key].mean())
                got = float(values[f"{key}_mean"])
                if not np.isclose(expected, got, rtol=1e-9, atol=1e-6):
                    agg_ok = False
                    details.append(f"{regime}/{condition}/{key}")
    report["aggregate_consistency"] = {"ok": agg_ok, "mismatches": details}

    # Protected state
    before_path = frozen.E3_DIR / "protected_state_before.json"
    with open(before_path, "r", encoding="utf-8") as fh:
        before = json.load(fh)
    after = frozen.protected_state_hashes()
    changed = {
        k: {"before": before.get(k), "after": after.get(k)}
        for k in after
        if before.get(k) != after.get(k)
    }
    report["protected_state"] = {
        "unchanged": len(changed) == 0,
        "changed": changed,
        "files": sorted(after.keys()),
    }

    # Artifact hashes
    hashes = _hash_tree(frozen.E3_DIR)
    with open(frozen.E3_DIR / "artifact_hashes.json", "w", encoding="utf-8") as fh:
        json.dump(hashes, fh, indent=2)
    report["artifact_hashes"] = {
        "n_artifacts": len(hashes),
        "file": "artifact_hashes.json",
    }

    report["overall_pass"] = bool(
        report["predictions_recompute"]["all_match"]
        and report["aggregate_consistency"]["ok"]
        and report["protected_state"]["unchanged"]
    )
    return report


def main() -> int:
    report = verify()
    path = frozen.E3_DIR / "reproducibility_check.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nREPRODUCIBILITY: {'PASS' if report['overall_pass'] else 'FAIL'}")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
