"""Experiment 3 execution driver (Phases 2-7).

Usage:
    python -m scripts.experiment_3.run_experiment train --regime location_grouped
    python -m scripts.experiment_3.run_experiment train --regime random
    python -m scripts.experiment_3.run_experiment finalize

Training is full and real: no pilot, timing-only, dry-run or unscored run.
Per-fold results are persisted incrementally so a long run can resume
deterministically. Nothing is committed, tagged or pushed.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from scripts.valuation_v2_1 import protocol as v21
from scripts.experiment_3 import data, frozen, graphs, metrics, train
from scripts.experiment_3.model import architecture_manifest, count_parameters


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(str(type(obj)))


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    return path


def _run_dir() -> Path:
    return frozen.E3_DIR / "runs"


def _run_path(regime: str, fold: int, condition: str) -> Path:
    return _run_dir() / f"{regime}_fold{fold}_{condition}.json"


def _prediction_path(regime: str, fold: int, condition: str) -> Path:
    return frozen.E3_DIR / "predictions" / regime / f"fold{fold}_{condition}.csv"


def write_model_configuration() -> Path:
    payload = {
        "architecture": architecture_manifest(),
        "k_neighbors": frozen.K_NEIGHBORS,
        "directed": frozen.DIRECTED,
        "distance": frozen.DISTANCE,
        "tie_break": frozen.TIE_BREAK,
        "edge_weight": frozen.EDGE_WEIGHT,
        "conditions": frozen.CONDITIONS,
        "primary_condition": frozen.PRIMARY_CONDITION,
        "primary_regime": frozen.PRIMARY_REGIME,
        "seed": frozen.RANDOM_STATE,
        "inner_validation_fraction": frozen.INNER_VALIDATION_FRACTION,
        "architecture_search": False,
    }
    path = _write_json(frozen.E3_DIR / "model_configuration.json", payload)
    return path


def capture_protected_state(stage: str) -> Path:
    path = frozen.E3_DIR / f"protected_state_{stage}.json"
    _write_json(path, frozen.protected_state_hashes())
    return path


def train_regime(regime: str, conditions: List[str]) -> None:
    dataset, _, _ = data.load_dataset()
    folds = v21.load_folds()
    fs = folds[regime]

    for fold in range(fs.n_folds()):
        for condition in conditions:
            out = _run_path(regime, fold, condition)
            if out.exists():
                print(f"[skip] {regime} fold{fold} {condition} (already done)")
                continue

            tr = np.asarray(fs.train[fold], dtype=np.int64)
            te = np.asarray(fs.valid[fold], dtype=np.int64)

            enc = data.fit_fold(dataset, tr, te)
            fit_idx, val_idx = train.inner_split(tr)

            graph = graphs.build_graph(condition, enc, enc.node_dim)
            result = train.train_graphsage(
                x_all=graph.x_all,
                train_edge_index=graph.train_edge_index,
                infer_edge_index=graph.infer_edge_index,
                y_log=enc.y_log,
                fit_idx=fit_idx,
                val_idx=val_idx,
                test_idx=te,
                seed=frozen.RANDOM_STATE,
            )

            price_true = enc.price[te]
            fold_metrics = metrics.compute_fold_metrics(
                price_true, result.predictions_log
            )

            pred_path = _prediction_path(regime, fold, condition)
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "node_index": te,
                    "price_true": price_true,
                    "log_price_true": enc.y_log[te],
                    "prediction_log": result.predictions_log,
                }
            ).to_csv(pred_path, index=False)

            payload = {
                "condition": condition,
                "regime": regime,
                "fold": fold,
                "train_rows": int(len(tr)),
                "test_rows": int(len(te)),
                "node_dim": result.node_dim,
                "e4_dim": enc.e4_dim,
                "parameters": result.parameters,
                "epochs_run": result.epochs_run,
                "best_epoch": result.best_epoch,
                "best_val_loss": result.best_val_loss,
                "train_seconds": result.train_seconds,
                "n_train_edges": result.n_train_edges,
                "n_inference_edges": result.n_inference_edges,
                "metrics": fold_metrics,
                "graph_stats": graph.stats,
                "graph_manifest": graph.manifest,
                "prediction_file": str(
                    pred_path.relative_to(frozen.ROOT)
                ),
                "history": result.history,
                "final_train_loss": result.history[-1]["train_loss"],
                "model_configuration_sha256": frozen.sha256_file(
                    frozen.E3_DIR / "model_configuration.json"
                ),
            }
            _write_json(out, payload)
            print(
                f"[done] {regime} fold{fold} {condition}  "
                f"R2_log={fold_metrics['R2_log']:.4f}  "
                f"MAE_log={fold_metrics['MAE_log']:.4f}  "
                f"{result.train_seconds:.1f}s  epochs={result.epochs_run}"
            )


def _load_runs() -> List[Dict]:
    out = []
    for path in sorted(_run_dir().glob("*.json")):
        with open(path, "r", encoding="utf-8") as fh:
            out.append(json.load(fh))
    return out


def finalize() -> None:
    runs = _load_runs()
    if not runs:
        raise SystemExit("No completed runs found. Train first.")
    capture_protected_state("after")

    metric_keys = metrics.METRIC_KEYS
    per_fold_rows = []
    for r in runs:
        row = {
            "condition": r["condition"],
            "regime": r["regime"],
            "fold": r["fold"],
            "train_rows": r["train_rows"],
            "test_rows": r["test_rows"],
            "node_dim": r["node_dim"],
            "parameters": r["parameters"],
            "epochs_run": r["epochs_run"],
            "best_epoch": r["best_epoch"],
            "train_seconds": r["train_seconds"],
            "n_train_edges": r["n_train_edges"],
            "n_inference_edges": r["n_inference_edges"],
        }
        row.update(r["metrics"])
        per_fold_rows.append(row)
    per_fold = pd.DataFrame(per_fold_rows).sort_values(
        ["regime", "condition", "fold"]
    )
    per_fold.to_csv(frozen.E3_DIR / "per_fold_metrics.csv", index=False)

    a0 = metrics.load_a0_catboost_per_fold()
    a0_rows = []
    for regime, rows in a0.items():
        for i, m in enumerate(rows):
            row = {"model": "A0", "regime": regime, "fold": i}
            row.update(m)
            a0_rows.append(row)
    pd.DataFrame(a0_rows).to_csv(
        frozen.E3_DIR / "a0_reference_per_fold.csv", index=False
    )

    aggregate: Dict[str, Dict] = {}
    for regime in frozen.REGIMES:
        for condition in frozen.CONDITIONS:
            block = per_fold[
                (per_fold["regime"] == regime)
                & (per_fold["condition"] == condition)
            ].sort_values("fold")
            if block.empty:
                continue
            rows = [dict(r) for _, r in block.iterrows()]
            agg = metrics.aggregate(rows)
            agg["n_folds"] = int(len(rows))
            agg["train_seconds_total"] = float(
                np.sum([r["train_seconds"] for r in rows])
            )
            aggregate.setdefault(regime, {})[condition] = agg
        # A0 reference aggregate
        if regime in a0:
            aggregate.setdefault(regime, {})["A0"] = metrics.aggregate(a0[regime])
            aggregate[regime]["A0"]["n_folds"] = len(a0[regime])
    _write_json(frozen.E3_DIR / "aggregate_metrics.json", aggregate)

    paired: Dict[str, Dict] = {}
    for regime in frozen.REGIMES:
        if regime not in aggregate:
            continue
        paired[regime] = {}
        for condition in frozen.CONDITIONS:
            if condition not in aggregate[regime]:
                continue
            block = per_fold[
                (per_fold["regime"] == regime)
                & (per_fold["condition"] == condition)
            ].sort_values("fold")
            a_metrics = [dict(r) for _, r in block.iterrows()]
            for other, label in (("A0", "vs_A0"), ("A1", "vs_A1"), ("A2", "vs_A2")):
                if condition == other:
                    continue
                if other == "A0":
                    b_metrics = a0.get(regime, [])
                elif other in aggregate[regime]:
                    bblk = per_fold[
                        (per_fold["regime"] == regime)
                        & (per_fold["condition"] == other)
                    ].sort_values("fold")
                    b_metrics = [dict(r) for _, r in bblk.iterrows()]
                else:
                    continue
                if len(a_metrics) != len(b_metrics):
                    continue
                diffs = metrics.paired_differences(a_metrics, b_metrics)
                paired[regime].setdefault(condition, {})[label] = {
                    "differences": diffs,
                    "mean_R2_log_diff": float(np.mean(diffs["R2_log"])),
                    "direction_consistent_R2_log": bool(
                        all(d > 0 for d in diffs["R2_log"])
                        or all(d < 0 for d in diffs["R2_log"])
                    ),
                }
    _write_json(frozen.E3_DIR / "paired_differences.json", paired)

    summary = {
        "per_fold_file": "per_fold_metrics.csv",
        "aggregate_file": "aggregate_metrics.json",
        "paired_differences_file": "paired_differences.json",
        "regimes": sorted(aggregate.keys()),
        "conditions": sorted({r["condition"] for r in runs}),
        "n_fits": len(runs),
        "total_train_seconds": float(
            np.sum([r["train_seconds"] for r in runs])
        ),
        "a0_source": str(frozen.PER_FOLD_METRICS_CSV.relative_to(frozen.ROOT)),
    }
    _write_json(frozen.E3_DIR / "evaluation_summary.json", summary)
    print(json.dumps(summary, indent=2))


def environment_record() -> Path:
    import torch
    import torch_geometric

    payload = {
        "recorded_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "device": "cpu",
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "note": "Experiment 3 isolated namespace v2_3; no GPU used.",
    }
    return _write_json(frozen.E3_DIR / "environment.json", payload)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Experiment 3 driver")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--regime", required=True, choices=frozen.REGIMES)
    p_train.add_argument(
        "--conditions", default="A1,A2,A3"
    )

    sub.add_parser("finalize")
    sub.add_parser("env")

    args = parser.parse_args(argv)

    if args.command == "env":
        print(environment_record())
        return 0
    if args.command == "finalize":
        finalize()
        return 0

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    write_model_configuration()
    environment_record()
    if not (frozen.E3_DIR / "protected_state_before.json").exists():
        capture_protected_state("before")
    train_regime(args.regime, conditions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
