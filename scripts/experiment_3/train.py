"""Training loop for the frozen GraphSAGE (spec Sections 6-7, 11).

Architecture and hyperparameters are frozen. The inner train'-validation split
is used only for early stopping; it is a deterministic partition of train_k and
selects nothing but the stopping epoch. Per-fold training time is recorded as a
Phase 3 output and is never used to alter the experiment.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
from torch import nn

from scripts.experiment_3 import frozen
from scripts.experiment_3.model import GraphSAGE


@dataclass
class FitResult:
    predictions_log: np.ndarray
    epochs_run: int
    best_epoch: int
    best_val_loss: float
    train_seconds: float
    history: List[Dict[str, float]]
    node_dim: int
    n_train_edges: int
    n_inference_edges: int
    parameters: int = 0
    device: str = "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def inner_split(
    train_idx: np.ndarray,
    frac: float = frozen.INNER_VALIDATION_FRACTION,
    seed: int = frozen.INNER_VALIDATION_SEED,
):
    """Deterministic train'-validation split of train_k (early stopping only)."""
    train_idx = np.asarray(train_idx, dtype=np.int64)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(train_idx))
    n_val = int(round(len(train_idx) * frac))
    val = np.sort(train_idx[perm[:n_val]])
    fit = np.sort(train_idx[perm[n_val:]])
    return fit, val


def train_graphsage(
    x_all: np.ndarray,
    train_edge_index: np.ndarray,
    infer_edge_index: np.ndarray,
    y_log: np.ndarray,
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int = frozen.RANDOM_STATE,
) -> FitResult:
    cfg = frozen.ARCHITECTURE
    torch.set_num_threads(max(1, torch.get_num_threads()))
    seed_everything(seed)

    x = torch.from_numpy(np.ascontiguousarray(x_all, dtype=np.float32))
    ei_train = torch.from_numpy(np.ascontiguousarray(train_edge_index, dtype=np.int64))
    ei_infer = torch.from_numpy(np.ascontiguousarray(infer_edge_index, dtype=np.int64))
    y = torch.from_numpy(np.ascontiguousarray(y_log, dtype=np.float32))
    fit_t = torch.from_numpy(np.ascontiguousarray(fit_idx, dtype=np.int64))
    val_t = torch.from_numpy(np.ascontiguousarray(val_idx, dtype=np.int64))
    test_t = torch.from_numpy(np.ascontiguousarray(test_idx, dtype=np.int64))

    in_dim = int(x.shape[1])
    model = GraphSAGE(in_dim, cfg["hidden_dim"], cfg["dropout"])
    if hasattr(model, "reset_parameters"):
        pass
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    patience = 0
    history: List[Dict[str, float]] = []

    start = time.perf_counter()
    for epoch in range(cfg["max_epochs"]):
        model.train()
        optimizer.zero_grad()
        out = model(x, ei_train)
        loss = loss_fn(out[fit_t], y[fit_t])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_out = model(x, ei_train)
            val_loss = float(loss_fn(eval_out[val_t], y[val_t]).item())
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "val_loss": val_loss,
            }
        )
        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg["patience"]:
                break
    train_seconds = time.perf_counter() - start

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        infer_out = model(x, ei_infer)
        predictions = infer_out[test_t].cpu().numpy().astype(np.float64)

    return FitResult(
        predictions_log=predictions,
        epochs_run=len(history),
        best_epoch=best_epoch,
        best_val_loss=best_val,
        train_seconds=train_seconds,
        history=history,
        node_dim=in_dim,
        n_train_edges=int(train_edge_index.shape[1]),
        n_inference_edges=int(infer_edge_index.shape[1]),
        parameters=int(sum(p.numel() for p in model.parameters())),
    )
