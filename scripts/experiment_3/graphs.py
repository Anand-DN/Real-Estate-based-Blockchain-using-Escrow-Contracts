"""Graph construction for Experiment 3 (spec Sections 3-5, 9).

A1 : no relational edges (self/root only).
A2 : permitted E1/E2/E5 locality/city relations via O(N) auxiliary hubs.
A3 : directed k=5 E4 property-similarity graph (PRIMARY).

All neighbour selection is fitted on training nodes only. Test nodes attach
to training nodes at prediction time; no test-test edge is ever built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from scripts.experiment_3 import frozen


def knn_indices(
    query: np.ndarray,
    ref: np.ndarray,
    k: int,
    ref_ids: np.ndarray | None = None,
    exclude_self: bool = False,
    block: int = 512,
    margin_pad: int = 256,
) -> np.ndarray:
    """Exact Euclidean k nearest neighbours with ordered tie-breaking.

    Ties in squared distance are broken by ascending positional node id, as
    required by spec Section 4.1 (``TIE_BREAK``). ``exclude_self`` removes the
    diagonal (only valid when ``query`` and ``ref`` are the same frame in the
    same order).

    Returns an integer array of shape (len(query), k) of neighbour ids
    expressed in the ``ref_ids`` space (defaults to ref row positions).
    """
    query = np.asarray(query, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    nq, nr = query.shape[0], ref.shape[0]
    if ref_ids is None:
        ref_ids = np.arange(nr, dtype=np.int64)
    ref_ids = np.asarray(ref_ids, dtype=np.int64)

    max_avail = nr - (1 if exclude_self else 0)
    if k > max_avail:
        raise ValueError(f"k={k} exceeds available neighbours {max_avail}")

    q2 = np.einsum("ij,ij->i", query, query)
    r2 = np.einsum("ij,ij->i", ref, ref)
    out = np.empty((nq, k), dtype=np.int64)

    for start in range(0, nq, block):
        stop = min(start + block, nq)
        qb = query[start:stop]
        d2 = q2[start:stop, None] + r2[None, :] - 2.0 * (qb @ ref.T)
        np.maximum(d2, 0.0, out=d2)
        if exclude_self:
            rows = np.arange(stop - start)
            d2[rows, start + rows] = np.inf

        for r in range(d2.shape[0]):
            row = d2[r]
            m = min(k + margin_pad, nr)
            while True:
                part = np.argpartition(row, m - 1)[:m]
                sub_d = row[part]
                order = np.lexsort((ref_ids[part], sub_d))
                sel = part[order][:k]
                dk = row[sel[-1]]
                n_le = int(np.count_nonzero(row <= dk))
                if n_le <= m or m >= nr:
                    break
                m = min(n_le + margin_pad, nr)
            out[start + r] = ref_ids[sel]
    return out


def _edge_index(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=np.int64).reshape(-1)
    dst = np.asarray(dst, dtype=np.int64).reshape(-1)
    if src.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    return np.vstack([src, dst])


def _dedupe_edges(edge_index: np.ndarray) -> np.ndarray:
    """Simple-graph guarantee: drop duplicate directed edges, keep order."""
    if edge_index.shape[1] == 0:
        return edge_index
    keys = edge_index[0].astype(np.int64) * (1 << 32) + edge_index[1]
    _, keep = np.unique(keys, return_index=True)
    keep = np.sort(keep)
    return edge_index[:, keep]


@dataclass
class GraphBundle:
    condition: str
    x_all: np.ndarray
    train_edge_index: np.ndarray    # edges used for message passing in training
    infer_edge_index: np.ndarray    # training edges + prediction-time test edges
    stats: Dict[str, float] = field(default_factory=dict)
    manifest: Dict[str, object] = field(default_factory=dict)


def build_a1(fold) -> GraphBundle:
    """A1: relationship-free control — no relational edges."""
    ei = np.empty((2, 0), dtype=np.int64)
    return GraphBundle(
        condition="A1",
        x_all=fold.x_all.astype(np.float32),
        train_edge_index=ei,
        infer_edge_index=ei.copy(),
        stats={
            "nodes": int(fold.x_all.shape[0]),
            "train_edges": 0,
            "test_edges": 0,
            "mean_degree": 0.0,
        },
        manifest={"condition": "A1", "edge_type": "none", "directed": False},
    )


def build_a3(fold) -> GraphBundle:
    """A3: directed k=5 E4 property-similarity (PRIMARY)."""
    k = frozen.K_NEIGHBORS
    tr = fold.train_idx
    te = fold.test_idx
    coords = fold.e4_all

    # Message flows neighbour -> node, so each node aggregates ITS k nearest
    # neighbours (PyG edge_index[0] = source, edge_index[1] = target).
    nn_tt = knn_indices(
        coords[tr], coords[tr], k, ref_ids=tr, exclude_self=True
    )
    train_ei = _edge_index(nn_tt.reshape(-1), np.repeat(tr, k))

    # Test nodes aggregate their k nearest TRAINING nodes; no test node is ever
    # a source, so no test feature enters a training node's aggregate.
    nn_te = knn_indices(coords[te], coords[tr], k, ref_ids=tr, exclude_self=False)
    test_ei = _edge_index(nn_te.reshape(-1), np.repeat(te, k))

    return GraphBundle(
        condition="A3",
        x_all=fold.x_all.astype(np.float32),
        train_edge_index=train_ei,
        infer_edge_index=_dedupe_edges(np.hstack([train_ei, test_ei])),
        stats={
            "nodes": int(fold.x_all.shape[0]),
            "train_nodes": int(len(tr)),
            "test_nodes": int(len(te)),
            "train_edges": int(train_ei.shape[1]),
            "test_edges": int(test_ei.shape[1]),
            "mean_degree": float(train_ei.shape[1] / len(tr)),
            "isolates_train": 0,
        },
        manifest={
            "condition": "A3",
            "edge_type": "E4",
            "feature_space": "46 permitted non-target coordinates",
            "distance": frozen.DISTANCE,
            "k": k,
            "directed": True,
            "tie_break": frozen.TIE_BREAK,
            "weight": frozen.EDGE_WEIGHT,
        },
    )


def build_a2(fold, node_dim: int) -> GraphBundle:
    """A2: permitted E1/E2/E5 locality/city relations through O(N) hubs.

    Locality and city hubs are auxiliary nodes (spec Section 1). Hub features
    are TRAINING-ONLY means of member training node features (E5). At inference
    only hub -> test edges are added, so no test feature ever enters a hub
    aggregate.
    """
    tr = fold.train_idx
    te = fold.test_idx
    loc = fold.location
    city = fold.source_city
    x = fold.x_all.astype(np.float32)
    n = x.shape[0]

    tr_loc = loc[tr]
    tr_city = city[tr]
    loc_levels = sorted(set(tr_loc.tolist()))
    city_levels = sorted(set(tr_city.tolist()))
    loc_index = {v: i for i, v in enumerate(loc_levels)}
    city_index = {v: i for i, v in enumerate(city_levels)}
    n_loc, n_city = len(loc_levels), len(city_levels)

    loc_hub = n + np.arange(n_loc, dtype=np.int64)
    city_hub = n + n_loc + np.arange(n_city, dtype=np.int64)

    loc_feat = np.zeros((n_loc, node_dim), dtype=np.float64)
    city_feat = np.zeros((n_city, node_dim), dtype=np.float64)
    loc_count = np.zeros(n_loc, dtype=np.int64)
    city_count = np.zeros(n_city, dtype=np.int64)
    for p in tr:
        li = loc_index[loc[p]]
        ci = city_index[city[p]]
        loc_feat[li] += x[p]
        city_feat[ci] += x[p]
        loc_count[li] += 1
        city_count[ci] += 1
    loc_feat = (loc_feat / np.maximum(loc_count, 1)[:, None]).astype(np.float32)
    city_feat = (city_feat / np.maximum(city_count, 1)[:, None]).astype(np.float32)

    x_combined = np.vstack([x, loc_feat, city_feat]).astype(np.float32)

    pairs = set()
    for p in tr:
        li = int(loc_hub[loc_index[loc[p]]])
        ci = int(city_hub[city_index[city[p]]])
        pairs.add((int(p), li))
        pairs.add((li, int(p)))
        pairs.add((int(p), ci))
        pairs.add((ci, int(p)))
        pairs.add((li, ci))
        pairs.add((ci, li))
    train_ei = np.array(sorted(pairs), dtype=np.int64).T

    infer_pairs = set(pairs)
    for p in te:
        l, c = loc[p], city[p]
        if l in loc_index:
            infer_pairs.add((int(loc_hub[loc_index[l]]), int(p)))
        if c in city_index:
            infer_pairs.add((int(city_hub[city_index[c]]), int(p)))
    infer_ei = np.array(sorted(infer_pairs), dtype=np.int64).T

    return GraphBundle(
        condition="A2",
        x_all=x_combined,
        train_edge_index=train_ei,
        infer_edge_index=infer_ei,
        stats={
            "nodes": int(n),
            "hub_nodes": int(n_loc + n_city),
            "locality_hubs": int(n_loc),
            "city_hubs": int(n_city),
            "train_edges": int(train_ei.shape[1]),
            "inference_edges": int(infer_ei.shape[1]),
        },
        manifest={
            "condition": "A2",
            "edge_types": ["E1", "E2", "E5"],
            "representation": "O(N) auxiliary hubs",
            "hub_features": "training-fold member means (E5)",
            "test_edges": "hub -> test only (no test -> hub)",
            "directed": True,
        },
    )


def build_graph(condition: str, fold, node_dim: int) -> GraphBundle:
    if condition == "A1":
        return build_a1(fold)
    if condition == "A2":
        return build_a2(fold, node_dim)
    if condition == "A3":
        return build_a3(fold)
    raise ValueError(f"Unknown condition: {condition!r}")
