"""Unit tests for the Experiment 3 implementation (Phase 2 validation).

These tests build graphs and instantiate the model but train nothing.
"""

from __future__ import annotations

import numpy as np
import torch

from scripts.valuation_v2_1 import protocol as v21
from scripts.experiment_3 import data, frozen, graphs, train
from scripts.experiment_3.model import GraphSAGE, count_parameters


def test_frozen_contract():
    assert len(frozen.INPUT_FEATURES) == 46
    assert len(frozen.FINAL_FEATURES) == 48
    assert len(frozen.E4_COORDS) == 46
    assert len(frozen.E4_NUMERIC) == 11
    assert len(frozen.E4_AMENITIES) == 35
    assert not (set(frozen.E4_COORDS) & set(frozen.CATEGORICAL_FEATURES))
    assert frozen.K_NEIGHBORS == 5


def test_knn_tie_break_prefers_smaller_id():
    query = np.array([[0.0, 0.0]], dtype=np.float32)
    ref = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    ids = np.array([10, 5, 20], dtype=np.int64)
    out = graphs.knn_indices(query, ref, k=2, ref_ids=ids, exclude_self=False)
    assert out.tolist() == [[5, 10]]


def test_knn_excludes_self_and_deterministic():
    rng = np.random.RandomState(0)
    ref = rng.randn(200, 5).astype(np.float32)
    out1 = graphs.knn_indices(ref, ref, k=5, exclude_self=True)
    out2 = graphs.knn_indices(ref, ref, k=5, exclude_self=True)
    assert np.array_equal(out1, out2)
    assert all(i not in out1[i] for i in range(len(ref)))


def test_model_forward_empty_edges():
    model = GraphSAGE(7, 8, 0.0)
    x = torch.randn(5, 7)
    ei = torch.empty((2, 0), dtype=torch.long)
    out = model(x, ei)
    assert out.shape == (5,)
    assert count_parameters(7) > 0


def test_inner_split_is_partition():
    idx = np.arange(1000)
    fit, val = train.inner_split(idx)
    assert len(set(fit) & set(val)) == 0
    assert len(fit) + len(val) == 1000


def test_build_graphs_on_frozen_fold():
    dataset, _, _ = data.load_dataset()
    fs = v21.load_folds()[frozen.PRIMARY_REGIME]
    tr = np.asarray(fs.train[0], dtype=np.int64)
    te = np.asarray(fs.valid[0], dtype=np.int64)
    enc = data.fit_fold(dataset, tr, te)

    a1 = graphs.build_a1(enc)
    assert a1.train_edge_index.shape[1] == 0

    a3 = graphs.build_a3(enc)
    k = frozen.K_NEIGHBORS
    assert a3.train_edge_index.shape[1] == len(tr) * k
    assert a3.infer_edge_index.shape[1] >= len(tr) * k
    assert not np.any(a3.train_edge_index[0] == a3.train_edge_index[1])
    assert set(a3.train_edge_index.reshape(-1).tolist()) <= set(tr.tolist())

    te_set = set(te.tolist())
    train_set = set(tr.tolist())
    # no test node is ever a source; test nodes only receive from training
    assert not (set(a3.infer_edge_index[0].tolist()) & te_set)
    mask = np.isin(a3.infer_edge_index[1], list(te_set))
    assert np.all(np.isin(a3.infer_edge_index[0][mask], list(train_set)))
    # each test node receives exactly k edges (from training nodes)
    dst_test = a3.infer_edge_index[1][mask]
    uniq, counts = np.unique(dst_test, return_counts=True)
    assert np.all(counts == k)

    a2 = graphs.build_a2(enc, enc.node_dim)
    assert not (set(a2.train_edge_index[0].tolist()) & te_set)
    assert not (set(a2.infer_edge_index[0].tolist()) & te_set)
