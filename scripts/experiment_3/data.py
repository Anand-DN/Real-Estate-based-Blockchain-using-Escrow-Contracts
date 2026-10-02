"""Node-feature and E4-coordinate representations (spec Sections 2 and 4).

All learned transforms are fitted on the training partition of the current
fold only (spec Section 6). Nothing here imports a target column or a
target-derived quantity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scripts.valuation_v2_1 import protocol as v21
from scripts.experiment_3 import frozen


@dataclass
class FoldEncoding:
    """Everything a fold needs to build graphs and run the model."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    x_all: np.ndarray            # node features, ALL nodes, float32
    e4_all: np.ndarray           # E4 similarity coordinates, ALL nodes, float32
    node_dim: int
    e4_dim: int
    node_feature_names: List[str]
    e4_feature_names: List[str]
    y_log: np.ndarray            # log1p(price), ALL nodes
    price: np.ndarray            # raw INR price, ALL nodes
    location: np.ndarray
    source_city: np.ndarray


def load_dataset() -> Tuple[pd.DataFrame, "v21.DuplicateReport", int]:
    return v21.build_dataset()


def _frequency_augmented(
    dataset: pd.DataFrame, train_idx: np.ndarray
) -> Tuple[pd.DataFrame, List[str]]:
    """Fit TrainOnlyFrequencyFeatures on train rows, transform all rows."""
    fitter = v21.TrainOnlyFrequencyFeatures()
    fitter.fit(dataset.iloc[train_idx][frozen.INPUT_FEATURES])
    augmented = fitter.transform(dataset[frozen.INPUT_FEATURES])
    return augmented, list(frozen.FINAL_FEATURES)


def fit_fold(dataset: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> FoldEncoding:
    """Fit the frozen V2.1 node-feature transform and E4 coordinates.

    Fitted ONLY on ``train_idx``; applied unchanged to every row.
    """
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)

    augmented, final_features = _frequency_augmented(dataset, train_idx)
    train_frame = augmented.iloc[train_idx]

    # --- Node features: frozen V2.1 numeric scaler + categorical one-hot ---
    node_pre = v21.create_preprocessor(dense_output=True)
    node_pre.fit(train_frame[final_features])
    x_all = np.asarray(
        node_pre.transform(augmented[final_features]), dtype=np.float32
    )

    # --- E4 coordinates: numeric standardised + amenities one-hot ---
    e4_pre = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(frozen.E4_NUMERIC)),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float64,
                ),
                list(frozen.E4_AMENITIES),
            ),
        ],
        remainder="drop",
    )
    e4_pre.fit(train_frame[list(frozen.E4_NUMERIC) + list(frozen.E4_AMENITIES)])
    e4_all = np.asarray(
        e4_pre.transform(
            augmented[list(frozen.E4_NUMERIC) + list(frozen.E4_AMENITIES)]
        ),
        dtype=np.float32,
    )

    node_feature_names = list(node_pre.get_feature_names_out())
    e4_feature_names = list(e4_pre.get_feature_names_out())

    price = dataset["price"].to_numpy(dtype=float)
    y_log = np.log1p(price)

    location = dataset["location"].to_numpy()
    source_city = dataset["source_city"].to_numpy()

    return FoldEncoding(
        train_idx=train_idx,
        test_idx=test_idx,
        x_all=x_all,
        e4_all=e4_all,
        node_dim=int(x_all.shape[1]),
        e4_dim=int(e4_all.shape[1]),
        node_feature_names=node_feature_names,
        e4_feature_names=e4_feature_names,
        y_log=y_log,
        price=price,
        location=location,
        source_city=source_city,
    )
