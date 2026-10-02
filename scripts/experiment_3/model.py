"""Frozen GraphSAGE architecture (spec Section 7, B1/B2).

Single preregistered model family. Two mean-aggregation SAGEConv layers,
ReLU, dropout 0.20, no residual/skip, scalar ``log1p(price)`` output.
No GCN symmetric normalisation. No architecture search.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv

from scripts.experiment_3 import frozen


class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = frozen.ARCHITECTURE["hidden_dim"],
        dropout: float = frozen.ARCHITECTURE["dropout"],
    ) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim, aggr="mean")
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.head = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        return self.head(x).squeeze(-1)


def architecture_manifest() -> dict:
    return dict(frozen.ARCHITECTURE)


def count_parameters(in_dim: int) -> int:
    model = GraphSAGE(in_dim)
    return int(sum(p.numel() for p in model.parameters()))
