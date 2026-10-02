"""Experiment 3 — GraphSAGE spatial/property-relationship modelling.

Isolated package for Experiment 3. Executed strictly against the frozen
documents:

  protocol            docs/research/EXPERIMENT_3_SPATIAL_MODELING_PROTOCOL_DRAFT.md
                      tag v1.5.2-experiment3-protocol-final
  implementation spec docs/research/EXPERIMENT_3_IMPLEMENTATION_SPEC.md
                      tag v1.5.3-experiment3-implementation-spec

Nothing in this package modifies Experiment 1 / Experiment 2 code, datasets,
folds, models, artifacts, blockchain state or src/config.json.
"""

from __future__ import annotations

__all__ = ["frozen", "data", "graphs", "model", "train", "metrics"]
