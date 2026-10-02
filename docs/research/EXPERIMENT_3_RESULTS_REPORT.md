# Experiment 3 — Spatial / Property-Relationship Modeling: Results Report

## 1. Experiment title and status

**Title:** Experiment 3 — Spatial / Property-Relationship Modeling (inductive graph evaluation).

**Status:** Experiment 3 is **complete and fully audited**. All preregistered computations
completed; the final read-only audit returned PASS on every check. This report is based only
on the frozen Experiment 3 artifacts under `artifacts/valuation/v2_3/`, the final protocol
(`v1.5.2`), the implementation specification (`v1.5.3`), and the verified audit results.
No retraining, graph rebuild, hyperparameter change, fold change, prediction change, metric
change, or protected-state change occurred in producing this report.

This is a reporting document only. It creates no model, prediction, or result artifact, and
it commits no Git ref.

## 2. Research question

**Primary research question (frozen, Protocol §3):**

> Does incorporating spatial/property relationships improve valuation generalization under
> location-held-out evaluation compared with the frozen CatBoost backbone?

Operational clauses (Protocol §3): *incorporating spatial/property relationships* means a
model whose inputs or architecture explicitly encode relations between properties/localities;
*valuation generalization* means out-of-fold predictive accuracy on unseen localities,
measured by the frozen metrics; *location-held-out evaluation* is the frozen
`location_grouped` regime; the *frozen CatBoost backbone* is the Experiment 1 CatBoost
configuration consumed read-only.

The question is deliberately **not** "Will the GNN perform better?" or "Which model is
best?". The protocol's explicit no-result prediction states that improvement, no measurable
improvement, or degradation are all admissible outcomes (Protocol §1). The closed secondary
questions are RQ2 (does relationship modeling reduce random→location-grouped degradation),
RQ3 (limited-locality performance), RQ4 (error behaviour across price bands), RQ5
(fold-robustness), and RQ6 (which relationship types contribute) (Protocol §4).

## 3. Frozen protocol and specification references and hashes

| Item | Value |
|---|---|
| Final protocol | `docs/research/EXPERIMENT_3_SPATIAL_MODELING_PROTOCOL_DRAFT.md` |
| Protocol tag | `v1.5.2-experiment3-protocol-final` |
| Protocol commit | `aee1055a0a4d1bf8a6e48a1eb54ffe155bdd54e2` |
| Protocol SHA-256 (re-verified on disk) | `5526dbae1f54430854f747ff2a94f193b612ec4e1ae505f50ce2a41feb386866` |
| Implementation specification | `docs/research/EXPERIMENT_3_IMPLEMENTATION_SPEC.md` |
| Spec tag | `v1.5.3-experiment3-implementation-spec` |
| Spec commit | `e7b14bcd5f8314cbbb7c989d105df3cbae5a0f19` |
| Spec SHA-256 (re-verified on disk) | `fded5f58268fc137507f52524b7e32b78ba75f8a93f752d636d2f07577e0550a` |
| Report HEAD at authoring | `e7b14bcd5f8314cbbb7c989d105df3cbae5a0f19` |

Both frozen documents are read-only. The architecture, graph definition, metrics contract,
fold contract, ablation mapping, leakage controls, and reproducibility identifiers were
frozen before training and are transcribed, not reinterpreted, in this report.

## 4. Dataset and evaluation regimes

**Dataset (read-only, Protocol §6.1):**

| Element | Value |
|---|---|
| File | `data/processed/MREID_property.csv` |
| SHA-256 | `1be10465221668ffde0fe7392efbc183ef77a654549a2724b26439978fad8877` |
| Post-deduplication rows (nodes) | 28,398 |
| Rows removed by V2.1 dedup | 737 (from 29,135) |
| Duplicate key | `(source_city, location, area, no_of_bedrooms, price)`, keep first |
| Target | `price`; learning space `log1p(price)`; presentation via `max(expm1(·), 0)` |
| Feature contract | 48 features (11 numeric + 37 one-hot); 46 pipeline inputs; 2 train-only frequency features derived inside the pipeline |
| Locality structure | 1,789 localities; min/median/max size 1 / 3 / 685; 6 cities |
| Geographic coordinates | **None available** — geographic-proximity edges (E3) are excluded by design |

**Frozen evaluation regimes (Protocol §6.5, §10, §11):**

| Regime | Role | Splitter | Frozen digest |
|---|---|---|---|
| `location_grouped` | **Primary** (unseen-localities generalization) | `GroupKFold`-equivalent grouping on `source_city__location`; `group(train) ∩ group(test) = ∅` | `4c1c331cc993ddedb58b0a177e118912eb657613e3c9e4f8b5ae2bc7963ec611` |
| `random` | Secondary reference (RQ2) | `KFold(shuffle=True)`, `random_state=42` | `1a9f896345492d60b4a8cc677a090b27c8affd745ef302f9fbbf4ea55afc415f` |

Both regimes use the same five frozen outer folds as Experiments 1 and 2, with identical
train/test rows across all conditions. Folds are never pooled as independent observations
(Protocol §12).

## 5. A0 / A1 / A2 / A3 definitions (as frozen)

The closed ablation set is A0–A3 (Protocol §16; Spec §9). No additional ablations were
introduced.

| ID | Condition | Frozen definition |
|---|---|---|
| **A0** | Frozen Experiment 1 CatBoost baseline | Consumed **read-only**; not retrained or retuned for Experiment 3 (Protocol §7; Spec §8). Parameters: `iterations=700`, `learning_rate=0.04`, `depth=7`, `l2_leaf_reg=2.0`, `random_seed=42`, `loss_function=RMSE`. |
| **A1** | Relationship-free GraphSAGE control | Same frozen node-feature representation and same frozen GraphSAGE architecture, but **self/root contribution only — no relational edges** (no locality edges, no similarity edges). |
| **A2** | Locality/city relationship GraphSAGE | Uses only the already-permitted locality/city relations **E1 (same city) / E2 (same locality) / E5 (training-fold locality aggregates)**; no new edge type. |
| **A3** | E4 property-similarity GraphSAGE (**PRIMARY graph model**) | Directed **k = 5** E4 property-characteristic similarity in the standardised non-target space (46 coordinates), with inductive test-node connectivity restricted to training nodes and no test→test relationships. The corrected message-passing edge orientation is neighbour→node; therefore test-side message-passing edges are training→test. |

A1 is the relationship-free control (self/root only); A2 isolates locality/city structure;
A3 isolates property-similarity structure. A1 and A3 are deliberately **not** the same
condition and must not be conflated (Protocol §16; Spec §9). The three GraphSAGE conditions
differ only in relationship structure, not by post-hoc tuning.

## 6. Graph construction and inductive inference rules

**Edge types (Protocol §8.3; Spec §3).** E4 (property-characteristic similarity) is the
primary relation. E1/E2/E5 are permitted for A2. E3 (geographic proximity) is **not
available** (no coordinates). E6 (price similarity) and E7 (target-derived neighbourhoods)
are **prohibited**.

**E4 similarity space (Protocol §8.8; Spec §4).** The exact 46 permitted non-target
coordinates: 3 basic (`area`, `no_of_bedrooms`, `resale`), 4 row-wise engineered
(`log_area`, `log_bedrooms`, `area_per_bedroom`, `area_bedroom_interaction`), 2 row-wise
indicators (`amenity_yes_count`, `amenities_fully_specified`), 2 training-only locality
aggregates (`location_frequency`, `log_location_frequency`), and the 35 raw amenity columns.
`location`, `source_city`, `price`, `log1p(price)`, `derived_price_per_sqft`, and all other
target-derived quantities are excluded. Numeric coordinates are standardised with the frozen
V2.1 `StandardScaler`; amenity coordinates use the frozen V2.1 one-hot encoding; both are fit
on training rows only, per fold, and applied unchanged to test rows. Distance is
deterministic Euclidean; **k = 5**; directed; exact ties broken by ascending positional node
id; neighbours unique; no duplicate edges; every edge weight `w_ij = 1` (binary; zero-distance
neighbours retained).

**Aggregation (Protocol §8.7, §17.1; Spec §5).** GraphSAGE mean-neighbourhood aggregation.
No GCN-style symmetric normalisation `A_hat = D^(-1/2) A D^(-1/2)`. Each node contributes a
self/root term under the frozen GraphSAGE implementation.

**Inductive inference contract (Protocol §9.3–§9.4; Spec §6).**

- *Training graph:* training rows only; learned transforms and the E4 kNN index fit on
  `train_k` only; no target-derived quantity; no edge uses `price` or any label-derived field.
- *Test/inference graph:* test nodes are attached at prediction time using prediction-time
  covariates only; each test node emits directed E4 edges to its five nearest permitted
  **training** nodes (test→training); no test→test E4 edge exists in the primary design.
- Explicitly prohibited: test labels in the graph, test-fitted scalers/encoders, test-derived
  graph statistics, target-derived similarity, and selection using the `location_grouped`
  outer test.

Verified graph statistics (frozen artifacts): A3 training graph contains only training
nodes, with exactly `5 × |train_k|` directed training→training edges and `5 × |test_k|`
directed test→training edges; A3 encoded E4 dimension `e4_dim = 116` (11 numeric + 35
amenities × 3); encoded node dimension is fold-dependent (e.g. `node_dim` 1540 for
`location_grouped` fold 0; 1782 for `random` fold 0).

## 7. A3 defect and correction provenance

The final A3 results incorporate a corrected implementation. This is disclosed in full; the
defect is an **implementation-direction defect that was corrected to restore the
preregistered graph contract**, not a scientific failure. The authoritative record is
`artifacts/valuation/v2_3/a3_defect_provenance.md` (SHA-256
`b265adff0434c029514f54f81832e09118f1168f51118f95f445249eb8bf6b51`).

- **Initial defect.** The initial A3 E4 implementation emitted edges in the node→neighbour
  direction for PyG, where `edge_index[0]` is the source and `edge_index[1]` is the
  target/aggregation destination. Consequences: training nodes aggregated reverse-kNN
  relationships; test features could flow into training nodes; test nodes had no incoming E4
  edges and relied on the root contribution; pre-fix A3 `R2_log` was approximately −16 to
  −41.
- **Detection.** Identified during Experiment 3 Phase 3 structural validation by checking the
  constructed E4 edge direction against PyG/GraphSAGE message-passing semantics.
- **Correction.** E4 direction changed to neighbour→node; test-side E4 edges are
  training→test; no test node is a source; structural tests were updated and enforced.
- **Invalidation.** All pre-fix A3 folds were invalidated and their run JSON/prediction CSV
  artifacts deleted before regeneration.
- **Regeneration.** All 10 A3 fits were regenerated (5 `location_grouped`, 5 `random`); all
  regenerated fits are post-fix; epochs 98–147; final A3 `R2_log` range 0.0952–0.2242.
- **Post-correction validation.** Experiment 3 tests: 6 passed; leakage audit: 10/10 passed;
  no test-source edges; no test→test edges; no duplicate edges; no self-edges; training edges
  remain training-only; test targets are sourced from training-side truth only.

**Scientific-status rule (verbatim, frozen provenance):** "Only the corrected post-fix A3
runs are part of the final Experiment 3 results. The pre-fix A3 runs are invalid
implementation outputs and are excluded from all reported metrics and comparisons." and "No
methodology, hyperparameter, fold, dataset, or model-selection decision was changed as a
consequence of the defect; the correction restored the preregistered E4 edge-direction
contract."

## 8. Training configuration

Single preregistered architecture, frozen before evaluation, no architecture search
(Protocol §17.1; Spec §7; `artifacts/valuation/v2_3/model_configuration.json`, SHA-256
`c4dd352482a80281d4fddaa727e7f38de3cf372edcf8b232bae9d1347d461292`):

| Field | Value |
|---|---|
| Model family | GraphSAGE (single preregistered family) |
| Message-passing layers | 2 |
| Hidden dimension | 64 |
| Aggregation | Mean-neighbourhood (no GCN symmetric normalisation) |
| Activation | ReLU |
| Dropout | 0.20 |
| Residual / skip connections | None |
| Output | Single scalar; predicted `log1p(price)` |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 1e-4 |
| Maximum epochs | 200 |
| Early stopping | Training-side validation loss, patience 20 |
| Loss | MSE in `log1p(price)` |
| Random seed | 42 |
| k (E4) | 5 |
| Directedness | Directed |
| Distance | Euclidean |
| Edge weight | 1 (binary) |
| Inner validation fraction | 0.10 |
| Architecture search | None |

**Fit accounting.** 30 intended fits (A1/A2/A3 × 2 regimes × 5 folds); A0 is consumed
read-only (0 fits). Measured total training time `3486.714895600009` seconds on CPU.
Environment (`environment.json`): Python 3.14.7 (CPython), Platform
Windows-11-10.0.26200-SP0, `torch` 2.14.1+cpu, `torch_geometric` 2.8.0.post1,
CUDA unavailable, device CPU, `numpy` 2.5.3, `pandas` 3.0.6, `scikit-learn` 1.9.1.

## 9. Evaluation metrics

Identical to the Experiment 1 metric contract, computed by the frozen
`scripts/valuation_v2_1/protocol.py::compute_metrics` (8 keys). No composite or overall
ranking score was created (Protocol §12; Spec §10).

- **Primary:** `R2_log`, `MAE_log`, `MedAPE_percent`.
- **Secondary:** `MAE_INR`, `RMSE_INR`, `R2_INR`, `RMSE_log`, `MAPE_percent`.

Reporting uses per-fold values plus fold mean and sample SD (`ddof = 1`) across the 5 folds.
Folds are never pooled. Regime comparisons rest on paired per-fold differences and direction
consistency. The primary regime is `location_grouped`; `random` is a secondary reference.

## 10. Primary `location_grouped` aggregate results

Fold mean ± sample SD (`ddof = 1`, 5 folds), all 8 metrics. Values are taken directly from
`artifacts/valuation/v2_3/aggregate_metrics.json`.

| Condition | MAE_INR | RMSE_INR | R2_INR | MAPE_percent | MedAPE_percent | MAE_log | RMSE_log | R2_log (primary) |
|---|---|---|---|---|---|---|---|---|
| **A0** | 6.743e6 ± 1.0e6 | 2.318e7 ± 6.2e6 | 0.065 ± 0.024 | 59.35 ± 6.1 | 37.29 ± 2.3 | 0.5128 ± 0.024 | 0.7013 ± 0.026 | **0.2677 ± 0.0378** |
| **A1** | 1.718e7 ± 2.1e7 | 7.425e8 ± 1.6e9 | −6792 ± 1.5e4 | 137.1 ± 190 | 43.43 ± 1.85 | 0.6140 ± 0.0565 | 0.8269 ± 0.074 | **−0.0170 ± 0.1158** |
| **A2** | 7.395e6 ± 1.1e6 | 2.600e7 ± 6.0e6 | −0.269 ± 0.60 | 54.63 ± 3.3 | 42.11 ± 1.36 | 0.5698 ± 0.0195 | 0.7631 ± 0.035 | **0.1340 ± 0.0270** |
| **A3** | 7.214e6 ± 1.0e6 | 2.439e7 ± 5.4e6 | −0.060 ± 0.15 | 55.72 ± 4.9 | 41.30 ± 2.69 | 0.5619 ± 0.0242 | 0.7620 ± 0.030 | **0.1360 ± 0.0268** |

**Observed empirical fact:** under the primary `location_grouped` regime, A0 has the highest
observed mean `R2_log` (0.2677). All GraphSAGE conditions (A1, A2, A3) have lower observed
mean `R2_log` than A0. Among GraphSAGE conditions, A3 (0.1360) and A2 (0.1340) are close and
both exceed A1 (−0.0170).

## 11. Secondary `random` aggregate results

Fold mean ± sample SD (`ddof = 1`, 5 folds), all 8 metrics, from `aggregate_metrics.json`.

| Condition | MAE_INR | RMSE_INR | R2_INR | MAPE_percent | MedAPE_percent | MAE_log | RMSE_log | R2_log (primary) |
|---|---|---|---|---|---|---|---|---|
| **A0** | 6.240e6 ± 6.7e4 | 2.263e7 ± 9.0e5 | 0.119 ± 0.016 | 50.96 ± 1.0 | 32.52 ± 0.74 | 0.4658 ± 0.0078 | 0.6545 ± 0.0089 | **0.3640 ± 0.0106** |
| **A1** | 1.037e7 ± 6.7e6 | 2.411e8 ± 4.9e8 | −400.4 ± 900 | 95.24 ± 56 | 48.48 ± 3.93 | 0.5944 ± 0.0239 | 0.7706 ± 0.020 | **0.1180 ± 0.0353** |
| **A2** | 7.011e6 ± 2.0e5 | 2.571e7 ± 5.3e6 | −0.161 ± 0.46 | 58.11 ± 2.2 | 40.26 ± 1.05 | 0.5350 ± 0.0068 | 0.7188 ± 0.007 | **0.2327 ± 0.0075** |
| **A3** | 7.012e6 ± 2.0e5 | 2.610e7 ± 5.8e6 | −0.202 ± 0.53 | 58.04 ± 2.9 | 41.18 ± 1.48 | 0.5403 ± 0.0070 | 0.7278 ± 0.0046 | **0.2133 ± 0.0099** |

**Observed empirical fact:** under the secondary `random` regime, A0 again has the highest
observed mean `R2_log` (0.3640). The GraphSAGE ordering is A2 (0.2327) above A3 (0.2133)
above A1 (0.1180). `random` evaluation yields higher observed `R2_log` than
`location_grouped` evaluation for every condition.

## 12. Per-fold primary results

Per-fold values are taken from `artifacts/valuation/v2_3/per_fold_metrics.csv` (A1–A3) and
`artifacts/valuation/v2_3/a0_reference_per_fold.csv` (A0). Fold indices 0–4 correspond to the
frozen folds.

### 12.1 Primary regime `location_grouped`

`R2_log` (primary metric):

| Condition | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| A0 | 0.2707 | 0.2983 | 0.3091 | 0.2413 | 0.2191 |
| A1 | 0.0613 | 0.0744 | −0.2019 | 0.0399 | −0.0587 |
| A2 | 0.1575 | 0.1329 | 0.0886 | 0.1409 | 0.1500 |
| A3 | 0.1416 | 0.1539 | 0.1258 | 0.0952 | 0.1634 |

`MAE_log`:

| Condition | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| A0 | 0.4999 | 0.4792 | 0.5145 | 0.5322 | 0.5382 |
| A1 | 0.5759 | 0.5561 | 0.6974 | 0.5990 | 0.6417 |
| A2 | 0.5524 | 0.5497 | 0.5975 | 0.5741 | 0.5754 |
| A3 | 0.5464 | 0.5302 | 0.5838 | 0.5866 | 0.5624 |

`MedAPE_percent`:

| Condition | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| A0 | 36.55 | 34.02 | 36.85 | 39.37 | 39.67 |
| A1 | 43.48 | 40.37 | 44.08 | 43.87 | 45.35 |
| A2 | 42.14 | 40.22 | 41.60 | 42.69 | 43.89 |
| A3 | 41.39 | 37.25 | 41.76 | 44.80 | 41.31 |

### 12.2 Secondary regime `random`

`R2_log` (primary metric):

| Condition | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| A0 | 0.3547 | 0.3704 | 0.3732 | 0.3505 | 0.3711 |
| A1 | 0.1017 | 0.1018 | 0.1534 | 0.0767 | 0.1566 |
| A2 | 0.2249 | 0.2430 | 0.2368 | 0.2261 | 0.2327 |
| A3 | 0.2066 | 0.2242 | 0.2000 | 0.2159 | 0.2200 |

`MAE_log`:

| Condition | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| A0 | 0.4763 | 0.4665 | 0.4542 | 0.4667 | 0.4652 |
| A1 | 0.6170 | 0.5967 | 0.5727 | 0.6181 | 0.5674 |
| A2 | 0.5450 | 0.5353 | 0.5280 | 0.5295 | 0.5372 |
| A3 | 0.5501 | 0.5411 | 0.5429 | 0.5352 | 0.5323 |

`MedAPE_percent`:

| Condition | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| A0 | 33.63 | 32.12 | 31.63 | 32.54 | 32.70 |
| A1 | 51.23 | 45.98 | 47.30 | 53.75 | 44.13 |
| A2 | 41.43 | 39.72 | 40.82 | 38.74 | 40.59 |
| A3 | 41.79 | 41.64 | 43.04 | 40.20 | 39.24 |

## 13. Paired ΔR2_log comparisons

Paired fold-level `R2_log` differences (same fold, same test rows), from
`artifacts/valuation/v2_3/paired_differences.json`. A positive value means the first
condition has higher `R2_log`.

| Pair | `location_grouped` mean Δ | `random` mean Δ |
|---|---|---|
| A3 − A1 | **+0.1530** | **+0.0953** |
| A3 − A2 | **+0.0020** | −0.0194 |
| A3 − A0 | −0.1317 | −0.1506 |
| A2 − A0 | −0.1337 | −0.1313 |
| A1 − A0 | −0.2847 | −0.2460 |

### 13.1 Per-fold ΔR2_log

`location_grouped` (fold 0 … fold 4):

| Pair | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| A3 − A1 | +0.0803 | +0.0795 | +0.3277 | +0.0553 | +0.2221 |
| A3 − A2 | −0.0159 | +0.0210 | +0.0372 | −0.0457 | +0.0134 |
| A3 − A0 | −0.1291 | −0.1444 | −0.1833 | −0.1461 | −0.0557 |
| A2 − A0 | −0.1132 | −0.1654 | −0.2205 | −0.1004 | −0.0691 |
| A1 − A0 | −0.2094 | −0.2239 | −0.5110 | −0.2014 | −0.2778 |

`random` (fold 0 … fold 4):

| Pair | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| A3 − A1 | +0.1049 | +0.1224 | +0.0466 | +0.1392 | +0.0634 |
| A3 − A2 | −0.0183 | −0.0188 | −0.0368 | −0.0102 | −0.0127 |
| A3 − A0 | −0.1481 | −0.1462 | −0.1732 | −0.1346 | −0.1511 |
| A2 − A0 | −0.1298 | −0.1274 | −0.1364 | −0.1244 | −0.1384 |
| A1 − A0 | −0.2530 | −0.2686 | −0.2198 | −0.2738 | −0.2145 |

**Fold direction consistency (from `paired_differences.json`).** A3 − A1 is direction
consistent (positive in all 5 folds) in both regimes. A3 − A2 is **not** direction
consistent: in `location_grouped` it is positive in 4 of 5 folds and negative in fold 3
(mean +0.0020), and in `random` it is negative in all 5 folds (mean −0.0194), i.e. the sign
of the A3−A2 difference is not consistent across regimes. A3 − A0, A2 − A0, and A1 − A0 are
negative and direction consistent in all 5 folds in both regimes.

## 14. Reproducibility and audit evidence

All items below are from the frozen artifacts and the completed read-only audit.

| Evidence | Result |
|---|---|
| Preflight | **PASS 43/43** (`preflight/preflight_report.json`, `n_failed=0`) |
| Intended fits | **30/30** (A1/A2/A3 × 2 regimes × 5 folds) |
| Metrics per fit | **8/8** present for every fit |
| Aggregation | Fold mean and sample SD (`ddof=1`) independently recomputed and matched for all conditions and metrics |
| Independent recomputation | All 30 fits re-scored from saved predictions; max absolute delta `R2_log` 0, `MAE_log` 0, `MedAPE_percent` 0, `MAE_INR` 1.86e-9, `RMSE_INR` 4.77e-7 (CSV round-trip noise) |
| Frozen fold node IDs | Prediction node indices exactly match the frozen folds' `valid` indices |
| Fold sizes | Per-run `train_rows`/`test_rows` match the frozen folds |
| Paired differences | All required pairs present and independently re-derived |
| Graph manifests/statistics | A3 `5 × |train|` train edges and `5 × |test|` test edges; manifest k=5, directed, E4; `e4_dim=116` |
| Structural graph audit | **10/10 PASS** (`leakage_audit.json`, `all_structural_passed=true`) |
| Model configuration hash | Embedded per-run SHA-256 `c4dd352482a80281d4fddaa727e7f38de3cf372edcf8b232bae9d1347d461292` equals the on-disk `model_configuration.json` SHA-256 |
| Environment recorded | `environment.json` (recorded at 2026-10-02T16:21:28.499892+00:00) |
| Reproducibility check | `reproducibility_check.json` `overall_pass=true` (30/30 recompute match, aggregate consistency OK, protected state unchanged) |
| Defect provenance | `a3_defect_provenance.md` (SHA-256 `b265adff0434c029514f54f81832e09118f1168f51118f95f445249eb8bf6b51`) |
| Protected state | Before vs after byte-identical; see §15 |
| Training after final experiment | None |

**Reproducibility note on a stale artifact hash entry (pre-existing, deliberately left
untouched).** During the final read-only audit, one pre-existing stale hash entry was
observed: `artifact_hashes.json` records hash `ee51e9c4d7237eca751f88b950308bc68fde423c6ec2574e505f5c390cbe91ef`
for `reproducibility_check.json`, whereas the current file hashes to
`8bd7abdb53aa5b91ced1edb48d4dc842af34967347a46f6d5b8d9b401f7b592f`. Both files carry the
same original write timestamp (22:17:23), i.e. the manifest entry was already stale before
the provenance step. It was **deliberately not modified**, because the final audit was
constrained to read-only verification after the experiment. This stale manifest entry did
**not** alter predictions, metrics, model artifacts, protected state, or any experimental
result; it is a manifest bookkeeping artifact only. The report does **not** claim that every
hash manifest is internally current.

## 15. Protected blockchain-state integrity result

Protected state was captured before and after the Experiment 3 run and is byte-identical
(`protected_state_before.json` == `protected_state_after.json`). Experiment 3 read, wrote,
and transacted against no blockchain state.

| Protected item | SHA-256 |
|---|---|
| `.chain\state.json` | `53c6f1d7b7165769c4f462baffc00ef638586c5f06fba71943e66ceb97f6e4f9` |
| `data\processed\millow_token_map.csv` | `91fcb8a3346d89a91360ce531b685e2e97683b7f80f2ac2c44dedf83042106f4` |
| `src\config.json` | `e0b6c7d3c1eb56cad5d4d43d2e958b724cf7c74018e6c693139bec9d82610c33` |
| `data\processed\MREID_property.csv` | `1be10465221668ffde0fe7392efbc183ef77a654549a2724b26439978fad8877` |
| `artifacts\valuation\v2_1\cv_folds_v2_1.json` | `989b79656b051b3b6c8c583b4170a5aaa1c31676260772247bbc98d1eb095156` |
| `artifacts\valuation\v2_1\protocol_v2_1.json` | `49e0383ce4045c793e7fae3d3e4a8d4ec8bc3bddccc57cb218adf85e776ce1d3` |

## 16. Limitations

1. **No formal significance testing was preregistered or performed.** Given only five folds,
   no significance claim is made; all comparisons are descriptive (means, sample SDs, paired
   fold-level differences, and direction consistency).
2. **Single seed (42).** Seed sensitivity is unquantified; results rest on one seed per
   condition.
3. **No geographic coordinates.** Geographic-distance modelling was excluded; spatial
   structure is expressed only through categorical locality/city keys and attribute
   similarity.
4. **INR-space metrics can be unstable for poor predictions** because of `expm1`
   amplification. In particular A1's `R2_INR`/`RMSE_INR` are extreme despite a near-zero
   `R2_log`; the `log`-space metrics are the appropriate headline for such comparisons.
5. **A2 locality/city implementation is a permitted realization of E1/E2/E5** (training-fold
   hub aggregates), not the only possible realization.
6. **The primary experiment is location-grouped / inductive.** Test-set covariates are not
   used during graph construction; the transductive variant is a secondary sensitivity
   analysis only and is not reported here.
7. **Five folds** limit variance estimation and fold-robustness assessment (RQ5).
8. **Single dataset and single backbone family.** Results are bounded by this dataset and the
   frozen CatBoost baseline and do not generalise to real-estate valuation in general.
9. **Baseline is frozen, not retuned.** A differently tuned CatBoost is out of scope.

## 17. Scientifically supported interpretation

The following statements separate observed empirical results from interpretation.

**Observed empirical results.**

- A0 has the highest observed mean `R2_log` in both evaluation regimes (0.2677
  `location_grouped`; 0.3640 `random`).
- A3 is consistently above A1 in `R2_log`: mean paired difference **+0.1530** in
  `location_grouped` and **+0.0953** in `random`, positive in all five folds in both regimes.
- A3 and A2 are close in the primary `location_grouped` regime: mean paired difference
  (A3 − A2) **+0.0020**; this difference is not direction consistent across the two regimes
  (in `random`, A3 − A2 = −0.0194 and negative in all five folds).
- All GraphSAGE conditions remain below A0 on the primary `R2_log` metric.
- `random` evaluation is easier than `location_grouped` evaluation for every condition.

**Interpretation.**

- Adding *some* relational structure through the E4 graph (A3) is associated with higher
  observed `R2_log` than the relationship-free control (A1), and this is directionally
  consistent across folds and regimes. This addresses part of RQ6: the E4 relation carries
  measurable signal beyond self/root-only aggregation.
- Observed differences between A3 (property similarity) and A2 (locality/city structure) are
  too small and direction-inconsistent to support a claim that one relationship type is
  better than the other.
- Under location-held-out evaluation — the primary and hardest setting — none of the
  GraphSAGE conditions reached the frozen CatBoost baseline on `R2_log`. This is the pattern
  the protocol anticipated as admissible: the primary research question can return a negative
  answer, and here the relationship-aware conditions did not exceed the frozen baseline on
  the primary metric.
- The `random` vs `location_grouped` gap persists for all conditions, consistent with RQ2's
  framing that location-held-out shift is materially harder; no claim is made that
  relationship modeling removed that gap.
- The A3 edge-direction defect was detected, invalidated, corrected, and regenerated before
  the final audit; only corrected A3 runs are included. This is an implementation-direction
  defect that was corrected to restore the preregistered E4 graph contract, not a scientific
  failure, and it does not change the methodology.

## 18. Claims explicitly NOT supported by this experiment

- No model or condition is called "best". A0 merely has the highest **observed mean**
  `R2_log` in both regimes.
- No result is described as "significantly better" or "significantly different": no formal
  significance test was preregistered or performed, and five folds give very low power.
- It is **not** claimed that GraphSAGE outperforms CatBoost; on the primary `R2_log` metric,
  every GraphSAGE condition is below A0.
- It is **not** claimed that E4 is universally superior; A3 and A2 are close and
  direction-inconsistent, and A3 is below A0.
- No causal claims are made; this is a predictive comparison, not an interventional study.
- No claim is made about production readiness, the full MILLOW system, or valuation in
  general.
- No transductive result is mixed into the primary result; the primary design is inductive.

## 19. Final conclusion

Experiment 3 completed as a pre-registered, leakage-controlled, inductive evaluation of
spatial/property-relationship modeling under location-held-out and random conditions, against
a frozen read-only CatBoost baseline. All preregistered checks passed (43/43 preflight, 30/30
fits, 8/8 metrics, structural audit 10/10, reproducibility PASS, protected state
byte-identical). The primary `location_grouped` result is a **negative result for the
primary comparison**: A0 has the highest observed mean `R2_log` (0.2677), and all GraphSAGE
conditions (A1 ≈ −0.0170, A2 ≈ 0.1340, A3 ≈ 0.1360) remain below it. At the same time,
adding E4 property-similarity relations (A3) is consistently associated with higher `R2_log`
than the relationship-free control (A1) in every fold and both regimes, while A3 and the
locality/city condition (A2) are close and not direction-consistent. The one implementation
defect (A3 edge direction) was detected during structural validation, invalidated,
corrected, and regenerated before the final audit; only corrected runs are included, with the
methodology unchanged. These results are reported descriptively, without significance,
best-model, superiority, or causal claims.

## 20. Artifact inventory / provenance

**Namespace:** `artifacts/valuation/v2_3/`

**Top-level artifacts.**

| Artifact | Purpose |
|---|---|
| `per_fold_metrics.csv` | Per-fit metrics for A1–A3 (30 rows) |
| `aggregate_metrics.json` | Per-regime/condition mean + sample SD (`ddof=1`) for all 8 metrics |
| `paired_differences.json` | Fold-level paired differences and direction consistency |
| `a0_reference_per_fold.csv` | Read-only A0 CatBoost per-fold reference |
| `evaluation_summary.json` | Evaluation index (30 fits, total train seconds, A0 source) |
| `model_configuration.json` | Frozen architecture, graph contract, seed (SHA-256 `c4dd3524…`) |
| `environment.json` | Recorded software/hardware environment |
| `leakage_audit.json` | Leakage discipline checklist + 10 structural checks |
| `reproducibility_check.json` | Recomputation, aggregate consistency, protected-state result |
| `protected_state_before.json` / `protected_state_after.json` | Byte-identical protected-state capture |
| `artifact_hashes.json` | SHA-256 manifest of produced artifacts |
| `a3_defect_provenance.md` | A3 defect/correction/regeneration provenance |
| `preflight/preflight_report.json` | 43/43 preflight checks |
| `runs/*.json` | 30 per-fit records (metrics, timing, graph stats/manifest, config SHA, epochs) |
| `predictions/{regime}/fold{k}_{cond}.csv` | Per-fit predictions (`node_index`, `price_true`, `log_price_true`, `prediction_log`) |

**Key identifiers and hashes.**

| Item | Value |
|---|---|
| Report HEAD | `e7b14bcd5f8314cbbb7c989d105df3cbae5a0f19` |
| Protocol tag / commit / SHA-256 | `v1.5.2-experiment3-protocol-final` / `aee1055a0a4d1bf8a6e48a1eb54ffe155bdd54e2` / `5526dbae1f54430854f747ff2a94f193b612ec4e1ae505f50ce2a41feb386866` |
| Spec tag / commit / SHA-256 | `v1.5.3-experiment3-implementation-spec` / `e7b14bcd5f8314cbbb7c989d105df3cbae5a0f19` / `fded5f58268fc137507f52524b7e32b78ba75f8a93f752d636d2f07577e0550a` |
| Dataset SHA-256 | `1be10465221668ffde0fe7392efbc183ef77a654549a2724b26439978fad8877` |
| Fold digests (`random` / `location_grouped`) | `1a9f896345492d60b4a8cc677a090b27c8affd745ef302f9fbbf4ea55afc415f` / `4c1c331cc993ddedb58b0a177e118912eb657613e3c9e4f8b5ae2bc7963ec611` |
| `cv_folds_v2_1.json` SHA-256 | `989b79656b051b3b6c8c583b4170a5aaa1c31676260772247bbc98d1eb095156` |
| Model configuration SHA-256 | `c4dd352482a80281d4fddaa727e7f38de3cf372edcf8b232bae9d1347d461292` |
| A3 defect provenance SHA-256 | `b265adff0434c029514f54f81832e09118f1168f51118f95f445249eb8bf6b51` |
| Manifest entries | 72 (`artifact_hashes.json`) |

**Provenance statements.** A0 is consumed read-only from
`artifacts/valuation/v2_1/model_comparison/per_fold_metrics.csv`; no Experiment 1/2 file,
dataset, fold, model, blockchain state, or `src/config.json` was modified. Only corrected
post-fix A3 runs are part of the final results. The pre-existing stale
`artifact_hashes.json` entry for `reproducibility_check.json` is noted in §14 and was
deliberately left untouched; it changes no prediction, metric, model artifact, protected
state, or result. No commit, tag, or push was performed in creating this report.
