# Experiment 3 — Implementation Specification (FINAL, v1.5.2)

| Field | Value |
|---|---|
| Document | `docs/research/EXPERIMENT_3_IMPLEMENTATION_SPEC.md` |
| State | **FINAL — implementation specification. Documentation only.** |
| Pre-registration | **Pre-registered before Experiment 3 training.** |
| Authoring date | 2026-10-02 |
| Depends on (final, frozen) | `docs/research/EXPERIMENT_3_SPATIAL_MODELING_PROTOCOL_DRAFT.md` |
| Final protocol tag | `v1.5.2-experiment3-protocol-final` |
| Final protocol commit | `aee1055a0a4d1bf8a6e48a1eb54ffe155bdd54e2` |
| Final protocol SHA-256 | `5526dbae1f54430854f747ff2a94f193b612ec4e1ae505f50ce2a41feb386866` |
| Blockchain impact | None. No blockchain state is read, written, or transacted against. |
| Implementation readiness | **YES — B1–B10 resolved (Section 14).** |

This document is the implementation specification required by the final Experiment 3
protocol (Sections 8.7, 17.1 and 21). It records the single preregistered architecture, the
frozen graph definition, the inductive inference contract, the leakage controls, the
reproducibility identifiers and the **analytical** computational budget. It introduces **no
scientific choice beyond what `v1.5.2-experiment3-protocol-final` already freezes**.

> **Nothing here authorises training.** The pre-training gate is
> **analytical budget complete AND architecture frozen** (Sections 11 and 14). No model is
> trained, no graph is built and no artifact is produced by authoring this document.

---

## 0. Resolution rule used for this specification

Every item below is marked **FROZEN** and is transcribed directly from the final protocol
(`v1.5.2`). No item is left **OPEN**, no methodological choice is invented here, and no
architecture-selection step is described. Where the protocol delegates a detail (for example,
the exact self/root term), this specification records it at the level the protocol fixes and
does not extend it.

---

## 1. Node definition

**Status: FROZEN.** (Protocol §8.1, §8.7, §6.1.)

| Element | Frozen value | Basis |
|---|---|---|
| Node population | The post-deduplication processed dataset: exactly **28,398** rows | Protocol §6.1 |
| Node ⟷ row mapping | **One property row = one node** | Protocol §8.1 |
| Node identifier | **Positional integer index** `0 … 28397` of the deterministic `build_dataset()` frame (`reset_index(drop=True)`) | Matches the frozen fold index space (`protocol_v2_1.json`); requires no target or identifier column |
| Duplicate / removed rows | Removed **before any split or graph step** by the V2.1 exact-duplicate policy (737 rows removed; key `source_city, location, area, no_of_bedrooms, price`, keep first) | Protocol §6.1; `scripts/valuation_v2_1/protocol.py::remove_duplicates` |
| `mreid_id` | **Not used**, not even as a node key (excluded column) | Protocol §6.2 |
| Train / test node separation | Determined entirely by the frozen V2.1 folds; for `location_grouped`, `group(train) ∩ group(test) = ∅` | Protocol §6.5, §9.1, §10 |
| Auxiliary (hub) nodes | Locality/city hubs permitted as *auxiliary* nodes, not primary nodes | Protocol §8.1, §8.3 (E1/E2/E5) |

Primary nodes are properties; locality and city are relations (edges or auxiliary hubs), so
held-out localities remain first-class test nodes (Protocol §8.1).

---

## 2. Node features (B8)

**Status: FROZEN.** (Protocol §8.7, §6.1, §9.3.)

| Constraint | Value | Basis |
|---|---|---|
| Feature source | The frozen **V2.1 node-feature contract**: **46 pipeline input columns** (`protocol_v2_1.json → pipeline_input_columns`) | Protocol §6.1, §7, §8.7 |
| Target-derived fields | **PROHIBITED**: `price`, `derived_price_per_sqft`, and any `price`-derived quantity | Protocol §6.2, §8.7, §9.3, §9.4 |
| Excluded raw columns | `mreid_id`, `source_file`, `derived_price_per_sqft`, `group` must not be reintroduced | Protocol §6.2 |
| Frequency features | `location_frequency`, `log_location_frequency` are derived **inside** the pipeline by `TrainOnlyFrequencyFeatures`, fitted on training rows only | Protocol §6.1, §9.3 |
| Fit discipline | Every learned transform (frequency counts, scaler, encoder vocabulary) is fitted on **training rows only**, per fold | Protocol §9.3, §9.4 |
| Target space | `log1p(price)`; presentation in INR via `max(expm1(·), 0)` | Protocol §6.1 |

The 46 pipeline inputs comprise: 3 basic (`area`, `no_of_bedrooms`, `resale`), 4 row-wise
engineered (`log_area`, `log_bedrooms`, `area_per_bedroom`, `area_bedroom_interaction`), 2
row-wise indicators (`amenity_yes_count`, `amenities_fully_specified`), 2 categorical keys
(`source_city`, `location`), and the 35 raw amenity columns. The 2 frequency features are
added at fit time (48 features total: 11 numeric + 37 one-hot).

**Fold-dependent input dimensionality.** Under the V2.1 one-hot contract the encoded node
dimension is **fold-dependent** and recorded per fold (Protocol §8.7): 11 numeric +
≤6 `source_city` levels + 105 amenity levels (35 × 3) + `#training-fold localities` (the
`location` vocabulary is the training-fold locality set; full data has 1,789 localities).
No value is fixed here beyond the frozen V2.1 contract.

---

## 3. Edge types and primary composition (B3)

**Status: FROZEN.** (Protocol §8.3, §8.7, §16.)

| ID | Relation | Classification | Direction | Basis |
|---|---|---|---|---|
| **E1** | Same city | **PERMITTED** (A2 only) | Undirected (or property→city hub) | Protocol §8.3 |
| **E2** | Same locality | **PERMITTED** (A2 only) | Undirected (or property→locality hub) | Protocol §8.3 |
| **E3** | Geographic proximity | **NOT AVAILABLE** (no coordinates) | — | Protocol §6.3, §8.3 |
| **E4** | Property-characteristic similarity | **PERMITTED — PRIMARY** (A3) | **Directed**, k=5 | Protocol §8.3, §8.7, §8.8, §21 |
| **E5** | Locality/property-neighbour via training-fold aggregates | **PERMITTED** (A2 only) | Property→hub, hub↔hub | Protocol §8.3 |
| **E6** | Price similarity | **PROHIBITED** | — | Protocol §8.3 |
| **E7** | Target-derived neighbourhoods | **PROHIBITED** | — | Protocol §8.3 |

**Primary graph = E4 property-similarity graph** (Protocol §8.7, §16). Condition A2 uses only
the already-permitted E1/E2/E5 relations. E3 is unavailable; E6 and E7 are prohibited.
Multi-edges are not permitted; the graph is simple.

---

## 4. E4 property-similarity edges (B4, B5, B6)

**Status: FROZEN.** (Protocol §8.3, §8.7, §8.8, §9.4, §21.)

### 4.1 Similarity construction

| Element | Frozen value | Basis |
|---|---|---|
| Feature space | The **46 permitted non-target E4 coordinates** (Section 4.2) | Protocol §8.8 |
| Representation | **Standardised**: numeric coordinates via the frozen V2.1 `StandardScaler`; amenity coordinates via the frozen V2.1 categorical (one-hot) encoding | Protocol §8.8 |
| Distance | deterministic **Euclidean** | Protocol §8.8 |
| k | **k = 5** | Protocol §8.7, §8.8, §21 |
| Direction | **Directed** | Protocol §8.7, §8.8, §9.4 |
| `location` | **EXCLUDED** from the similarity space | Protocol §8.8 |
| `source_city` | **EXCLUDED** from the similarity space | Protocol §8.8 |
| Target-derived variables | **EXCLUDED / PROHIBITED** | Protocol §8.8, §9.3, §9.4 |
| Preprocessing fit | **Training rows only**, per fold; applied unchanged to validation/test rows | Protocol §8.8, §9.3, §9.4 |
| Tie handling | exact distance ties broken by **ascending positional node id** | Protocol §8.7, §8.8 |
| Neighbours | **unique** neighbour ids; **no duplicate edges** | Protocol §8.7, §8.8 |

### 4.2 The 46 permitted E4 coordinates (exact, Protocol §8.8)

The E4 similarity space is every frozen V2.1 feature **except** `price`, `log1p(price)`,
`derived_price_per_sqft`, any other target-derived quantity, `location`, and `source_city`:

- **basic (3):** `area`, `no_of_bedrooms`, `resale`;
- **row-wise engineered (4):** `log_area`, `log_bedrooms`, `area_per_bedroom`,
  `area_bedroom_interaction`;
- **row-wise indicators (2):** `amenity_yes_count`, `amenities_fully_specified`;
- **training-only locality aggregates (2):** `location_frequency`,
  `log_location_frequency`;
- **the 35 raw amenity columns (35):** `maintenancestaff`, `gymnasium`, `swimmingpool`,
  `landscapedgardens`, `joggingtrack`, `rainwaterharvesting`, `indoorgames`,
  `shoppingmall`, `intercom`, `sportsfacility`, `atm`, `clubhouse`, `school`,
  `24x7security`, `powerbackup`, `carparking`, `staffquarter`, `cafeteria`,
  `multipurposeroom`, `hospital`, `washingmachine`, `gasconnection`, `ac`, `wifi`,
  `children_splayarea`, `liftavailable`, `bed`, `vaastucompliant`, `microwave`,
  `golfcourse`, `tv`, `diningtable`, `sofa`, `wardrobe`, `refrigerator`.

`location` and `source_city` are excluded deliberately: the primary evaluation holds out
localities, and one-hot location identity would inject an artificial train/test category
asymmetry into the similarity distance. No new property feature is introduced. The realised
encoded dimension is fold-dependent (the training-fold level sets of the amenity
coordinates) and is recorded per fold.

### 4.3 Edge weights (B6)

**Binary:** `w_ij = 1` for every present edge. **No** distance weighting, **no** clipping,
**no** distance transformation. **Zero-distance neighbours are valid and retained.**

---

## 5. Graph aggregation and normalisation (B7)

**Status: FROZEN.** (Protocol §8.7, §17.1.)

- The propagation rule is **GraphSAGE mean-neighbourhood aggregation**.
- **No GCN-style symmetric normalisation** `A_hat = D^(-1/2) A D^(-1/2)` is used; it would
  contradict the frozen GraphSAGE architecture.
- **Self/root handling** follows the frozen GraphSAGE configuration (Protocol §8.7, §17.1):
  each node contributes a self/root term under standard GraphSAGE mean aggregation. This
  specification records the self/root term at the level the protocol fixes and does not
  extend it.
- Direction/weighting follow Sections 3–4.

---

## 6. Inductive inference contract

**Status: FROZEN.** (Protocol §9.2, §9.3, §9.4, §10.)

### 6.1 TRAIN (per frozen outer fold `k`)

```
1. Load frozen dataset; apply V2.1 clean + dedup + row-wise features
   (deterministic, no per-fold variation).                 [protocol §6.1]
2. Select train_k from the frozen fold; test_k (valid_k) is untouched.
3. Fit all learned transforms on train_k only:
   frequency counts, scalers, encoder vocabularies.        [§9.3, §9.4]
4. Build the TRAINING graph using train_k nodes only:
   - training-only preprocessing;
   - no test_k node and no test-derived statistic;
   - no edge uses price / derived_price_per_sqft / any target. [§9.3, §9.4]
5. Fit the frozen GraphSAGE architecture on
   (training graph, train_k labels).                       [§17.1]
6. Record the graph manifest + digest for the fold.        [§20]
```

### 6.2 TEST (per frozen outer fold `k`)

```
1. Take test_k rows only.
2. Transform test nodes with the FROZEN training-fitted preprocessing
   (prediction-time transformation; no test-fitted scaler). [§9.4]
3. Attach test nodes using only prediction-time covariates; each test
   node emits directed E4 edges to its k = 5 nearest permitted
   TRAINING nodes only (test→training). No test→test edge.  [§9.4, §8.8]
4. Run graph inference; emit predictions in log1p(price) space.
5. Score with the V2.1 metric contract.                     [protocol §12]
```

### 6.3 Explicitly prohibited

- test targets / test labels in the graph (train or inference);
- **test→test primary edges** (transductive only, and then only as labelled secondary);
- target-derived similarity (E6/E7);
- test-fitted scalers / encoders;
- test-derived graph statistics;
- future labels.

A transductive variant is permitted **only** as a labelled secondary sensitivity analysis and
must never be mixed with the primary result (Protocol §9.2).

---

## 7. Model architecture (B1, B2)

**Status: FROZEN.** (Protocol §17, §17.1.) **GraphSAGE is the single preregistered model
family. No architecture search is performed or permitted, and no value is selected using
`location_grouped` outer-test results.**

| Field | Frozen value |
|---|---|
| Model family | **GraphSAGE** (single preregistered family) |
| Number of message-passing layers | **2** |
| Hidden dimension | **64** |
| Aggregation | **mean-neighbourhood aggregation** (standard GraphSAGE); **no** GCN symmetric normalisation |
| Activation | **ReLU** |
| Dropout | **0.20** |
| Residual / skip connections | **none** |
| Output layer | single scalar; predicted **`log1p(price)`** |
| Optimizer | **Adam** |
| Learning rate | **0.001** |
| Weight decay | **1e-4** |
| Maximum epochs | **200** |
| Early stopping | on **training-side validation loss**, **patience = 20** |
| Loss | **MSE in `log1p(price)`** |
| Random seed | **42** |
| Self/root handling | fixed by the frozen GraphSAGE implementation (Section 5) |

No multi-architecture comparison is run. This is the single preregistered model.

---

## 8. Baseline (B9 / A0)

**Status: FROZEN.** (Protocol §7.)

The comparison baseline is the **frozen Experiment 1 CatBoost** configuration, consumed
**read-only**:

| Parameter | Value |
|---|---|
| `iterations` | 700 |
| `learning_rate` | 0.04 |
| `depth` | 7 |
| `l2_leaf_reg` | 2.0 |
| `random_seed` | 42 |
| `loss_function` | `RMSE` |

- **Not retrained or retuned** for Experiment 3 (Protocol §7).
- Per-fold baseline predictions/metrics are consumed read-only from the Experiment 1
  artifacts.
- Experiment 2 nested models are explicitly **not** the baseline (Protocol §7).

Frozen reference metrics (`std_ddof=1`, `n_folds=5`), to be re-read from the authoritative
Experiment 1 files at evaluation time (Protocol §7):

| Regime | `R2_log` mean ± SD | `MAE_log` mean ± SD | `MedAPE_percent` mean ± SD |
|---|---|---|---|
| `random` | 0.363993 ± 0.010551 | 0.465792 ± 0.007833 | 32.524296 ± 0.743129 |
| `location_grouped` | 0.267686 ± 0.037828 | 0.512773 ± 0.024097 | 37.293596 ± 2.312190 |

---

## 9. Ablations (B9)

**Status: FROZEN.** (Protocol §16.) The closed set is A0–A3. No additional ablations are
introduced.

| ID | Condition | Description |
|---|---|---|
| **A0** | Frozen Experiment 1 CatBoost baseline | Consumed **read-only**; not retrained or tuned (Section 8). |
| **A1** | **Relationship-free GraphSAGE control** | Same frozen node-feature representation and same frozen GraphSAGE architecture, but **self/root contribution only — no relational edges** (no locality edges, no similarity edges). |
| **A2** | **Locality/city relationship GraphSAGE** | Uses only the already-permitted locality/city relations **E1 / E2 / E5**; no new edge type. |
| **A3** | **E4 property-similarity GraphSAGE** | Directed **k = 5** E4, standardised non-target similarity space (Section 4), **inductive test→training relationships only**, no test→test edges. **A3 is the PRIMARY graph model.** |

A1 is the relationship-free control (self/root only), A2 isolates locality/city structure,
and A3 isolates property-similarity structure. **A1 and A3 are deliberately not the same
condition**: A1 has **no relational edges**, whereas A3 is the **E4 graph**. They must not be
conflated, and A3 is the primary model whose composition is frozen at pre-registration
(Protocol §16, §17).

---

## 10. Metrics and reporting

**Status: FROZEN.** (Protocol §12.)

- **Primary:** `R2_log`, `MAE_log`, `MedAPE_percent`.
- **Secondary:** `MAE_INR`, `RMSE_INR`, `R2_INR`, `RMSE_log`, `MAPE_percent`.
- Metrics computed by the frozen `scripts/valuation_v2_1/protocol.py` `compute_metrics` (all
  8 keys).
- **Reporting:** fold mean, sample SD (`ddof = 1`) across the 5 folds, and per-fold values.
  Folds are **never** pooled as independent observations. Regime comparison rests on paired
  per-fold differences and direction consistency (Protocol §12, §18).
- Primary regime `location_grouped`; `random` is a secondary reference (Protocol §10, §11).

---

## 11. Computational budget (B10)

**Status: FROZEN — analytical budget required before training; measured timing is a Phase 3
output.** (Protocol §21.)

### 11.1 Pre-training gate

The pre-training gate is:

> **ANALYTICAL COMPUTATIONAL BUDGET COMPLETE + ARCHITECTURE FROZEN.**

Measured per-fold and total training-time figures are **NOT** a pre-training requirement;
they are **Phase 3 outputs**.

### 11.2 Analytical estimates (established from frozen anchors)

| Quantity | Estimate | Basis |
|---|---|---|
| Graph nodes (full dataset) | **28,398** | Protocol §21 |
| Nodes per training fold (`random`) | ~22,718 | Protocol §21 |
| Nodes per test fold (`random`) | ~5,680 | Protocol §21 |
| Nodes per training fold (`location_grouped`) | 21,409–23,540 | Protocol §21 |
| Nodes per test fold (`location_grouped`) | 4,858–6,989 | Protocol §21 |
| E4 similarity edges (directed, k=5, full data) | **~141,990** (= 5 × 28,398) | Protocol §21 |
| Per training fold, E4 train→train | ~5 × |train_k| ≈ 107k–118k | derived |
| Per test fold, E4 test→train | ~5 × |test_k| ≈ 24k–35k | derived |
| E2 complete same-locality (full data, A2 bound) | 1,850,513 undirected | Protocol §21 |
| E1 complete same-city (full data; **not a default**) | 75,609,350 undirected | Protocol §21 |
| E1/E2 hub representation | O(N) ≈ 28,398 | Protocol §21 |
| Feature dimensionality / range | 46 pipeline inputs → encoded ≈ 122 + `#training-fold localities`, fold-dependent | Protocol §8.7, §8.8 |
| Storage — E4 adjacency index arrays | ~1.1 MB (2 × ~141,990 `int32`) | derived |
| Storage — feature matrix | 28,398 × d × 4 B ≈ 23 MB (d≈200) to ~193 MB (d≈1,700); 75.6M-edge graph ~600 MB excluded by default | Protocol §21 |
| Parameter estimate — frozen GraphSAGE | ≈ 128·d + 8,513 params (2 mean-aggregation layers, 64 hidden, scalar head), where d = encoded node dimension; ≈34k at d≈200, ≈252k at d≈1,900 | derived from Protocol §17.1 |
| Number of model fits | A3 primary: 5; A1–A3: 15; +`random` secondary: 30; A0: 0 (read-only); **no architecture search** | Section 9 |
| Computational scale | per-epoch O(N·d·hidden) ≈ 3.6e8 MACs at d≈200; 200 epochs × folds — CPU-feasible | derived |

### 11.3 Measured timing (Phase 3 output)

During the actual Phase 3 training run, record:

- **per-fold training time**;
- **total training time**.

These measurements are recorded as **empirical computational measurements** and **must NOT**
be used to alter the architecture, graph definition, hyperparameters, folds, or evaluation
protocol.

> **No pilot, timing-only, dry-run, or unscored training run is required or permitted solely
> to satisfy the computational-budget gate.**

All computation is CPU-feasible at this scale; GPU use is optional and must be recorded if
used (Protocol §21).

---

## 12. Reproducibility

**Status: FROZEN.** (Protocol §20, §6.5, §22.)

| Item | Value / requirement | Basis |
|---|---|---|
| Global seed | `random_state = 42` inherited; no undocumented seed | Protocol §20, §6.1 |
| Fold IDs / provenance | Frozen V2.1 folds, re-asserted in code; `random` digest `1a9f896345492d60b4a8cc677a090b27c8affd745ef302f9fbbf4ea55afc415f`; `location_grouped` digest `4c1c331cc993ddedb58b0a177e118912eb657613e3c9e4f8b5ae2bc7963ec611` | Protocol §6.5, §20 |
| Dataset hash | `1be10465221668ffde0fe7392efbc183ef77a654549a2724b26439978fad8877` | Protocol §6.1, §20 |
| Graph manifest | per fold: node ids, edge counts by type, construction parameters, content digest | Protocol §20 |
| Graph statistics | degree distribution, isolates, locality coverage | Protocol §20 |
| Model configuration | frozen before evaluation (Section 7); hashed before the first outer-fold evaluation | Protocol §17, §17.1, §20 |
| Environment | recorded environment information | Protocol §20 |
| Prediction artifacts | per-fold predictions, per-fold metrics, aggregate metrics | Protocol §20 |
| Artifact hashes | SHA-256 of every produced artifact | Protocol §20 |
| Protected state | verified unchanged before and after any Experiment 3 run (Protocol §22): blockchain state (`.chain/state.json`), `data/processed/millow_token_map.csv`, contracts, `src/config.json` (`e0b6c7d3c1eb56cad5d4d43d2e958b724cf7c74018e6c693139bec9d82610c33`), V2.1 dataset and folds (`989b79656b051b3b6c8c583b4170a5aaa1c31676260772247bbc98d1eb095156`), all Experiment 1/2 models, artifacts and tags | Protocol §22 |

Generated artifacts must remain **regenerable** from the frozen inputs and the recorded
configuration. Fold-level reporting uses per-fold values + mean ± sample SD (`ddof=1`) over
5 folds; never pooled (Protocol §12).

---

## 13. Leakage audit checklist

**Status: FROZEN.** Complete from the final protocol (Protocol §9). Every transformation is
mapped to **TRAIN-ONLY**, **PREDICTION-TIME**, or **PROHIBITED**.

| Step | Discipline | Basis |
|---|---|---|
| Load dataset; V2.1 clean + dedup | TRAIN-INDEPENDENT (deterministic whole-data preprocessing, no target) | Protocol §6.1 |
| Row-wise feature engineering | TRAIN-INDEPENDENT (pure function of the same row) | `protocol.py` |
| `location_frequency` / `log_location_frequency` | **TRAIN-ONLY** (fit on `train_k`) | Protocol §9.3, §9.4 |
| Numeric scaler (V2.1 `StandardScaler`) | **TRAIN-ONLY** | Protocol §9.3, §9.4 |
| One-hot encoder vocabulary | **TRAIN-ONLY** | Protocol §9.3, §9.4 |
| E4 kNN index | **TRAIN-ONLY** | Protocol §9.3, §9.4 |
| E4 training edges | **TRAIN-ONLY** graph | Protocol §9.2, §9.4 |
| E4 test edges | **PREDICTION-TIME** (test→training only) | Protocol §9.4 |
| E1 same-city edges (A2) | **PREDICTION-TIME AVAILABLE** (assignable feature) | Protocol §8.3, §8.4 |
| E2 same-locality edges (A2) | **PREDICTION-TIME AVAILABLE** but no unseen-locality reach | Protocol §8.6 |
| E5 locality aggregates (A2) | **TRAIN-ONLY**; prediction-time attach where attributable | Protocol §8.3 |
| Any target/price-based similarity | **PROHIBITED** | Protocol §8.3 (E6/E7), §9.3 |
| `derived_price_per_sqft` | **PROHIBITED** | Protocol §6.2, §9.4 |
| Test labels in graph | **PROHIBITED** | Protocol §9.4 |
| Test→test edges (primary) | **PROHIBITED** | Protocol §9.3, §9.4 |
| Fitting on test rows | **PROHIBITED** | Protocol §9.3, §9.4 |
| Graph defined using test results | **PROHIBITED** | Protocol §8.7, §17 |
| Selection using `location_grouped` outer test | **PROHIBITED** | Protocol §8.7, §17, §17.1 |

A transductive variant is permitted **only** as a labelled secondary sensitivity analysis,
reported under a distinct name, never mixed with the primary result (Protocol §9.2).

---

## 14. Decision gate

**IMPLEMENTATION READY: YES**

Every B1–B10 item is resolved and frozen by
`v1.5.2-experiment3-protocol-final`, and this specification is consistent with it. The
pre-training gate is the **analytical** budget (Section 11) plus the frozen architecture
(Section 7).

| # | Item | Status | Frozen value / basis |
|---|---|---|---|
| B1 | Model family | **RESOLVED** | GraphSAGE, single preregistered family, no search (Section 7; §17) |
| B2 | GNN hyperparameters | **RESOLVED** | 2 layers, 64 hidden, mean aggregation, ReLU, dropout 0.20, no skip, scalar `log1p(price)`, Adam, lr 0.001, wd 1e-4, 200 epochs, patience 20, MSE, seed 42 (Section 7; §17.1) |
| B3 | Primary edge composition | **RESOLVED** | E4 property-similarity is primary (Section 3; §8.7, §16) |
| B4 | Similarity space + metric | **RESOLVED** | 46 non-target coordinates (Section 4.2), standardised, Euclidean, `location`/`source_city` excluded, training-only fit (§8.8) |
| B5 | Tie / duplicate handling | **RESOLVED** | k=5, directed, positional-ID tie-break, unique neighbours, no duplicate edges (Section 4.1; §8.7, §8.8) |
| B6 | Edge weights | **RESOLVED** | binary `w_ij=1`, no weighting/clipping, zero-distance retained (Section 4.3; §8.7, §8.8) |
| B7 | Normalisation / aggregation | **RESOLVED** | GraphSAGE mean-neighbourhood; no GCN symmetric normalisation; self/root per frozen GraphSAGE (Section 5; §8.7, §17.1) |
| B8 | Node-feature representation | **RESOLVED** | frozen V2.1 contract; fold-dependent encoded dimension documented; no target-derived features (Section 2; §8.7) |
| B9 | Ablation mapping | **RESOLVED** | A0 CatBoost read-only; A1 relationship-free (self/root only); A2 E1/E2/E5; A3 E4 primary; A1 ≠ A3 (Section 9; §16) |
| B10 | Computational budget | **RESOLVED** | analytical budget is the pre-training gate; measured timing is a Phase 3 output (Section 11; §21) |

No scientific choice has been introduced beyond what `v1.5.2` already freezes. No
implementation, graph, training or evaluation is performed by this document.

---

## 15. Version provenance

- Protocol: `docs/research/EXPERIMENT_3_SPATIAL_MODELING_PROTOCOL_DRAFT.md`
- Tag: **`v1.5.2-experiment3-protocol-final`**
- Commit: **`aee1055a0a4d1bf8a6e48a1eb54ffe155bdd54e2`**
- SHA-256: **`5526dbae1f54430854f747ff2a94f193b612ec4e1ae505f50ce2a41feb386866`**
- State: **Pre-registered before Experiment 3 training.**

This specification is written against the final protocol only; no earlier protocol revision
is referenced.

---

## Appendix A — Actions explicitly not taken

- No model was trained or instantiated.
- No graph was built (including no kNN index, no adjacency, no hub nodes).
- No graph/model artifact was generated.
- No Experiment 3 evaluation or `location_grouped` outer-test inspection was performed.
- No Experiment 1 / Experiment 2 file, dataset, fold, blockchain state or `src/config.json`
  was modified.
- No commit or tag was created.
