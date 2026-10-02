# Experiment 3 — Spatial / Property-Relationship Modeling Protocol (DRAFT)

## 1. Status

| Field | Value |
|---|---|
| Document | `docs/research/EXPERIMENT_3_SPATIAL_MODELING_PROTOCOL_DRAFT.md` |
| State | **DRAFT — design only. No implementation, no training, no results.** |
| Authoring date | 2026-10-02 |
| Pre-registration | Not yet frozen. This draft must be reviewed and approved before any code is written. |
| Depends on | Experiment 1 (valuation backbone), Experiment 2 (uncertainty characterisation) |
| Research checkpoints at authoring time | `v1.2.0-experiment1-xgb-anchor`, `v1.3.0-experiment1-catboost-backbone`, `v1.4.0-experiment2-conformal`, `v1.4.1-experiment2-results` |
| Blockchain impact | None. No blockchain state is read, written, or transacted against. |

> **No-result prediction.** The experiment does not assume that relationship-aware
> modeling will outperform CatBoost. Improvement, no measurable improvement, or
> degradation are all admissible outcomes.

This document defines research questions, hypotheses, data and fold contracts, candidate
graph constructions, leakage controls, evaluation, ablations, failure criteria,
reproducibility and the computational budget. It trains nothing and produces no artifacts.

---

## 2. Research Motivation

Experiment 1 compared valuation models and froze **CatBoost** as the V2.1 valuation
backbone. Experiment 2 placed conformal uncertainty around nested CatBoost models and
established:

- random-regime marginal coverage close to nominal for the evaluated methods;
- location-grouped marginal coverage changing modestly for most methods;
- **strong conditional coverage non-uniformity across price bands and cities**;
- failure of the pre-registered usability gate (`median_relative_width ≤ 2.00` at 90%) for
  every method in every fold;
- important **locality structure** in the data (1,789 localities, median size 3, highly
  skewed; 6 cities).

Experiment 2 treated properties as exchangeable rows within a fold (the conformal methods
used per-row or per-stratum scores). It did **not** model relationships *between* properties
or localities. The next research layer is therefore to ask whether explicitly representing
**spatial / property relationships** — locality structure, city structure, and
property-characteristic similarity — changes valuation generalization under
**location-held-out** evaluation, where generalization is hardest.

This is a modeling-layer question. It is distinct from the uncertainty layer (Experiment 2)
and from the downstream decision/escrow layer (future work).

---

## 3. Primary Research Question

> **Does incorporating spatial/property relationships improve valuation generalization
> under location-held-out evaluation compared with the frozen CatBoost backbone?**

Operational meaning of each clause:

- **incorporating spatial/property relationships** — a model whose inputs or architecture
  explicitly encode relations between properties/localities, as defined in Section 8;
- **valuation generalization** — out-of-fold predictive accuracy on unseen localities,
  measured by the metrics in Section 12;
- **location-held-out evaluation** — the frozen `location_grouped` regime of V2.1
  (Section 10);
- **frozen CatBoost backbone** — the Experiment 1 CatBoost configuration, consumed
  read-only as the baseline (Section 7).

This question is deliberately **not** phrased as "Will the GNN perform better?", "Can the
GNN beat CatBoost?", or "Which model is best?". Those framings presuppose an outcome and a
family. The experiment must be able to return a negative answer with equal validity.

---

## 4. Secondary Research Questions

A small, closed set. No additional RQs are added without amending this document before any
training.

| ID | Question |
|---|---|
| **RQ2** | Does relationship-aware modeling **reduce the degradation** observed when moving from random evaluation to location-held-out evaluation? |
| **RQ3** | Do spatial/property relationships improve performance for **localities with limited training observations**? |
| **RQ4** | Does the relationship-aware model change **error behaviour across price bands** relative to the baseline? |
| **RQ5** | Are any observed improvements **robust across held-out folds** rather than driven by one or two folds? |
| **RQ6** | **What types** of graph construction and relationship information contribute to the observed result? |

RQ2 is answered by the paired random-vs-location-grouped comparison (Section 13). RQ3 is
answered by the limited-locality analysis (Section 14). RQ4 by the price-band analysis
(Section 15). RQ5 by fold-level paired differences and direction consistency (Sections 13
and 18). RQ6 by the ablation design (Section 16).

---

## 5. Hypotheses

**H1 (directional, testable).** A relationship-aware model may improve location-held-out
generalization by exploiting information shared between related properties and localities —
for example, that properties in the same micro-market share location-driven price level and
that localities with similar attributes behave similarly.

**H0 (null).** The relationship-aware model does **not** improve the predefined
generalization metrics relative to the frozen CatBoost baseline, under location-held-out
evaluation.

Clarifications that are binding on interpretation:

- H1 is a **testable hypothesis, not an expected outcome**. The word "may" is deliberate.
- The null and the negative result are **scientifically admissible**. A clean demonstration
  that relationship modeling does not help at this data scale and locality structure is a
  valid Experiment 3 outcome, not a failure of the experiment.
- H1 is directional only in the sense that it names a plausible mechanism; it does not
  license a one-sided test that ignores degradation (see Section 18).

---

## 6. Dataset and Frozen Protocol

All inherited items are **read-only**. Experiment 3 may not regenerate, extend, tune or
overwrite them.

### 6.1 Dataset

| Element | Value |
|---|---|
| File | `data/processed/MREID_property.csv` |
| SHA-256 | `1be10465221668ffde0fe7392efbc183ef77a654549a2724b26439978fad8877` |
| Post-deduplication rows | **28,398** |
| Duplicates previously removed | 737 exact on `(source_city, location, area, no_of_bedrooms, price)`, keep first |
| Target | `price` |
| Training / prediction space | `log1p(price)` |
| Presentation space | INR via `max(expm1(·), 0)` |
| Feature count | **48** (11 numeric + 37 one-hot); pipeline inputs **46** (the 2 train-only frequency features are derived inside the pipeline) |
| Group key | `source_city + '__' + location` |
| `random_state` | 42 |

### 6.2 Excluded raw columns (must not be reintroduced)

| Column | Reason |
|---|---|
| `mreid_id` | unique identifier; no signal |
| `source_file` | scrape provenance; near-duplicate of `source_city` |
| `derived_price_per_sqft` | equals `price / area`; using it **is target leakage** |
| `group` | split key only |

Text normalisation carried forward: `location` and `source_city` are
`astype(str).str.strip().str.lower()`.

### 6.3 No reliable geographic coordinates

**Important design fact.** The dataset contains **no latitude/longitude or any geographic
coordinate field**. Its columns are the identifier, price, area, location label,
bedrooms, resale flag, 35 amenity flags, `source_city`, `source_file`, and the
target-derived `derived_price_per_sqft`.

Therefore **geographic-distance edges are unavailable** and must not be invented
(Section 8, E3). Reliable geographic coordinates are absent from the frozen dataset, so
**no edge type may assume that physical distance can be computed.** Spatial structure can
only be expressed through the categorical `source_city` / `location` keys and through
property-characteristic similarity. Where this document refers to spatial modelling, the
intended term is **"spatial / property-relationship modelling"**, which does not imply that
geographic distance is available.

### 6.4 Locality structure (frozen, post-deduplication)

| Statistic | Value |
|---|---|
| Localities (`source_city__location`) | 1,789 |
| Min / median / max locality size | 1 / 3 / 685 |
| Localities with ≥ 2 rows | 1,149 (64.2%) |
| Localities with ≥ 5 rows | 713 (39.9%) |
| Localities with ≥ 10 rows | 499 (27.9%) |
| Localities with ≥ 50 rows | 139 (7.8%) |
| Cities | 6 |

The median locality has **3 rows**, which constrains every locality-based graph
construction and motivates the limited-locality analysis (Section 14).

### 6.5 Frozen folds

The V2.1 fold definitions are read-only and **must not be regenerated**:

| Regime | Splitter | Frozen digest |
|---|---|---|
| `random` | `KFold(shuffle=True)`, `random_state=42` | `1a9f896345492d60b4a8cc677a090b27c8affd745ef302f9fbbf4ea55afc415f` |
| `location_grouped` | `GroupKFold(shuffle=True)` on `group` | `4c1c331cc993ddedb58b0a177e118912eb657613e3c9e4f8b5ae2bc7963ec611` |

Experiment 3 uses the **same frozen outer folds** as Experiment 1 and Experiment 2. It does
not create a new split, including any split that would produce a more convenient result.

---

## 7. Baseline

**Primary baseline:** the frozen Experiment 1 CatBoost configuration, consumed read-only.

| Parameter | Value |
|---|---|
| `iterations` | 700 |
| `learning_rate` | 0.04 |
| `depth` | 7 |
| `l2_leaf_reg` | 2.0 |
| `random_seed` | 42 |
| `loss_function` | `RMSE` |

**Do not retrain or tune the baseline for Experiment 3.** The Experiment 1 location-grouped
evaluation is the main generalization baseline. If per-fold baseline predictions and metrics
already exist, they are consumed read-only from the Experiment 1 artifacts; they are not
recomputed.

Reference Experiment 1 metrics for the frozen backbone (`std_ddof = 1`, `n_folds = 5`), to
be re-read from the authoritative Experiment 1 files at evaluation time rather than trusted
from this table:

| Regime | `R2_log` mean ± SD | `MAE_log` mean ± SD | `MedAPE_percent` mean ± SD |
|---|---|---|---|
| random | 0.363993 ± 0.010551 | 0.465792 ± 0.007833 | 32.524296 ± 0.743129 |
| location_grouped | 0.267686 ± 0.037828 | 0.512773 ± 0.024097 | 37.293596 ± 2.312190 |

Experiment 2 nested models are **not** the Experiment 3 baseline. They were fitted on
`T'_k ⊂ train_k` and are therefore not comparable on equal footing to a model trained on all
of `train_k`. Experiment 3 compares against the Experiment 1 frozen backbone.

---

## 8. Graph Construction Candidates

### 8.1 Node definition

The primary node is an **individual property listing** (one row of the processed dataset).
Locality and city are represented as relations (edges or auxiliary nodes), not as the
primary nodes, so that held-out localities remain first-class test nodes.

### 8.2 Design principle

No graph structure is adopted without documenting (a) where the relation information comes
from, (b) whether it is available at prediction time for a genuinely new property,
(c) whether it can carry target information, and (d) whether it crosses the train/test
boundary. Every candidate below is evaluated against those four questions before
implementation.

### 8.3 Candidate edge types

| ID | Relation | Source information | Construction rule | Available at prediction time? | Target-leakage risk | Crosses train/test boundary? | Cost | Expected interpretation |
|---|---|---|---|---|---|---|---|---|
| **E1** | Same city | `source_city` | Connect two properties iff same city (optionally: connect property to a city hub, not all pairs) | Yes (city is a feature) | None (no price used) | Only if constructed globally; must be built per fold | All-pairs is O(N²); city-hub is O(N) | Coarse regional shared structure |
| **E2** | Same locality | normalised `source_city__location` | Connect two properties iff same locality (or property→locality hub) | Yes (locality is a feature) | None | Must be built per fold; a locality may not span train and test in `location_grouped` | locality-hub O(N); complete-within-locality sum of C(n_g,2) | Micro-market shared structure |
| **E3** | Geographic proximity | — | **NOT AVAILABLE** | — | — | — | — | Excluded; no reliable coordinates exist |
| **E4** | Property-characteristic similarity | frozen non-target V2.1 attribute vector (excludes `price`, `derived_price_per_sqft`, `location`, `source_city`; Section 8.8) | **directed** k-nearest neighbours, **k = 5**, in a standardised attribute space; deterministic positional-ID tie-break; unique neighbours; no duplicate edges | Yes | None if labels are never used; scaler/encoder must be fit on training rows only | Must restrict edges to avoid test→test label paths (see Section 9); directed | ~k·N = ~141,990 directed emissions (Section 21) | Similar-property analogues; **PRIMARY** relation (A3) |
| **E5** | Locality/property-neighbour relationships from training data | training-fold aggregates | Auxiliary locality nodes with attributes = training-fold attribute means/counts; connect properties to their locality hub; connect localities by attribute similarity | Yes for cities/localities attributable from features; for unseen test localities the hub is absent | None if aggregates use features only, never price | Built only from training folds | O(N) | Learned locality-level context |
| **E6** | Price similarity | — | **PROHIBITED** | — | Direct target leakage | — | — | Not permitted |
| **E7** | Target-derived neighbourhoods | — | **PROHIBITED** (kNN on price, price rank, price quantile) | — | Direct target leakage | — | — | Not permitted |

### 8.4 Key scientific subtlety: locality edges cannot reach an unseen locality

Under `location_grouped`, a held-out property's locality does **not** appear in training.
Consequently:

- **E2 (same-locality) edges contribute nothing directly to a held-out locality node**,
  because no training node shares that locality. E2 can still shape the model's shared
  representation through training-side locality structure.
- **E1 (same-city) edges remain available** for a held-out locality, because other
  localities in the same city are in training. This is legitimate: `source_city` is an
  assignable feature and would be known for a genuinely new property.
- **E4/E5 edges remain available**, since they depend on property attributes, not on the
  held-out locality being seen.

This asymmetry is itself part of RQ6: if the relationship model helps under
location-held-out evaluation, the ablation must show *which* relations carry the effect.

### 8.5 Density caution

Complete same-locality edges over the full dataset would be 1,850,513 undirected edges; a
fully connected same-city graph would be 75,609,350 undirected edges (Section 21). All-pairs
city edges are therefore **not** a default design. Where a super-node (hub) representation is
sufficient, it is preferred.

### 8.6 Same-locality edge limitation (scientific property, not a bug)

Under `location_grouped` evaluation, an **unseen test locality has no same-locality training
nodes by construction**, because `group(train) ∩ group(test) = ∅`. Consequently
same-locality edges **cannot provide cross-locality information for the primary
location-held-out test**. This is a **scientific property of the experiment, not an
implementation bug**, and it is precisely the condition the primary research question
targets.

It follows that if the model is to transfer information to an unseen locality, it must do so
through a relationship that is **explicitly available at prediction time for a genuinely new
property** — for example property-characteristic similarity (E4), same-city structure (E1),
or training-fold-derived locality hubs/attributes (E5). Same-locality edges (E2) may still
shape the model's shared representation through training-side locality structure, but they
provide no direct neighbourhood for a held-out locality.

### 8.7 Graph specification freeze (binding pre-implementation gate)

**Before any Experiment 3 training**, the graph definition must be frozen in full. The
frozen specification is:

- **Node definition.** One node per processed property row (Section 8.1). The node id is the
  positional row index of the deterministic V2.1 frame; the 737 exact duplicates are removed
  before any node exists (Section 6.1).
- **Node-feature set (model input).** The frozen V2.1 feature contract, read-only
  (`pipeline_input_columns`: 46 columns; the two frequency features are derived
  training-only by `TrainOnlyFrequencyFeatures`). No target-derived field is used. The
  encoded dimension is fold-dependent and is recorded per fold.
- **Edge definitions.** The primary relationship (A3, Section 16) is **E4
  property-characteristic similarity**. Condition A2 uses only the already-permitted
  locality/city relations E1, E2 and E5. E3 is not available; E6 and E7 are prohibited
  (Section 8.3).
- **k.** For E4, **k = 5**, directed (each node emits edges to its five nearest permitted
  neighbours); see Section 21.
- **Edge weighting.** Binary: `w_ij = 1` for every present edge. No distance weighting, no
  clipping and no distance transformation. Zero-distance neighbours are valid and retained.
- **Direction.** E4 is **directed**. Training nodes emit directed edges to training
  neighbours; a test node emits directed edges only to training nodes (Section 9.4). No
  test→test E4 edge exists in the primary design.
- **Self-loop / root handling.** Under GraphSAGE mean aggregation each node contributes a
  self/root term; the exact self/root handling is fixed by the frozen GraphSAGE
  configuration (Section 17.1).
- **Normalisation / aggregation.** The propagation rule is **GraphSAGE mean-neighbourhood
  aggregation**. Symmetric GCN-style normalisation `A_hat = D^(-1/2) A D^(-1/2)` is **not**
  used, because it would make the layer GCN-style and contradict the frozen GraphSAGE
  architecture.
- **Determinism.** Exact distance ties are broken by ascending positional node id; neighbour
  ids are unique; duplicate edges are not permitted.
- **Graph construction procedure.** Every learned transform (scaler/encoder/frequency) and
  the E4 kNN index are fitted on training rows only and applied unchanged to validation/test
  rows. No test-derived statistic enters any graph and no future/test-label information is
  used.
- **E4 similarity coordinates.** The exact permitted non-target coordinates are frozen in
  Section 8.8 and are identical in definition across folds.

**No graph definition may be selected using `location_grouped` test results.** The frozen
graph specification, together with a content digest, is part of the pre-registration and is
verified unchanged before evaluation.

### 8.8 E4 similarity coordinates (frozen)

The E4 property-similarity space uses only non-target attributes from the frozen V2.1
feature contract. The exact permitted coordinates are every V2.1 feature **except** `price`,
`log1p(price)`, `derived_price_per_sqft`, any other target-derived quantity, `location`, and
`source_city`:

- basic: `area`, `no_of_bedrooms`, `resale`;
- row-wise engineered: `log_area`, `log_bedrooms`, `area_per_bedroom`,
  `area_bedroom_interaction`;
- row-wise indicators: `amenity_yes_count`, `amenities_fully_specified`;
- training-only locality aggregates: `location_frequency`, `log_location_frequency`;
- the 35 raw amenity columns: `maintenancestaff`, `gymnasium`, `swimmingpool`,
  `landscapedgardens`, `joggingtrack`, `rainwaterharvesting`, `indoorgames`,
  `shoppingmall`, `intercom`, `sportsfacility`, `atm`, `clubhouse`, `school`,
  `24x7security`, `powerbackup`, `carparking`, `staffquarter`, `cafeteria`,
  `multipurposeroom`, `hospital`, `washingmachine`, `gasconnection`, `ac`, `wifi`,
  `children_splayarea`, `liftavailable`, `bed`, `vaastucompliant`, `microwave`,
  `golfcourse`, `tv`, `diningtable`, `sofa`, `wardrobe`, `refrigerator`.

That is **46 coordinates** (the 48 V2.1 features minus `source_city` and `location`).
`location` and `source_city` are excluded deliberately: the primary evaluation holds out
localities, and one-hot location identity would inject an artificial train/test category
asymmetry into the similarity distance. No new property feature is introduced.

Preprocessing is training-only and identical in definition across folds:

- the numeric coordinates are standardised with the frozen V2.1 `StandardScaler`, and the
  amenity coordinates use the frozen V2.1 categorical (one-hot) encoding; both are **fitted
  on training rows only** and applied unchanged to validation/test rows, with no test-fitted
  statistic;
- distance is deterministic **Euclidean**; **k = 5**, directed;
- exact distance ties are broken by ascending positional node id; neighbour ids are unique;
  duplicate edges are not permitted;
- zero-distance neighbours are valid and retained;
- every E4 edge weight is `w_ij = 1` (binary); no distance weighting, no clipping and no
  distance transformation.

The realised encoded dimension is fold-dependent (the training-fold level sets of the
amenity coordinates) and is recorded per fold.

---

## 9. Leakage Model

This section is binding and is the primary correctness gate (F0/F4).

### 9.1 Fold boundary

For every outer fold, the graph used for fitting must respect the location-held-out
boundary. For the primary `location_grouped` experiment:

```
group(train) ∩ group(test) = ∅
```

The graph construction must not allow **target information** from test properties to enter
training, directly or by path.

### 9.2 Transductive vs inductive

| | **A. Transductive** | **B. Inductive** |
|---|---|---|
| Graph at fit time | Training and test nodes both present | Training nodes only |
| Test features in construction | Yes | No (test nodes attached at inference) |
| Test labels in construction | Never | Never |
| Deployment semantics | Requires the full test batch before prediction | Predicts a genuinely new property |
| Role in Experiment 3 | Secondary sensitivity analysis | **Primary design** |

**Primary choice: inductive.** Because the research question concerns *unseen localities*,
and because a real valuation request arrives for a property not in any batch, the primary
design fits the model on a training-only graph and attaches test nodes at inference using
only prediction-time-available relations. A transductive variant may be run as a labelled
secondary sensitivity analysis to quantify how much of any effect depends on seeing test
covariates during construction; it must be reported under a distinct name and never mixed
with the primary result.

### 9.3 Rules (both designs)

1. No edge may be defined using `price` or any target-derived quantity.
2. Feature standardisation and any attribute scaler are fit on training rows only.
3. `TrainOnlyFrequencyFeatures` semantics are preserved; no full-dataset aggregate is
   consulted for any feature.
4. In the inductive design, test nodes may connect only to training nodes (or to hubs built
   from training nodes), never to other test nodes.
5. Cross-boundary edges are permissible only when the relation is prediction-time
   available and target-free (e.g. same-city). Their scientific effect is isolated by
   ablation (Section 16).
6. Graph construction is recorded per fold in a graph manifest with node/edge counts and a
   content digest (Section 20).

### 9.4 Inductive inference contract (`location_grouped`)

For the primary inductive design, the following contract is binding and distinguishes
**training graph construction** from **test / inference graph construction**.

**Training graph construction.**

- May use **training rows only**.
- **Test targets are never used**, and no target-derived quantity is ever used.
- No edge may be defined using `price`, `derived_price_per_sqft`, or any label-derived
  quantity.
- Any scaler, encoder, feature standardisation, or training-fold aggregate is fit on
  training rows only.

**Test / inference graph construction.**

- Test-node edges may use **only prediction-time covariates** that would be known for a
  genuinely new property (for example `source_city`, normalised `location`, `area`,
  bedrooms, amenities).
- **No test labels or target-derived quantities may influence edges.**
- If kNN / property similarity is used, the rule is fixed in advance; the primary inductive
  rule is that **test nodes may connect only to training nodes** (or to hubs built from
  training nodes) and **never to other test nodes**. A variant permitting test→test edges is
  transductive and belongs to the labelled secondary sensitivity analysis only.
- For E4 specifically (Section 8.8), the relation is **directed**: each **training** node
  emits directed edges to its **k = 5** nearest permitted training neighbours, and each
  **test** node emits directed edges to its **k = 5** nearest permitted **training** nodes
  only (test→training). No test→test E4 edge exists in the primary design.
- Self-loops, edge weighting/direction, and normalisation follow the frozen graph
  specification (Section 8.7).

This contract makes the primary result interpretable as "predicting a genuinely new
property", which is the deployment setting the research question concerns.

---

## 10. Primary Experimental Design

| Element | Choice |
|---|---|
| Folds | Frozen V2.1 folds, unchanged |
| Primary regime | `location_grouped` |
| Training set | `train_k` (all of it), as in Experiment 1 |
| Test set | `valid_k`, bit-identical to the frozen fold |
| Graph | Inductive, training-only at fit time (Section 9) |
| Comparison | Frozen Experiment 1 CatBoost, per-fold and aggregate |
| Test rows | Identical for both models |

`location_grouped` is the primary regime because the research question is about unseen
localities. It is the harder and more informative setting, not a setting chosen for
convenience.

---

## 11. Secondary / Random Evaluation

The `random` regime is retained as a **secondary reference** to support RQ2 (does
relationship modeling reduce the random→location-grouped degradation?). It is reported with
the same folds, metrics and estimators as the primary regime. It is **not** used to select
the architecture against the `location_grouped` test folds (Section 17).

---

## 12. Metrics

Identical to the established Experiment 1 metric contract. No composite score and no
overall ranking score are created.

**Primary:** `R2_log`, `MAE_log`, `MedAPE_percent`.

**Secondary:** `MAE_INR`, `RMSE_INR`, `R2_INR`, `RMSE_log`, `MAPE_percent`.

**Reporting (all metrics):** fold mean, sample SD (`ddof = 1`) across the 5 folds, and the
per-fold values. Folds are **never** pooled as if they were independent observations. Any
regime comparison rests on paired per-fold differences and their direction consistency, not
on an SD magnitude alone.

---

## 13. Generalization Analysis

For both the frozen CatBoost and the relationship-aware model, explicitly compare
`random → location_grouped`:

- metric degradation (per metric, per fold, and fold-mean);
- fold-level **paired** differences (same fold, same metric);
- direction consistency (how many folds move the same way);
- variance across folds.

No claim is made from a single aggregate number. A claimed improvement under shift must be
visible in the paired per-fold differences and be directionally consistent, or it is
reported as fold-driven (see F3, Section 19).

---

## 14. Limited-Locality Analysis

Because locality size is highly skewed (median 3; 640 localities have exactly 1 row; 1,805
rows live in localities with fewer than 5 rows), the experiment defines a limited-locality
diagnostic.

- **Strata are defined using training-side information only.** Candidate stratifiers: the
  training-fold count of the property's locality and/or city (the same quantity the V2.1
  `location_frequency` feature already expresses for training localities). Test-set target
  values are never used to define strata.
- **Minimum sample requirements are documented before analysis.** A stratum is treated as
  inferentially usable only if it contains a pre-declared minimum number of test rows across
  the folds; otherwise the stratum is reported descriptively only.
- **Power honesty.** If a stratum is too small to support inference, it is reported as a
  descriptive characterisation, not forced into a significance claim. Under
  `location_grouped`, every test locality is unseen by construction, so "training
  representation" means the size of *other* localities available to the training graph and
  the presence/absence of same-city training nodes, not the test locality's own count.

Answers RQ3.

---

## 15. Price-Band Analysis

Retained because Experiment 2 found strong conditional coverage differences by price level.

- Use **prediction-time-assignable** bands wherever a band is used as a model input or as a
  graph construction key (for example, bands of the model's predicted price or of
  `area`/attribute ranges).
- **True test price must not determine model inputs or graph construction.**
- True-price bands may be retained **only as a post-hoc diagnostic**, clearly labelled,
  in the same spirit as Experiment 2's diagnostic band reporting.

Answers RQ4.

---

## 16. Ablations

A small, closed set. Each ablation must answer a scientific question and its graph
construction must be leakage-safe. The experiment is not exploded into dozens of variants.

| ID | Model | Answers |
|---|---|---|
| **A0** | Frozen Experiment 1 CatBoost baseline (consume read-only, no retraining or tuning) | Reference |
| **A1** | **Relationship-free GraphSAGE control**: same frozen node-feature representation and same frozen GraphSAGE architecture, but **self/root contribution only — no relational edges** (no locality edges, no similarity edges) | Does relationship structure improve valuation beyond the raw property features? |
| **A2** | **Locality/city relationship GraphSAGE condition**: uses only the already-permitted locality/city relations **E1 / E2 / E5**, as applicable; no new edge type | Do locality/city relationships improve valuation generalization? |
| **A3** | **E4 property-similarity GraphSAGE condition**: directed **k = 5**, standardised non-target property-similarity space (Section 8.8), test→training only during primary inductive inference, no test→test edges. **A3 is the PRIMARY graph model.** | Do property-similarity relationships improve valuation generalization? |

A1 is the relationship-free control (self/root only), A2 isolates locality/city structure, and
A3 isolates property-similarity structure; the three conditions differ only in relationship
structure, not by post-hoc tuning. A1 and A3 are deliberately **not** the same condition: A1
has no relational edges, whereas A3 is the E4 graph. Ablations A1–A3 must not be conflated
with one another, and A3 is the primary model whose composition is frozen at pre-registration
(Section 17). If A1 captures nearly all of any effect, RQ6's answer is that relational
structure was not the source.

---

## 17. Model-Selection Procedure

**GraphSAGE, GAT and GCN were candidate model families.** The pre-registration selects
**GraphSAGE** as the single model family, with **no architecture search** and **no selection
against the `location_grouped` outer test folds**. There is no second architecture and no
post-hoc family choice; the frozen configuration is part of the pre-registration
(Section 17.1).

- The model class and the exact architecture are both fixed before implementation.
- The final architecture is frozen **before evaluation**, as recorded in Section 17.1.
- **No hyperparameter tuning against the `location_grouped` test folds** is permitted. Any
  training-side choice uses only a **training-side validation procedure** — for example, an
  inner split of `train_k` (comparable in spirit to Experiment 2's nested `T'_k / C_k`
  construction) — and **must never inspect `location_grouped` outer test metrics**.
- The chosen architecture, seeds, and all hyperparameters are frozen and hashed before the
  first evaluation on the frozen outer folds.

### 17.1 Model-selection freeze (binding)

The single preregistered architecture is frozen as follows. No architecture search is
performed, and no value below is selected using `location_grouped` outer-test results.

| Field | Frozen value |
|---|---|
| Model family | **GraphSAGE** |
| Number of message-passing layers | **2** |
| Hidden dimension | **64** |
| Aggregation | **mean-neighbourhood aggregation** (standard GraphSAGE); **no** GCN-style symmetric normalisation |
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
| Self/root handling | fixed by the frozen GraphSAGE implementation and recorded in the implementation specification |

No multi-architecture comparison is run. The architecture above is the single preregistered
model.

---

## 18. Statistical Comparison

Defined in advance; no arbitrary "wins".

**Primary comparison:** paired fold-level metric differences between the relationship-aware
model and the frozen CatBoost baseline, on the same fold and the same test rows.

Report for each primary metric:

- mean paired difference across folds;
- the five fold-level differences;
- sample SD (`ddof = 1`) of the differences;
- sign consistency (how many folds share the mean direction).

**If a formal test is proposed**, it is specified here, before training, with its
assumptions stated. Given only five folds, any test has very low power; a single test result
is not used to manufacture significance. If no test is defensible, the result is reported
descriptively with the fold-level differences shown in full. **No significance is
manufactured from five folds.**

---

## 19. Failure Criteria

Objective outcomes, fixed before training. A negative scientific result remains valid; F1
is **not** defined as "the GNN performed worse".

| ID | Condition | Interpretation |
|---|---|---|
| **F0** | Implementation/data-leakage checks fail (fold boundary violated, target-derived features used, test labels reachable, protected state changed) | Invalid; discard all results |
| **F1** | The relationship-aware model does **not improve** the predefined primary `location_grouped` metric(s) relative to the frozen CatBoost baseline | Valid negative result; relationship modeling did not help at this scale/structure |
| **F2** | Improvement occurs **only** in `random` evaluation but not in `location_grouped` | Improvement does not transfer to the unseen-locality setting |
| **F3** | Improvement is **driven by one fold** (fails direction-consistency / fold-robustness) | Not a robust result; reported as fold-driven |
| **F4** | Graph construction introduces leakage or depends on unavailable prediction-time information | Design invalid for the affected variant |
| **F5** | Computational instability or reproducibility failure (non-deterministic results, un-reproducible artifacts) | Invalid; fix before interpreting |

F0, F4 and F5 invalidate the affected run. F1, F2 and F3 are **findings**, not failures of
the experiment.

---

## 20. Reproducibility

Required for any implementation:

- deterministic seeds (`random_state = 42` inherited; no undocumented seed);
- frozen fold provenance (V2.1 digests, re-asserted in code);
- a **graph-construction manifest** per fold (node ids, edge counts by type, construction
  parameters, content digest);
- **graph statistics** (degree distribution, isolates, locality coverage);
- model configuration (frozen before evaluation) and environment information;
- per-fold predictions, per-fold metrics, and aggregate metrics;
- hashes of all produced artifacts.

Explicitly recorded identifiers:

| Item | Source |
|---|---|
| Dataset hash | `data/processed/MREID_property.csv` → `1be10465221668ffde0fe7392efbc183ef77a654549a2724b26439978fad8877` |
| Frozen fold digest | `random` → `1a9f8963…c415f`; `location_grouped` → `4c1c331c…3ec611` |
| Baseline model identifiers | Experiment 1 CatBoost artifacts and their manifests |
| Graph manifest hash | produced by Experiment 3 (not yet existing) |
| Experiment configuration hash | produced by Experiment 3 (not yet existing) |

Generated artifacts must remain **regenerable** from the frozen inputs and the recorded
configuration. Protected-state verification is run before and after any implementation.

---

## 21. Computational Budget

The budget must be documented **before training begins**. Node counts are exact from the
frozen data; edge counts are design-dependent and are given as bounds for the candidate
constructions.

| Quantity | Estimate | Basis |
|---|---|---|
| Graph nodes (full dataset) | 28,398 | frozen post-dedup rows |
| Nodes per training fold (`random`) | ~22,718 | frozen `train_k` |
| Nodes per test fold (`random`) | ~5,680 | frozen `valid_k` |
| Nodes per training fold (`location_grouped`) | 21,409–23,540 | frozen `train_k` ranges |
| Nodes per test fold (`location_grouped`) | 4,858–6,989 | frozen `valid_k` ranges |
| E2 complete same-locality edges (full data) | **1,850,513** undirected | sum of C(n_g,2) over 1,789 localities |
| E1 complete same-city edges (full data) | **75,609,350** undirected (73.76M beyond same-locality) | all pairs within each city — **not a default** |
| E1/E2 hub representation | O(N) = ~28,398 edges | property→city or property→locality hub |
| E4 similarity edges (directed, k = 5) | ~141,990 | 5 × 28,398 |

Indicative memory: a sparse edge list of 1.85M undirected edges is ~30 MB as two `int32`
index arrays; the 75.6M-edge all-pairs city graph is ~600 MB and is excluded by default.
Feature matrices are small (28,398 × tens of features). Training-time and storage estimates
must be completed once the architecture is frozen; **no training may begin until this
section is filled with measured per-fold and total time figures**. All computation is
CPU-feasible at this scale; GPU use is optional and must be recorded if used.

---

## 22. Protected-State Requirements

The following are **read-only** and must be verified unchanged before and after any
Experiment 3 run:

- blockchain state (`.chain/state.json`) and all blockchain/token/escrow state;
- `data/processed/millow_token_map.csv`;
- contracts;
- `src/config.json` (SHA-256 `e0b6c7d3c1eb56cad5d4d43d2e958b724cf7c74018e6c693139bec9d82610c33`);
- V2.1 dataset and frozen folds (`artifacts/valuation/v2_1/cv_folds_v2_1.json`,
  digest `989b79656b051b3b6c8c583b4170a5aaa1c31676260772247bbc98d1eb095156`);
- all Experiment 1 models and results;
- the Experiment 2 protocol, models and artifacts;
- all existing Git tags (`v1.0.0-reproducible-state`, `v1.1.0-research-checkpoint`,
  `v1.2.0-experiment1-xgb-anchor`, `v1.3.0-experiment1-catboost-backbone`,
  `v1.4.0-experiment2-conformal`, `v1.4.1-experiment2-results`).

**Experiment 3 must not touch blockchain state. No blockchain transactions are required.**

---

## 23. Limitations

1. **No geographic coordinates.** Spatial structure is limited to categorical locality/city
   keys and attribute similarity. True geographic proximity cannot be modelled.
2. **Median locality size is 3.** Locality-level signal is sparse; relationship edges for
   small localities are thin, and many localities (640) have a single row.
3. **The baseline is frozen, not retuned.** Any comparison is against the Experiment 1
   CatBoost as-is; a differently tuned CatBoost is out of scope.
4. **Five folds.** Variance estimates are weak, and fold-level robustness (RQ5) is limited
   by the number of folds.
5. **Inductive restriction.** The primary design cannot use test-set covariates during
   construction, which is the honest deployment setting but may understate transductive
   upper bounds; the transductive variant is secondary only.
6. **Single dataset and single backbone family.** Results generalise to this dataset and
   this baseline, not to real-estate valuation in general.
7. **Segmentation is exploratory.** Limited-locality and price-band analyses are
   exploratory; no multiplicity correction is planned unless pre-registered, and subgroup
   claims must be labelled accordingly.
8. **Model-class dependence.** Conclusions about "relationship-aware modeling" are bounded
   by the single frozen architecture; a different family might behave differently.

---

## 24. Expected Contribution

- A leakage-safe, inductive evaluation of whether spatial/property relationships change
  valuation generalization under **location-held-out** conditions, against a frozen,
  already-characterised baseline.
- A clear answer to RQ6: which relationship types (city, locality, attribute similarity)
  contribute, if any.
- A fold-level robustness characterisation, avoiding single-fold artifacts.
- Either a positive or a negative result, reported without overclaiming, plus a
  reproducible graph-construction manifest that later work can extend.

It does **not** claim to prove the complete MILLOW system, production readiness, or
superiority of any model family.

---

## 25. Implementation Plan

Phased, mirroring the discipline of Experiments 1 and 2. Nothing in this plan is executed
until the protocol is frozen.

| Phase | Work | Output | Trains models? |
|---|---|---|---|
| 0 | Freeze this protocol; record approvals and the no-result-prediction statement | Frozen protocol hash | No |
| 1 | Graph-construction library (pure, no I/O), with unit tests for leakage invariants | `experiment_3/` graph module + tests | No |
| 2 | Per-fold graph derivation for frozen folds (inductive), manifest + digests | graph manifest | No |
| 3 | Architecture selection on a training-only inner validation; freeze config | frozen architecture record | Yes (validation only) |
| 4 | Fold-wise training/evaluation on frozen outer folds | per-fold predictions/metrics | Yes |
| 5 | Aggregation, ablations, limited-locality and price-band analysis, figures | results tables/figures | No (analysis) |
| 6 | Manifest, deterministic re-run verification, protected-state check | verification report | No |
| 7 | Review and freeze as a research checkpoint | tag (only after approval) | No |

Implementation order and gates (F0 before any interpretation) follow the Experiment 2
pattern. No phase may consume `valid_k` for tuning.

---

## 26. Pre-registration Checklist

All items must be resolved and recorded before Phase 3 training.

- [ ] Primary and secondary RQs frozen (Sections 3–4).
- [ ] H1 / H0 frozen, with the negative-result clause (Section 5).
- [ ] Dataset and fold digests re-verified against the frozen values (Section 6).
- [ ] Baseline consumed read-only and identified (Section 7).
- [ ] Edge types E1–E7 resolved: which are used, which are excluded, and why (Section 8).
- [ ] Inductive vs transductive roles fixed; inductive is primary (Section 9).
- [ ] Primary regime and test rows fixed (Section 10).
- [ ] Primary and secondary metrics and reporting format fixed (Section 12).
- [ ] Random→location-grouped analysis fixed (Section 13).
- [ ] Limited-locality strata and minimum-n rule fixed (Section 14).
- [ ] Price-band rule fixed; true-price bands restricted to post-hoc diagnostics (Section 15).
- [ ] Ablation set A0–A3 fixed (Section 16).
- [ ] Architecture-selection procedure and training-only validation fixed (Section 17).
- [ ] Statistical comparison and any test fixed in advance (Section 18).
- [ ] Failure criteria F0–F5 fixed (Section 19).
- [ ] Reproducibility identifiers and manifest schema fixed (Section 20).
- [ ] Computational budget completed with measured training-time estimates (Section 21).
- [ ] Protected-state list fixed (Section 22).
- [ ] Limitations acknowledged (Section 23).
- [ ] No-result-prediction statement retained verbatim (Section 1).

---

## 27. Status / Next Action

**Status: DRAFT — design only.** No model has been trained, no graph has been built, no new
dataset or fold file exists, and no results are claimed. This document is the only intended
new file. No existing research file, model, artifact, protocol, dataset, fold, blockchain
state, or Git ref is modified by authoring it.

**Next action:** review this protocol, resolve the pre-registration checklist (Section 26),
freeze the architecture-selection procedure, and only then proceed to Phase 1. No training
may begin until the computational budget (Section 21) is complete and the protocol is
approved.

---

## Appendix A — MILLOW research chain position

```
Experiment 1   valuation backbone selection (XGBoost anchor, CatBoost backbone)
Experiment 2   uncertainty characterisation (conformal calibration)
Experiment 3   spatial / property-relationship modelling   ← this document

Future layer   valuation
                   ↓
               uncertainty
                   ↓
               transaction risk
                   ↓
               decision policy
                   ↓
               blockchain / escrow
```

Experiment 3 connects the valuation layer to a richer structural representation. It does
**not** prove the complete MILLOW system, and it does not integrate with the blockchain.
