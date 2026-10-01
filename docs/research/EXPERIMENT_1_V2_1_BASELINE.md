# MILLOW RESEARCH EXPERIMENT 1 — V2.1 BASELINE PROTOCOL

**Status:** frozen research protocol. No model trained yet.
**Supersedes:** `scripts/train_valuation_v2.py` (retained unmodified).
**Code:** `scripts/valuation_v2_1/protocol.py`
**Records:** `artifacts/valuation/v2_1/`

> The historical V2 numbers in §11 are labelled
> **"Historical V2 reference — pre-leakage-control baseline"**.
> They are **not** the publication-grade baseline. The V2.1 protocol below is.

---

## 1. Dataset

| Property | Value |
|---|---|
| File | `data/processed/MREID_property.csv` |
| Raw shape | 29,135 rows × 44 columns |
| Invalid rows removed by validity filter | **0** |
| Exact duplicate rows removed | **737** |
| **Final rows** | **28,398** |

Validity filter (unchanged from V2): `price > 0 AND area > 0 AND no_of_bedrooms > 0`.

Text normalisation (unchanged from V2): `location` and `source_city` are
`astype(str).str.strip().str.lower()`.

The validity filter is retained even though it currently removes nothing — it
guards the MAPE denominator (§10).

### Excluded columns

| Column | Reason |
|---|---|
| `mreid_id` | unique row identifier, carries no signal |
| `source_file` | scrape provenance; near-duplicate of `source_city` |
| `derived_price_per_sqft` | equals `price / area`; including it **is** target leakage |
| `group` | the location-grouped split key, used only by the splitter |

`derived_price_per_sqft` was confirmed equal to `price/area` to a maximum
relative error of 2.6e-16. Excluding it is the single most important
correctness decision carried over from V2.

---

## 2. Duplicate policy

**Rule.** Drop rows that are exact duplicates on the five-column key
`(source_city, location, area, no_of_bedrooms, price)`, keeping the **first**
occurrence.

**Explicitly not part of the key:**

- `mreid_id` — unique per row, so a key including it could never match across
  rows and de-duplication would be a no-op.
- `source_file` — the same listing signature can legitimately arrive from more
  than one scrape run.

**Not performed:** no fuzzy matching, no near-duplicate detection, no
within-location de-duplication, no price-tolerance matching.

Consequence: only rows identical on *all five* fields are removed. Distinct
properties that merely share a location, an area or a bedroom count are always
retained. Two tests assert exactly this.

**Measured effect:** 737 of 29,135 rows removed (2.53%).

Under the V2 random split, 234 of 5,827 test rows (4.02%) had an exact twin in
train. That leakage is now structurally impossible.

---

## 3. Target

| Space | Definition |
|---|---|
| Raw column | `price` (INR) |
| **Training space** | `log1p(price)` |
| **Prediction space** | `log1p(price)` — models emit log-space values |
| **Evaluation space** | INR, via `max(expm1(pred), 0)` |

`log_target()` and `to_rupees()` in `protocol.py` are the only sanctioned
conversions. A fitted pipeline's `.predict()` returns **log-space** numbers;
inversion is the caller's responsibility and must be explicit.

The inversion floor at 0 exists because `expm1` of a negative log value is
negative. It does not trigger on this dataset.

---

## 4. Final feature definition — 48 features

`FINAL_FEATURE_COUNT = 48`. See §5 for the two removed feature groups.

The count is derived, not asserted by hand:

```
3 basic + 6 row-wise + 2 train-only frequency + 2 categorical + 35 amenities = 48
```

| Block | Count |
|---|---|
| Basic numeric | 3 |
| Row-wise engineered | 6 |
| Train-only frequency | 2 |
| Categorical keys | 2 |
| Raw amenities | 35 |
| **Total** | **48** |

### 4a. Standardised numeric (11)

| Feature | Definition |
|---|---|
| `area` | raw |
| `no_of_bedrooms` | raw |
| `resale` | raw (0/1) |
| `log_area` | `log1p(area)` |
| `log_bedrooms` | `log1p(no_of_bedrooms)` |
| `area_per_bedroom` | `area / no_of_bedrooms.clip(lower=1)` |
| `area_bedroom_interaction` | `area * no_of_bedrooms` |
| `amenity_yes_count` | count of the 35 amenities equal to `1` |
| `amenities_fully_specified` | **provenance flag**, see below. `1` if all 35 values ∈ {0,1}, else `0` |
| `location_frequency` | **train-only** count of the location |
| `log_location_frequency` | `log1p(location_frequency)`, **train-only** |

The six row-wise engineered features are pure functions of a row's own raw
columns, so computing them before the split **cannot** leak. The two frequency
features are aggregate and are fitted inside the Pipeline instead.

**`amenities_fully_specified` is a data-recording / provenance indicator, not
an ordinary property amenity.** It records whether the source listing filled in
every amenity field. It is *not* a statement about the property's quality or
furnishings. This framing is asserted by a test so it cannot silently drift
back to being read as a normal amenity.

`city_frequency` is **dropped** — see §5.

### 4b. One-hot encoded (37)

`source_city`, `location`, plus all 35 raw amenity columns.

Each amenity expands to **three** columns (`_0`, `_1`, `_9`).

---

## 5. Amenity treatment

**Value semantics — unchanged from V2, and unchanged here:**

| Value | Meaning | Treatment |
|---|---|---|
| `0` | No | one-hot level `_0` |
| `1` | Yes | one-hot level `_1` |
| `9` | Not specified / unknown | one-hot level `_9` |

**`9` is never reinterpreted as `0` or `1`.** It is never imputed. It is never
collapsed. All 35 columns still contain exactly `{0, 1, 9}` after
preprocessing, and the fitted encoder still reports three levels per amenity.
Two tests assert this.

### What changed

V2 carried **both** `amenity_known_count` and `amenity_unknown_count`. Because
the 35 amenity cells are either definite or `9`, the two satisfy
`known + unknown == 35` identically, so `corr = -1.0` exactly — two features
carrying one degree of freedom.

V2.1 replaces the pair with a single explicit binary:

```
amenities_fully_specified = 1  if all 35 amenity values are in {0,1}
                          0  if at least one value is 9
```

Measured on the real dataset: **7,584** rows fully specified, **20,814** rows
with at least one unknown.

### `city_frequency` is dropped

`city_frequency` is a deterministic 1:1 restatement of `source_city`, which is
already a one-hot categorical feature. It added zero information. Dropped.

This takes the count from 49 to **48**. The count is **not** padded back to 49
or 50 for continuity with V2.

### Feature count: 48

V2 had 50. V2.1 has **48**, from two independent removals:

- the `amenity_known_count` + `amenity_unknown_count` pair (perfectly
  collinear) replaced by one binary → −1
- `city_frequency` dropped as a deterministic restatement of `source_city` → −1

Neither removal is cosmetic. Both delete degrees of freedom that the model
could otherwise spend on noise. `amenity_yes_count` is retained because it is
*not* redundant: a row can be fully specified and have zero amenities, so the
provenance flag and the yes-count carry independent information (asserted by a
test).

### An honest caveat on the provenance flag

The `9` pattern is all-or-nothing per row: `amenity_unknown_count` took only
the values `0` and `35`. The flag is therefore a *provenance* indicator, not a
per-amenity availability measure. It carries **0.155 nats** of mutual
information with `source_city` (Kolkata 99.1% unspecified, Hyderabad 4.2%).

Removing the collinear pair and `city_frequency` did **not** remove this:
`amenities_fully_specified` remains partly a city proxy. Documented, not fixed
— the decision is to keep it, framed as provenance rather than as a property
characteristic.

---

## 6. Frequency-feature policy — the core V2.1 fix

### The V2 defect

`train_valuation_v2.py` called `engineer_features()` on all 29,135 rows
**before** splitting, so `location_frequency` and `city_frequency` for a test
row included that row and its test-mates in the count. Transductive leakage.

### The V2.1 fix

`TrainOnlyFrequencyFeatures` is the **first step of the Pipeline**. sklearn
calls its `fit()` with training rows only and `transform()` separately for
train and for validation. No validation or test row can contribute to any
frequency value. Because it lives inside the Pipeline, the guarantee holds
identically in every cross-validation fold, not just in a one-off split.

Callers pass `INPUT_FEATURES` (46 columns), which **excludes** the two
frequency columns — they are derived, never supplied. Removing them from the
input is what makes it structurally impossible to re-introduce the V2 defect.

### Unseen-category fallback

A location absent from the training partition receives **exactly `0.0`**,
meaning zero training observations, and `log1p(0.0) = 0.0`.

**No smoothing constant. No global count. No full-dataset count is consulted.**

`0.0` rather than `1.0` so an unseen locality is never confused with a
singleton locality that genuinely appears once in training. A test asserts the
two stay distinct.

The fallback is keyed on **location alone**, since that is the only aggregate
feature retained. `city_frequency` is no longer computed, so an unfamiliar city
string cannot zero out a location count for a location that genuinely appears
in training. A test asserts this.

### Measured effect

Fold 0, random regime, 22,718 training rows:

- 127 validation rows have a location unseen in training → all get `0.0`
- the V2 pre-split count would have inflated those same rows by
  **mean 1.3222×, max 4.0000×**

`city_frequency` is dropped rather than retained, so the V2.1 frequency block is
`location_frequency` + `log_location_frequency` only.

---

## 7. Preprocessing

Identical construction for **all five** models. No model-specific preprocessing
decision exists.

```
Pipeline
├── 1. TrainOnlyFrequencyFeatures     train-only aggregate features
├── 2. ColumnTransformer
│     ├── numeric (11)   StandardScaler()
│     └── categorical (37) OneHotEncoder(handle_unknown="ignore")
└── 3. estimator        supplied per model
```

### Why a scaler was added

V2 used `"passthrough"` for numerics. That is harmless for XGBoost, which is
invariant to feature scale, but **Ridge is not scale-invariant** and the
numeric block ranges from `amenity_yes_count` ∈ [0, 35] to
`area_bedroom_interaction` reaching 1.0e5. Without a scaler, Ridge would be
unevaluable and the five-model comparison could not proceed at all.

Tree models are standardised too. They do not need it, and applying it to all
five removes any possibility of a model-specific preprocessing advantage.

### Fit isolation

The `StandardScaler` sits inside the `ColumnTransformer` inside the
`Pipeline`, so it is re-fitted on training rows only, in every fold. Verified
empirically: fitted means match a train-only reference to 1e-6 and **differ**
from the full-dataset reference.

### Documented model-specific accommodation

`make_pipeline(dense_output=True)` sets `OneHotEncoder(sparse_output=False)`
for CatBoost, whose estimator does not accept scipy sparse matrices. This
changes **no fitted statistic** — identical categories, identical StandardScaler
means. It is an input-format accommodation only.

---

## 8. CV strategy

Fold assignments are generated **once**, persisted to
`artifacts/valuation/v2_1/cv_folds_v2_1.json`, and reused by all five models.
All five models must **load** them, never re-derive them.

| Regime | Splitter | `n_splits` | `random_state` | Groups |
|---|---|---|---|---|
| `random` | `KFold(shuffle=True)` | 5 | 42 | none |
| `location_grouped` | `GroupKFold(shuffle=True)` | 5 | 42 | `source_city + "__" + location` |

### Regime A — random / standard CV

| Fold | Train rows | Valid rows |
|---|---|---|
| 0 | 22,718 | 5,680 |
| 1 | 22,718 | 5,680 |
| 2 | 22,718 | 5,680 |
| 3 | 22,719 | 5,679 |
| 4 | 22,719 | 5,679 |

Digest: `1a9f896345492d60b4a8cc677a090b27c8affd745ef302f9fbbf4ea55afc415f`

### Regime B — location-grouped generalisation

| Fold | Train rows | Valid rows |
|---|---|---|
| 0 | 23,540 | 4,858 |
| 1 | 22,693 | 5,705 |
| 2 | 23,384 | 5,014 |
| 3 | 21,409 | 6,989 |
| 4 | 22,566 | 5,832 |

Digest: `4c1c331cc993ddedb58b0a177e118912eb657613e3c9e4f8b5ae2bc7963ec611`

**Max group overlap: 0 in every fold.** No `source_city__location` group appears
in both train and validation.

### Why GroupKFold rather than GroupShuffleSplit

K-fold reuses every row for validation exactly once, so fold-to-fold variance
is comparable across models. `GroupShuffleSplit` draws one partition and cannot
produce a variance estimate.

### Why the grouped folds are uneven

`GroupKFold` balances by **group count**, not row count. With 1,789 groups of
highly skewed size (median 3, min 1, max 687), validation row counts range from
4,858 to 6,989. This is expected and is a property of grouped CV, not a defect.
Metrics must therefore be reported as fold-level mean ± std, never pooled.

### Verified guarantees

- every row validated exactly once, in both regimes
- zero train/valid row overlap, all 10 folds
- zero exact duplicates across any train/valid boundary (all 5 random folds)
- zero group overlap (all 5 grouped folds)
- byte-identical digests on regeneration
- digest match after a JSON persist/reload round-trip

---

## 9. Random seed

`RANDOM_STATE = 42` throughout. Applied to `KFold`, to `GroupKFold`, and
required of every estimator's own `random_state` at training time. Fold
digests are recorded so a silent protocol drift is detectable.

---

## 10. Metrics

Eight metrics, in two explicitly separated spaces.

| Key | Space | Definition |
|---|---|---|
| `MAE_INR` | INR | `mean_absolute_error` after inversion |
| `RMSE_INR` | INR | `sqrt(mean_squared_error)` after inversion |
| `R2_INR` | INR | `r2_score` after inversion |
| `MAPE_percent` | INR | mean absolute percentage error × 100 |
| `MedAPE_percent` | INR | median absolute percentage error × 100 |
| `MAE_log` | log1p(price) | `mean_absolute_error`, no inversion |
| `RMSE_log` | log1p(price) | `sqrt(mean_squared_error)`, no inversion |
| `R2_log` | log1p(price) | `r2_score`, no inversion |

### Why MedAPE and log-space metrics were added

The dataset is heavy-tailed — max/min price ratio **427**, median ₹6.88M vs
99th percentile ₹82.5M. MAPE weights a ₹2M error on a ₹2M property equally with
a ₹2M error on an ₹80M property, so it describes the typical mid-market
property and says nothing about the 2,647 properties above ₹20,000/sqft. MedAPE
is robust to that.

`R2_INR` is dominated by a small luxury tail, so `R2_log` — the space the model
actually optimises — is the headline goodness-of-fit measure. A test asserts the
two can diverge.

### MAPE numerical safety

Both percentage metrics divide by `max(|y_true|, eps)` as sklearn does. The
minimum price is **₹2,000,000** and the validity filter enforces `price > 0`,
so the smallest denominator is 2e6 and `eps` (≈2.2e-16) never activates.
**MAPE is numerically sound on this dataset.** The guard exists only so a
future unfiltered dataset cannot silently produce `inf`.

`MAPE` and `MedAPE` are implemented directly in `protocol.py` rather than
imported, because scikit-learn 1.9.1 exposes `mean_absolute_percentage_error`
but no median counterpart. Writing both keeps them on one denominator
convention.

### Reporting requirement

Per regime, report **fold-level mean ± std** for all eight metrics. A single
pooled number hides fold-to-fold variance and is not acceptable for a
five-model comparison.

---

## 11. Historical V2 reference — pre-leakage-control baseline

**These are NOT the publication-grade baseline.** They are retained as the
record of what V2 produced, before leakage control.

| Split | Train | Test | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|---|---|
| random | 23,308 | 5,827 | ₹6,313,796.49 | ₹25,979,803.19 | 0.090639 | 48.0975% |
| location-grouped | 24,180 | 4,955 | ₹6,194,757.08 | ₹20,899,053.70 | 0.055779 | 54.3962% |

Files: `models/valuation/valuation_v2_xgboost_*.joblib`,
`artifacts/valuation/valuation_v2_*`. **Untouched by V2.1 and never
regenerated.**

These came from a single train/test draw per regime, with frequency leakage,
4.02% duplicate leakage, no scaler, and no variance estimate. They are not
comparable to V2.1 numbers and must not be placed in the same table as V2.1
results without this label.

### Non-equivalence of the two protocols

Placing the V2.1 anchor numbers (§16) next to the table above would be
misleading. The protocols differ on five axes at once:

| Axis | Historical V2 | V2.1 |
|---|---|---|
| Frequency features | computed pre-split (leaky) | train-only, inside Pipeline |
| Duplicates | retained, 4.02% spanned train/test | 737 exact duplicates removed |
| Feature set | 50 features | 48 features |
| Preprocessing | `passthrough` numerics | `StandardScaler` inside Pipeline |
| Evaluation | one train/test draw per regime | 5-fold CV, two regimes |

**No better-or-worse claim is made, and none is supportable.** V2's numbers are
flattered by leakage; V2.1's are honest under a harder protocol. The two are
not the same measurement.

---

## 12. Environment

Recorded in `artifacts/valuation/v2_1/environment_v2_1.json`.

| Package | Version |
|---|---|
| Python | 3.14.7 |
| pandas | 3.0.6 |
| numpy | 2.5.3 |
| scikit-learn | 1.9.1 |
| xgboost | 3.4.1 |
| joblib | 1.6.0 |
| lightgbm | **not installed** |
| catboost | **not installed** |

**No package was installed, upgraded or downgraded.** LightGBM and CatBoost
installations are deferred to the model-training step, as instructed.

Note: `scripts/` has no lockfile. The versions above plus the fold digests are
the reproducibility contract.

---

## 13. What changed from V2 → V2.1, and why

| # | Area | V2 | V2.1 | Why necessary |
|---|---|---|---|---|
| 1 | Frequency features | computed on all 29,135 rows before split | fitted inside Pipeline on training rows only | **CONFIRMED leakage.** Test-row counts inflated up to 4.0× |
| 2 | Unseen frequency | no fallback existed | explicit `0.0`, documented, tested | V2 had no defined behaviour for unseen categories |
| 3 | Duplicates | kept (4.02% of test rows had twins in train) | 737 exact duplicates removed pre-split | **POTENTIAL leakage.** Identical records spanned train and test |
| 4 | Amenity degeneracy | `known` + `unknown`, corr = −1.0 exactly | single `amenities_fully_specified` binary | Two features, one degree of freedom |
| 5 | Scaling | `"passthrough"` | `StandardScaler` in Pipeline | **Blocks the comparison.** Ridge is not scale-invariant |
| 6 | Evaluation | single split per regime | 5-fold, two regimes, persisted | A single draw cannot support a 5-model ranking |
| 7 | Metrics | MAE, RMSE, R², MAPE | + MedAPE, + 3 log-space metrics | 427× price range makes MAPE unrepresentative |
| 8 | Reproducibility | unpinned | environment record + fold digests | Silent drift undetectable without them |

### Deliberately unchanged

These V2 decisions were audited, found correct, and carried forward untouched:

- `derived_price_per_sqft` excluded (target leakage)
- no price-per-area feature anywhere
- `log1p` target
- `handle_unknown="ignore"` on the encoder
- `city__location` grouping — coarser than `location`, the conservative choice
- `price > 0` filter retained despite removing 0 rows
- no imputation, justified by zero nulls across all 44 columns
- amenities one-hot as three discrete levels

---

## 14. Reproduction

```powershell
# Build dataset, folds and records. Trains nothing.
python -m scripts.valuation_v2_1.build_protocol

# Non-live tests.
python -m pytest scripts/valuation_v2_1/test_protocol.py -v
```

Outputs, all under `artifacts/valuation/v2_1/`:

| File | Contents |
|---|---|
| `cv_folds_v2_1.json` | fold assignments + digests |
| `environment_v2_1.json` | package versions |
| `protocol_v2_1.json` | machine-readable protocol definition |
| `xgb_anchor_per_fold_v2_1.csv` | per-fold, per-regime anchor metrics |
| `xgb_anchor_metrics_v2_1.csv` | same table |
| `xgb_anchor_metrics_v2_1.json` | anchor run incl. fold digests and mean ± std |
| `xgb_anchor_run_v2_1.json` | anchor run record |

Fold models: `models/valuation/v2_1/xgb_anchor_folds/`.

---

## 15. Frozen decisions (closed)

Resolved before the anchor run:

1. **Feature count = 48.** Collinearity fix kept: `amenity_known_count` and
   `amenity_unknown_count` are NOT restored. `amenity_yes_count` and
   `amenities_fully_specified` are kept, and the completeness indicator is
   documented as a data-recording/provenance indicator, not an ordinary
   property amenity. A test enforces that framing.
2. **`city_frequency` dropped** as a deterministic 1:1 restatement of the
   already one-hot `source_city`.
3. **No padding.** 48 is the count; it is not held at 49 or 50 for continuity.
4. **Unchanged:** `log1p(price)` target, `derived_price_per_sqft` exclusion,
   train-only `location_frequency` and `log_location_frequency`, unseen
   fallback `0.0`, amenity `9` as its own category, `city__location` grouped
   evaluation, `StandardScaler` inside the Pipeline, the duplicate rule,
   persisted fold assignments, seed 42.

## 16. V2.1 XGBoost anchor run

Executed with the frozen protocol and the **unchanged** historical V2 XGBoost
configuration. Not a tuning exercise: no early stopping, no search, no
parameter change. 10 models trained (5 folds × 2 regimes). Predictions scored
per fold and **not pooled**.

Reproduce with:

```powershell
python -m scripts.valuation_v2_1.train_xgboost_anchor
```

### 16a. Configuration

```
objective           = reg:squarederror      n_estimators    = 700
max_depth           = 7                     learning_rate   = 0.04
subsample           = 0.85                  colsample_bytree= 0.85
min_child_weight    = 5                     reg_alpha       = 0.1
reg_lambda          = 2.0                   tree_method     = hist
random_state        = 42                    n_jobs          = -1
eval_metric         = rmse
```

`eval_metric="rmse"` was also set by V2. With no early stopping it affects
logging only, not the fitted model.

### 16b. Regime A — random 5-fold CV

| Fold | n_valid | MAE_INR | RMSE_INR | R²_INR | MAPE% | MedAPE% | MAE_log | RMSE_log | R²_log |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5,680 | 6,073,823 | 21,704,525 | 0.1236 | 51.21 | 31.34 | 0.4624 | 0.6564 | 0.3655 |
| 2 | 5,680 | 6,196,552 | 23,196,493 | 0.1269 | 51.20 | 30.19 | 0.4534 | 0.6553 | 0.3771 |
| 3 | 5,680 | 6,087,407 | 23,983,680 | 0.1286 | 49.38 | 29.95 | 0.4446 | 0.6392 | 0.3767 |
| 4 | 5,679 | 6,161,316 | 21,946,489 | 0.1069 | 49.12 | 30.27 | 0.4539 | 0.6562 | 0.3592 |
| 5 | 5,679 | 6,005,100 | 21,800,283 | 0.1539 | 51.53 | 30.87 | 0.4513 | 0.6434 | 0.3834 |

Mean ± std (ddof=1, n=5):

| Metric | mean | std |
|---|---|---|
| MAE_INR | 6,104,839.67 | 75,522.73 |
| RMSE_INR | 22,526,293.98 | 1,013,867.82 |
| R²_INR | 0.1280 | 0.0169 |
| MAPE_percent | 50.4908 | 1.1423 |
| MedAPE_percent | 30.5240 | 0.5713 |
| MAE_log | 0.4531 | 0.0064 |
| RMSE_log | 0.6501 | 0.0082 |
| R²_log | 0.3723 | 0.0098 |

### 16c. Regime B — location-grouped 5-fold CV

| Fold | n_valid | MAE_INR | RMSE_INR | R²_INR | MAPE% | MedAPE% | MAE_log | RMSE_log | R²_log |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4,858 | 6,421,620 | 21,037,519 | 0.0600 | 59.31 | 37.02 | 0.5070 | 0.6988 | 0.2511 |
| 2 | 5,705 | 5,836,112 | 18,430,930 | 0.1149 | 62.76 | 36.94 | 0.5006 | 0.6814 | 0.2557 |
| 3 | 5,014 | 8,228,930 | 32,505,300 | 0.0390 | 54.36 | 36.90 | 0.5143 | 0.7170 | 0.3096 |
| 4 | 6,989 | 6,535,451 | 17,484,605 | 0.0871 | 75.11 | 42.27 | 0.5590 | 0.7425 | 0.1669 |
| 5 | 5,832 | 7,553,485 | 25,871,702 | 0.0682 | 72.54 | 41.55 | 0.5560 | 0.7445 | 0.1865 |

Mean ± std (ddof=1, n=5):

| Metric | mean | std |
|---|---|---|
| MAE_INR | 6,915,119.58 | 959,890.28 |
| RMSE_INR | 23,066,011.19 | 6,199,252.55 |
| R²_INR | 0.0739 | 0.0287 |
| MAPE_percent | 64.8159 | 8.7973 |
| MedAPE_percent | 38.9338 | 2.7274 |
| MAE_log | 0.5274 | 0.0279 |
| RMSE_log | 0.7168 | 0.0274 |
| R²_log | 0.2340 | 0.0575 |

### 16d. The regime gap is the finding

Grouped CV is materially harder than random CV on every metric: `R²_log` falls
from 0.3723 to 0.2340, `MedAPE` rises from 30.5% to 38.9%, and fold-to-fold
variance roughly quadruples for `RMSE_INR` (1.01M → 6.20M).

Fold-to-fold spread within the grouped regime is large enough to matter for
the comparison: `RMSE_INR` ranges 17.5M–32.5M, and folds 4 and 5 are visibly
harder than folds 1–3. Any single-number ranking of the five models would hide
this, which is why §10 requires mean ± std.

## 17. Remaining open decisions

1. **Hyperparameter budget for the five-model comparison.** V2's
   `n_estimators=700` with no early stopping is now fixed by decision. All five
   models must get equal budgets, and whether that means equal round counts or
   early stopping on a train-internal slice is still open.
2. **`amenities_fully_specified` is a partial city proxy** (0.155 nats MI). Kept
   as provenance per decision 1. Worth revisiting only if a future regime
   study shows it is doing city identification rather than provenance work.
