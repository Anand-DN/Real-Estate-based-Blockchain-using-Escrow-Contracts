# Experiment 2 — Conformal Prediction Protocol (DRAFT)

**Status: PHASE 0 APPROVED. Phase 1 + Phase 2 implemented and validated.
Phase 3 (nested training) NOT yet executed. NO EXPERIMENT 2 RESULTS EXIST.**

No Experiment 2 result has been generated. No model was trained. No Experiment 1 file was
modified. No commit, tag, or push was made. Conformal validity is **not** claimed.

| Field | Value |
|---|---|
| Document | `docs/research/EXPERIMENT_2_CONFORMAL_PROTOCOL_DRAFT.md` |
| Status | **Phase 0 APPROVED**; Phase 1 + Phase 2 complete |
| Frozen predecessor tag | `v1.3.0-experiment1-catboost-backbone` |
| Frozen predecessor commit | `101a771e37509eafdc05c2a913909b8d91bf8b00` |
| Protocol | V2.1 (unchanged, read-only) |
| Experiment | 2 |
| Blockchain integration | **None. Explicitly out of scope.** |
| Conformal validity claimed | **No. Nothing is claimed before it is tested.** |

---

## 0. Phase 0 — Pre-registration decisions (APPROVED, BINDING)

Recorded here because they are now fixed. They were decided **before** any coverage
number exists and may not be revisited in response to observed coverage.

### 0.1 Calibration fraction

**`cal_frac = 0.20`** — the single pre-registered value.

Explicitly **not** to be compared against 0.10 or 0.25 after seeing results.

### 0.2 Method set (closed)

| ID | Method | Status |
|---|---|---|
| **A** | Split conformal, absolute residual | **INCLUDED** |
| **B1** | Normalized conformal, `sigma_hat(x) = y_hat(x)` | **INCLUDED** |
| **B3** | Normalized conformal, `sigma_hat(x) = 1` | **INCLUDED** — implementation control |
| **C1** | Mondrian conformal by `source_city` | **INCLUDED** |
| **C2** | Mondrian conformal by predicted-price tertile | **INCLUDED** |
| **C3** | Grouped conformal by `source_city__location` | **INCLUDED — SECONDARY ONLY** |
| B2 | kNN-normalized conformal | **EXCLUDED** |
| C4 | Weighted conformal under covariate shift | **EXCLUDED** |

**No additional methods may be introduced after results are observed.** In particular, no
method may be added in response to a method performing poorly (Section 10, §16 F7).

### 0.3 Nominal coverage levels (fixed)

`0.80`, `0.90`, `0.95` — i.e. `alpha ∈ {0.20, 0.10, 0.05}`.

### 0.4 F3 usability gate (fixed)

Metric: **row-level relative interval width**

```
relative_width = (upper - lower) / point_prediction
```

Gate, evaluated at **nominal 0.90**:

```
median_relative_width <= 2.00   ->  method is USABLE
median_relative_width >  2.00   ->  method is NOT USABLE
```

Two constraints carried with this threshold:

1. This is an **Experiment-specific operational threshold**. It is **not** a claim about
   a universal real-estate industry standard, and must never be described as one.
2. **Calibration and usability are reported separately.** A method may be calibrated and
   still fail the usability gate. The two are independent axes and neither substitutes
   for the other.

> **Interaction with Section 10.1 — recorded deliberately, not overlooked.** For the
> homogeneous methods (A, B3, C1, C2), `relative_width` is effectively constant and
> equal to `2*sinh(q_hat_alpha)` (Section 10.1), so the gate reduces to a single
> comparison `q_hat_0.90 <= asinh(1.0) = 0.88137`. The diagnostic in Section 10.3
> indicates `q_hat_0.90 ~ 1.05`, i.e. `relative_width ~ 2.52`, so **the homogeneous
> methods are predicted to fail this gate on the random regime at minimum**. That
> prediction is recorded now, before results, precisely so that a failure cannot later be
> presented as a surprise or explained away. It is not a reason to alter the gate.

> **Numerical precision of the identity — corrected after Phase 1 implementation.** The
> identity is exact when the denominator is `exp(yhat)`. The protocol divides by
> `expm1(yhat)`, which introduces a correction factor `exp(yhat)/(expm1(yhat))`. At
> `yhat = 15` that factor is `1 + 3.06e-7`, so the identity holds to roughly **7
> significant figures, not to machine epsilon**. Measured deviations:
> `yhat=14 -> 8.3e-7`, `yhat=15 -> 3.1e-7`, `yhat=20 -> 2.1e-9` (all relative). The
> F3 boundary in `q` is therefore `asinh(1.0) + 4.1e-7`, not exactly `asinh(1.0)`. This
> is recorded so that no later test or report claims more precision than exists. It does
> not affect any gate outcome at the planned interval widths.

### 0.5 F5 correction — monotonicity is descriptive, not a validity requirement

The original F5 required `Spearman(nominal_level, coverage_error) > 0`. That is
**withdrawn**. It incorrectly imposed a particular statistical sign as a validity
requirement, which was not justified.

**Corrected F5:** evaluate the ordered empirical-coverage / coverage-error sequence at
80%, 90% and 95% and report **whether the observed sequence is directionally monotone**.
Report it descriptively. **Do not impose a particular Spearman sign as a validity
requirement.**

### 0.6 Nested training approval

**Explicitly approved:** train exactly **10** new nested CatBoost models
(2 regimes × 5 outer folds) using the **exact** frozen Experiment 1 configuration:

```
iterations       = 700
learning_rate    = 0.04
depth            = 7
l2_leaf_reg      = 2.0
random_seed      = 42
loss_function    = RMSE
dense preprocessing
```

No hyperparameter tuning. No early stopping. No feature changes. No model selection.

Note: approval is granted, but Phase 3 is **not** to be executed until Phase 1 and
Phase 2 are implemented, unit-tested and validated — per the mandated order in §0.7.

### 0.7 Mandated implementation order

**Do not run the full experiment immediately.**

1. **Phase 1 only** — `conformal_scores()`, `conformal_quantile()`,
   `build_intervals()`, `coverage_metrics()`, `sharpness_metrics()`,
   `subgroup_coverage()`. No filesystem writes. No model training. Run the pure unit
   tests.
2. **Phase 2** — deterministic nested fold derivation; write
   `calibration_folds_v2_2.json`; verify all fold invariants; verify frozen V2.1 fold
   SHA-256.
3. **STOP and report** Phase 1 + Phase 2 validation.
4. Only after that may Phase 3 nested CatBoost training be executed.

### 0.8 Protected state

Absolutely not to be modified:

`.chain/state.json` · `data/processed/MREID_property.csv` ·
`data/processed/millow_token_map.csv` · `artifacts/valuation/v2_1/cv_folds_v2_1.json` ·
`artifacts/valuation/v2_1/**` · `models/valuation/v2_1/**` ·
`models/valuation/valuation_v2*` · `artifacts/valuation/valuation_v2*` ·
`scripts/valuation_v2_1/**` · `backend/**` · existing tags.

Experiment 1 remains frozen at `v1.3.0-experiment1-catboost-backbone` /
`101a771e37509eafdc05c2a913909b8d91bf8b00`.

### 0.9 Research discipline (binding)

- No claim of conformal validity before results exist.
- Location-grouped coverage must **never** be described as having a formal
  distribution-free guarantee.
- No method added because an initial method performs poorly.
- `cal_frac`, `m`, thresholds and methods must **not** be selected on observed coverage.
- No commit, no tag, no push.
- No final Experiment 2 results yet.

---

## 1. Research objective

Determine, empirically, whether split-conformal-style prediction intervals can supply
**calibrated uncertainty** for the frozen Experiment 1 CatBoost property valuation
backbone, and specifically whether that calibration **survives location-held-out
distribution shift**.

The experiment is a measurement, not a construction. It is designed so that the
following are all acceptable findings:

- coverage near nominal (calibration succeeded),
- systematic undercoverage (calibration failed),
- overcoverage with uselessly wide intervals (calibration succeeded, method not useful),
- calibration that holds in the random regime but degrades under location shift,
- monotone degradation with locality novelty.

None of these is a failure of the experiment. Only a **leakage-induced** result, or a
result produced without a pre-registered analysis, would be a failure of the protocol.

### 1.1 Explicit non-goal

Experiment 2 does **not** attempt to improve point-prediction accuracy. The backbone is
frozen. No hyperparameter search, no refitting for accuracy, no ensembling, no feature
changes. If nested refitting (Section 5) is approved, it is a *consequence of the
conformal requirement*, not an accuracy experiment, and its point accuracy will be
**worse** than Experiment 1 by construction (Section 15.2).

---

## 2. Research questions

Pre-registered. Each is answerable from the planned artifacts without further choices.

| ID | Question | Answered by |
|---|---|---|
| **RQ1** | In the random-CV regime, does split conformal attain nominal marginal coverage at 80/90/95%? | `per_fold_coverage.csv`, `summary_coverage.csv` |
| **RQ2** | Does normalized conformal improve sharpness over split conformal *without* degrading coverage? | paired per-fold width and coverage deltas |
| **RQ3** | Does marginal coverage degrade under location-grouped evaluation relative to random CV, at matched nominal level? | `regime_shift_coverage.csv` |
| **RQ4** | Is degradation under location shift **monotone** in nominal level (80 → 90 → 95)? | `coverage_vs_nominal` figures, Spearman rank |
| **RQ5** | Is coverage **non-uniform across cities** within a single regime and nominal level? | `subgroup_coverage.csv` (city) |
| **RQ6** | Is coverage **non-uniform across price bands** within a single regime and nominal level? | `subgroup_coverage.csv` (price band) |
| **RQ7** | Are intervals **sharp enough to be actionable**, in absolute INR and in relative terms? | width metrics + pre-declared usability gate (F3) |
| **RQ8** | Do residuals exhibit structure (non-normality, heavy tails, heteroscedasticity) that would undermine a symmetric absolute-residual score? | `residual_diagnostics` figures + tables |
| **RQ9** | Does any pre-declared **location-aware** strategy (Section 6.3) recover coverage relative to plain split conformal under location shift? | paired regime-grouped comparison |
| **RQ10** | Are observed deviations from nominal **statistically distinguishable from Monte-Carlo noise** at the achieved calibration size? | Clopper–Pearson intervals vs nominal |

RQ1 is the **negative control**: it must pass, or the implementation is broken and no
other result is interpretable. RQ3 is the headline.

---

## 3. Frozen dependencies from Experiment 1

Everything in this section is **read-only input**. Nothing here may be regenerated,
tuned, or overwritten.

### 3.1 Frozen backbone configuration

Taken verbatim from `CATBOOST_CONFIG` in
`scripts/valuation_v2_1/run_model_comparison.py`. **Not to be tuned under any
circumstance in Experiment 2.**

| Parameter | Value |
|---|---|
| `iterations` | 700 |
| `learning_rate` | 0.04 |
| `depth` | 7 |
| `l2_leaf_reg` | 2.0 |
| `random_seed` | 42 |
| `loss_function` | `RMSE` (library default, as fitted in Experiment 1) |
| `verbose` | `false` |
| `allow_writing_files` | `false` |
| dense preprocessing | **required** — CatBoost rejects scipy sparse matrices |

Experiment 1 recorded the rationale verbatim: iterations/learning_rate/depth equal the
frozen XGBoost anchor exactly, `l2_leaf_reg` matches the anchor's `reg_lambda`, there is
no early stopping and no search, and `requires_dense_output=True` is an input-format
accommodation that changes no fitted statistic.

Verified at authoring time by reloading a persisted pipeline:

```
steps: ['frequency', 'preprocessor', 'model']
params: {'iterations': 700, 'learning_rate': 0.04, 'depth': 7,
         'l2_leaf_reg': 2.0, 'loss_function': 'RMSE',
         'random_seed': 42, 'verbose': False, 'allow_writing_files': False}
```

### 3.2 Frozen V2.1 protocol

| Element | Value | Source |
|---|---|---|
| Dataset | `data/processed/MREID_property.csv` | `protocol.DATA_PATH` |
| Rows after cleaning + de-duplication | 28,398 | `build_dataset()` |
| Duplicates removed | 737 exact on `['source_city','location','area','no_of_bedrooms','price']` | `DuplicateReport` |
| Target column | `price` | `protocol.TARGET_COLUMN` |
| Training space | `log1p(price)` | `protocol.TRAINING_SPACE` |
| Prediction space | `log1p(price)` | `protocol.PREDICTION_SPACE` |
| Presentation space | INR via `max(expm1(·), 0)` | `protocol.to_rupees` |
| Feature count | **48** | `protocol.FINAL_FEATURE_COUNT` |
| Pipeline input columns | 46 (the 2 frequency columns are derived inside the `Pipeline`) | `pipeline_input_columns` |
| Numeric features | 11 | `NUMERIC_FEATURES` |
| One-hot features | 37 | `ONEHOT_FEATURES` |
| Categorical | `source_city`, `location` | `CATEGORICAL_FEATURES` |
| Location-group key | `source_city + '__' + location` | `add_group_labels` |
| `random_state` | 42 | `protocol.RANDOM_STATE` |
| `n_splits` | 5 | `protocol.N_SPLITS` |
| Metric contract | `MAE_INR, RMSE_INR, R2_INR, MAPE_percent, MedAPE_percent, MAE_log, RMSE_log, R2_log` | `METRIC_KEYS` |

Pipeline step order is frozen and asserted by an existing test
(`test_pipeline_has_expected_step_order`):

```
1. TrainOnlyFrequencyFeatures   # fit() on training rows only
2. ColumnTransformer            # StandardScaler (numeric) + OneHotEncoder (categorical+amenities)
3. CatBoostRegressor
```

**Leak-relevant properties of step 1**, carried over from V2.1:

- `location_frequency` and `log_location_frequency` are fitted on the rows passed to
  `fit()`. In a nested design (Section 5) that means **only `T'_k`**, never `T'_k ∪ C_k`.
- An unseen location receives exactly `0.0`, and `log1p(0.0) = 0.0`. No smoothing
  constant, no global count. `0.0` (not `1.0`) is used so an unseen locality is never
  confused with a singleton. **In the location-grouped regime every test row has
  `location_frequency == 0.0` by construction.** This is a structural property of the
  regime and must be discussed in the results, not treated as a bug.

Target-exclusion check performed at authoring time: `'price' in INPUT_FEATURES` is
`False`, and `46 + 2 == 48`. The target is not a feature.

### 3.3 Frozen fold contract

`artifacts/valuation/v2_1/cv_folds_v2_1.json` — **read-only, must not be regenerated,
extended, or edited.**

| Regime | Splitter | `group_column` | Per-fold train/valid | `digest` |
|---|---|---|---|---|
| `random` | `KFold(shuffle=True)` | `None` | 22718/5680, 22718/5680, 22718/5680, 22719/5679, 22719/5679 | `1a9f896345492d60b4a8cc677a090b27c8affd745ef302f9fbbf4ea55afc415f` |
| `location_grouped` | `GroupKFold(shuffle=True)` | `group` | 23540/4858, 22693/5705, 23384/5014, 21409/6989, 22566/5832 | `4c1c331cc993ddedb58b0a177e118912eb657613e3c9e4f8b5ae2bc7963ec611` |

Verified at authoring time:

- Every fold is a sorted, disjoint `list[int]`; `train ∩ valid = ∅` for all 10 folds.
- `⋃(train_k ∪ valid_k)` = all 28,398 rows, indices `0…28397`; each row validated
  exactly once across the 5 folds.
- `location_grouped`: **0 shared groups** in every fold.
- `random`: 83.8–87.7% of validation *groups* also appear in training, and
  **96.8–97.7% of validation rows** come from a locality the model has seen. This is the
  intended contrast between the two regimes, not a defect.

### 3.4 Frozen Experiment 1 artifacts

- `artifacts/valuation/v2_1/model_comparison/` — 27 files: `experiment_manifest.json`,
  `per_fold_metrics.{csv,json}`, `summary_metrics.csv`, `comparison_table.csv`,
  `model_configurations.json`, `environment.json`, `training_time.csv`,
  `analysis_results.json`, `analysis_manifest.json`, `artifact_verification.json`,
  `fold_stability.{csv,md}`, `generalization_table.{csv,md}`,
  `metric_discrimination.{csv,md}`, `regime_shift_all_metrics.{csv,md}`,
  `per_metric_ordering_by_regime.{csv,md}`, `thesis_summary_table.{csv,md}`, `figures/`
  (8 PNG + 8 PDF).
- `models/valuation/v2_1/model_comparison/CatBoost/` — **10 persisted pipelines**
  (`random_fold1..5`, `location_grouped_fold1..5`), ~16.5 MB each.

**Verified at authoring time — all 10 reload read-only and reproduce their recorded
Experiment 1 metrics exactly (agreement to < 1e-12):**

| Regime | Fold | `MAE_log` reproduced | recorded | `R2_log` reproduced | recorded |
|---|---|---|---|---|---|
| random | 1 | 0.476294 | 0.476294 | 0.354678 | 0.354678 |
| random | 2 | 0.466494 | 0.466494 | 0.370420 | 0.370420 |
| random | 3 | 0.454245 | 0.454245 | 0.373206 | 0.373206 |
| random | 4 | 0.466745 | 0.466745 | 0.350525 | 0.350525 |
| random | 5 | 0.465182 | 0.465182 | 0.371134 | 0.371134 |
| location_grouped | 1 | 0.499855 | 0.499855 | 0.270661 | 0.270661 |
| location_grouped | 2 | 0.479162 | 0.479162 | 0.298298 | 0.298298 |
| location_grouped | 3 | 0.514472 | 0.514472 | 0.309120 | 0.309120 |
| location_grouped | 4 | 0.532171 | 0.532171 | 0.241292 | 0.241292 |
| location_grouped | 5 | 0.538207 | 0.538207 | 0.219062 | 0.219062 |

### 3.5 Experiment 1 reference metrics for the frozen backbone

From `summary_metrics.csv` (`std_ddof = 1`, `n_folds = 5`):

| Regime | `R2_log` mean ± SD | `MAE_log` mean ± SD | `MedAPE_percent` mean ± SD |
|---|---|---|---|
| random | 0.363993 ± 0.010551 | 0.465792 ± 0.007833 | 32.524296 ± 0.743129 |
| location_grouped | 0.267686 ± 0.037828 | 0.512773 ± 0.024097 | 37.293596 ± 2.312190 |

Point-accuracy degradation under location shift is **already established**. Experiment 2
must not re-litigate it; it must test whether *interval* calibration degrades too, and by
how much relative to this point-accuracy loss.

### 3.6 Explicitly protected — must not be touched

| Path | SHA-256 |
|---|---|
| `.chain/state.json` | `53c6f1d7b7165769c4f462baffc00ef638586c5f06fba71943e66ceb97f6e4f9` |
| `data/processed/MREID_property.csv` | `1be10465221668ffde0fe7392efbc183ef77a654549a2724b26439978fad8877` |
| `data/processed/millow_token_map.csv` | `91fcb8a3346d89a91360ce531b685e2e97683b7f80f2ac2c44dedf83042106f4` |
| `artifacts/valuation/v2_1/cv_folds_v2_1.json` | `989b79656b051b3b6c8c583b4170a5aaa1c31676260772247bbc98d1eb095156` |
| `models/valuation/valuation_v2_xgboost_random.joblib` | `0e658e818ea46657b9a4b59e32034f55fd3f8ee032ea2978b7a7d734cfa04081` |
| `models/valuation/valuation_v2_xgboost_location_grouped.joblib` | `d49826367a778739830562416e222aaa51d3a87296e6fd8754aa72f7da85e3db` |
| `artifacts/valuation/valuation_v2_metrics.csv` | `353c150fc4afaf63fb11e3224754a4db4dd0220014d6a9e66c6fa89e03b4e802` |

Plus, by instruction: all V2.1 fold artifacts, all Experiment 1 models and artifacts,
and all four existing tags (`v1.0.0-reproducible-state`, `v1.1.0-research-checkpoint`,
`v1.2.0-experiment1-xgb-anchor`, `v1.3.0-experiment1-catboost-backbone`).

---

## 4. Data/fold contract for Experiment 2

### 4.1 What Experiment 2 may reuse without retraining

| Leg | Reusable from Experiment 1? | Reason |
|---|---|---|
| Held-out **test** predictions | **Yes** | The 10 persisted pipelines reproduce Experiment 1 exactly and predict their own `valid_k` rows, which they never saw. |
| Calibration residuals | **No** | See Section 5.2. There is no admissible source. |
| Fold indices | **Yes** | `cv_folds_v2_1.json` supplies `train_k` / `valid_k` unchanged. |
| Feature pipeline | **Yes** | `make_pipeline` + `build_dataset` are reused verbatim. |

### 4.2 What Experiment 2 must create

A **new, derived** fold file for the nested calibration split, at a new path
(`artifacts/valuation/v2_2/conformal/calibration_folds_v2_2.json`). It records, for each
`(regime, outer_fold)`, the three index sets `fit`, `calibrate`, `test`, plus digests and
provenance pointing back at the frozen V2.1 file. It **does not replace, extend, or
reinterpret** `cv_folds_v2_1.json`; `test` in the new file must be bit-identical to
`valid_k` in the frozen file, and this equality will be asserted in code.

### 4.3 Group semantics carried forward

`source_city__location`, 1,789 groups over 28,398 rows:

| Statistic | Value |
|---|---|
| Groups | 1,789 |
| Min / median / max group size | 1 / 3 / 685 |
| Groups with ≥ 2 rows | 1,149 (64.2%) |
| Groups with ≥ 5 rows | 713 (39.9%) |
| Groups with ≥ 10 rows | 499 (27.9%) |
| Groups with ≥ 50 rows | 139 (7.8%) |
| Cities | 6 |
| Rows per city | mumbai 6,820; kolkata 6,270; bangalore 5,438; chennai 4,208; delhi 3,859; hyderabad 1,803 |

The median group has **3 rows**. This single fact constrains every group-aware method in
Section 6.3 and is the reason group-level calibration power is reported rather than
assumed.

---

## 5. Calibration/evaluation split design

### 5.1 Proposed design — nested 3-way split inside `train_k`

For each regime `r ∈ {random, location_grouped}` and each outer fold `k ∈ 1…5`:

```
train_k        (from frozen cv_folds_v2_1.json; NOT the test fold)
  ├── T'_k    ⊂ train_k   →  fits the nested CatBoost (frequency, scaler, one-hot, model)
  └── C_k     ⊂ train_k   →  produces calibration residuals only; never fitted on

test_k         = valid_k from the frozen file, unchanged, used for NOTHING but scoring
```

with hard invariants, asserted at runtime and in tests:

```
T'_k ∩ C_k   = ∅
T'_k ∩ test_k = ∅
C_k   ∩ test_k = ∅
T'_k ∪ C_k  = train_k          (exactly; no row dropped)
test_k       = frozen valid_k  (exactly; bit-identical)
```

**Invariant verification for `location_grouped`** (stricter than above):

```
group(T'_k) ∩ group(C_k)   = ∅     ← calibration localities unseen by the model
group(T'_k) ∩ group(test_k) = ∅     ← already guaranteed by the frozen contract
group(C_k)   ∩ group(test_k) = ∅     ← required by this design
```

The last line is the one that matters scientifically. It makes the calibration set
**structurally identical** to the test set under location shift: both consist of
localities the nested model has never seen. This is what lets RQ3 measure shift-induced
miscalibration rather than a train/test mismatch in the calibration regime itself.

### 5.2 Why the frozen 5-fold contract alone is NOT sufficient — audit evidence

The frozen contract provides exactly two disjoint sets per fold: `train_k` and
`valid_k`. Conformal calibration requires residuals from rows that are neither fitted
on nor scored. Three independent lines of evidence show no admissible source exists.

**(a) Model coverage leaves nothing over.**
`model_k` was fitted on `train_k`. The rows it has never seen are exactly `valid_k`,
which is the evaluation fold. Therefore the *only* out-of-sample residuals obtainable
from the frozen artifacts are test residuals. Using them is contamination by definition.

**(b) Cross-conformal / CV+ reuse is in-sample.**
For `j ≠ k`, `model_j` was fitted on `train_j`, and `train_j` covers everything except
`valid_j` — which **includes all of `valid_k`**. Measured:

| Regime | `valid_k` rows inside `model_j`'s training set |
|---|---|
| random | **5,680 / 5,680 = 100%** |
| location_grouped | **4,858 / 4,858 = 100%** |

`model_j` has seen every test row *and its label*. Its residuals on `valid_k` are
in-sample and inadmissible as calibration scores. Mutual calibration across the 5
persisted models is therefore **not available**.

**(c) Jackknife+ is not constructible and LOO is infeasible.**
Jackknife+ requires *n* predictions per test point from models that each never saw that
point. Only `model_k` never saw `valid_k`, giving **1** clean prediction per test point
where *n* = 5 is required. LOO jackknife would need 28,398 refits. Both are ruled out.

**(d) In-sample calibration would actively mislead.**
Measured on `random_fold1` (frozen model, read-only):

| Quantity | `train_1` (in-sample) | `valid_1` (out-of-sample) | ratio |
|---|---|---|---|
| mean \|residual\| (log) | 0.449492 | 0.476294 | 0.9437 |
| 95th pct \|residual\| (log) | 1.296735 | 1.359095 | 0.9541 |

In-sample residuals are ~5–6% optimistically small. Calibrating on them would yield
systematically **too-narrow** intervals and manufactured undercoverage.

### 5.3 Verdict on the audit question

> **The existing V2.1 five-fold structure is NOT sufficient for Experiment 2. A nested
> calibration structure is scientifically necessary.**

This is a structural property of how the frozen artifacts were produced, not a defect
in them and not a reason to alter the frozen contract. The nested split is **derived
from** `train_k` and written to a **new** file; `cv_folds_v2_1.json` is never opened for
writing. The fold contract is honoured, not changed.

### 5.4 Consequence that requires explicit approval

A nested `T'_k` is strictly smaller than `train_k`, so **the nested model is not the
frozen Experiment 1 backbone** — it is the same hyperparameter configuration fitted on
~80% of `train_k`. Experiment 2's intervals are therefore valid for the *nested* model,
and will be **wider** than intervals the frozen backbone would produce. This is the
standard cost of split conformal and is not avoidable while preserving the no-leakage
rule.

**This is flagged as a required-approval item:** implementing the design in Section 5.1
means training new CatBoost models. The current instruction forbids training, so this
report stops here.

### 5.5 Sizing the inner split (feasibility, not a decision)

Measured on outer fold 1, using the frozen `train_k` sizes:

| Regime | `cal_frac` | `n_fit` | `n_cal` | 80% estimable | 90% estimable | 95% estimable |
|---|---|---|---|---|---|---|
| random | 0.10 | 20,446 | 2,272 | yes | yes | yes |
| random | 0.20 | 18,174 | 4,544 | yes | yes | yes |
| random | 0.25 | 17,038 | 5,680 | yes | yes | yes |
| location_grouped | 0.10 | 21,186 | 2,354 | yes | yes | yes |
| location_grouped | 0.20 | 18,832 | 4,708 | yes | yes | yes |
| location_grouped | 0.25 | 17,655 | 5,885 | yes | yes | yes |

All levels are estimable at every candidate size. The binding constraint is **not**
estimability but statistical power for RQ3 and the group-level power in Section 15.5.
`cal_frac` is to be fixed by pre-registration, not chosen after seeing coverage.

### 5.6 Inner split generation (deterministic, declared before any result)

- `random_state = 42`, inherited from `protocol.RANDOM_STATE`. No other seed is used
  anywhere in Experiment 2.
- Regime `random`: inner split is `KFold`-compatible and row-wise (`shuffle=True`,
  `random_state=42`), mirroring the outer splitter's semantics.
- Regime `location_grouped`: inner split is **`GroupKFold(shuffle=True, random_state=42)`
  on the same `source_city__location` key**, guaranteeing
  `group(T'_k) ∩ group(C_k) = ∅`.
- Inner assignment uses a **single** deterministic split per outer fold, not nested
  K-fold loops, so the design stays interpretable and cheap.
- The derived split file records: parent fold, both index lists, `group_column`,
  `random_state`, `cal_frac`, and a SHA-256 digest per entry, mirroring the frozen
  file's digest convention.

---

## 6. Conformal methods

Notation: `α = 1 − nominal_coverage`. Nominal levels **0.80, 0.90, 0.95**, so
`α ∈ {0.20, 0.10, 0.05}`.

All methods operate on **`log1p(price)`**, the space the backbone predicts and the space
`MAE_log`/`R2_log` are defined in. INR intervals are produced only at presentation time
by `max(expm1(·), 0)`.

### 6.1 Method A — Split conformal (absolute residual)

Established methodology: Vovk, Gammerman & Shafer (2005); Lei et al. (2018);
Angelopoulos & Bates (2023) tutorial.

Score: `s_i = |y_i − ŷ_i|` in log space.

Interval: `[ŷ − q̂_α, ŷ + q̂_α]`.

Properties: **homogeneous** — constant width in log space for every test point.
Requires only exchangeability of calibration and test scores; no model refitting
internally.

**Design-motivation diagnostic** (frozen models, read-only, *not* an Experiment 2
result) — the score is heteroscedastic in predicted level, which is what motivates
Method B:

| Regime (fold 1) | quintile 1 mean \|resid\| | quintile 5 mean \|resid\| | ratio | p90 ratio |
|---|---|---|---|---|
| random | 0.372968 | 0.526653 | **1.412** | 1.583 |
| location_grouped | 0.381048 | 0.625006 | **1.640** | 2.000 |

Residual scale grows with predicted price, and grows *more* under location shift
(ratio 1.64 vs 1.41). A single global `q̂_α` will therefore over-cover cheap properties
and under-cover expensive ones. That is the hypothesis RQ6 tests.

### 6.2 Method B — Normalized conformal

Established methodology: Lei et al. (2018), *Distribution-Free Predictive Inference for
Regression* (§4.2); Romano, Patterson & Candès (2019), *Conformalized Quantile
Regression*.

Score: `s_i = |y_i − ŷ_i| / σ̂(x_i)`.

**Scale mechanism — explicitly declared.** No scale may be chosen after seeing coverage.

Method set is **CLOSED per Phase 0 §0.2**: only **A, B1, B3, C1, C2, C3** are included.
**B2 is EXCLUDED.** The row below is retained to document what was considered and
rejected, not as a live option.

| ID | Scale `σ̂(x)` | Rationale | Status |
|---|---|---|---|
| **B1** | `σ̂(x) = ŷ(x)` — the frozen model's own log-price prediction | Directly targets the measured heteroscedasticity (Section 6.1). Requires **no new model**. Cannot be zero or negative since `log1p(price) ∈ [14.51, 20.57]` on this dataset. | **INCLUDED** |
| B2 | `σ̂(x) =` mean `\|residual\|` of the `m` nearest calibration rows in standardised 48-feature space | Localised conformal; would adapt beyond a pure price scale | **EXCLUDED** (§0.2) |
| **B3** | `σ̂(x) = 1` | Degenerate case; must reproduce Method A to within floating-point tolerance | **INCLUDED** (control) |

**B3 is an implementation self-test, not a method.** If normalised conformal with
`σ̂ ≡ 1` does not reproduce Method A, the normalisation code is wrong. This is asserted
in tests.

**Why B2 was excluded.** It was designed (standardised 48-feature L2 distance, `m`
neighbours, fitted on `C_k` only) and then dropped at Phase 0. Excluding it removes the
only method whose interval width would have varied for reasons *other* than the price
scale, which makes the A-vs-B1 width comparison interpretable: under B1, any width
heterogeneity is attributable to the price scale and nothing else. No `m` needs to be
pre-declared, since B2 is not in scope.

**Guard on `ŷ` as a divisor (B1).** `ŷ` is used as a divisor. It cannot be zero or
negative on this dataset, but the implementation must still floor it explicitly and
record any activation in the manifest. Silent clipping is prohibited (Section 7).

Interval: `[ŷ − q̂_α · σ̂(x), ŷ + q̂_α · σ̂(x)]` → **heterogeneous width**, which is the
entire point.

### 6.3 Method C — Location-aware / grouped calibration

The instruction for this section is explicit: *do not invent a method merely because it
sounds novel*, and *clearly distinguish established conformal methodology from the
MILLOW-specific experimental design*. Accordingly, only established constructions are
listed, each with its standing, and the honest negative result is stated first.

#### C0 — The honest primary finding: there is no guarantee under location shift

Location-grouped evaluation breaks the exchangeability assumption that every
distribution-free conformal guarantee rests on. Calibration localities and test
localities are, by construction, disjoint sets. **Therefore no method in this document
provides a distribution-free coverage guarantee in the `location_grouped` regime.**
Coverage there is *measured*, not guaranteed. This is a property of the question, not a
gap in the design, and the results must be phrased accordingly.

#### C1 — Mondrian (category-conditional) conformal on `source_city` — **established**

Vovk et al. (2005); Vovk (2012). Coverage is guaranteed **conditional on the category**,
under per-category exchangeability.

- Stratum: `source_city` — a *feature*, so assignable at prediction time. No leakage.
- All 6 cities appear in both `C_k` and `test_k` in both regimes, so every category is
  estimable in every fold.
- Power: smallest city (hyderabad) has 1,803 rows total → ~1,447 in `train_1` → ~289 in
  `C_1` at `cal_frac=0.20`. Adequate for a 95% level; **thin** for tight interval
  estimates. Reported per-city, never pooled away.

#### C2 — Mondrian on a *predicted* price band — **established, and the assignable variant**

- Stratum: **predicted** price band, e.g. tertiles of `ŷ(x)` → **low / mid / high**,
  cutpoints computed on `T'_k` predictions and applied unchanged to `C_k` and `test_k`.
- Assignable at prediction time (depends only on `x`), so this is a usable deployment
  strategy, unlike a true-label band.
- **Explicit design distinction:** the *true*-price band is used **only for diagnostic
  reporting** in Section 12 (RQ6) and **never** to assign a stratum, because it depends
  on the unknown label. Reporting coverage by true band is legitimate; calibrating on it
  would not be usable.

#### C3 — Cluster/grouped conformal — **established, secondary**

Coverage with the *group* (locality) as the exchangeable unit rather than the row.

- Established as grouped/cluster conformal prediction.
- **Standing: clearly weaker.** The exchangeability argument applies to whole groups
  being simultaneously exchangeable, and with a **median group size of 3** the
  group-level effective sample size is roughly an order of magnitude below the row count:
  at `cal_frac=0.20`, outer fold 1 has ~4,708 calibration rows in only **~286 groups**.
- Reported as secondary, always alongside the row-level result, never as the headline.

#### C4 — Weighted conformal under covariate shift — **established method, but NOT PROPOSED here**

Tibshirani, Barber, Candès & Ramdas (2019) provide a distribution-free guarantee for
weighted conformal under covariate shift, given an importance weight
`w(x) = p_deployment(x) / p_calibration(x)`.

**It is deliberately excluded from the primary design.** The weight must estimate the
ratio between the held-out-locality distribution and the training distribution. Here
that ratio is dominated by "is this locality in the test fold at all", which is a
deterministic 0/∞ event under the frozen grouped contract — not a smooth density ratio
estimable from features. Adopting it would require inventing an estimator for a weight
that is not identifiable here, and would produce intervals that are formally guaranteed
and empirically meaningless. Recorded as **considered and rejected, with reason**.

#### Not proposed, deliberately

- Any method that pools calibration residuals across folds (transductive schemes).
  Calibration is strictly per-fold in this design.
- Any method that uses `location_frequency` or the unseen-location flag to *inflate*
  width, except as an explicitly labelled post-hoc diagnostic in Section 12. Adjusting
  width by a hand-set factor would be curve-fitting to a coverage target, not conformal
  prediction.
- Any adaptive/conformal-risk-control procedure that tunes a parameter against
  validation coverage. That is a second tuning loop on a frozen backbone, and is out of
  scope.

### 6.4 Established methodology vs MILLOW-specific design — separation

| Established conformal methodology (cited, not invented here) | MILLOW-specific experimental design choices (this document) |
|---|---|
| Split conformal with absolute residual (Vovk et al. 2005) | Using `log1p(price)` as the conformal space |
| Normalised / CQR conformal (Lei et al. 2018; Romano et al. 2019) | Choosing `ŷ(x)` as the scale (B1) rather than a quantile model |
| Mondrian / category-conditional conformal (Vovk et al. 2005) | Stratifying on `source_city` (C1) and on *predicted* price band (C2) |
| Cluster/grouped conformal | Treating `source_city__location` as the cluster unit |
| Weighted conformal under covariate shift (Tibshirani et al. 2019) | **Excluded**, with the reason recorded in C4 |
| Gneiting & Raftery (2007) interval score | Using it as the primary sharpness-plus-penalty metric |
| Clopper–Pearson coverage intervals | Using them for RQ10 rather than asymptotic Wald intervals |
| Nested / cross-validation conformal | The specific 3-way `T'_k / C_k / test_k` split and its group invariants |

No method in the left column is new. Every design choice in the right column is a
pre-registered protocol decision, and none of them changes what the left column
guarantees.

---

## 7. Nonconformity scores

| Method | Score (log space) | Heterogeneous? | Scale fitted on |
|---|---|---|---|
| A | `s_i = \|y_i − ŷ_i\|` | No | — |
| B1 | `s_i = \|y_i − ŷ_i\| / ŷ_i` | Yes | — (frozen predictor) |
| B3 | `s_i = \|y_i − ŷ_i\|` (σ̂ ≡ 1) | No | — control, identical to A |
| C1 | A, computed within `source_city` | No (per-category constant) | `C_k`, per city |
| C2 | A, computed within predicted band | No (per-band constant) | `C_k`, per band |
| C3 | A on group-level aggregate scores | Yes | `C_k`, per group |
| ~~B2~~ | ~~`s_i = \|y_i − ŷ_i\| / σ̂_knn(x_i)`~~ | — | **EXCLUDED** (§0.2) |

**Space decision, stated once and applied everywhere.** Scores are computed on
`log1p(price)` residuals because (i) the backbone predicts in that space, (ii)
`R2_log`/`MAE_log` are the Experiment 1 primary metrics, and (iii) the score is then
scale-free w.r.t. price level. INR scores are **not** used for the headline analysis;
they would be dominated by luxury rows, exactly the failure mode `R2_log` was adopted
to avoid in Experiment 1. INR widths are reported as a secondary presentation metric.

**Degenerate-score handling.** `ŷ` is used as a divisor in B1. It cannot be zero or
negative on this dataset (`log1p(price) ∈ [14.51, 20.57]`), but the implementation must
still guard with an explicit floor and record any activation in the manifest. Silent
clipping is prohibited.

---

## 8. Quantile calculation

**The conformal quantile, computed exactly, with the finite-sample correction.**

For calibration scores `s_1 … s_n` sorted ascending, and miscoverage `α`:

```
k        = ceil( (n + 1) * (1 - alpha) )
q̂_α     = s_(k)              if k <= n
        = +inf               if k > n      → interval reported as unbounded
```

Implementation requirements:

- `numpy.quantile` **linear interpolation must not be used**. Take the exact order
  statistic `s_(k)`. Interpolation between order statistics silently breaks the
  finite-sample guarantee, since the guarantee is stated for the `k`-th order statistic
  specifically.
- The `k > n` branch must be **implemented and unit-tested**, even though it is
  unreachable at the planned sizes (Section 5.5). An unreachable-but-untested guard is
  how a protocol acquires a silent failure mode.
- Verified estimability at the planned sizes, `n_cal = 4,708` (`location_grouped`,
  outer fold 1, `cal_frac=0.20`):

  | nominal | `α` | `k = ceil((n+1)(1−α))` | `k ≤ n` | quantile resolution `1/n` |
  |---|---|---|---|---|
  | 0.80 | 0.20 | 3,768 | yes | 0.00021 |
  | 0.90 | 0.10 | 4,239 | yes | 0.00021 |
  | 0.95 | 0.05 | 4,474 | yes | 0.00021 |

- **One quantile per `(method, regime, outer_fold, nominal_level, stratum)`.** It is
  computed from `C_k` only. It is never computed from, or adjusted using, `test_k`.
- For C1/C2, quantiles are computed per stratum with the same exact rule, and a
  stratum with insufficient `n` falls back to the pooled `C_k` quantile with an explicit
  `fallback_used` flag recorded in the output — never silently.

### 8.1 Resolution of the C1/C2 fallback and C3 aggregation ambiguities (Phase 4 pre-registration addendum)

> **Status.** Recorded before any Phase 4 conformal score or interval is generated.
> No Phase 4 scoring has occurred. Phase 1, Phase 2 and Phase 3 are complete; the ten
> nested CatBoost models exist and are reload-verified. This addendum resolves two
> ambiguities found while preparing the Phase 4 scoring layer, and **supersedes** the
> conflicting wording in the preceding bullet and in the §7 C3 row.

#### 8.1.1 Original ambiguity

**(a) C1/C2 insufficient strata.** The prose in §8 above states that a stratum with
insufficient `n` "falls back to the pooled `C_k` quantile". The frozen Phase 1
implementation (`scripts/experiment_2/conformal.py::build_intervals`) instead emits an
unbounded interval for a fallback/unseen stratum. The two behaviours disagree, and
"insufficient" was not given an explicit numeric definition.

**(b) C3 group aggregation.** §7 describes C3 as "A on group-level aggregate scores"
with heterogeneous width and "per group" scaling, whereas §15.5 and the Phase 2
estimability statement describe a single group-level quantile resting on ~287 groups
(`k = 274` at 95%). The aggregation function that turns a group's row scores into one
group score was not defined.

#### 8.1.2 Selected behaviour (binding)

**C1/C2 — follow the frozen Phase 1 implementation.** A required C1/C2 stratum that is
unseen in `C_k`, or whose calibration size is insufficient, receives:

```
lower        = -inf
upper        = +inf
fallback_used = True
```

"Insufficient" is defined explicitly and only as:

```
k > n_stratum,   where k = ceil((n_stratum + 1) * (1 - alpha))
```

A pooled `C_k` quantile is **not** substituted for an insufficient stratum.

**C3 — one score per calibration group, one quantile per cell.** For each calibration
group `g` defined by `source_city__location`:

```
s_g = mean( |y_i - yhat_i| )   for i in calibration group g
```

Then, for each `(regime, outer_fold, nominal_level)`, take exactly one finite-sample
conformal quantile over the group scores `{s_g}` using the exact order-statistic rule
of §8. Therefore `n = number of calibration groups`; for the location-grouped folds
`n = 287` and at 95% `k = ceil((287 + 1) * 0.95) = 274`. The test interval is:

```
[yhat - q, yhat + q]   in log1p(price) space
```

C3 is a **secondary group-aggregated diagnostic**. The following are explicitly NOT
implemented: max-residual-per-group aggregation, per-group quantiles, and any pooled
fallback for unseen test groups.

#### 8.1.3 Why this is consistent with the frozen protocol and implementation

- **C1/C2.** The frozen, already-validated Phase 1 module is the executable contract;
  adopting its `build_intervals` behaviour keeps Phase 4 a pure consumer of frozen
  code and avoids editing `conformal.py`. An unbounded interval is the honest
  statement that no calibrated interval exists for that stratum, and `fallback_used`
  makes the condition visible rather than silent.
- **C3.** The effective `n` in §15.5 (`287` calibration groups, `k = 274` at 95%) and
  the Phase 2 "30/30 cells, `k <= n`" group-level estimability count only make sense if
  a single quantile is taken over group-level scores. Mean aggregation is the direct
  reading of "group-level aggregate score", and produces exactly 30 C3 cells
  (`2 regimes x 5 folds x 3 levels`).
- No other protocol decision is altered by this addendum.

#### 8.1.4 Phase 4 status

This addendum is recorded **before** Phase 4 scoring begins. No conformal score,
quantile, interval, coverage, width or subgroup quantity has been computed or
persisted. Phase 4 has not run.

---

### 8.2 Resolution of the F0 constant-width / homogeneity ambiguity (Phase 5 pre-interpretation addendum)

Recorded 2026-10-02, after Phase 4 scoring and while implementing the Phase 5
analysis, **before any coverage, sharpness or decision number was interpreted**.

#### 8.2.1 Original ambiguity

Section 10.1 and Section 16 F0 treated the set `(A, B3, C1, C2)` as a single
class of "homogeneous" methods whose interval width is the constant `2·q̂_α`,
so that `sd_width` is expected to be exactly `0`, and any non-zero `sd_width`
for such a method is a **scale leak** requiring all results to be discarded.

That classification contradicts Section 7, which defines C1 as
`|y − ŷ|` computed **within `source_city`** and C2 as `|y − ŷ|` computed
**within predicted price band**. Both are therefore constant-width **within a
stratum** and heterogeneous **across strata**, so their pooled/global
`sd_width` is expected to be non-zero even with a correct implementation.

The contradiction is observable in the frozen Phase 4 artifacts themselves
(read-only): C1 (random fold 1, 0.90) has within-city width SD `~5e-9` but
per-city widths ranging from `0.947` (hyderabad) to `2.428` (mumbai); C2 has
pooled width SD `~0.33`. Neither indicates a scale leak.

A second, independent issue: Phase 4 persisted intervals with
`float_format="%.10g"`, so a mathematically constant width is only recoverable
to roughly `1e-9` relative after serialization and reload. A literal
`sd_width == 0` can therefore never be satisfied from the persisted artifacts,
even for A and B3.

#### 8.2.2 Selected behaviour (binding)

- **Global constant-width F0 check applies only to A and B3**, the genuinely
  globally homogeneous methods. C1 and C2 are **not** members of the global
  homogeneous set.
- A/B3 pass the global check when, for every cell,

  ```
  relative_sd = sd_width / max(|mean_width|, epsilon)  <=  1e-6
  ```

  The `1e-6` is an **implementation/serialization tolerance**, not a
  statistical or scientific performance threshold.
- The A/B3 equivalence check is retained independently and exactly:
  `max|lower_A − lower_B3| == 0` and `max|upper_A − upper_B3| == 0` within
  `1e-12`.
- **C1/C2 are validated structurally instead of by a global SD.** Within each
  `source_city` stratum for C1, and within each predicted-price-band stratum
  for C2, the log width must be constant within the same `1e-6` relative
  tolerance. The pooled/global SD of C1/C2 is descriptive only and must not
  trigger F0 failure.

#### 8.2.3 Why this is consistent with the frozen protocol and implementation

- It follows Section 7's own definition of C1/C2 rather than Section 10.1's
  over-generalisation.
- It does not alter `conformal.py`, whose frozen `HOMOGENEOUS_METHODS`
  constant is left untouched; the analysis module defines its own
  `GLOBAL_HOMOGENEOUS_METHODS = ("A", "B3")` scope.
- The exact-zero requirement is replaced by an explicit relative tolerance
  solely to account for `%.10g` persistence; A and B3 were already verified
  bit-identical (`0.0` bound difference) in Phase 4.

#### 8.2.4 Status and non-effects

- This addendum is recorded **after** Phase 4 scoring completed and is applied
  only to the Phase 5 analysis and its decision table.
- It does **not** alter any coverage, sharpness, subgroup, interval-score or
  method-selection value, and it changes no Phase 4 interval, model,
  calibration fold or protected file.
- Phase 5 is re-run only to update the F0 rows and the decision table; all
  non-F0 rows and all figures are unchanged. Deterministic reproduction is
  re-verified after the change.

---

## 9. Coverage metrics

Recorded for **every** `(method, regime, fold, level, stratum)` cell.

| Metric | Definition | Space |
|---|---|---|
| `n_evaluated` | number of scored test observations | — |
| `empirical_coverage` | `mean( lower ≤ y ≤ upper )` | — |
| `coverage_error` | `empirical_coverage − nominal` | — |
| `abs_coverage_error` | `\|coverage_error\|` | — |
| `coverage_cp_lower`, `coverage_cp_upper` | Clopper–Pearson 95% bounds | — |
| `coverage_significant` | nominal outside the Clopper–Pearson interval | — |
| `miscoverage_low`, `miscoverage_high` | `mean(y < lower)`, `mean(y > upper)` | — |

Required extras:

- **Clopper–Pearson, not Wald.** Coverage near 0.8–0.95 with `n` in the thousands is
  where Wald intervals are least trustworthy. This directly serves RQ10: it separates
  "deviation from nominal that is real" from "deviation that is Monte-Carlo noise".
- `miscoverage_low` and `miscoverage_high` are recorded **separately**, because
  asymmetry is diagnostic. An absolute-residual score on a right-skewed target can
  under-cover from one side only, and pooling the two tails would hide it.
- Aggregation across folds: **mean and sample SD (`ddof=1`)**, matching the existing
  `summary_metrics.csv` convention (`std_ddof = 1`, `n_folds = 5`).
- **Every fold's raw value is retained.** No pooling of predictions across folds —
  the Experiment 1 manifest already asserts `predictions_pooled_across_folds: false`, and
  Experiment 2 will carry the same assertion.

---

## 10. Sharpness metrics

| Metric | Definition | Space |
|---|---|---|
| `mean_width` | `mean(upper − lower)` | log1p(price) — primary |
| `median_width` | `median(upper − lower)` | log1p(price) — primary |
| `sd_width` | sample SD of width, `ddof=1` | log1p(price) |
| `min_width`, `max_width` | extremes | log1p(price) |
| `mean_width_inr` | `mean` of INR width | INR — secondary |
| `median_width_inr` | `median` of INR width | INR — secondary |
| `mean_relative_width` | `mean( (upper − lower) / point )` | ratio — see caveat |
| `interval_score` | Gneiting & Raftery (2007) | both |
| `winkler_score` | same quantity, `β = α`, for `α ∈ {0.10, 0.20}` | both |

**Interval score** (primary combined metric — rewards both coverage and sharpness):

```
IS_α = (u − l) + (2/α)·(l − y)·1{y < l} + (2/α)·(y − u)·1{y > u}
```

It is the proper scoring rule for interval predictions and is reported because a
method can hit nominal coverage with uselessly wide intervals, and mean width alone
would not reveal that.

### 10.1 A non-obvious sharpness caveat discovered during the audit

For **homogeneous** methods (A, B3, C1, C2), the log-space width is the constant `2·q̂_α`
by construction, so:

```
relative width = ( exp(ŷ+q̂) − 1 ) − ( exp(ŷ−q̂) − 1 )
               = exp(ŷ) · ( exp(q̂) − exp(−q̂) )
               = point_estimate × 2·sinh(q̂)
```

so `relative_width = 2·sinh(q̂)` is a **deterministic constant**, identical for every test
row. Measured confirmation at nominal 90% on `random_fold1`: `mean = median = p90 =
251.9%`, and `2·sinh(1.0534) = 2.5196`.

Consequences, both mandatory:

1. **`mean_relative_width` carries zero information for homogeneous methods.** It is a
   restatement of `q̂_α`. It must not be used to compare A against B1, because the
   comparison would be vacuous by construction.
2. For homogeneous methods, sharpness is a **single scalar per cell**: report
   `mean_width` (log) as primary and `median_width_inr` as the business-facing number.
   `sd_width` **will be exactly 0** and this is expected, not a bug — a test will assert
   `sd_width == 0` for homogeneous methods to catch an accidental scale leak.

`mean_relative_width` becomes informative **only** for heterogeneous methods — in scope
that is **B1** alone (B2 excluded per §0.2) — where it measures how much width tracks
price level. It is reported for all methods, flagged with a `width_is_constant` boolean
so downstream readers cannot misread it.

### 10.2 INR asymmetry must be stated, not assumed away

`expm1` is convex, so a log-space symmetric band becomes an **asymmetric** INR band.
Measured at nominal 90%, `random_fold1`:

| Quantity | Value |
|---|---|
| `q̂_0.90` (log) | 1.0534 |
| mean log width | 2.1067 |
| mean INR width | 21,660,982 |
| mean gap, point → lower | 5,601,087 |
| mean gap, point → upper | 16,059,896 |
| **upper / lower gap ratio** | **2.867** |

The INR interval extends ~2.9× further above the point estimate than below it. This is
mathematically expected and is **not** a defect, but it materially affects how the
interval must be communicated and how any downstream consumer (including a future
blockchain consumer) would read it. It is called out here so that no reader assumes INR
symmetry.

### 10.3 A prior expectation worth stating before results exist

A symmetric-in-log band at 90% on this backbone implies, per the identity in 10.1, a
median relative width of ~252% of the point estimate, i.e. roughly a factor
`exp(±1.05)` ≈ ÷2.87 / ×2.87 multiplicatively. **Even a perfectly calibrated split
conformal interval for this backbone is expected to be extremely wide.** Experiment 2
should therefore be read primarily as a test of *calibration*, with sharpness treated as
a separate axis and gated by F3 (Section 16). This expectation is recorded now, before
any coverage number exists, so it cannot be rationalised after the fact.

---

## 11. Random vs location-grouped evaluation

Both regimes are evaluated with **identical** method, level, and metric definitions. The
only difference is the outer fold structure from the frozen file.

| | `random` | `location_grouped` |
|---|---|---|
| Outer splitter | `KFold(shuffle=True, rs=42)` | `GroupKFold(shuffle=True, rs=42)` |
| Test-fold locality novelty | **96.8–97.7%** of test rows are from a seen locality | **100%** of test rows are from an unseen locality |
| `location_frequency` on test rows | mostly > 0 | **exactly 0.0 for every test row** |
| Conformal guarantee | holds under exchangeability | **none — exchangeability broken** |
| Role | positive control (RQ1 must pass) | stress test (RQ3, headline) |

Regime comparison is the core of the experiment and is reported as:

- `regime_shift_coverage.csv` — per method × level, `random` vs `location_grouped`
  coverage, coverage error, and paired delta, with the sign convention
  `Δ = grouped − random` declared explicitly. Negative `Δ` = worse under location shift.
- paired per-fold deltas (same fold index, both regimes) so the comparison is paired and
  not confounded by fold-difficulty differences.
- the same for every sharpness metric and the interval score.

A reference point already established by Experiment 1, for scale: CatBoost `R2_log` fell
0.3640 → 0.2677 (−26%) and `MedAPE` rose 32.5% → 37.3% under location shift. Interval
degradation should be reported **alongside** these, not in isolation, so that a reader
can see whether interval miscalibration is *proportionally worse* than point-accuracy
degradation.

---

## 12. City and price-band subgroup analysis

**Both are diagnostic reporting. Neither is used to assign a calibration stratum except
where explicitly noted (C1, C2).**

### 12.1 City-level (RQ5)

6 cities. All 6 appear in every `test_k` in both regimes. Reported per
`(method, regime, level, city)`.

Rows: mumbai 6,820; kolkata 6,270; bangalore 5,438; chennai 4,208; delhi 3,859;
hyderabad 1,803. Per-fold test counts will be roughly one fifth of these, so
hyderabad contributes ~360 test rows per fold — adequate for a coarse coverage
estimate, thin for a tight one. Reported with `n_evaluated` and Clopper–Pearson bounds
always attached.

**Pre-declared suppression rule:** any subgroup cell with `n_evaluated < 100` is emitted
with `coverage` present but `coverage_reliable = false`, and is excluded from any
summary statement. This prevents a 12-row cell from becoming a headline claim.

### 12.2 Price-band (RQ6)

Two distinct bandings, deliberately separated:

**(a) True-price bands — diagnostic only.** Fixed INR edges, pre-declared:

| Band | Rows | Share |
|---|---|---|
| < 0.5 cr | 9,637 | 33.94% |
| 0.5–1 cr | 9,689 | 34.12% |
| 1–2 cr | 5,608 | 19.75% |
| 2–4 cr | 2,365 | 8.33% |
| 4–10 cr | 886 | 3.12% |
| ≥ 10 cr | 213 | 0.75% |

These edges are **absolute and fixed**, not dataset quantiles, so they stay meaningful
if the dataset changes. Reported per `(method, regime, level, band)` with the suppression
rule above — the ≥10 cr band has 213 rows dataset-wide, so most per-fold cells will fall
below 100 and be marked unreliable.

**(b) Predicted-price bands — the assignable variant (C2).** Tertiles of `ŷ(x)` with
cutpoints computed on `T'_k`. This is the only price-based stratification usable at
prediction time, and it is the one whose coverage should be compared against (a).

**Honest note on why (a) is diagnostic and not a calibration stratum.** Assigning a
stratum from the true price requires the unknown label, so a "true-price-band Mondrian
calibrator" cannot be deployed. It is reported because the *scientific* question — does
coverage vary with price level — is real and worth answering, and because a coverage
profile that degrades monotonically with price band is the clearest possible
demonstration of the heteroscedasticity measured in Section 6.1.

### 12.3 Residual diagnostics (RQ8)

Per `(regime, fold)`, using frozen-model held-out residuals as a design-motivation
reference and nested-model residuals for the actual analysis:

- residual vs fitted (log space), with city colouring
- QQ plot of residuals against the normal reference
- histogram + kernel density of residuals, with the calibrated `±q̂_α` overlaid
- scale–location plot: `|residual|` vs `ŷ`, with the B1 scale overlaid
- residual vs `location_frequency` (expecting structure at 0.0 under location shift)
- Shapiro–Wilk and a tail-index check; reported as diagnostics, not as tests with
  significance claims

Heavy tails are expected: `price` has a max/min ratio of **427.3**
(2,000,000 → 854,599,999). If residuals are heavy-tailed, an absolute-residual score
produces intervals that are conservative in the bulk and can still under-cover the
extremes — and the subgroup and tail diagnostics, not the marginal coverage number,
are what would reveal it.

### 12.4 Plots (all PNG **and** PDF, per Experiment 1 convention)

1. coverage vs nominal, `random` regime, one line per method, diagonal = perfect
2. coverage vs nominal, `location_grouped` regime, same
3. coverage vs nominal, both regimes overlaid, to make RQ4's monotonicity visible
4. mean width vs nominal coverage, both regimes, one line per method (RQ7)
5. paired regime-shift coverage delta, per method × level
6. interval score vs nominal, both regimes
7. city-level coverage, grouped bar with Clopper–Pearson error bars, per method × level
8. price-band coverage profile, per method × level, both bandings
9. residual vs fitted, faceted by regime
10. QQ plot, faceted by regime
11. scale–location plot with B1 scale overlaid
12. calibration curve: empirical coverage of `s ≤ q̂_α` on `C_k` vs on `test_k`, both
    regimes — the single most direct visualisation of calibration transfer

All plots must be byte-deterministic, using the same
`metadata={"CreationDate": None}` mechanism established and verified in Experiment 1.

---

## 13. Leakage controls

Enforced in code as assertions, and covered by tests. Every item is a hard failure.

| # | Control | Assertion |
|---|---|---|
| L1 | Calibration ∩ test | `set(C_k) ∩ set(test_k) == ∅` |
| L2 | Fit ∩ test | `set(T'_k) ∩ set(test_k) == ∅` |
| L3 | Fit ∩ calibration | `set(T'_k) ∩ set(C_k) == ∅` |
| L4 | Partition completeness | `T'_k ∪ C_k == train_k`, no row lost |
| L5 | Test identity | `test_k ==` frozen `valid_k`, bit-identical |
| L6 | **Quantile source** | every `q̂_α` computed from `C_k` scores only; enforced by passing scores through a function that receives `C_k` indices and nothing else |
| L7 | **Frequency-feature refit scope** | `TrainOnlyFrequencyFeatures` fitted on `T'_k` only — **never** on `T'_k ∪ C_k`. This is the subtlest trap in the design: refitting on `T'_k ∪ C_k` would leak calibration-row locality counts into both the model and the calibration residuals. |
| L8 | Scaler / one-hot fit scope | `StandardScaler` and `OneHotEncoder` fitted on `T'_k` only |
| L9 | Group disjointness | `group(T'_k) ∩ group(C_k) == ∅` (location-grouped) |
| L10 | Test-fold locality novelty | `group(T'_k) ∩ group(test_k) == ∅` (location-grouped) |
| L11 | Target not a feature | `'price' not in INPUT_FEATURES` |
| L12 | Fold immutability | `cv_folds_v2_1.json` SHA-256 equals `989b7965…` before **and** after every run |
| L13 | Protected-file hashes | all 7 Section 3.6 hashes re-verified after every run |
| L14 | Tag immutability | all 4 tags retain their recorded hashes |
| L15 | Write-path safety | `assert_safe_write_targets()` extended with every protected path in Section 3.6, refusing to run on collision |
| L16 | No fold regeneration | `cv_folds_v2_1.json` opened **read-only**; `generate_folds` / `generate_all_folds` never called |
| L17 | Single seed | `random_state == 42` everywhere; no undeclared seed anywhere in Experiment 2 |
| L18 | No post-hoc selection | `cal_frac`, method set, scale definition, band edges, and `m` all fixed in this document **before** any coverage number is computed |
| L19 | No blockchain write | no module under Experiment 2 imports `.chain`, `backend`, or any blockchain writer |
| L20 | No pooled-across-fold quantiles | one quantile per `(method, regime, fold, level, stratum)`; never pooled |

### 13.1 The three controls that would actually break this experiment

Stated explicitly because generic leakage checklists miss them:

- **L7** is the one that produces *plausible-looking but invalid* numbers. Fitting the
  frequency transformer on `T'_k ∪ C_k` inflates the apparent locality support of
  calibration rows, shrinking their residuals and silently improving apparent
  calibration. Nothing crashes; the coverage just looks better than it is.
- **L6** is the one that is easiest to violate by accident, because it is one refactor
  away: a helper that receives the whole dataset and slices internally cannot be
  statically checked. The design therefore passes **indices**, not data, into the
  quantile function.
- **L9/L10 together** are what make the location-grouped result interpretable. If
  calibration localities overlap test localities, the experiment silently stops
  measuring location shift and starts measuring nothing.

---

## 14. Reproducibility requirements

Reusing established Experiment 1 conventions rather than inventing new ones.

| Requirement | Convention reused |
|---|---|
| Environment record | `protocol.environment_record()` → `environment_v2_2.json`, same schema as `environment_v2_1.json` |
| Folds provenance | record `folds_loaded_from` and `folds_regenerated: false`; **add** `parent_folds_sha256` |
| Fold digests | SHA-256 per entry, mirroring the frozen file's `digest` convention |
| Manifest | `experiment_manifest_v2_2.json` mirroring `experiment_manifest.json` key-for-key where applicable, including the boolean discipline flags |
| Determinism | `deterministic: true`, input + output SHA-256 manifest, `mismatches: 0`, mirroring `analysis_manifest.json` |
| Zero-model-training accounting | manifest must state `backbone_retrained_for_calibration: true` and record nested fit timings; **no** claim of zero models trained (contrast Experiment 1, where `models_trained_this_run` was 4) |
| Verification script | `verify_conformal.py` mirroring `verify_model_comparison.py`: reload artifacts and re-verify recorded values within tolerance |
| Tests | `test_conformal.py` mirroring `test_protocol.py`/`test_model_comparison_analysis.py` conventions |
| Aggregation | sample SD, `ddof=1`, `n_folds` recorded |
| Plots | PNG + PDF, byte-deterministic via `metadata={"CreationDate": None}` |
| Metric keys | declared as an explicit constant, asserted by a test, exactly as `METRIC_KEYS` is |

### 14.1 Known repository-level caveat carried forward

`analysis_manifest.json` (Experiment 1) records SHA-256 of **working-tree** bytes, while
Git stores 18 affected text artifacts LF-normalised under `core.autocrlf=true`. A fresh
Linux clone therefore cannot reproduce those specific hashes. This is **pre-existing**,
identical for the already-committed V2.1 artifacts, and applies identically to
Experiment 2's manifest. It is **not** introduced by Experiment 2 and **not** fixed
here, because fixing it would require adding a `.gitattributes` rule and re-hashing
committed artifacts — an Experiment 1 change, which is forbidden. Experiment 2 will
follow the same convention and the same caveat, and will additionally record the
`core.autocrlf` setting in its manifest so the discrepancy is diagnosable rather than
mysterious.

### 14.2 Runtime expectation

Measured from Experiment 1 `training_time.csv`, CatBoost `random_fold1_fit_seconds`
≈ 11.0 s, `location_grouped_fold1_fit_seconds` ≈ 11.2 s. A nested design trains **10 new
models** (2 regimes × 5 folds), plus predictions on `C_k` and `test_k`.

```
≈ 10 × ~11 s  ≈  ~110 s   fits
+ ~10 × (2 × ~0.5 s)      predict passes
≈ ~2-4 minutes total, single-threaded-feasible
```

Comfortably cheap. Runtime is not a constraint on this experiment.

---

## 15. Statistical limitations

Stated now, before results, so they constrain interpretation rather than being
retro-fitted to it.

### 15.1 The guarantee does not extend to the location-grouped regime

Exchangeability fails by construction. Every coverage number in that regime is a
**descriptive measurement**. Phrasing must be "observed coverage under location shift",
never "coverage is 90%". This is the single most important limitation in the document.

### 15.2 The calibrated model is not the frozen backbone

Per Section 5.4, nested `T'_k` ⊂ `train_k`, so the nested model is strictly weaker in
point accuracy than the Experiment 1 backbone, and intervals calibrated for it are
wider than the frozen backbone's would be. Consequence: **Experiment 2 results
generalise to "a CatBoost with this configuration fitted on ~80% of `train_k`", not to
"the Experiment 1 backbone"**. Reporting intervals for the frozen backbone from these
calibration sets would be invalid.

### 15.3 Five folds is a weak variance estimate

Mean and SD across `n = 5` folds are reported per the existing convention, but a sample
SD from 5 points has very wide uncertainty and no useful confidence interval. Fold-level
values are always reported alongside the aggregate, never instead of it. Any claim of
the form "regime A is worse than regime B" must be supported by the paired per-fold
deltas and their sign consistency, not by the SD magnitude alone.

### 15.4 Rows within a locality are not independent

Properties in one locality share location characteristics, and localities recur across
folds in the random regime. Coverage is computed over correlated rows, so **binomial
confidence intervals — including the Clopper–Pearson bounds used for RQ10 — are
anti-conservative** (too narrow). This is a property of the data, not of the estimator.
Mitigation: Clopper–Pearson bounds are reported as a *necessary but not sufficient*
significance screen, and clustered/block bootstrap by locality is **declared out of
scope** rather than silently omitted. Any "significant undercoverage" statement is
qualified accordingly.

### 15.5 Group-level power is an order of magnitude below row-level power

Median group size is 3. At `cal_frac = 0.20`, `location_grouped` outer fold 1 has ~4,708
calibration rows in only **~286 groups** (~143 at `cal_frac = 0.10`). A group-level
quantile at 95% therefore rests on ~14 groups in the tail. C3 results must carry the
group count alongside every number, and are secondary by construction.

### 15.6 Only one backbone

CatBoost only. Nothing here licenses a claim about Ridge, RandomForest, LightGBM or
XGBoost, or about conformal prediction in general for this dataset. Cross-model
generalisation would require re-running the whole design per model.

### 15.7 Multiple comparisons

> **Corrected.** The draft previously stated "≈ 126 primary cells" for
> `3 × 2 × (A, B1, B2, B3, C1, C2, C3)`. That product is 42, not 126, and B2 is now
> excluded. The arithmetic is restated correctly below rather than silently adjusted.

Counting **stratum-level** cells, where each Mondrian stratum is its own cell:

| Component | Cells per (level, regime) |
|---|---|
| A, B1, B3 (pooled) | 3 |
| C1 — 6 `source_city` strata | 6 |
| C2 — 3 predicted-price tertiles | 3 |
| C3 — group-level aggregate | 1 |
| **subtotal** | **13** |

```
13 x 3 levels x 2 regimes = 78 primary stratum-level cells
```

plus secondary outputs: city coverage (6 bands), true-price-band coverage (6 bands),
predicted-band coverage (3), group-level rows for C3, and any `fallback_used` cells.
Total examined quantities comfortably exceed 150.

**No multiplicity correction is planned.** Consequently the experiment is **exploratory
and descriptive**: subgroup and method comparisons are reported as observed patterns,
and no p-value-driven selection is permitted. This is declared in advance so that a
striking-looking subgroup is not later presented as a confirmatory finding. It is also
the reason F2 (§16) requires the rejection criterion to hold at **all three** nominal
levels rather than any one of them.

### 15.8 Honest recall correction for correlated residuals

Interval score and relative width are computed over correlated rows and inherit the same
inflation concerns as coverage. They are used as **comparative** metrics across folds and
methods, not as calibrated population quantities.

### 15.9 Single inner split per outer fold

One deterministic inner split, not repeated resampling. Consequences: the coverage
estimate carries inner-split variability that is not quantified; and no
split-averaging is available to stabilise `q̂_α`. Deliberate, to keep the design
interpretable and cheap, and recorded as a limitation.

### 15.10 Audit limitation: F1/F2/F6 fold-aggregation ambiguity (post-audit note)

> **Post-audit note — not a binding criterion and not a resolution.** This records a
> limitation identified during the Phase 5 decision audit. It does not change any
> pre-registered criterion, any F1–F7 decision value, or any implementation behaviour,
> and it does not retroactively declare the `any()` interpretation to be the intended
> protocol rule.

The frozen protocol does not explicitly specify whether the Clopper–Pearson screening
used by F1/F2/F6 is evaluated independently per fold with an any-fold rule or on pooled
observations across folds. The implementation uses the per-fold `any()` interpretation.
Because the alternative pooled interpretation changes F1 and F2 outcomes and makes F6
testable rather than vacuous, these 19 rows are treated as ambiguity-affected and are not
interpreted as definitive findings. No post-hoc reclassification was performed.

---

## 16. Failure conditions and falsifiers

Pre-registered. These are decision rules, fixed before any coverage number exists.

### F0 — Implementation validity gate (must pass before anything is interpreted)

| Condition | Verdict |
|---|---|
| B3 (`σ̂ ≡ 1`) does not reproduce Method A within `1e-12` | **Implementation broken.** Stop; no result interpretable. |
| Any L1–L20 assertion fails | **Leakage.** Stop; discard all results. |
| `sd_width != 0` for a homogeneous method | Scale leaked into a constant-width method. Stop. |
| Protected hashes (Section 3.6) changed | Protected file modified. Stop. |
| Any metric key outside the declared constant | Contract violation. Stop. |

### F1 — Negative control on the method (RQ1)

If `random`-regime empirical coverage at any nominal level falls **outside** the
Clopper–Pearson 95% interval around nominal, split conformal is not working even where
its assumptions hold. This would indicate an implementation or conceptual error, **not**
a limitation of conformal prediction. The experiment's headline claim would be void.

Conversely, RQ1 passing does **not** license any claim about the location-grouped regime.

### F2 — Headline falsifier (RQ3)

**Hypothesis under test:** conformal prediction provides calibrated intervals for this
backbone *including under location-held-out distribution shift*.

**Rejected** if, in the `location_grouped` regime, `abs_coverage_error > 0.05` at **all
three** nominal levels for Method A. **Not rejected** (i.e. the hypothesis survives) if
at least one level achieves `abs_coverage_error ≤ 0.05` **and** its Clopper–Pearson
interval contains nominal.

The all-three-levels conjunction is deliberate: a single lucky level among three, with
~126 cells examined overall, is not evidence.

### F3 — Sharpness / usability gate (RQ7) — THRESHOLD NOW FIXED

Calibration without usability is not a success. Per Section 10.3, a perfectly
calibrated log-symmetric 90% band on this backbone implies ~252% relative width. If that
is confirmed, the correct finding is **"calibrated but not actionable"**, and it must be
reported that way rather than as a success.

**The threshold is no longer open.** Per Phase 0 §0.4 it is fixed as:

```
metric : relative_width = (upper - lower) / point_prediction      [row-level]
level  : nominal 0.90
gate   : median_relative_width <= 2.00   -> usable
         median_relative_width >  2.00   -> not usable
```

This is an **Experiment-specific operational threshold**, not a universal
real-estate industry standard. **Calibration and usability are reported separately**; a
method may be calibrated and still fail this gate.

> Recorded expectation, stated before results exist: for the homogeneous methods
> (A, B3, C1, C2) `relative_width` is the constant `2*sinh(q_hat_alpha)`, so the gate is
> equivalent to `q_hat_0.90 <= 0.8814`. The Section 10.3 diagnostic suggests
> `q_hat_0.90 ~ 1.05` (`relative_width ~ 2.52`), so those methods are **predicted to fail
> the gate**. B1 and B3-vs-A behaviour is the informative comparison. This expectation is
> recorded so a failure cannot later be presented as a surprise. It is **not** grounds to
> change the gate.

### F4 — Overcoverage falsifier

If `abs_coverage_error ≤ 0.05` but `coverage_error ≥ +0.05` systematically while widths
exceed the F3 gate, the finding is "conservative and useless", distinct from F2. Must be
reported as its own outcome, not folded into success.

### F5 — Monotonicity check (DESCRIPTIVE — corrected per Phase 0 §0.5)

> **CORRECTED.** The original wording required `Spearman(nominal_level,
> coverage_error) > 0`. That requirement is **withdrawn**: it imposed a particular
> statistical sign as a validity condition, which was not justified.

**Corrected definition.** For each method × regime × fold, take the ordered sequence of
empirical coverage (equivalently, coverage error) at nominal 0.80 → 0.90 → 0.95 and
report **whether the observed sequence is directionally monotone**. Report the sequence,
the direction, and whether it is monotone, as a **descriptive** characterisation.

**No particular Spearman sign is a validity requirement**, and non-monotonicity is
**not** a falsifier of conformal calibration. It is simply reported as observed. There is
no pass/fail outcome attached to F5.

### F6 — Uniformity falsifier (RQ5, RQ6)

If any city or price band shows `coverage_significant == true` at a nominal level where
the pooled coverage is calibrated, then **marginal** coverage is hiding conditional
miscalibration, and the headline claim must be reported as valid only on average.
City and price-band coverage must therefore be reported even in a "successful" run.

### F7 — C1/C2 improvement falsifier (RQ9)

If neither Mondrian variant narrows or corrects location-grouped coverage relative to
Method A, the honest conclusion is that the shift is not addressable by category
conditioning at this granularity, and **no further method should be invented to fix
it** within Experiment 2. Declared in advance to prevent a post-hoc escalation to C4 or
an ad-hoc variant.

---

## 17. Exact implementation plan

Sequenced. Each step is independently verifiable. **No step has been executed.**

### Phase 0 — Approval gate ✅ COMPLETE

**All Phase 0 items are decided and recorded in Section 0. No item remains open.**

| Item | Decision | §|
|---|---|---|
| 0.1 `cal_frac` | **`0.20`** | §0.1 |
| 0.2 Method set | **A, B1, B3, C1, C2, C3**; B2 and C4 excluded | §0.2 |
| 0.3 F3 threshold | **`median_relative_width <= 2.00`** at nominal 0.90 | §0.4 |
| 0.4 B2 scope | **Excluded**; no `m` required | §0.2 |
| 0.5 Nested training approval | **Granted** — exactly 10 models, frozen config | §0.6 |

Remaining open item, deliberately: the **F3 interaction note** (§0.4) records that
homogeneous methods are *predicted* to fail the usability gate. This is a prediction,
not a decision, and does not reopen Phase 0.

### Phase 1 — Pure library, no I/O, no training ✅ IMPLEMENTED

1.1 `conformal_scores(...)` → **A, B1, B3** score arrays (B2 excluded per §0.2).
1.2 `conformal_quantile(scores, alpha)` → exact `k = ceil((n+1)(1−α))` order statistic,
     plus the `k > n → +inf` branch.
1.3 `build_intervals(...)` → `(lower, upper)` per method, in log space.
1.4 `coverage_metrics(...)`, `sharpness_metrics(...)` including the Gneiting–Raftery
     interval score and Clopper–Pearson bounds.
1.5 `subgroup_coverage(...)` → city and both price-band variants, with the `n < 100`
     suppression rule.
1.6 Pure functions, no filesystem, no global state → trivially unit-testable.

Implemented in `scripts/experiment_2/conformal.py`, tested in
`scripts/experiment_2/test_conformal.py`.

### Phase 2 — Nested fold derivation (writes one new file, no training) ✅ IMPLEMENTED

2.1 `build_calibration_folds()` → deterministic inner split per `(regime, fold)` from
     the frozen `train_k`, group-aware in the grouped regime.
2.2 Assert L1–L5, L9, L10, L16 on the derived structure **before** writing.
2.3 Write `calibration_folds_v2_2.json` with `parent_folds_sha256`, per-entry digests,
     `cal_frac`, `random_state`.
2.4 Re-verify the frozen `cv_folds_v2_1.json` hash (L12).

**Implemented** in `scripts/experiment_2/build_calibration_folds.py`, tested in
`scripts/experiment_2/test_calibration_folds.py`.

**Validation result: complete and verified.** 145 tests pass in
`scripts/experiment_2/` (93 Phase 1 + 52 Phase 2). The artifact
`artifacts/valuation/v2_2/conformal/calibration_folds_v2_2.json` (5,576,976 bytes)
was written, then re-read from disk and re-verified independently.

Evidence, stated as measured rather than assumed:

| Check | Result |
| --- | --- |
| Frozen `cv_folds_v2_1.json` SHA-256 before and after derivation | `989b7965…5156`, unchanged |
| `folds_regenerated` / `frozen_fold_contract_modified` | `false` / `false` |
| Outer fold count × regimes | 5 × 2 = 10 cells, all present |
| L1–L5 (row-level) | pass on all 10 cells |
| L9/L10 + `groups(C) ∩ groups(test)` | pass on all 5 grouped cells |
| `test_k` bit-identity vs frozen `valid_k` | exact on all 10 cells |
| `T' ∪ C_k == train_k` (no row dropped) | exact on all 10 cells |
| Group-level quantile estimability | 30/30 cells, `k ≤ n` |
| Finest quantile resolution | `1/n = 1.86e-04` |
| Protected-path write refusal | 8 protected prefixes refused, both `v2_2` namespaces allowed |
| Verifier negative controls | 6 injected corruptions each caught by the expected rule |

Realised sizes:

| Regime | `n_train` | `n_fit` | `n_calibrate` | `n_test` | realised cal_frac |
| --- | --- | --- | --- | --- | --- |
| random 1–3 | 22,718 | 18,174 | 4,544 | 5,680 | 0.200018 |
| random 4–5 | 22,719 | 18,175 | 4,544 | 5,679 | 0.200009 |
| grouped 1–5 | 21,409–23,540 | 16,266–19,295 | 3,668–5,369 | 4,858–6,989 | **0.1616–0.2402** |

**One deviation from the `cal_frac = 0.20` target, now recorded rather than
smoothed over.** The random regime realises 0.2000 as intended. The
location-grouped regime realises 0.1616–0.2402 because `GroupKFold` balances the
**number of groups**, not the number of rows, and group sizes span 1–685 rows
(global mean 15.87, median 3). Every grouped fold therefore contains exactly 287
calibration groups, while the row count follows wherever the large localities
land. This is inherent to group-disjoint calibration, not a defect: `cal_frac` in
this regime is a target for group count, not an achievable row fraction. It is
pinned by a test so it cannot drift silently.

The consequence for C3 is quantified rather than assumed (§15.5): the effective
`n` for a group-level 0.95 quantile is 287 groups, `k = 274`, which is estimable
but roughly six times thinner than the row-level equivalent. That supports the
existing decision to treat C3 as secondary.

### Phase 3 — Nested training (approved per §0.6, NOT YET EXECUTED)

> **Execution is gated.** Per §0.7, Phase 3 may run only after Phase 1 and Phase 2 are
> implemented, unit-tested and validated. Phase 1 and Phase 2 are now complete and
> validated, so Phase 3 is *unblocked* — but it has **not** been run, and no Experiment 2
> result exists.

3.1 Extend `assert_safe_write_targets()` with all Section 3.6 paths.
3.2 For each `(regime, fold)`: `make_pipeline(build_estimator("CatBoost"), dense_output=True)`
     and `.fit(X[T'_k], y[T'_k])` — identical construction to `run_model.run_model`,
     unchanged hyperparameters.
3.3 Predict on `C_k` and on `test_k`. Persist to new paths only.
3.4 Record fit/predict timings per fold.

### Phase 4 — Calibration and scoring

4.1 Per `(regime, fold, method, level)`: scores on `C_k` → `q̂_α` → intervals on `test_k`.
4.2 Mondrian variants C1 (city) / C2 (predicted band) per stratum, with
     `fallback_used` flags.
4.3 Grouped C3 with group counts attached.
4.4 Emit `per_fold_coverage.csv` and `per_fold_sharpness.csv` (one row per
     `(method, regime, fold, level, stratum)`).

### Phase 5 — Aggregation, subgroup analysis, plots

5.1 `summary_coverage.csv` / `summary_sharpness.csv` — mean, sample SD (`ddof=1`),
     min, max, `n_folds`, `std_ddof`, mirroring `summary_metrics.csv` columns exactly.
5.2 `regime_shift_coverage.csv` — paired `grouped − random` deltas.
5.3 `subgroup_coverage.csv` — city + both price-band variants, with suppression flags.
5.4 The 12 figures of Section 12.4, PNG + PDF, byte-deterministic.
5.5 `residual_diagnostics` outputs.

### Phase 6 — Manifest, verification, tests

6.1 `experiment_manifest_v2_2.json` with the full boolean discipline block
     (`tuning_performed: false`, `hyperparameter_search: false`,
     `early_stopping_used: false`, `post_hoc_configuration_changes: false`,
     `predictions_pooled_across_folds: false`, plus
     `backbone_hyperparameters_changed: false`).
6.2 `analysis_manifest.json` equivalent: `deterministic`, input/output hashes, `mismatches`.
6.3 `environment_v2_2.json` via `protocol.environment_record()`.
6.4 `verify_conformal.py` — reload and re-verify within tolerance; mirror
     `verify_model_comparison.py`.
6.5 `test_conformal.py` — including all L1–L20 assertions as tests, plus the F0 gates.
6.6 Write `EXPERIMENT_2_CONFORMAL_RESULTS.md` **only after** results exist, with no
     validity claim beyond what the data supports.

### Phase 7 — Review and freeze

7.1 Report findings against RQ1–RQ10 and F1–F7 **including negative outcomes**.
7.2 Independent review of leakage controls before any external claim.
7.3 Only then consider a tag. **Not now.**

---

## 18. Files that would be created or modified during implementation

**All new. No existing file is modified — except this draft, and only to record
pre-registration decisions made in Phase 0.**

### 18.1 New documentation

| Path | Purpose |
|---|---|
| `docs/research/EXPERIMENT_2_CONFORMAL_PROTOCOL_DRAFT.md` | this document |
| `docs/research/EXPERIMENT_2_CONFORMAL_RESULTS.md` | written only after results exist |

### 18.2 New code

| Path | Purpose |
|---|---|
| `scripts/experiment_2/__init__.py` | package marker |
| `scripts/experiment_2/conformal.py` | Phase 1 pure library (scores, quantiles, intervals, metrics) |
| `scripts/experiment_2/build_calibration_folds.py` | Phase 2 nested fold derivation |
| `scripts/experiment_2/run_conformal.py` | Phase 3–4 nested training, calibration, scoring |
| `scripts/experiment_2/analyze_conformal.py` | Phase 5 aggregation, subgroups, figures |
| `scripts/experiment_2/verify_conformal.py` | Phase 6 artifact verification |
| `scripts/experiment_2/test_conformal.py` | Phase 6 tests, including L1–L20 |

`scripts/experiment_2/` is a **new** package. `scripts/valuation_v2_1/` is imported
read-only and **not** modified — including `protocol.py`, `run_model_comparison.py`,
`build_estimator`, `CATBOOST_CONFIG`, and `assert_safe_write_targets`. The write-target
guard is **re-implemented and extended** inside `scripts/experiment_2/`, not edited in
place, so the frozen module stays byte-identical.

### 18.3 New artifacts

| Path | Purpose |
|---|---|
| `artifacts/valuation/v2_2/conformal/calibration_folds_v2_2.json` | derived nested splits |
| `artifacts/valuation/v2_2/conformal/experiment_manifest_v2_2.json` | provenance + discipline flags |
| `artifacts/valuation/v2_2/conformal/analysis_manifest_v2_2.json` | determinism + hashes |
| `artifacts/valuation/v2_2/conformal/environment_v2_2.json` | environment record |
| `artifacts/valuation/v2_2/conformal/per_fold_coverage.csv` / `.json` | per-cell coverage |
| `artifacts/valuation/v2_2/conformal/per_fold_sharpness.csv` / `.json` | per-cell sharpness |
| `artifacts/valuation/v2_2/conformal/summary_coverage.csv` | mean + sample SD across folds |
| `artifacts/valuation/v2_2/conformal/summary_sharpness.csv` | mean + sample SD across folds |
| `artifacts/valuation/v2_2/conformal/regime_shift_coverage.csv` | paired regime deltas |
| `artifacts/valuation/v2_2/conformal/subgroup_coverage.csv` | city + price-band cells |
| `artifacts/valuation/v2_2/conformal/residual_diagnostics.csv` | RQ8 diagnostics |
| `artifacts/valuation/v2_2/conformal/artifact_verification.json` | verification output |
| `artifacts/valuation/v2_2/conformal/figures/*.png` / `*.pdf` | 12 figures, 24 files |

`v2_2` is a **new** artifact namespace. Nothing under `artifacts/valuation/v2_1/` or
`artifacts/valuation/valuation_v2*` is touched.

### 18.4 New models

| Path | Purpose |
|---|---|
| `models/valuation/v2_2/conformal/CatBoost/{regime}_fold{k}.joblib` | 10 nested pipelines |

`models/valuation/v2_1/**` and `models/valuation/valuation_v2*` are untouched.

### 18.5 Explicitly NOT touched

`.chain/state.json` · `data/processed/MREID_property.csv` ·
`data/processed/millow_token_map.csv` · `artifacts/valuation/v2_1/cv_folds_v2_1.json` ·
`artifacts/valuation/v2_1/model_comparison/**` ·
`models/valuation/v2_1/model_comparison/**` ·
`models/valuation/valuation_v2*` · `artifacts/valuation/valuation_v2*` ·
`scripts/valuation_v2_1/**` · `backend/**` · existing tags ·
`requirements.txt` / `ai/requirements.txt` (no new dependency — see 18.6)

### 18.6 Dependency decision: no new library

**No conformal prediction library will be added.** `mapie`, `nonconformist`, and
`conformal` are **not installed**, and none is declared in `requirements.txt` or
`ai/requirements.txt`. SciPy is installed but is not declared in either file.

The plan is to implement split/normalized/Mondrian conformal directly in NumPy, which:

- keeps the finite-sample quantile explicit and auditable rather than delegated,
- matches an established repository convention — `protocol.compute_metrics` implements
  `MAPE`/`MedAPE` **by hand** rather than importing them, with the stated reason that
  keeping both on one denominator convention makes the protocol runnable across
  versions,
- adds no dependency to an already minimal declared set,
- makes the exact `ceil((n+1)(1−α))` rule (Section 8) visible in this repository's own
  code rather than in a third party's.

If a reviewer prefers a vetted library, that is a legitimate decision, but it is a
dependency change and belongs in a separately approved step.

---

## 19. Statement on implementation status

**Phase 1 and Phase 2 are implemented and validated. Phase 3 has NOT been executed.**

Phase 1 (pure conformal library) and Phase 2 (nested fold derivation) are complete.
Validation evidence is recorded in Section 17. The boundary below is what has *not*
been crossed.

### 19.1 What has been done

- **Phase 1 implemented and unit-tested.** `scripts/experiment_2/conformal.py` with
  `scripts/experiment_2/test_conformal.py`; 93 tests pass. Pure functions, no
  filesystem, no global state.
- **Phase 2 implemented, unit-tested, and executed.** `scripts/experiment_2/build_calibration_folds.py`
  with `scripts/experiment_2/test_calibration_folds.py`; 52 tests pass.
- **One new artifact created** — and only one:
  `artifacts/valuation/v2_2/conformal/calibration_folds_v2_2.json`. It contains index
  sets and SHA-256 digests only. It contains no prediction, no residual, no interval,
  and no coverage measurement.
- **The frozen fold contract was read, never written.** Its SHA-256 is
  `989b79656b051b3b6c8c583b4170a5aaa1c31676260772247bbc98d1eb095156` before
  derivation, after derivation, and after the artifact write. The written artifact
  was re-read from disk and re-verified against the frozen contract independently of
  the code that produced it.

### 19.2 What has NOT been done

- **No model was trained.** No CatBoost, nested or otherwise, was fitted. The 10 nested
  models approved in §0.6 do **not** exist.
- **No model was retrained**, including the frozen Experiment 1 backbone. All 10
  persisted pipelines and all 27 Experiment 1 artifacts remain byte-identical to
  commit `101a771e…`.
- **No fold was regenerated.** `cv_folds_v2_1.json` was opened read-only.
- **No result artifact was created.** No CSV, PNG, PDF, joblib, manifest, or
  conformance report exists. `calibration_folds_v2_2.json` is a split definition, not
  a result.
- **No blockchain file was touched.** `.chain/state.json` is byte-identical;
  `backend/**` was not modified and no blockchain code was invoked.
- **No commit, no tag, no push** was performed during Phase 1 or Phase 2.
- **No Experiment 2 result was generated.** The quantitative numbers appearing in this
  document are **repository audit measurements, design-motivation diagnostics, and
  fold-structure measurements**, each explicitly labelled at its point of use:
  - fold sizes, digests, group statistics, group-overlap counts (Section 3, 4, 5);
  - the 10-pipeline reproduction check (Section 3.4) and the in-sample bias check
    (Section 5.2d) — both read-only diagnostics on frozen artifacts;
  - the heteroscedasticity quintile table (Section 6.1) and the INR asymmetry /
    relative-width identities (Section 10.1–10.3) — design motivation, computed from
    frozen fold-1 models on their own held-out folds;
  - the nested split sizes, realised calibration fractions, group counts and quantile
    estimability in Section 17 Phase 2 — **fold-structure measurements only**. These
    describe index sets, not predictions.
  **None of these is a conformal coverage result.** No interval, no coverage, no width,
  and no interval score has been computed for any nominal level. No model has produced
  a single prediction on behalf of Experiment 2.
- **No conformal validity is claimed.** Nothing in this document asserts that conformal
  prediction will achieve nominal coverage. The claim under test is stated in F2, and
  its rejection is an acceptable outcome (Section 1).

Work stopped here, as instructed, at the Phase 2 / Phase 3 boundary. Phase 3 (the 10
nested fits approved in §0.6) is now **unblocked** but has not been started. It
requires a separate explicit go-ahead.

---

## Appendix A — Audit findings summary

| # | Question | Finding |
|---|---|---|
| A1 | V2.1 protocol | 48 features (11 numeric + 37 one-hot), `log1p(price)` target, `random_state=42`, 5 folds. Target verified absent from `INPUT_FEATURES` (`46 + 2 == 48`). |
| A2 | Fold contract | 10 folds, all disjoint, sorted, `⋃` = all 28,398 rows. `random`: 96.8–97.7% of test rows from seen localities. `location_grouped`: **0** shared groups, **100%** unseen localities, `location_frequency == 0.0` on every test row. |
| A3 | Group structure | 1,789 groups (`source_city__location`), min/median/max size **1/3/685**. 6 cities, 1,803–6,820 rows each. |
| A4 | CatBoost config | Exactly as frozen: `iterations=700, lr=0.04, depth=7, l2_leaf_reg=2.0, random_seed=42, loss_function=RMSE`, dense output required. Verified on reload. |
| A5 | Persisted models | **10** pipelines. All 10 reload and reproduce recorded Experiment 1 metrics to < 1e-12. Held-out predictions need **no retraining**. |
| A6 | Reusable residuals | **None.** No row-level prediction or residual artifact exists anywhere under `artifacts/`. Experiment 1 stored per-fold *metrics* only, not per-observation predictions. |
| A7 | Calibration feasibility | **Infeasible from the frozen contract alone.** (a) model coverage leaves only test rows; (b) cross-conformal reuse is 100% in-sample; (c) jackknife+ needs 5 clean predictions, only 1 exists; LOO needs 28,398 fits; (d) in-sample residuals are 5–6% optimistically small. |
| A8 | **Verdict** | **Nested calibration structure is scientifically necessary.** It does not alter the frozen fold contract; it derives from `train_k` into a **new** file. Requires approval to train 10 nested models. |
| A9 | Heteroscedasticity | `\|residual\|` grows with predicted level: quintile-5/quintile-1 ratio **1.412** (random), **1.640** (grouped). Motivates Method B. |
| A10 | Expected sharpness | A log-symmetric 90% band implies **~252%** relative width (identity-derived, verified). Conformal intervals for this backbone are expected to be very wide. Calibration and usability must be judged as separate axes. |
| A11 | Conformal libraries | `mapie` / `nonconformist` / `conformal` **not installed**, not declared. NumPy implementation proposed, consistent with the repo's hand-written-metrics convention. |
| A12 | Manifest conventions | Boolean discipline flags, `deterministic` + input/output hash manifest, `ddof=1` sample SD, PNG+PDF byte-deterministic plots — all reusable as-is. |
| A13 | Subgroup power | Smallest city×true-price-band cell = **1** row. Pre-declared `n < 100` suppression rule required. True-price bands are diagnostic only (label-dependent); predicted bands are the assignable variant. |
| A14 | Runtime | ~2–4 minutes for 10 nested fits + prediction passes. Not a constraint. |

## Appendix B — Pre-registration checklist

To be completed in Phase 0, **before** any coverage number exists.

- [x] `cal_frac` fixed — **`0.20`** (§0.1, §5.5)
- [x] F3 usability threshold fixed — **`median_relative_width <= 2.00` at nominal 0.90** (§0.4, §16 F3)
- [x] B2 in scope? — **NO. Excluded.** (§0.2) `m` therefore does not arise
- [x] B1 scale definition confirmed as `σ̂(x) = ŷ(x)` (§0.2)
- [x] True-price band edges confirmed fixed (§12.2(a))
- [x] Predicted-band tertile cutpoints computed on `T'_k` only (§12.2(b))
- [x] `n < 100` subgroup suppression rule confirmed (§12.1)
- [x] C3 group-level status confirmed as **secondary** (§0.2, §6.3 C3)
- [x] C4 confirmed **excluded**, with reason (§0.2, §6.3 C4)
- [x] Clopper–Pearson (not Wald) confirmed for RQ10 (§9)
- [x] 3 nominal levels confirmed: **0.80 / 0.90 / 0.95** (§0.3)
- [x] **Approval to train 10 nested models granted** (§0.6) — execution gated on Phase 1+2 validation
- [x] Confirmed: no blockchain integration in Experiment 2 (§14, §18.5)
- [x] Confirmed: exploratory/descriptive framing, no multiplicity correction (§15.7)
- [x] F5 corrected to descriptive monotonicity, Spearman sign not a validity requirement (§0.5, §16 F5)
