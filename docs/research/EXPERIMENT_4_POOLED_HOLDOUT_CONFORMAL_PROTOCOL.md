# Experiment 4 — Pooled-Holdout Conformal Production Calibration Protocol (DRAFT)

**Status: DRAFT — design only. No training executed. No results exist. No model has been
fit, no interval has been produced, no artifact has been written.**

No Experiment 4 result has been generated. No model was trained. No Experiment 1, 2 or 3
file was modified. No application file was modified. No blockchain file was touched. No
commit, tag, or push was made. **Conformal validity is not claimed.**

| Field | Value |
|---|---|
| Document | `docs/research/EXPERIMENT_4_POOLED_HOLDOUT_CONFORMAL_PROTOCOL_DRAFT.md` |
| Revision | **2 — E4-L1 disambiguated; method renamed; fold provenance and leakage disclosures added (§0.13, §0.14). No registered method, coverage level, gate or research question was changed.** |
| Previous draft filename | `EXPERIMENT_4_CROSS_CONFORMAL_PROTOCOL_DRAFT.md` (revision 1). Renamed because "cross-conformal" is **not** the correct name for this construction — see §0.2.1. |
| Status | **DRAFT — awaiting review and freeze** |
| Experiment | 4 |
| Frozen predecessor tag | `v1.6.0-experiment3-results` |
| Frozen predecessor commit | `78830a80d58c9f98066c0ffe0609b6cd4c3efffa` |
| Protocol baseline | V2.1 (frozen, read-only) for data/folds; V5 production configuration for modelling |
| SHA-256 of this revision-2 file | Computed at freeze time and recorded in the Phase 0 freeze record and the review report. **Deliberately not embedded here:** writing the hash into the file changes the file, so an embedded value can never describe the bytes that contain it. |
| Registered method name | **`POOLED-HOLDOUT-CONFORMAL`** (§0.2). **Not** cross-conformal, **not** CV+, **not** jackknife+. |
| Guarantee-bearing reference | `HOLDOUT-REFERENCE` — per-fold split conformal on `calibrate_k` only (§0.2.4). Diagnostics only; never the deployed quantity. |
| Target model | **MILLOW V5 Log-Price XGBoost** (`models/valuation/final/millow_valuation_model.json`) — loaded **read-only**, never refit, never used to score |
| Relationship to Experiment 2 | **Independent follow-on. NOT a correction, NOT a re-run, NOT a replacement.** |
| Application integration | **Out of scope. Not performed by this experiment.** |
| Blockchain / escrow integration | **None. Explicitly out of scope.** |
| Training executed | **No. Protocol authoring only.** |
| Conformal validity claimed | **No. Nothing is claimed before it is tested.** |

---

## 0.0 What revision 2 changed, and why

Revision 1 was reviewed and **must not be executed as written**. Three defects were
found. Each is corrected here. Nothing else was touched.

| # | Defect in revision 1 | Correction | Section |
|---|---|---|---|
| **D1** | E4-L1 stated the rule in prose ("no calibration score … may be produced by the production V5 model") but left the **point estimate feeding every reported metric ambiguous**: §6.1 computed `yhat_test_k` *and* applied the interval to `yhat_V5`, without saying which one produces coverage, width, subgroup and interval-score numbers. Read either way, the experiment could report coverage built on in-sample production residuals — the exact failure E4-L1 exists to prevent. | The point estimate is bound to each metric **explicitly and individually**. Every calibration, coverage, width, interval-score and subgroup number comes from fold-model predictions only. Production predictions are **not permitted for any diagnostic**. Enforced by an interface that has no parameter through which V5 could enter. | §0.14, §3.2, §6.1, E4-L1, E4-L26 |
| **D2** | The procedure was labelled **"cross-conformal"**, which is a specific, named construction in the literature and is *not* what is registered here. | The exact construction is named (`POOLED-HOLDOUT-CONFORMAL`), the three nearby named constructions it is **not** are identified and contrasted, and the term "cross-conformal" is retired as a description of this experiment's procedure everywhere in the document (it survives only in the naming contrast of 0.2.1 / 0.2.5 and in the prohibition at 0.12). | §0.2.1 |
| **D3** | The protocol asserted a **finite-sample coverage guarantee for `M_k`** (§3.2) while simultaneously registering a **single pooled quantile shared by all five folds**. Because `calibrate_k ⊂ train_k = frame \ test_k` and the five `test_k` partition the frame, a pooled quantile necessarily draws calibration scores from models that were fitted on the very test rows it is then used to evaluate (§5.4). The guarantee as written was therefore false for the deployed quantity. | The guarantee is re-stated where it is true (**per-fold** holdout conformal) and **withdrawn** for the pooled quantity. The pooling decision itself is **unchanged**; a `HOLDOUT-REFERENCE` variant is registered as a diagnostic so that the withdrawn guarantee stays measurable. | §0.2.4, §3.2, §5.4, §15 E4-L27 |

Two further **factual errors** found during the read-only re-verification were corrected:

| Fact | Revision 1 said | Verified value |
|---|---|---|
| `location_grouped` pooled calibration count | `21,579` (§0.2, §7.2, §17.2) | **22,579** (= 4,245+3,668+5,369+5,143+4,154) |
| `random` frozen `train_k` size | `22,718` "for all five folds" (§5.2) | **22,718** for folds 1–3, **22,719** for folds 4–5 |

And one **pre-existing production defect** was discovered that would have silently
invalidated the fold models; it is now a fixed, testable part of the protocol (§0.13).

---

## 0. Phase 0 — Pre-registration decisions (BINDING)

Recorded here because they are now fixed. They were decided **before** any coverage number
exists and may not be revisited in response to observed results.

### 0.1 Relationship to Experiment 2 — this is NOT a correction

Experiment 2 is frozen, complete and correct **as research**. It answered its own question
about the CatBoost backbone and reported a negative usability outcome (F3 gate, Section 17).
Experiment 4 does **not** revisit, repair, re-interpret, re-run or supersede any part of it.

Experiment 4 asks a **different question about a different model**:

| | Experiment 2 | Experiment 4 |
|---|---|---|
| Question | Is the CatBoost research backbone calibratable? | Is the **production V5 XGBoost** serving path calibratable? |
| Model | CatBoost, V2.1 config | XGBoost, **V5 production config** |
| Feature contract | 48-feature V2.1 protocol | **V5 production pipeline (7 numeric + `location`)** |
| Purpose | characterise research uncertainty | characterise **deployment** uncertainty |
| Status | frozen (`v1.4.1-experiment2-results`) | draft |

Experiment 2's results, artifacts, protocol and tags are **read-only inputs** and are listed
in Section 4.6 as protected state. Nothing in this document modifies them.

### 0.2 Registered method: `POOLED-HOLDOUT-CONFORMAL` (fixed)

**One sentence:** five nested holdout conformal models are fit; their five
**disjointly-held-out residual sets are concatenated into one calibration multiset**; a
single exact finite-sample quantile is read off that multiset; and that one quantile is
applied to every fold's own model's test predictions. Nothing else.

```
for regime in {location_grouped (primary), random (secondary)}:
    for k in 1..5:
        M_k       = XGBRegressor(V5 config)    fitted on  fit_k
        P_k       = ColumnTransformer(V5)     fitted on  fit_k
        s_k       = score( y[calibrate_k], M_k(P_k(X[calibrate_k])) )     # out-of-sample for M_k

    S_regime = concat( s_1, ..., s_5 )          # ONE calibration multiset per regime
    qhat[(regime, method, level, stratum)] = exact_order_statistic( S_regime, alpha )

    for each row i in test_k:
        [lower, upper] = M_k(x_i)  +/-  qhat        # M_k never saw test_k
```

### 0.2.1 Naming: "cross-conformal" is WRONG and is retired

Revision 1 called this "cross-conformal". It is not, and the distinction is not cosmetic —
it is the difference between a construction with a published coverage bound and one
without.

| Named construction | Test point scored by | Calibration pool for that test point | Bound | Is this Experiment 4? |
|---|---|---|---|---|
| **Split conformal** (Vovk et al. 2005) | the single model fit on the fit-part | `calibrate_k` only | `>= 1 − alpha` | **This is the `HOLDOUT-REFERENCE` variant** (§0.2.4), per fold |
| **K-fold cross-conformal** (Vovk 2015; Vovk et al. 2018; Barber et al. 2021 eq. 12) | **all K** leave-one-fold-out models `mu_{-S_k}` | the **full** pool of all n cross-validation residuals, plus a `1/(n+1)` rank term | `>= 1 − 2alpha` | **NO** |
| **CV+ / jackknife+** (Barber et al. 2021 eq. 11) | one model **per residual** `mu_{-S_{k(i)}}(x)` — the interval is built from `mu(x_i) ± R_i` pairs | leave-one-fold-out score pairs | `>= 1 − 2alpha` | **NO** |
| **`POOLED-HOLDOUT-CONFORMAL`** (this experiment) | **one** model `M_k`, fit on `fit_k` (which excludes `test_k` entirely) | the **concatenated** residual multiset of all five *differently-fitted* models, no fold excluded, no rank correction | **none claimed** | **yes** |

Three structural facts separate this construction from cross-conformal, and all three are
verifiable from §0.14:

1. **The models are not leave-one-fold-out over a common training set.** `M_k` is fit on
   `fit_k = (frame \ test_k) \ calibrate_k`. Cross-conformal's `mu_{-S_k}` is fit on *all*
   data outside fold `k` and no fold's rows are additionally held out for calibration.
2. **The test point is scored by exactly one model, not by all five.** Cross-conformal
   requires `mu_{-S_k}(x)` for every `k`; this experiment never forms a second prediction
   for a test row from another fold's model.
3. **No leave-one-fold-out calibration pool and no rank correction.** The pool is the
   *same* five-fold concatenation for every test row, and the plain `k = ceil((n+1)(1−a))`
   order statistic of §7.2 is used. Cross-conformal's `1/(n+1)` term and its
   randomisation device are absent, and correctly so — those terms belong to a different
   construction.

**The word "cross-conformal" therefore appears nowhere in Experiment 4 as a description of
this procedure.** It survives only in §0.2.1/§0.2.5 (to name what this is *not* and what a
genuine cross-conformal construction would require), in §0.12 and in §23 (to record the
retirement), and in §5.4/§19.13 (to name the cross-*fold* contamination, which is a
different phenomenon and is not a cross-conformal method).

### 0.2.2 Aggregation rule (unchanged from revision 1 — pooling is retained)

**Aggregation rule (fixed):** scores are pooled across folds (concatenate the five
`calibrate_k` score arrays within a regime, then take the exact order statistic). This is
the deployed quantity. **Per-fold quantiles are recorded as diagnostics and are never the
deployed quantity.** No other aggregation (mean of quantiles, median of quantiles,
fold-weighted quantiles) may be computed or reported.

**Rejected aggregations, declared now so they cannot appear later:** mean-of-fold-quantiles;
median-of-fold-quantiles; per-fold intervals reported as if they were one interval; taking
the widest or narrowest fold quantile.

**Pooling is retained despite D3.** It was chosen for a stated reason — a single quantile
from ≈22.6k scores is estimated far more stably than five quantiles from ≈4.5k each, and it
is the quantity whose deployment form `yhat ± qhat` is actually wanted. The price of
pooling is stated in §3.2 and §5.4 and is **paid openly**, not removed by quietly switching
aggregation after results exist. Switching to per-fold aggregation, or to a genuine
cross-conformal construction, after any coverage number is observed would violate E4-L20.

### 0.2.3 What pooling costs, stated as a registered assumption

Pooling residuals from five **differently fitted** models into one quantile requires the
five residual distributions to be exchangeable with each other and with the test fold's
residuals. They are not exchangeable in general: `n_fit` ranges 16,266–19,295, and in the
`location_grouped` regime the five `calibrate_k` sets have entirely different locality
compositions (287, 287, 287, 287, 287 groups over 3,668–5,369 rows drawn from disjoint
regions of the locality space).

This is registered as the **cross-fold residual homogeneity assumption**, it is an
assumption and not a proven property, and it is falsifiable from artifacts already
registered: `per_fold_coverage.csv` and the per-fold quantiles show the pooled versus
per-fold coverage gap directly. If `HOLDOUT-REFERENCE` and `POOLED-HOLDOUT-CONFORMAL`
coverage differ by more than 0.05 at nominal 0.90 in the `random` regime, the assumption
is rejected and that fact is reported as a finding, not explained away.

### 0.2.4 `HOLDOUT-REFERENCE` (guarantee-bearing, diagnostic only)

For each `(regime, k)`:

```
qhat_ref[(regime, method, level, stratum, k)] = exact_order_statistic( s_k, alpha )
[lower_ref, upper_ref]_i = M_k(x_i) +/- qhat_ref        for i in test_k
```

- **This is ordinary split conformal** on `calibrate_k`, and it is the **only** cell in
  Experiment 4 for which a finite-sample marginal coverage statement is available (§3.2).
- It is **never** the deployed quantity and **never** the headline. `POOLED-HOLDOUT-CONFORMAL`
  remains the primary reported construction; `HOLDOUT-REFERENCE` is reported beside it.
- It exists so that the guarantee withdrawn from the pooled construction (§3.2) remains
  directly measurable, and so that §0.2.3 can be checked.
- Adding it **does not add a method, a coverage level, a gate or a research question.** It
  is the same six methods, the same three levels, the same `qhat` rule, applied to the same
  score arrays, restricted to one fold at a time.

### 0.2.5 What a genuine cross-conformal construction would require (declared, NOT done)

Recorded now so that it cannot be claimed later. Implementing Vovk's K-fold cross-conformal
on this dataset would require: (a) re-fitting five models each on `frame \ S_k` for a fold
partition `S_1..S_5` of the *training* data; (b) scoring every test point under **all five**
of those models; (c) a `1/(n+1)` rank correction with the randomisation device of Vovk
eq. 12; and (d) abandoning the frozen nested `fit`/`calibrate` split entirely, which would
destroy the bit-identical test-row provenance with Experiment 2 (§4.3). **Experiment 4 does
none of this and does not claim cross-conformal validity.**

### 0.3 Method set (closed — reused unchanged from Experiment 2)

| ID | Method | Status |
|---|---|---|
| **A** | Split conformal, absolute residual | **INCLUDED** — primary |
| **B1** | Normalized conformal, `sigma_hat(x) = y_hat(x)` | **INCLUDED** |
| **B3** | Normalized conformal, `sigma_hat(x) = 1` | **INCLUDED** — implementation control |
| **C1** | Mondrian conformal by `source_city` | **INCLUDED** |
| **C2** | Mondrian conformal by predicted-price tertile | **INCLUDED** |
| **C3** | Grouped conformal by `source_city__location` | **INCLUDED — SECONDARY ONLY** |
| B2 | kNN-normalized conformal | **EXCLUDED** (as in Experiment 2) |
| C4 | Weighted conformal under covariate shift | **EXCLUDED** (as in Experiment 2) |

The set is **identical to Experiment 2 §0.2 — nothing added, nothing removed** — so that
Experiment 2 and Experiment 4 are directly comparable on the same six methods.

**No additional method may be introduced after results are observed.** In particular no
method may be added because a method performs poorly (Section 21, F7).

### 0.4 Nominal coverage levels (fixed)

`0.80`, `0.90`, `0.95`, i.e. `alpha ∈ {0.20, 0.10, 0.05}`. Identical to Experiment 2.

### 0.5 Nonconformity score and target space (fixed)

**Target space:** `log1p(price)` — identical to Experiment 2 and to the V5 model's own
training target (`metadata.json: target = "log1p(price)"`, `inverse_transform =
"expm1(prediction)"`).

**Scores** are computed in `log1p(price)` space by `scripts/experiment_2/conformal.py`,
imported **read-only** (§16.1):

```
A, B3, C1, C2, C3 :  s_i = | y_i - yhat_i |
B1                 :  s_i = | y_i - yhat_i | / yhat_i      (calibration-set yhat)
```

with `y = log1p(price)` and `yhat` the fold model's log-space prediction.

INR quantities are derived **at presentation time only**, via `expm1` with a floor at 0.

### 0.6 Usefulness criteria (fixed, U1–U3)

**Three** independent criteria, all fixed now. **Calibration and usefulness are reported
separately and neither substitutes for the other** (carried forward from Experiment 2
§0.4). A method may pass U1 and fail U2, or the reverse; both facts are reported.

> **Revision 3 removed U4, D-T and F8 entirely.** U4 was a transfer-direction criterion
> whose only measurement was a comparison of residuals between the production V5 model and
> the fold models. D-T was the transfer diagnostic that reported `scale_v5 / scale_fold`.
> F8 was the cross-experiment divergence gate comparing Experiment 4's
> `median_relative_width` to Experiment 2's. Because the production V5 model was fitted on
> all 29,135 rows, every such comparison necessarily uses production residuals on rows the
> production model was trained on — in-sample residuals. These have therefore been removed,
> not reinterpreted, and **no replacement production-residual diagnostic is registered**. U1,
> U2 and U3 are unchanged, including their thresholds, their sign conventions and their
> aggregation rules. The registered method set is unchanged: {A, B1, B3, C1, C2, C3}.

**U1 — sharpness / usability gate.** At nominal `0.90`:

```
median_relative_width = median( (upper - lower) / point_prediction )   in INR
U1 PASSES  <=>  median_relative_width <= 2.00
```

The threshold `2.00` is **reused unchanged from Experiment 2 §0.4** so that the two
experiments are directly comparable. It remains an **Experiment-specific operational
threshold**, not a universal real-estate industry standard, and must never be described as
one.

**U2 — calibration gate (random regime, positive control).** At nominal `0.90` in the
`random` regime, pooled over folds:

```
U2 PASSES  <=>  | empirical_coverage - 0.90 | <= 0.02
```

**U3 — location-shift degradation gate.** At nominal `0.90`:

```
delta = coverage(location_grouped) - coverage(random)
U3 PASSES  <=>  delta >= -0.05      (degradation under shift at most 5 percentage points)
```

Sign convention `Δ = grouped − random` is declared here, as in Experiment 2 §11.

**Pre-registered expectation for U1, stated before any result exists:** Experiment 2
observed `median_relative_width` of 2.30–2.76 for the CatBoost backbone against this same
2.00 threshold, i.e. **every method failed**. Experiment 4 records in advance that **U1 is
expected to fail again for the homogeneous methods (A, B3, C1, C2)**, because for those
methods `relative_width = 2*sinh(qhat)` depends only on `qhat`, and `qhat_0.90 ~ 1.0` on
this dataset. This expectation is written now so that a failure cannot later be presented
as a surprise or rationalised. **It is not a reason to alter the threshold.**

### 0.7 Transfer to the served V5 model — unmeasured, and not measured here (revision 3)

The fold models are fitted on `fit_k` ≈ 16,266–19,295 rows (roughly 57–68% of the 28,398
row population). The production V5 model is fitted on **all 29,135 raw rows**.

**What follows from that alone, and is the only thing claimed:** a model fitted on strictly
more data has, in expectation, no larger residual scale than the same configuration fitted
on less data. This is a **directional expectation about model fitting, not a measurement of
the served artifact, and it confers no coverage claim of any kind.**

**Revision 3 removed the transfer measurement.** U4, F8 and diagnostic D-T are therefore
**removed from the protocol**, with no substitute registered. The expectation above is
retained as context for why a fold-model calibration *might* be conservative for the served
model; it is **not** evidence, it is not tested, and no report may cite it in support of
deploying an interval.

**Consequence, registered as a hard constraint rather than a limitation to note:** the
served V5 artifact is **out of scope for measurement** in this experiment. Experiment 4
produces numbers about `M_k` only. Any statement about the served model's coverage is
outside what this experiment can establish, in either direction, and is not made.

### 0.8 Data and fold sources (fixed, read-only reuse)

| Input | Path | SHA-256 |
|---|---|---|
| Dataset (raw) | `data/processed/MREID_property.csv` | `1be10465221668ffde0fe7392efbc183ef77a654549a2724b26439978fad8877` |
| Frozen outer folds | `artifacts/valuation/v2_1/cv_folds_v2_1.json` | `989b79656b051b3b6c8c583b4170a5aaa1c31676260772247bbc98d1eb095156` |
| Frozen nested calibration folds | `artifacts/valuation/v2_2/conformal/calibration_folds_v2_2.json` | `47b89d8b41e98301129860cbf253f8c30255e272ec79f8277da54bb73acc6ea9` |

All three are opened **read-only**. Experiment 4 **regenerates no fold file** and calls no
`generate_folds` / `generate_all_folds` path.

### 0.9 Model configuration (fixed — V5 production configuration)

The fold models use the **production V5 XGBoost configuration verbatim**, so that the only
*modelling* difference between a fold model and the served model is training-set size.

```
objective        = reg:squarederror
n_estimators     = 900
max_depth        = 7
learning_rate    = 0.03
subsample        = 0.85
colsample_bytree = 0.85
min_child_weight = 5
reg_alpha        = 0.1
reg_lambda       = 2
tree_method      = hist
random_state     = 42
n_jobs           = -1
```

Provenance of this block, asserted in Phase 2 (E4-L25): `objective` and `n_jobs` are read
from `scripts/train_final_valuation.py:473-486`; the remaining ten are asserted equal to
`models/valuation/final/metadata.json -> model_parameters` **key for key and value for
value**. A missing key, an extra key or any value mismatch is a hard stop (§18.1).

**The word "only" is now bounded.** It is accurate for the hyperparameter block and for the
feature *column* contract. It is **not** accurate in two respects that §0.13 makes explicit
and that are now part of the registered contract:

- the fold models use a **training-set-restricted** preprocessor, whereas production V5's
  encoder was fitted on all 29,135 rows, so the one-hot category space differs by fold;
- production V5's *serving* transform is **not** identical to its *training* transform
  (§0.13.1), a pre-existing production defect. Experiment 4 reproduces the **training**
  transform, and that choice is registered rather than assumed.

**No hyperparameter tuning. No early stopping. No second seed. No feature change. No model
selection. No architecture search.** Any deviation voids the experiment.

> Note: this is deliberately **not** the Experiment 1 XGBoost *anchor* configuration
> (`n_estimators = 700`, `learning_rate = 0.04`), which was a research anchor only. The
> served artifact is V5, so V5's configuration is the one that must be characterised.

### 0.10 Mandated implementation order

**Do not run the full experiment immediately.**

1. **Phase 1 only** — pure module, no filesystem writes, no training. Reuse
   `scripts/experiment_2/conformal.py` unmodified; add only the V5 feature/fold driver.
   Run the pure unit tests.
2. **Phase 2** — load and verify the frozen datasets, folds and nested calibration folds;
   assert every partition and hash invariant; write nothing outside the new Experiment 4
   artifact directory. **Still no training.**
3. **STOP and report** Phase 1 + Phase 2 validation for review.
4. Only after **explicit written authorisation to execute training** may Phase 3 begin.

**Training is not authorised by the existence of this protocol.** Authorisation to execute
Phase 3 is a separate, explicit instruction.

### 0.11 Protected state

Absolutely not to be modified by any phase of Experiment 4:

`.chain/state.json` · `data/processed/MREID_property.csv` ·
`data/processed/millow_token_map.csv` · `chain-manifest.json` · `src/config.json` ·
`artifacts/valuation/v2_1/**` · `models/valuation/v2_1/**` ·
`artifacts/valuation/v2_2/**` · `models/valuation/v2_2/**` ·
`artifacts/valuation/v2_3/**` · `models/valuation/v2_3/**` ·
`models/valuation/final/**` · `models/valuation/v5/**` ·
`scripts/valuation_v2_1/**` · `scripts/experiment_2/**` · `scripts/experiment_3/**` ·
`docs/research/EXPERIMENT_1_*` · `docs/research/EXPERIMENT_2_*` ·
`docs/research/EXPERIMENT_3_*` · `contracts/**` · `backend/**` · `src/**` · `server/**` ·
existing tags.

Existing tags that must retain their recorded identities:

`v1.0.0-reproducible-state`, `v1.1.0-research-checkpoint`, `v1.2.0-experiment1-xgb-anchor`,
`v1.3.0-experiment1-catboost-backbone`, `v1.4.0-experiment2-conformal`,
`v1.4.1-experiment2-results`, `v1.5.0-experiment3-protocol`,
`v1.5.1-experiment3-protocol-amended`, `v1.5.2-experiment3-protocol-final`,
`v1.5.3-experiment3-implementation-spec`, `v1.6.0-experiment3-results`.

### 0.12 Research discipline (binding)

- No claim of conformal validity before results exist.
- **Location-grouped coverage must never be described as having a formal distribution-free
  guarantee.**
- **Coverage of the served production V5 model must never be described as formally
  guaranteed** (Section 3.2). It is a measurement, at best.
- No method added because an initial method performs poorly.
- `cal_frac`, method set, thresholds, band edges and aggregation rule must **not** be
  selected on observed results.
- No retraining of the production V5 model. No change to the production model.
- No modification of Experiments 1, 2 or 3.
- Experiment 4 is **not** described as a fix, correction, improvement or successor of
  Experiment 2 anywhere in any document.
- No application integration in this experiment.
- No blockchain, escrow, mint, deploy or contract work.
- No commit, no tag, no push without explicit instruction.
- The term "confidence interval" is **never** used for a conformal interval. The correct
  term is "N% conformal prediction interval".
- The term **"cross-conformal"** is never used to describe this experiment's procedure
  (§0.2.1). The registered name is `POOLED-HOLDOUT-CONFORMAL`.
- The registered name `POOLED-HOLDOUT-CONFORMAL` must never be shortened to
  "cross-conformal", "CV", "CV+" or "jackknife+" in any artifact, figure, table key, column
  header or report. Those are different constructions with different bounds (§0.2.1).

---

## 0.13 Feature-pipeline facts fixed before execution (new in revision 2)

All three facts below were established by **read-only** inspection of frozen artifacts
during review. Nothing was trained. Each is now a testable part of the contract, because
getting any of them wrong silently produces models that are not V5-compatible while still
looking plausible.

### 0.13.1 V5 has **two different feature transforms**: training-time and serving-time

`scripts/train_final_valuation.py::create_features` (used to fit the served model) and
`scripts/predict_property_value.py::create_features` (used to serve it) are **not the same
function**. The training version derives its amenity column set dynamically:

```python
exclude_cols = {"mreid_id","price","derived_price_per_sqft","source_file","source_city","location"}
amenity_cols = [c for c in df.columns if c not in exclude_cols
                and pd.api.types.is_numeric_dtype(df[c])]      # -> 38 columns
```

On this dataset that is the 35 amenity columns **plus `area`, `no_of_bedrooms` and
`resale`**, and the three amenity summary features are then computed over all 38. The
serving version uses a hard-coded list of the 35 amenity columns only. Verified
consequence on all 29,135 raw rows:

| Feature | Rows differing between training-time and serving-time transform |
|---|---|
| `amenity_yes_count` | **12,386 / 29,135 (42.51%)**, mean 2.8592 vs 2.3642 |
| `amenity_known_count` | **29,135 / 29,135 (100.00%)**, mean 10.9999 vs 9.8819 |
| `amenity_unknown_count` | 1 / 29,135 (0.00%) |

`area` never equals 1, but `no_of_bedrooms == 1` on 3,436 rows and `resale` is 0/1 on every
row, which is where the difference comes from.

**Registered decision.** Experiment 4 fold models reproduce the **training-time** transform,
because that is the transform the served model's weights were fitted on. Reproducing the
serving-time transform would produce models whose inputs differ from the served model's on
42–100% of rows, and their residuals would not be residuals of "a V5-configured model" in
any meaningful sense.

**Declared consequence.** The served artifact therefore does not consume the features it was
trained on. This is a **pre-existing production defect, discovered by read-only inspection
and out of scope to fix here**: `models/valuation/final/**` must not be modified, retrained
or re-preprocessed by Experiment 4. It is recorded as a limitation (§19.18) and as a
finding for a future, separately-authorised task. Experiment 4 makes no claim about
production serving correctness.

### 0.13.2 The V2.1 fold frame cannot be used as the feature matrix

`scripts/valuation_v2_1/protocol.py::build_dataset()` is the frozen source of the 28,398-row
frame the fold indices are keyed to, and it must still be used for **row identity, index
arithmetic, group labels and subgroup labels**. It may **not** be used to build the V5 model
matrix, because:

- it emits `amenity_yes_count` computed over the **35** amenity columns, which differs from
  the V5 value on **16,704 / 28,398 rows (58.82%)**;
- it emits `amenities_fully_specified` and **no** `amenity_known_count` and **no**
  `amenity_unknown_count`, so two of V5's seven numeric features are simply absent;
- it adds `group`, which V5 does not use and which is not a feature.

**Registered decision.** Two views of the same 28,398 rows are maintained, joined by
`mreid_id`:

| View | Built by | Used for |
|---|---|---|
| **fold view** | `protocol.build_dataset()` (frozen, read-only, imported) | index arithmetic, `source_city` / `location` (lowercased) group keys, C1/C3 strata, city subgroups, price bands |
| **V5 feature view** | V5 training-time `create_features` re-implemented per §6.3 and parity-tested against `train_final_valuation.py` | the 8 model input columns only |

`mreid_id` is unique in both the 29,135-row raw file and the 28,398-row deduplicated frame,
and the join is verified 1:1 with order preserved (28,398/28,398 positional parity, prices
bit-identical). Phase 2 asserts this rather than assuming it (E4-L25).

### 0.13.3 The fold frame lowercases `location`; V5 does not

`protocol._clean` applies `.str.strip().str.lower()` to `location` and `source_city`.
`train_final_valuation` applies `.str.strip()` **only**. Verified consequences:

| Quantity | Value |
|---|---|
| distinct `location` under strip-only (V5 / production) | **1,776** |
| distinct `location` under strip+lower (V2.1 fold frame) | **1,775** |
| lowercased values absent from the production encoder's category set | **1,722 (97.0%)** |
| strip-only values absent from the production encoder's category set | **0** |
| rows where the two views disagree on `location` | 28,056 / 28,398 |

**Registered decision and hard stop.** Lowercased `location` strings are used **only** for
group keys and subgroup labels, never as model input. Every `location` value passed to any
encoder must be byte-identical to a category of the production encoder. If a lowercased
value ever reaches a model encoder, `handle_unknown="ignore"` silently maps it to an
all-zero categorical block, production predictions become meaningless, and the run aborts
(E4-L25). Verified currently: **0 unknown** strip-only values.

---

## 0.14 Fold provenance and leakage proofs (new in revision 2)

Every number in this section was measured read-only against the frozen artifacts and is
re-asserted at runtime. **No model was fit to obtain any of them.**

### 0.14.1 Which frozen file supplies training, calibration and test data

```
artifacts/valuation/v2_1/cv_folds_v2_1.json          (SHA-256 989b7965…)
    └── per regime: 5 x (train_k, valid_k)      positional indices into the 28,398-row
                                                  deduplicated frame
        train_k = frame \ valid_k

artifacts/valuation/v2_2/conformal/calibration_folds_v2_2.json   (SHA-256 47b89d8b…)
    └── per regime: 5 x (fit, calibrate, test)
        test_k      = valid_k                     bit-identical, frozen
        fit_k ∪ calibrate_k = train_k             exactly, no row lost
        fit_k       = inner split of train_k, fold 0 vs folds 1-4
        calibrate_k = inner split of train_k
        cal_frac = 0.20, random_state = 42, n_splits = 5
        inner splitter: random          -> KFold(shuffle=True, random_state=42)
                        location_grouped -> GroupKFold(shuffle=True, random_state=42)
                                           on group = source_city__location

Experiment 4 consumes this file verbatim. It regenerates nothing, calls no
generate_folds / generate_all_folds path, and re-derives no inner split.
```

| Regime | fold | `fit_k` | `calibrate_k` | `test_k` | `train_k` | groups fit | groups cal | groups test |
|---|---|---|---|---|---|---|---|---|
| random | 1 | 18,174 | 4,544 | 5,680 | 22,718 | 1,542 | 898 | 953 |
| random | 2 | 18,174 | 4,544 | 5,680 | 22,718 | 1,518 | 880 | 950 |
| random | 3 | 18,174 | 4,544 | 5,680 | 22,718 | 1,519 | 871 | 979 |
| random | 4 | 18,175 | 4,544 | 5,679 | 22,719 | 1,528 | 882 | 975 |
| random | 5 | 18,175 | 4,544 | 5,679 | 22,719 | 1,500 | 906 | 970 |
| location_grouped | 1 | 19,295 | 4,245 | 4,858 | 23,540 | 1,144 | 287 | 358 |
| location_grouped | 2 | 19,025 | 3,668 | 5,705 | 22,693 | 1,144 | 287 | 358 |
| location_grouped | 3 | 18,015 | 5,369 | 5,014 | 23,384 | 1,144 | 287 | 358 |
| location_grouped | 4 | 16,266 | 5,143 | 6,989 | 21,409 | 1,144 | 287 | 358 |
| location_grouped | 5 | 18,412 | 4,154 | 5,832 | 22,566 | 1,145 | 287 | 357 |

Verified for all ten `(regime, fold)` cells: `fit ∩ calibrate = ∅`, `fit ∩ test = ∅`,
`calibrate ∩ test = ∅`, `fit ∪ calibrate = train_k`, `test_k = valid_k` element-for-element,
`fit`/`calibrate` sorted ascending. In `location_grouped`, additionally
`groups(fit) ∩ groups(calibrate) = ∅`, `groups(fit) ∩ groups(test) = ∅` and
`groups(calibrate) ∩ groups(test) = ∅` — all three are **exactly zero overlap**, which is
what makes calibration localities as unfamiliar to `M_k` as test localities are. In
`random`, locality recurs across all three partitions **by design**; it is the positive
control and makes no locality claim.

Also verified: the five `test_k` sets **partition the 28,398-row frame exactly** in both
regimes (sum = 28,398; union = all rows; pairwise disjoint), so every row is evaluated
exactly once per regime and no row is evaluated twice.

### 0.14.2 The six required proofs

| # | Required proof | Status | Evidence / mechanism |
|---|---|---|---|
| **P1** | No row used to fit a fold model is scored as calibration data **for that same model** | **PROVEN, exactly** | `fit_k ∩ calibrate_k = ∅` for all ten cells, measured 0. The encoder is fit on `fit_k` only (E4-L8), so no calibration row contributes a category. **Caveat P1' below.** |
| **P2** | No calibration row is used as test data | **PROVEN per fold; FALSE across folds** | `calibrate_k ∩ test_k = ∅` for all ten cells. Across folds, **54–62%** of `test_k` rows are calibration rows in some *other* fold. See P2' below. |
| **P3** | No test target is used to determine `qhat` | **PROVEN structurally** | `qhat` is a function of score arrays built only from `calibrate_k` indices. Test targets enter `coverage_metrics` only. The scoring entry point has no test-index parameter (E4-L1). |
| **P4** | Preprocessing is fitted only on the relevant training data | **PROVEN structurally** | `ColumnTransformer`/`OneHotEncoder` fit on `fit_k` only; `transform` on `calibrate_k` and `test_k`. `preprocessor_fit_rows` recorded per fold. The production preprocessor is **never** fit, only `transform`-ed (E4-L8, E4-L21). |
| **P5** | No target-derived feature enters calibration or test construction | **PROVEN by contract** | The 8 input columns are fixed and enumerated (§0.13.2, §6.3). `price` and `derived_price_per_sqft` are excluded by name; `resale`, `area`, `no_of_bedrooms`, `source_city`, `group`, `amenities_fully_specified` are never model inputs. Verified: V5's amenity set excludes `derived_price_per_sqft` and its values are not `1`, not `9`, not in `{0,1}`, so it could not contribute to an amenity count even if it leaked into the frame. |
| **P6** | Model / feature configuration matches production V5 **exactly** | **PROVEN for hyper-parameters and column contract; corrected for feature *values*** | §0.9 table asserted key-for-key against `metadata.json`; column contract asserted against `metadata.json.features`; feature **values** now required to match `train_final_valuation.create_features` by parity test (§0.13.1). Revision 1 failed P6 on values — it pointed at `predict_property_value.py`. |

**P1' — the qualification to P1, and why the §3.2 guarantee had to be withdrawn.** The
per-fold statement `fit_k ∩ calibrate_k = ∅` holds exactly, and every individual
calibration score is out-of-sample for the model that produced it. But the deployed `qhat`
is a **pooled** statistic, and the pool is not exchangeable with `test_k`:

| Regime | fold | `test_k` rows fit by some `M_j`, `j ≠ k` | `test_k` rows that are calibration rows of some other fold | pooled score entries produced by a model that saw `>= 1` row of `test_k` |
|---|---|---|---|---|
| random | 1 | 5,665 / 5,680 (99.7%) | 3,356 (59.1%) | 18,176 / 22,720 (80.0%) |
| random | 2 | 5,663 / 5,680 (99.7%) | 3,375 (59.4%) | 18,176 / 22,720 (80.0%) |
| random | 3 | 5,676 / 5,680 (99.9%) | 3,291 (57.9%) | 18,176 / 22,720 (80.0%) |
| random | 4 | 5,672 / 5,679 (99.9%) | 3,295 (58.0%) | 18,176 / 22,720 (80.0%) |
| random | 5 | 5,665 / 5,679 (99.8%) | 3,288 (57.9%) | 18,176 / 22,720 (80.0%) |
| location_grouped | 1 | 4,841 / 4,858 (99.7%) | 2,689 (55.4%) | 18,334 / 22,579 (81.2%) |
| location_grouped | 2 | 5,705 / 5,705 (100.0%) | 3,495 (61.3%) | 18,911 / 22,579 (83.8%) |
| location_grouped | 3 | 5,005 / 5,014 (99.8%) | 2,919 (58.2%) | 17,210 / 22,579 (76.2%) |
| location_grouped | 4 | 6,989 / 6,989 (100.0%) | 4,334 (62.0%) | 17,436 / 22,579 (77.2%) |
| location_grouped | 5 | 5,823 / 5,823 (99.8%) | 3,169 (54.3%) | 18,425 / 22,579 (81.6%) |

**Not one test row is untouched by the other four folds.** Zero rows fall outside all other
folds' `fit ∪ calibrate`. This is **structural, not a defect of the fold file**: because
`test_1..test_5` partition the frame and `calibrate_j ⊆ train_j = frame \ test_j`, every
test row necessarily lies in four other folds' `fit ∪ calibrate`. **No pooling scheme over
this frozen structure can avoid it.** The per-fold construction of §0.2.4 avoids it, which
is exactly why that variant is registered as the guarantee-bearing reference.

The magnitude of the resulting contamination is expected to be second-order — a handful of
test rows among ~18,000 fit rows is a negligible perturbation of any one fold model, and
the calibration residuals remain out-of-sample for their own model — but "expected to be
second-order" is not a proof, and no proof exists. It is therefore recorded as a limitation
(§19.13–19.15) and as an empirical question the experiment answers (F1/U2, §0.2.3).

**P2' — the qualification to P2.** Per fold, `calibrate_k ∩ test_k = ∅` exactly, which is
the invariant that governs every individual score and every per-fold interval. Across folds
the pooled pool unavoidably contains 54–62% of `test_k`'s rows, per the table above. Because
the pooled quantile is a single scalar, this does not let a test row be *scored* by a model
that saw it; but it does mean the pooled quantile is not an independent calibration set for
`test_k`. Reported honestly as E4-L27 and §19.14, not suppressed.

### 0.14.3 Pooled calibration multiset is not a partition — multiplicity disclosure

The pooled multiset has 22,720 entries (random) and 22,579 (grouped), but those entries come
from far fewer distinct rows, because `calibrate_j` and `calibrate_k` overlap for `j ≠ k`:

| Regime | pooled score entries | distinct rows | multiplicity 1 | 2 | 3 | 4 | Kish effective n (row-equalised) |
|---|---|---|---|---|---|---|---|
| random | 22,720 | **16,605** | 11,400 | 4,352 | 796 | 57 | **13,995** |
| location_grouped | 22,579 | **16,606** | 11,353 | 4,568 | 650 | 35 | **14,148** |

Consequences, registered: the exact order statistic of §7.2 is still computed on the
**22,720 / 22,579 entries**, because that is what pooling means; but `n` in the quantile
diagnostics must be reported **alongside** `n_distinct_rows` and `n_effective_rows`, and
the `1-alpha` resolution is not `1/n_entries`. `k > n` remains unreachable at all three
levels either way. Registered as E4-L28.

---

## 1. Research objective

Determine, empirically, whether a pooled-holdout conformal calibration built on the **production V5
XGBoost configuration and the production V5 feature pipeline** yields prediction intervals
that are **calibrated for V5-class models**, and characterise — honestly — how far that
calibration can be said to transfer to the already-trained served V5 artifact.

The experiment is a **measurement**, not a construction. Every one of the following is an
acceptable finding:

- coverage near nominal in the random regime (the procedure is calibrated),
- systematic undercoverage (the procedure is not calibrated),
- overcoverage with uselessly wide intervals (calibrated but not useful),
- calibration that holds under random CV but degrades under location shift,
- U1 failing for every method, as it did in Experiment 2.

None of these is a failure of the experiment. Only a **leakage-induced** result, a result
produced without a pre-registered analysis, or an execution performed without explicit
authorisation would be a failure of the protocol.

### 1.1 Explicit non-goals

Experiment 4 does **not**:

1. improve point-prediction accuracy — the production V5 model is untouched and is not
   refit, retuned, replaced or ensembled;
2. train or serve any model as a replacement for V5 — the fold models are **calibration
   instruments only** and are never exposed to the application;
3. integrate anything into the live application, API, schemas or frontend — that is a
   separate, later, separately-authorized task;
4. modify, correct, re-run, re-interpret or supersede Experiment 2;
5. modify blockchain, escrow, contracts, canonical chain state or token supply;
6. produce a "guaranteed value", "guaranteed price", "guaranteed range", investment
   recommendation or certified appraisal.

---

## 2. Research questions

Pre-registered. Each is answerable from the planned artifacts without further choices.

| ID | Question | Answered by |
|---|---|---|
| **RQ1** | In the random regime, does the pooled-holdout procedure attain nominal marginal coverage at 80/90/95% for the fold models? | `per_fold_coverage.csv`, `summary_coverage.csv` |
| **RQ2** | Does normalized conformal (B1) improve sharpness over split conformal (A) *without* degrading coverage? | paired per-fold width and coverage deltas |
| **RQ3** | **(primary)** How much does marginal coverage degrade under location-grouped evaluation, at matched nominal level? | `regime_shift_coverage.csv` |
| **RQ4** | Is coverage degradation under location shift directionally monotone across 80 → 90 → 95? | `coverage_vs_nominal` figures |
| **RQ5** | Is coverage non-uniform across the 6 cities within a regime and level? | `subgroup_metrics.csv` (city) |
| **RQ6** | Is coverage non-uniform across fixed INR price bands? | `subgroup_metrics.csv` (band) |
| **RQ7** | Are intervals sharp enough to be actionable, in absolute INR and relative terms? | width metrics + **U1** |
| **RQ8** | Do V5-configuration residuals show structure (heteroscedasticity, heavy tails) that undermines a symmetric absolute-residual score? | residual diagnostic figures |
| **RQ9** | Does any pre-declared location-aware strategy (C1, C2, C3) recover coverage relative to A under location shift? | paired regime-grouped comparison |
| **RQ10** | Are deviations from nominal statistically distinguishable from Monte-Carlo noise at the achieved calibration size? | Clopper–Pearson bounds vs nominal |

RQ1 is the **negative control**: it must pass, or the implementation is broken and no other
result is interpretable. RQ3 is the primary research question.

---

## 3. The served-model transfer problem — what Experiment 4 can and cannot establish

This section is the methodological core of the protocol. It is written **before** any
result so that it constrains interpretation rather than being retro-fitted to it.

### 3.1 The production model has no held-out data

Verified during the read-only audit and re-verified as a Phase 2 assertion:

`scripts/train_final_valuation.py` fits the served artifact as

```python
X_full_encoded = final_preprocessor.fit_transform(X_full)
final_model.fit(X_full_encoded, np.log1p(df["price"].values))
# prints: "Production training rows: 29135"
```

i.e. **the served V5 model is fitted on every row of the dataset**, with the explicit
comment *"After validation, the final production model is trained on ALL valid rows."*

Consequences, three binding:

1. **Every MREID row is in-sample for the served model.** No row exists on which the served
   model's residual is out-of-sample. Therefore **the out-of-sample coverage of the served
   V5 artifact cannot be measured on this dataset by any means.**
2. **The production V5 model is never a source of a nonconformity score, in any regime,
   for any method, at any level, for any stratum.** Scoring a model on data it was fitted on
   produces residuals biased small, hence a `qhat` too narrow and intervals that
   under-cover. This is the Experiment 4 analogue of Experiment 2's controls L2/L6, and it
   is the single most dangerous failure mode in this design. Encoded as **E4-L1** and
   **E4-L26** (§15) and as hard-stop conditions 4 and 11 (§18.1). **Every row of the
   28,398-row deduplicated frame is inside the served model's training set**, so this is not
   a "mostly fine except a few rows" situation: there is no admissible row.
3. **The served model is never a point estimate in any reported metric.** Coverage, width,
   interval score, usability gates, subgroups and residuals are computed from fold-model
   predictions `M_k(x)`. See §0.14 and §6.1 for the metric-by-metric binding.

### 3.2 What is and is not guaranteed — corrected (revision 2)

**Revision 1's §3.2 claimed a finite-sample coverage guarantee for the deployed procedure.
That claim was wrong and is withdrawn here.** It was written for split conformal on
`calibrate_k`, then applied without re-checking to a quantile that is *pooled across five
folds*. §0.14.2 (P1') shows the pool is not exchangeable with any `test_k`: 99.7–100% of
`test_k` rows were used to fit at least one of the other four fold models, and 76–84% of the
pooled score entries come from such models.

Precisely:

| Quantity | Guarantee | Conditions |
|---|---|---|
| `HOLDOUT-REFERENCE` interval (§0.2.4), i.e. per-fold split conformal on `calibrate_k` | **finite-sample marginal coverage `>= 1 − alpha`** | `fit_k ∩ calibrate_k = ∅` (proven, all ten cells) and **exchangeability**, which holds in the **`random` regime only** |
| `POOLED-HOLDOUT-CONFORMAL` interval (§0.2), the deployed quantity | **none. No distribution-free claim of any kind** | — |
| Any interval centred on the **served V5** estimate | **none** | `V5 != M_k` (Section 3.1) |

The `location_grouped` regime breaks exchangeability by construction, so **no** guarantee
attaches there even for `HOLDOUT-REFERENCE`: `M_k` has never seen the localities of `test_k`,
so a "fresh exchangeable point" does not exist. Phrasing must be "observed coverage under
location shift", never "coverage is 90%".

The pooled construction is **not** thereby invalid — it is **unguaranteed**, which is an
empirical question this experiment answers (F1/U2), not a licence to assume calibration. The
homogeneity assumption it rests on is registered and falsifiable in §0.2.3.

Finally, the served model is `V5`, fitted on all rows — never `M_k`. Whether the calibrated
width transfers to it at all is a separate, unproven question. No transfer diagnostic is
registered in this experiment.

> Every coverage number involving the served V5 point estimate is a **descriptive
> measurement**, never a guarantee. Phrasing must be "observed coverage of the served
> estimate under this calibration", never "the served estimate has 90% coverage". The same
> sentence, with "measured for `M_k` under `POOLED-HOLDOUT-CONFORMAL`", applies to every
> headline number in this experiment.

### 3.3 Consequence for any future application integration

Any later integration (a separate, separately-authorized task — **not** part of this
experiment) would have to state, in the product and in the API, that the reported interval
is a **"90% conformal prediction interval calibrated on V5-configuration fold models"**,
with coverage measured for those fold models and an explicit, unproven transfer to the
served estimate. If that framing is not acceptable for the product, the scientifically
valid alternative is a re-fit of V5 with a calibration holdout — which changes the
production point prediction and is **out of scope and unauthorised**.

---

## 4. Frozen dependencies (read-only inputs)

### 4.1 Dataset

| Property | Value |
|---|---|
| Path | `data/processed/MREID_property.csv` |
| SHA-256 | `1be10465221668ffde0fe7392efbc183ef77a654549a2724b26439978fad8877` |
| Raw rows | **29,135** (header excluded) |
| Post-dedup rows | **28,398** (737 exact duplicates removed) |
| Duplicate key | `['source_city', 'location', 'area', 'no_of_bedrooms', 'price']`, keep first; `mreid_id` and `source_file` excluded from the key |
| Cities | Bangalore, Chennai, Delhi, Hyderabad, Kolkata, Mumbai (6) |
| Rows by city (dedup) | mumbai 6,820 · kolkata 6,270 · bangalore 5,438 · chennai 4,208 · delhi 3,859 · hyderabad 1,803 |

Deduplication is applied by `scripts/valuation_v2_1/protocol.py::remove_duplicates`,
imported read-only. No fuzzy, near-duplicate or within-location deduplication is performed.

### 4.2 Frozen outer folds

`artifacts/valuation/v2_1/cv_folds_v2_1.json`, SHA-256 `989b79656b051b3b6c8c583b4170a5aaa1c31676260772247bbc98d1eb095156`,
protocol V2.1, `n_splits = 5`, `random_state = 42`, regimes `random` and `location_grouped`.
Indices are **positional into the 28,398-row deduplicated frame**.

### 4.3 Frozen nested calibration folds — reused verbatim

`artifacts/valuation/v2_2/conformal/calibration_folds_v2_2.json`, SHA-256
`47b89d8b41e98301129860cbf253f8c30255e272ec79f8277da54bb73acc6ea9`, derived from
`cv_folds_v2_1.json` (`parent_folds_sha256 = 989b7965…`), `cal_frac = 0.20`,
`random_state = 42`.

Per `(regime, outer_fold)` entries carry `fit`, `calibrate`, `test` index arrays plus
`n_fit`, `n_calibrate`, `n_test`, `n_frozen_train`, `n_frozen_valid`, `digests`.

| Regime | fold | `fit` | `calibrate` | `test` |
|---|---|---|---|---|
| random | 1 | 18,174 | 4,544 | 5,680 |
| random | 2 | 18,174 | 4,544 | 5,680 |
| random | 3 | 18,174 | 4,544 | 5,680 |
| random | 4 | 18,175 | 4,544 | 5,679 |
| random | 5 | 18,175 | 4,544 | 5,679 |
| location_grouped | 1 | 19,295 | 4,245 | 4,858 |
| location_grouped | 2 | 19,025 | 3,668 | 5,705 |
| location_grouped | 3 | 18,015 | 5,369 | 5,014 |
| location_grouped | 4 | 16,266 | 5,143 | 6,989 |
| location_grouped | 5 | 18,412 | 4,154 | 5,832 |

**Reusing this file is the central reuse decision of Experiment 4.** It means the
train/calibrate/test separation is *bit-identical to Experiment 2*, so any difference in
outcome between the two experiments is attributable to the **model and feature contract**,
not to a different data split.

**Where the training / calibration / test data comes from — the full chain, stated
explicitly because §6.1 of revision 1 left it implicit:**

1. `cv_folds_v2_1.json` supplies, per regime, five `train_k` / `valid_k` pairs, positional
   into the 28,398-row deduplicated frame, with `train_k = frame \ valid_k`.
2. `calibration_folds_v2_2.json` supplies the **three-way** split. It was derived once,
   in Experiment 2, by holding out one inner fold of `train_k`:
   `calibrate_k` = inner fold 0, `fit_k` = inner folds 1-4, `test_k` = `valid_k` unchanged.
   `cal_frac = 0.20`, `n_splits = 5`, `random_state = 42`. Inner splitter is
   `KFold(shuffle=True, rs=42)` for `random` and `GroupKFold(shuffle=True, rs=42)` on
   `source_city__location` for `location_grouped` — which is precisely why calibration
   localities are as unfamiliar to `M_k` as test localities are.
3. Experiment 4 opens that file **read-only**, re-verifies its SHA-256 and its per-entry
   `digests`, and consumes `fit`, `calibrate` and `test` arrays unchanged. It calls no
   splitter, derives no inner split, and regenerates no fold file.

### 4.4 Production V5 model (read-only; never refit; never a source of scores)

| Artifact | Path | SHA-256 (recorded in revision 2) |
|---|---|---|
| Model | `models/valuation/final/millow_valuation_model.json` | `9ce41db56e28d0c75bf949c99068628c8a102fba111e5f33a9db347356d5c6e9` |
| Preprocessor | `models/valuation/final/millow_valuation_preprocessor.joblib` | `ff96478256704ec54365b576074483314778fd1063956cf95b530a67374388b4` |
| Metadata | `models/valuation/final/metadata.json` | `edb08b6f930687adddfab56452f3d154942c160ce4b9dc7ea29f9e0688e824a0` |

All three are re-verified before and after every run (E4-L21). **The only permitted
operation on them is `load_model` / `joblib.load` followed by `transform` and `predict`.
`.fit()` on either the production model or the production preprocessor is a hard stop
(§18.1).** The production preprocessor's fitted category set is additionally compared against
the fold frame's feature values (E4-L25, §0.13.3).

Metadata facts (asserted in Phase 2): `model_name = "MILLOW V5 Log-Price XGBoost"`,
`target = log1p(price)`, `inverse_transform = expm1(prediction)`,
`training_rows = 29135`, `location_count = 1776`, categorical `["location"]`, numerical
`["log_area","log_bedrooms","area_per_bedroom","area_bedroom_interaction","amenity_yes_count","amenity_known_count","amenity_unknown_count"]`.

Reported validation (context only, not a target to match):
`mean_MAE = 5,810,457.42`, `mean_RMSE = 15,857,144.84`, `mean_R2 = 0.142516`,
`mean_MAPE = 54.8991`, `GroupShuffleSplit by location`, `test_size = 0.2`,
`random_state = 42`.

### 4.5 Experiment 2 results (reference only; unchanged)

Carried as context, not re-derived: six methods, three levels, ten fold models;
`median_relative_width` at 90% = 2.30–2.76 against the 2.00 gate; **F3 not_usable ×12**;
largest absolute coverage error 0.0092 (C3, 90%, random); report explicitly does not claim
production readiness.

### 4.6 Explicitly protected — must not be touched

Everything in Section 0.11. Additionally, all Experiment 2 protocol/result documents and
all Experiment 1/2/3 artifacts, models, manifests and tags.

---

## 5. Data and fold construction

### 5.1 Two row populations — stated, not assumed away

| Population | Rows | Used by |
|---|---|---|
| Raw | 29,135 | production V5 training; live application serving |
| Deduplicated | 28,398 | frozen V2.1 folds, frozen nested calibration folds, **Experiment 4 calibration and evaluation** |

Experiment 4 operates on the **28,398-row deduplicated frame**, because that is the frame
the frozen folds are keyed to. The 737 excluded rows are **exact duplicates** on the
duplicate key, so excluding them from calibration removes no information and avoids
weighting identical rows twice.

This population difference is recorded as a limitation (§19, item 8) and is **not**
silently reconciled.

### 5.2 Training / calibration / test separation

Per `(regime, outer fold) k`, three disjoint index sets, all read verbatim from
`calibration_folds_v2_2.json`:

| Set | Source | Sole permitted role | Rows it may never take |
|---|---|---|---|
| `fit_k` | frozen nested `fit` | fit `M_k` **and** fit `P_k` (the encoder) | may not be scored; may not be evaluated |
| `calibrate_k` | frozen nested `calibrate` | compute nonconformity scores → `qhat` | may not be fitted on; may not be evaluated |
| `test_k` | frozen nested `test` == frozen `valid_k` | evaluation, subgroups | may not be fitted on; may not be scored; **its targets may not enter `qhat`** |

Required invariants (asserted, §15/§18):

```
fit_k ∩ calibrate_k = ∅        measured: 0, all ten cells
fit_k ∩ test_k      = ∅        measured: 0, all ten cells
calibrate_k ∩ test_k = ∅        measured: 0, all ten cells
fit_k ∪ calibrate_k  = frozen train_k    measured: exact, all ten cells
test_k               = frozen valid_k    measured: element-for-element, all ten cells
```

Partition-completeness facts, verified during the read-only audit and re-asserted at runtime:

- `random`: `n_fit + n_calibrate` = **22,718** for folds 1-3 and **22,719** for folds 4-5
  (revision 1 wrongly said 22,718 for all five); `n_test` sums to **28,398**.
- `location_grouped`: `n_train` = 23,540 / 22,693 / 23,384 / 21,409 / 22,566;
  `n_test` sums to **4,858 + 5,705 + 5,014 + 6,989 + 5,832 = 28,398**.
- Hence the five `test_k` sets **partition the deduplicated frame exactly** (verified: union
  = all 28,398 rows, pairwise disjoint) — every row is evaluated exactly once per regime,
  and no row is evaluated twice.

### 5.3 Group semantics carried forward

`group_column = source_city__location` for `location_grouped`. Group disjointness:

```
group(fit_k)      ∩ group(test_k) = ∅     (location_grouped)
group(fit_k)      ∩ group(calibrate_k) = ∅ (location_grouped)
```

The `random` regime is a positive control where locality recurs across splits by design.

### 5.4 Cross-fold overlap of the pooled pool — structural, disclosed, not fixable here

Within a fold, the three partitions are exactly disjoint (§5.2). **Across folds they are
not, and for the pooled construction they cannot be**, because `test_1..test_5` partition the
frame while `calibrate_j ⊆ train_j = frame \ test_j`. Consequence, measured:

| | `random` | `location_grouped` |
|---|---|---|
| `test_k` rows fit by some `M_j`, `j ≠ k` | 5,663–5,676 of 5,679/5,680 (**99.7–99.9%**) | 4,841–6,989 of 4,858–6,989 (**99.7–100.0%**) |
| `test_k` rows that are calibration rows of another fold | 3,288–3,375 (**57.9–59.4%**) | 2,689–4,334 (**54.3–62.0%**) |
| `test_k` rows untouched by all other folds | **0** | **0** |
| pooled score entries produced by a model that saw ≥1 `test_k` row | 18,176 / 22,720 (**80.0%**) | 17,210–18,911 / 22,579 (**76.2–83.8%**) |

Per-fold table: §0.14.2 P1'/P2'. Registered as E4-L27; the withdrawal of the §3.2 guarantee
follows from it.

Three things this is **not** meant to be read as saying:

- **It is not in-sample scoring.** No calibration score is ever produced by a model that saw
  the row being scored: `fit_k ∩ calibrate_k = ∅`, verified 0/10, and each `s_i` is
  out-of-sample for `M_k`. Per-fold intervals from `HOLDOUT-REFERENCE` are clean.
- **It does not let a test row be calibrated by itself.** The pooled statistic is a single
  scalar, not a per-row quantity.
- **It does not mean the pooled result is worthless.** It means the pooled result is
  *unguaranteed*, and that the size of the resulting deviation is an empirical question, to
  be answered by F1/U2 and by the `HOLDOUT-REFERENCE` comparison in §0.2.3 — not asserted.

No alternative pooling scheme over this frozen structure can remove the overlap; only the
per-fold construction (§0.2.4) can, which is why it is registered.

---

## 6. `POOLED-HOLDOUT-CONFORMAL` method

### 6.1 Construction — with every point estimate bound to its metric

This is the section revision 1 got wrong. **Each array below is labelled with the only
metrics it is permitted to feed.**

```
for regime in {location_grouped (primary), random (secondary)}:

    # ---- fit partition: the ONLY rows any model or encoder ever sees -------------
    for k in 1..5:
        P_k = ColumnTransformer(V5 contract)        fitted on fit_k      # §6.3
        M_k = XGBRegressor(V5 config, §0.9)         fitted on fit_k

        # ---- calibration partition: scored, never fitted --------------------------
        yhat_cal_k  = M_k( P_k.transform( X_v5[calibrate_k] ) )     # out-of-sample for M_k
        s_k         = score( y[calibrate_k], yhat_cal_k )          # the ONLY score source
        #   no test row, no test target and no production prediction appears above.

        # ---- test partition: evaluated, never fitted, never scored ---------------
        yhat_test_k = M_k( P_k.transform( X_v5[test_k] ) )          # out-of-sample for M_k

        # ---- deployment-form prediction: NOT USED IN ANY REPORTED METRIC (E4-L1) ----
        yhat_v5_k   = V5( V5_preprocessor.transform( X_v5[test_k] ) )   # IN-SAMPLE

    S_regime      = concat( s_1 ... s_5 )                 # POOLED-HOLDOUT-CONFORMAL pool
    qhat_pooled   = exact_order_statistic( S_regime, alpha )       # deployed quantity
    qhat_ref[k]   = exact_order_statistic( s_k, alpha )            # HOLDOUT-REFERENCE (§0.2.4)
```

**Point-estimate binding — the single most important table in this protocol.**

| Reported quantity | Point estimate it MUST use | May it ever use `yhat_v5`? |
|---|---|---|
| nonconformity score `s_i` | `yhat_cal_k` | **No — hard stop** |
| `qhat` (pooled or reference) | derived from `s_i` only | **No — hard stop** |
| `empirical_coverage`, `miscoverage_low/high`, Clopper–Pearson | `yhat_test_k` | **No — hard stop** |
| `mean/median/sd/min/max_width`, `*_inr`, `*_relative_width`, `upper_lower_gap_ratio` | `yhat_test_k` | **No — hard stop** |
| `interval_score`, `winkler_score` | `yhat_test_k` | **No — hard stop** |
| U1 `median_relative_width`, U2, U3, F1, F2, F3, F4, F6, F7, F9, F10 | `yhat_test_k` | **No — hard stop** |
| city and price-band subgroup coverage (RQ5, RQ6) | `yhat_test_k` | **No — hard stop** |
| residual diagnostics RQ8 | `yhat_test_k` | **No — hard stop** |
| "deployment-form" interval described in prose | `yhat_v5_k` | **Yes — description only, never a metric, never exported as an artifact** |

Consequently there is **no** coverage number anywhere in Experiment 4 that is a property of
the served model, and no artifact may contain an interval centred on `yhat_v5` labelled as
a result. §3.2 governs how any such quantity may be described if it is mentioned at all.

**Enforcement.** `scripts/experiment_2/conformal.py` is used unmodified and **none of its
functions can be handed a production prediction** for a reported number, because every
reported number is produced by an Experiment 4 driver that only ever receives `yhat_cal_k`
or `yhat_test_k`. The production predictions live in a separate object
(`DeploymentPointEstimates`) with no accessor on the scoring path, and the manifest asserts
`n_scores_from_production_model = 0` (E4-L1, E4-L26). This is structural, not a convention.

### 6.2 Fold model definition

Fold models are **calibration instruments only**. They are written to
`models/valuation/v2_4/conformal/**` and are **never imported by `backend/**`**, never
exposed through any API, and never presented to a user.

### 6.3 Feature pipeline — the V5 contract, pinned exactly

`P_k` is, per fold, fit on `fit_k` **only**:

```
categorical = ["location"]        OneHotEncoder(handle_unknown="ignore", sparse_output=True)
numerical   = ["log_area", "log_bedrooms", "area_per_bedroom",
               "area_bedroom_interaction", "amenity_yes_count",
               "amenity_known_count", "amenity_unknown_count"]
transformer = ColumnTransformer([("cat", OneHotEncoder(...), categorical),
                                ("num", "passthrough", numerical)])
```

The **input columns to that transformer are exactly 8**, asserted key-for-key against
`metadata.json.features` (E4-L7). No scaler. No `source_city`. No `resale`. No `group`. No
`amenities_fully_specified`. **No frequency features** — V5's contract has neither
`location_frequency` nor `log_location_frequency`, so Experiment 2's L7 trap does not exist
here; the assertion that no frequency column is present is still made (E4-L7).

**Feature *values* — pinned to the training-time transform (§0.13.1).** The eight columns are
not enough; their values must be the ones the served model was fitted on:

```
X_v5 = create_features_training_time( rows )        # §0.13.1, parity-tested
  log_area                  = log1p(area)
  log_bedrooms              = log1p(no_of_bedrooms)
  area_per_bedroom          = area / no_of_bedrooms
  area_bedroom_interaction  = area * no_of_bedrooms
  amenity_yes_count         = (V5_amenity_set == 1).sum()      # 38 columns, NOT 35
  amenity_known_count       = (V5_amenity_set in {0,1}).sum()  # 38 columns, NOT 35
  amenity_unknown_count     = (V5_amenity_set == 9).sum()      # 38 columns, NOT 35
  location                  = str(location).strip()           # strip ONLY, never lower()

  V5_amenity_set = all numeric columns of the frame except
                   {mreid_id, price, derived_price_per_sqft, source_file,
                    source_city, location}                      # -> 38 columns here
```

Three rules that are each independently checked (E4-L25):

1. **The amenity set is the 38-column training-time set, not the 35-column serving set.**
   Using 35 changes `amenity_yes_count` on 42.51% of rows and `amenity_known_count` on 100%
   (§0.13.1). The set is recomputed from the frame's columns by the recorded rule, not
   hard-coded, and its size is asserted.
2. **`location` is stripped, never lowercased** (§0.13.3). Lowercased values are permitted
   only as group keys and subgroup labels. Every `location` value that reaches an encoder
   must appear in the production encoder's category set; verified currently **0 unknown**.
3. **The V2.1 fold frame is never used as the feature matrix.** It supplies indices, group
   keys and strata; the eight feature columns come from the V5 feature view (§0.13.2),
   joined by `mreid_id`. `price`, `derived_price_per_sqft` and every non-listed column are
   not inputs (E4-L11).

`handle_unknown="ignore"` is retained verbatim from production. It is what makes the
`location_grouped` result meaningful (unseen locality → all-zero categorical block) **and**
it is why rule 2 is a hard stop: an accidentally lowercased `location` degrades silently to
an all-zero block rather than raising.

### 6.4 Aggregation — restated from §0.2

One pooled quantile per `(regime, method, level, stratum)`, from the concatenated
`calibrate_k` score multiset. `HOLDOUT-REFERENCE` per-fold quantiles are recorded alongside
as diagnostics (§0.2.4), never as the deployed quantity.

### 6.5 Why this is not a new conformal method

Every conformal operation — scoring, exact finite-sample quantiles, interval construction,
coverage metrics, sharpness metrics, interval score, subgroup coverage, usability gate —
is executed by **`scripts/experiment_2/conformal.py`, imported read-only and unmodified**
(§16.1). Experiment 4 adds **no** scoring rule, **no** quantile rule, **no** interval
formula and **no** metric. The only genuinely new content in Experiment 4 is the *model
being calibrated*, the *V5 feature transform* (§6.3), and the honest analysis of the
transfer problem (§3).

---

## 7. Nonconformity scores, target space, quantiles

### 7.1 Scores

As fixed in §0.5, computed in `log1p(price)` space. `B1` divides by the **calibration-set**
fold-model prediction (`yhat_cal_k`), never by the served prediction.

### 7.2 Quantile calculation

Exact finite-sample order statistic, no interpolation:

```
k    = ceil( (n + 1) * (1 - alpha) )
qhat = s_(k)    if k <= n
     = +inf     if k > n
```

`numpy.quantile` interpolation is deliberately **not** used — the finite-sample statement is
defined for the `k`-th order statistic, and interpolating breaks it (Experiment 2 §8).

Pooled calibration sizes, for method A (the others share them):

| Regime | pooled entries `n` | distinct rows | `n_effective` (Kish) |
|---|---|---|---|
| `random` | **22,720** (4,544 × 5) | 16,605 | 13,995 |
| `location_grouped` | **22,579** (4,245+3,668+5,369+5,143+4,154) | 16,606 | 14,148 |

(Revision 1 printed `21,579` for `location_grouped` in three places; the correct value is
**22,579**.)

`k > n` is unreachable at all three levels; the guard is implemented and unit-tested
regardless. Every quantile record must carry `n`, `n_distinct_rows` and `n_effective_rows`
(E4-L28), because the `1 − alpha` resolution is **not** `1/n` when rows repeat across folds
(§0.14.3).

### 7.3 Mondrian strata

- **C1** — `source_city` (6 strata). Available as a data column even though V5 does not use
  it as a feature.
- **C2** — predicted-price tertile, cutpoints computed on `calibrate_k` predictions and
  applied unchanged to `test_k`.
- **C3** — grouped by `source_city__location`, **secondary only**, group count attached to
  every number.

Strata with `n < min_n` produce a recorded fallback (`fallback_used = True`, null quantile,
unbounded interval) rather than a silent pool — reusing Experiment 2 §8.1 behaviour.

---

## 8. Interval construction and target space

In `log1p(price)` space, per method. **`yhat` here is always the fold model's prediction
`yhat_test_k`** (§6.1 binding table) — never the served estimate:

```
pooled   (A, B3, B1) :  lower = yhat_test_k - qhat*sigma ,  upper = yhat_test_k + qhat*sigma
                          sigma = 1 except B1 where sigma = yhat_test_k   (B1 normalises by
                          the model's own prediction at the row being bounded, matching
                          the calibration-time divisor yhat_cal_k)
Mondrian (C1, C2)    :  per-stratum qhat, same point estimate
C3                   :  per-group qhat (caller-supplied)
```

`qhat` is the **pooled** quantile of §0.2 for the headline; the `HOLDOUT-REFERENCE` variant
of §0.2.4 is built from the same arrays restricted to `s_k` and reported alongside, flagged
`reference_only = true` in every row.

INR presentation at the boundary only:

```
lower_inr = max(expm1(lower), 0)
upper_inr = max(expm1(upper), 0)
width_inr = upper_inr - lower_inr
```

**INR asymmetry is reported, not assumed away** (Experiment 2 §10.2): `expm1` is convex, so
a log-symmetric band becomes an asymmetric INR band. The upper/lower gap ratio is recorded
for every cell.

**Terminology (binding):** outputs are described as a **"90% conformal prediction
interval"**. Never "confidence interval", "guaranteed value", "guaranteed price",
"guaranteed range", "probability of price", or "investment recommendation".

---

## 9. Coverage metrics

Recorded for **every** `(regime, method, level, stratum, fold)` cell and for the pooled
regime cell.

| Metric | Definition |
|---|---|
| `n_evaluated` | scored test observations |
| `empirical_coverage` | `mean(lower <= y <= upper)` |
| `coverage_error` | `empirical_coverage - nominal` |
| `abs_coverage_error` | absolute value |
| `coverage_cp_lower`, `coverage_cp_upper` | Clopper–Pearson 95% bounds |
| `coverage_significant` | nominal outside the Clopper–Pearson interval |
| `miscoverage_low`, `miscoverage_high` | `mean(y < lower)`, `mean(y > upper)` |

Mandatory:

- **Clopper–Pearson, not Wald** (RQ10).
- `miscoverage_low` / `miscoverage_high` recorded **separately** — asymmetry is diagnostic.
- Aggregation across folds: **mean and sample SD, `ddof=1`**, `n_folds` recorded.
- **Every fold's raw value retained. No pooling of predictions across folds.**
- Every coverage row carries `point_estimate = "fold_model_M_k"` and
  `construction = "POOLED-HOLDOUT-CONFORMAL"` (or `"HOLDOUT-REFERENCE"` with
  `reference_only = true`). A row lacking either field is a schema violation, not a default.
- **No coverage row may have `point_estimate = "production_V5"`.** Such a row would be an
  in-sample residual measurement presented as a coverage result, which is E4-L1.

---

## 10. Interval-width metrics

| Metric | Definition | Space |
|---|---|---|
| `mean_width`, `median_width` | `mean`/`median(upper - lower)` | log1p — primary |
| `sd_width`, `min_width`, `max_width` | dispersion and extremes | log1p |
| `mean_width_inr`, `median_width_inr` | INR width | INR — business-facing |
| `mean_relative_width`, `median_relative_width` | `(upper-lower)/point` | ratio — **U1 input** |
| `width_is_constant` | boolean, `method in {A,B3,C1,C2}` | — |
| `upper_lower_gap_ratio` | INR upper gap / INR lower gap | ratio |

`width_is_constant` is carried on every row so a reader cannot misread
`mean_relative_width` for the homogeneous methods, where it is a deterministic restatement
of `qhat` (Experiment 2 §10.1).

---

## 11. Interval score

Gneiting & Raftery (2007), the proper scoring rule for interval predictions:

```
IS_alpha = (u - l) + (2/alpha)*(l - y)*1{y < l} + (2/alpha)*(y - u)*1{y > u}
```

Reported as `mean` and `median` per cell, in both log space and INR. It is the primary
combined metric because a method can hit nominal coverage with uselessly wide intervals and
mean width alone would not reveal that. (Also reported as the `winkler_score` alias at
`beta = alpha`, matching Experiment 2's convention.)

---

## 12. Primary evaluation — location-grouped

`location_grouped` is the **primary** evaluation (RQ3), because it is the regime that
matches MILLOW's deployment question: a property in a locality the model has not seen.

| | `location_grouped` (primary) | `random` (secondary, guarantee-bearing) |
|---|---|---|
| Outer splitter | `GroupKFold(shuffle=True, rs=42)` | `KFold(shuffle=True, rs=42)` |
| Test-fold locality novelty | **100% unseen locality** | 96.8–97.7% seen locality |
| `group(fit_k) ∩ group(test_k)` | `∅` | overlaps by design |
| Conformal guarantee | **none — exchangeability broken** | holds for `M_k` under exchangeability |
| Role | **primary deployment-relevant measurement** | **positive control; RQ1 must pass** |

**Both roles are preserved deliberately.** Making location-grouped *primary* does not
relegate random to a formality: random is where the formal guarantee lives and where a
failure (U2, RQ1) invalidates the implementation. Reporting must always show the pair
together with the declared sign convention `Δ = grouped − random`.

Reporting: `regime_shift_coverage.csv` per `(method, level)` with coverage, coverage error
and paired delta; paired per-fold deltas at matched fold index; the same for every width
metric and the interval score.

Reference point for scale (Experiment 1, unchanged): CatBoost `R2_log` fell 0.3640 → 0.2677
(−26%) under location shift. Interval degradation must be reported **alongside** point
accuracy degradation, not in isolation.

---

## 13. Subgroup analysis

**Both are diagnostic reporting. Neither assigns a calibration stratum except C1/C2 as
specified in §7.3.**

### 13.1 City (RQ5)

6 cities, all present in every `test_k` in both regimes. Reported per
`(regime, method, level, city)` with `n_evaluated` and Clopper–Pearson bounds always
attached.

**Pre-declared suppression rule:** any subgroup cell with `n_evaluated < 100` is emitted
with `coverage_reliable = false` and excluded from every summary statement.

### 13.2 Price band (RQ6)

Fixed absolute INR edges, pre-declared (identical to Experiment 2 §12.2a):

| Band | Rows (dedup) | Share |
|---|---|---|
| < 0.5 cr | 9,637 | 33.94% |
| 0.5–1 cr | 9,689 | 34.12% |
| 1–2 cr | 5,608 | 19.75% |
| 2–4 cr | 2,365 | 8.33% |
| 4–10 cr | 886 | 3.12% |
| ≥ 10 cr | 213 | 0.75% |

Banding by **true** price depends on the label, so it is **diagnostic only** and is never
used to assign a calibration stratum. The assignable banding is C2's predicted-price
tertile.

### 13.3 Residual diagnostics (RQ8)

Residual-vs-fitted, QQ, scale-location and per-city residual summaries for the fold models,
mirroring Experiment 2's figure set.

### 13.4 Figures

All PNG **and** PDF, byte-deterministic via `metadata={"CreationDate": None}`, matching the
Experiment 1/2/3 convention.

---

## 14. Usefulness criteria — summary

| ID | Criterion | Threshold | Fixed at |
|---|---|---|---|
| **U1** | median relative width @0.90 | `<= 2.00` | §0.6 (reused from Exp 2 §0.4) |
| **U2** | \|coverage − 0.90\| @0.90, random | `<= 0.02` | §0.6 |
| **U3** | coverage delta grouped − random @0.90 | `>= −0.05` | §0.6 |

Reported **independently**, never combined into a single score. Pre-registered expectation:
**U1 is expected to fail** for A/B3/C1/C2 (§0.6). The protocol registers no U4
(any production-residual comparison); U4, D-T and F8 were removed per revision 3.

---

---

## 15. Leakage audit

Enforced in code as assertions, covered by tests. Every item is a **hard failure**: on
violation the run halts immediately and **no result is reported**.

### 15.1 E4-L1 — the critical control, stated without ambiguity

**The production V5 model was fitted on all 29,135 rows. Therefore every one of the 28,398
rows Experiment 4 operates on is in-sample for it. There is no admissible row, no regime, no
stratum and no method for which a production residual is usable. Concretely, E4-L1 forbids
all eight of the following, unconditionally:**

| # | Forbidden | Enforcement |
|---|---|---|
| 1 | a `yhat_v5` value entering any nonconformity score | the scoring driver has no parameter that can carry it |
| 2 | a `yhat_v5` value entering `qhat` | same |
| 3 | a `yhat_v5`-centred interval being reported as a result | no artifact writes an interval with `point_estimate = production_V5`; §6.1 binding table |
| 4 | a coverage, width, interval-score, subgroup or usability number computed from `yhat_v5` | §6.1 binding table marks all of those fold-model-only |
| 5 | the production preprocessor being `.fit()`-ed, even on `fit_k` | only `.load_model` / `joblib.load` + `.transform` / `.predict` are reachable; E4-L21 |
| 6 | the production model being retrained, re-tuned, re-early-stopped or replaced | no `.fit()` on the production artifact; hashes in §4.4 |
| 7 | `scale_v5` being used anywhere for any reported metric or diagnostic | no diagnostic consumes `scale_v5`; manifest asserts `n_scores_from_production_model = 0` |
| 8 | any residual distribution, quantile or figure derived from `scale_v5` being mixed with a fold-model distribution | separate columns/series; the manifest flags them distinctly |

**No production-model use is permitted for any diagnostic or reported metric.** The
production predictions are held in a type with no accessor on the scoring path (E4-L26).

### 15.2 Full control table

| # | Control | Assertion |
|---|---|---|
| **E4-L1** | **No production-model scores or point estimates** | As §15.1: eight prohibitions, enforced structurally and by manifest counters `n_scores_from_production_model = 0`, `n_intervals_centred_on_production = 0`. **This is the control that makes the whole experiment valid.** |
| **E4-L2** | `fit ∩ calibrate = ∅` | `set(fit_k) ∩ set(calibrate_k) == ∅` (measured 0/10) |
| **E4-L3** | `fit ∩ test = ∅` | `set(fit_k) ∩ set(test_k) == ∅` (measured 0/10) |
| **E4-L4** | `calibrate ∩ test = ∅` | `set(calibrate_k) ∩ set(test_k) == ∅` (measured 0/10) |
| **E4-L5** | Partition completeness | `fit_k ∪ calibrate_k == frozen train_k`, no row lost (measured exact 10/10) |
| **E4-L6** | Test identity | `test_k ==` frozen `valid_k`, element-for-element (measured 10/10) |
| **E4-L7** | **Feature-column contract** | the 8 columns fed to `ColumnTransformer` equal `metadata.json.features` exactly — key for key, no extra, no missing — and contain **no** frequency column (`location_frequency`, `log_location_frequency`, `city_frequency`) and **no** `amenities_fully_specified` |
| **E4-L8** | Preprocessor fit scope | `OneHotEncoder` fitted on `fit_k` only; `preprocessor_fit_rows == n_fit_k` recorded per fold; never fitted on `fit_k ∪ calibrate_k`, never on `calibrate_k`, never on `test_k`, never on all rows; production preprocessor never fitted at all |
| **E4-L9** | Group disjointness | `group(fit_k) ∩ group(calibrate_k) == ∅` (location_grouped) (measured 0/10) |
| **E4-L10** | Test locality novelty | `group(fit_k) ∩ group(test_k) == ∅` (location_grouped) (measured 0/10) |
| **E4-L11** | Target not a feature | `'price'`, `'derived_price_per_sqft'`, `'resale'`, `'area'`, `'no_of_bedrooms'`, `'group'`, `'source_city'`, `'mreid_id'` all absent from the 8 input columns; and `derived_price_per_sqft` absent from the amenity set used to build the three amenity counts (§6.3 rule 1) |
| **E4-L12** | Outer-fold immutability | `cv_folds_v2_1.json` SHA-256 equals `989b7965…` before **and** after every run |
| **E4-L13** | Nested-fold immutability | `calibration_folds_v2_2.json` SHA-256 equals `47b89d8b…` before **and** after every run |
| **E4-L14** | Dataset immutability | dataset SHA-256 equals `1be10465…` before **and** after every run |
| **E4-L15** | Protected-file hashes | all Section 0.11 paths re-verified after every run |
| **E4-L16** | Tag immutability | all 11 existing tags retain their identities |
| **E4-L17** | Write-path safety | write-target assertion refuses to run on any protected path collision |
| **E4-L18** | No fold regeneration | fold files opened **read-only**; no `generate_folds` / `derive_inner_split` / splitter call anywhere in Experiment 4 code |
| **E4-L19** | Single seed | `random_state == 42` everywhere; no undeclared seed; no second seed |
| **E4-L20** | No post-hoc selection | `cal_frac`, method set, aggregation rule, band edges, thresholds U1–U3, feature contract and §0.14 disclosure tables all fixed in this document before any result |
| **E4-L21** | No production refit | `models/valuation/final/**` SHA-256s in §4.4 re-verified before and after every run; no `.fit()` on model or preprocessor |
| **E4-L22** | No pooled-across-regime quantiles | one quantile per `(regime, method, level, stratum)`; regimes never pooled |
| **E4-L23** | No blockchain write | no Experiment 4 module imports `.chain`, `backend`, or any blockchain writer |
| **E4-L24** | Fold models not served | no model under `models/valuation/v2_4/**` is imported by `backend/**` or any API path |
| **E4-L25** | **V5 feature-value parity** (new in rev 2) | (a) `create_features_training_time` output equals `train_final_valuation.create_features` row-for-row and value-for-value, on a unit-test fixture (E4-L7 covers names, this covers values); (b) the amenity set has **38** members, recomputed by the recorded rule, not hard-coded; (c) every `location` value passed to any encoder is **byte-identical** to a category of the production encoder (measured: **0 unknown** strip-only; 1,722/1,775 lowercased values are unknown and are therefore barred); (d) `mreid_id` join between fold view and V5 feature view is 1:1 with order preserved and `price` bit-identical; (e) the hyper-parameter block equals `metadata.json.model_parameters` key-for-key |
| **E4-L26** | **Production predictions off the reporting path** (new in rev 2) | production predictions are held in a type with no accessor reachable from the scoring/quantile/coverage/width/interval-score/subgroup call graph; a static check asserts no Experiment 4 driver module both imports `models/valuation/final` and calls `conformal_scores` / `conformal_quantile` / `build_intervals` / `coverage_metrics`; every emitted result row carries `point_estimate` and `construction` (§9) |
| **E4-L27** | **Cross-fold disclosure matches** (new in rev 2) | the measured cross-fold overlap table (§0.14.2 P1'/P2', §5.4) is reproduced at runtime: `test_k ∩ ∪_{j≠k} fit_j`, `test_k ∩ ∪_{j≠k} calibrate_j`, and the contaminated-pool fraction must equal the §5.4 values exactly. This is a **disclosure-match** control, not a zero-tolerance control: the overlap is structural (§5.4) and must be **reported**, not forbidden. A mismatch means the frozen inputs changed and every number in §5.4 must be re-derived before results are interpreted |
| **E4-L28** | **Pooled multiplicity disclosed** (new in rev 2) | `n`, `n_distinct_rows`, `n_effective_rows` and the multiplicity histogram are recorded for every pooled quantile and must match §0.14.3 (22,720/16,605/13,995 and 22,579/16,606/14,148); no `1 − alpha` resolution claim is made from `n` alone |

### 15.3 The controls that would actually break this experiment

- **E4-L1 / E4-L26** are the ones unique to Experiment 4 and the most dangerous. Because
  the production V5 model was fitted on all 29,135 rows, any score, quantile, interval or
  reported metric that uses a production-model prediction is structurally impossible: the
  scoring driver has no code path that carries a V5 prediction, and the manifest asserts
  `n_scores_from_production_model = 0` and `n_intervals_centred_on_production = 0`. A
  violation halts the run. **This is the single control whose removal would invalidate the
  entire experiment.**

- **E4-L25** is critical: the feature-values parity test (38-column amenity set, strip-only
  `location`, 0 unknown encoder categories, mreid_id join, hyper-parameter block). If this
  check fails, every subsequent number is uninterpretable because the input contract is
  wrong. Revision 1 pointed at `predict_property_value.py`, which disagrees with the
  training transform on 42–100% of rows; revision 2 corrected it to `train_final_valuation.create_features`.

- **E4-L8** is the subtlest generic trap: fitting the encoder on `fit_k ∪ calibrate_k`
  leaks calibration-set locality composition into the model. The encoder is asserted to be
  fit on `fit_k` only; `transform` on `calibrate_k` and `test_k` only.

- **E4-L9 / E4-L10 together** make the location-grouped primary result interpretable. If
  calibration localities overlap test localities, the experiment stops measuring location
  shift. Group disjointness (`groups(fit) ∩ groups(calibrate) = ∅`) and test locality
  novelty (`groups(fit) ∩ groups(test) = ∅`) are exactly zero; both are measured at
  runtime.

- **E4-L6** preserves provenance: without bit-identical `test_k`, Experiment 4 is not
  comparable to Experiment 2 on the same rows. The outer-fold (`cv_folds_v2_1.json`) and
  nested-fold (`calibration_folds_v2_2.json`) SHA-256s must match before and after every run
  (§4.4, E4-L12/L13).

- The removed controls (U4, F8, D-T) are noted here for historical transparency:
  they were removed in revision 3 because their sole purpose was a production-model residual
  comparison that is forbidden by E4-L1. No replacement diagnostic is registered.

## 16. Reproducibility requirements

Reusing established conventions rather than inventing new ones.

| Requirement | Convention |
|---|---|
| Environment record | `protocol.environment_record()` → new `environment_v2_4.json`, same schema as `environment_v2_1.json` |
| Fold provenance | record `folds_loaded_from`, `folds_regenerated: false`, `parent_folds_sha256`, `nested_folds_sha256` |
| Fold digests | SHA-256 per entry, mirroring the frozen `digests` convention |
| Manifest | `experiment_manifest_v2_4.json` mirroring the Experiment 1/2 manifest key-for-key where applicable, including the boolean discipline flags |
| Determinism | `deterministic: true`, input + output SHA-256 manifest, `mismatches: 0` |
| Model accounting | manifest states `models_trained_this_run: 10`, `production_model_refit: false`, `backbone_retrained: false` |
| Verification script | `verify_conformal_v2_4.py` mirroring `verify_model_comparison.py`: reload artifacts and re-verify recorded values within tolerance |
| Tests | `test_conformal_v2_4.py` mirroring `test_conformal.py` / `test_protocol.py` conventions |
| Aggregation | sample SD, `ddof=1`, `n_folds` recorded |
| Plots | PNG + PDF, byte-deterministic via `metadata={"CreationDate": None}` |
| Metric keys | declared as an explicit constant, asserted by a test |
| Repository caveat | the `core.autocrlf` working-tree/committed-hash discrepancy documented in Experiment 2 §14.1 is carried forward unchanged; Experiment 4 records the same caveat and the `core.autocrlf` setting |

### 16.1 Reused code (read-only imports) — corrected in revision 2

| Module | Reused for |
|---|---|
| `scripts/experiment_2/conformal.py` | scores, quantiles, intervals, coverage, sharpness, interval score, subgroup coverage, usability gate — **unmodified, imported** |
| `scripts/valuation_v2_1/protocol.py` | dataset load/clean/dedup, fold loading and validation, `log_target`, `to_rupees`, `environment_record` — **unmodified, imported** |
| `scripts/train_final_valuation.py` | the **training-time** `create_features` and the V5 hyper-parameter block, as the parity reference for Experiment 4's own `create_features_training_time` (§6.3, E4-L25) — **not imported at runtime**; see below |

**Revision 2 removed `scripts/predict_property_value.py` from the reuse table, and this is a
correction, not a deletion of scope.** Two reasons, both established in §0.13:

1. Its `create_features` builds amenity counts from a **35-column** serving list, while the
   served model was fitted on a **38-column** training-time set (§0.13.1). Using it as the
   reference reproduces the training/serve skew this experiment is supposed to avoid.
2. Importing the module executes its **module-level `load_model` + joblib load** (§0.13.4),
   which puts the production artifact on the import graph of every Experiment 4 driver —
   the exact proximity E4-L26 exists to eliminate.

`format_indian_price` is therefore reimplemented locally as a formatting-only helper if
needed; it touches no model and no data path.

Experiment 4 therefore ships its own `create_features_training_time` (§6.3), **pinned by
E4-L25(a) to produce values identical to `train_final_valuation.create_features`**. The
`train_final_valuation.py` import is a *test-time* import only: the parity test compares the
two implementations on a fixture and asserts equality, and the parity test is the thing that
guards the contract. It is never called from the fitting, scoring, quantile or reporting
path.

New code is limited to: a V5-config fold-model driver, an Experiment 4 artifact writer, an
analysis/aggregation script, and tests.

---

## 17. Computational budget

Recorded **before training**, as an **analytical** budget, per Experiment 3 §21 discipline.

### 17.1 Pre-training gate

**Analytical budget complete AND V5 configuration frozen (§0.9).** No pilot, timing-only,
dry-run or unscored training run is permitted or required to satisfy this gate. Measured
per-fold and total timing is a **Phase 3 output**, recorded as an empirical measurement and
never used to select or modify configuration, folds, methods or evaluation.

### 17.2 Analytical quantities

| Quantity | Estimate | Basis |
|---|---|---|
| Fits required | **10** | 2 regimes × 5 folds; no architecture search, no second seed |
| Rows per fit | 16,266 – 19,295 | frozen `fit_k` (§4.3) |
| Rows per calibration | 3,668 – 5,369 | frozen `calibrate_k`; pooled entries 22,720 (random), **22,579** (grouped); distinct rows 16,605 / 16,606 (§0.14.3) |
| Rows per test | 4,858 – 6,989 | frozen `test_k`; union = 28,398 per regime |
| Trees per fit | 900 | V5 config |
| Features | 7 numeric passthrough + up to ~1,776 one-hot `location` levels ≈ **~1,783** | V5 contract; sparse OHE |
| Predict passes | 20 | `calibrate_k` + `test_k` per fold |
| Quantile computations | 10 folds-regimes × 6 methods × 3 levels × strata | exact order statistics |

### 17.3 Analytical time estimate

Reference anchors from frozen Experiment 1 timing (700-tree models on ~18–23k rows):
CatBoost **22.9–28.0 s**/fold, LightGBM **11.8–14.1 s**/fold, RandomForest **55.2–70.4 s**/fold.

XGBoost `hist` with 900 trees on ≤19,295 rows and ~1,783 sparse features is expected to be
**of the same order or faster** than the CatBoost anchor. Analytical order:

```
10 fits × O(10^1 – 10^2 s)  ≈  O(10^2 – 10^3 s) total   (single host, n_jobs = -1)
+ 20 predict passes          ≈  O(10^0 – 10^1 s) each
+ quantiles / metrics        ≈  negligible (sort of 22,720 floats)
────────────────────────────────────────────────────────────────
Expected total: low single-digit minutes on one machine
```

Runtime is **not** a constraint on this experiment. Measured figures are recorded in
Phase 3 (§17.1).

### 17.4 Analytical storage estimate

| Artifact | Estimate | Basis |
|---|---|---|
| Fold models | ~30 MB | 10 × ~2.95 MB (production model JSON is 2.95 MB) |
| Fold preprocessors | ~0.3 MB | 10 × ~31 KB |
| OOF predictions + scores | < 5 MB | 28,398 × 2 regimes × a few float64 columns |
| Interval CSVs (Exp 2 schema) | ~200 MB | Experiment 2 produced 197.3 MB across 10 files at the same 6-method × 3-level × 56,796-row scale |
| Quantile JSONs | < 2 MB | 10 files |
| Figures (PNG + PDF) | ~30 MB | ~12 figures × 2 formats, sized to Experiment 2's set |
| **Total** | **≈ 270 MB** | within the scale already present in `artifacts/valuation/` |

No new third-party dependency is required: `xgboost 3.4.1`, `scikit-learn 1.9.1`,
`pandas 3.0.6`, `numpy 2.5.3`, `scipy 1.18.1`, `joblib 1.6.0` are already present.

---

## 18. Failure and stop criteria

### 18.1 Hard stop conditions — immediate abort, no results reported

Any of the following halts the run at once. Nothing is written, nothing is reported as a
result, and the run is not retried without a documented cause and fresh authorisation.

1. Dataset, outer-fold or nested-fold SHA-256 mismatch (E4-L12/L13/L14).
2. Any partition assertion failure (E4-L2…L6).
3. Any group-disjointness failure (E4-L9/L10).
4. **Any attempt to produce a calibration score, quantile, interval or reported metric from
   the production model or its predictions** — the eight prohibitions of E4-L1, including any
   `yhat_v5`-centred interval, any coverage/width/interval-score number derived from
   `yhat_v5`, and any `.fit()` on the production model or preprocessor (E4-L1, E4-L26).
5. Production model or preprocessor file hash change (E4-L21).
6. Protected-path hash change or tag change (E4-L15/L16).
7. Fitted feature list ≠ V5 contract (E4-L7/L11), or **V5 feature-value parity failure**
   — amenity set ≠ 38 columns, non-stripped `location`, or any `location` value unknown to
   the production encoder (E4-L25).
8. Any write outside `artifacts/valuation/v2_4/**`, `models/valuation/v2_4/**`,
   `docs/research/EXPERIMENT_4_*` and the new `scripts/experiment_4/**` directory.
9. Execution of training without explicit written authorisation (§0.10).
10. Any blockchain, escrow, contract or canonical chain-state access.
11. **Cross-fold disclosure mismatch** — the runtime-measured overlap table does not match
    §5.4 / §0.14.2 (E4-L27). The overlap itself does **not** stop the run; a *mismatch* does,
    because it means the frozen inputs are not the ones analysed here.
12. **Any result row missing `point_estimate` or `construction`**, or any result row whose
    `point_estimate` is not a fold model (E4-L26, §9).

### 18.2 Interpretation gates — evaluated after results exist

| ID | Gate | Fail meaning |
|---|---|---|
| **F0** | **Implementation validity** — must pass before anything is interpreted. B3 reproduces A to `1e-12`; homogeneous methods show `sd_width == 0`; pooled quantile equals the exact order statistic of concatenated scores; feature list equals the V5 contract; **E4-L1…L28 all pass**, including V5 feature-value parity (L25), point-estimate binding (L26), cross-fold disclosure match (L27) and multiplicity disclosure (L28). | Implementation is broken; **no result is interpretable**. |
| **F1** | **Negative control** (RQ1, U2) — random-regime coverage within ±0.02 of 0.90 at 0.90. | The procedure is not calibrated even where exchangeability holds. |
| **F2** | **Headline** (RQ3, U3) — location-grouped coverage degradation ≥ 5 pp vs random. | Calibration does not survive location shift. |
| **F3** | **Sharpness / usability** (RQ7, U1) — `median_relative_width <= 2.00` at 0.90. | **Pre-registered as expected to fail** for A/B3/C1/C2 (§0.6). Failure is an outcome, not a surprise. |
| **F4** | **Overcoverage falsifier** — coverage at 0.90 exceeds 0.97 in either regime. | Intervals are uninformatively wide; calibration is nominal-shaped but useless. |
| **F5** | **Monotonicity** (RQ4) — *descriptive only*. Report whether the 80→90→95 coverage sequence is directionally monotone. No particular direction required; non-monotonicity is not a falsifier. | — |
| **F6** | **Uniformity** (RQ5, RQ6) — any city or price band with `|coverage_error| > 0.10` at 0.90 and `n >= 100`. | Coverage is not uniform conditional on city or price level. |
| **F7** | **Location-aware improvement** (RQ9) — `min(C1, C2, C3) < A` on interval score at 0.90. | Location-aware strategies do not improve on pooled scoring. |
| **F10** | **Pooled vs reference divergence (diagnostic criterion only)** — the pooled quantile's 0.90 coverage differs from the `HOLDOUT-REFERENCE` per-fold coverage by more than 5 pp in either regime, or U2 fails for the pooled construction while passing for the reference. | The pooled construction's deviation is large in this frozen structure. Reported alongside the primary gates; it does not retract them. **F10 does NOT establish coverage. F10 does NOT provide a coverage guarantee for the pooled quantity. F10 must NOT be used for post-hoc conformal-method selection. The registered method set remains exactly {A, B1, B3, C1, C2, C3}. All methods and thresholds remain pre-registered.** |

Gates are reported **individually** and never merged into a single verdict. Calibration
(U2) and usability (U1) remain independent axes.

---

## 19. Statistical limitations

Stated now, before results, so they constrain interpretation.

1. **No formal guarantee for the served estimate.** The most important limitation in this
   document. See Section 3.2. Every served-estimate coverage number is descriptive.
2. **The served model's true out-of-sample coverage is unmeasurable on this dataset.**
   See Section 3.1. No transfer diagnostic is registered; the directional expectation in
   §0.7 is retained as context only and is not tested.
3. **The guarantee does not extend to the location-grouped regime.** Exchangeability fails
   by construction. Phrasing: "observed coverage under location shift", never "coverage is
   90%".
4. **The calibrated models are not the served model.** Fold models are fitted on
   `fit_k` ⊂ dataset, so results generalise to "an XGBoost with the V5 configuration fitted
   on ~57–68% of the rows", not to "the served V5 artifact".
5. **Five folds is a weak variance estimate.** Mean and SD over `n = 5`; fold-level values
   always reported alongside aggregates.
6. **Rows within a locality are not independent.** Clopper–Pearson bounds are
   **anti-conservative** here; they are a necessary but not sufficient significance screen.
   Clustered/bootstrap alternatives are declared out of scope rather than silently omitted.
7. **Group-level power is far below row-level power.** Median locality size is 3; C3
   numbers must always carry the group count.
8. **Two row populations.** Calibration runs on 28,398 deduplicated rows; the served model
   was trained on 29,135 raw rows (Section 5.1).
9. **One model family, one configuration.** XGBoost V5 only. Nothing here licenses claims
   about CatBoost, Ridge, RandomForest, LightGBM, or about conformal prediction in general.
10. **Multiple comparisons.** 3 levels × 2 regimes × 6 methods = 36 primary cells, plus
    subgroup cells. No multiplicity correction is planned; every subgroup claim must be
    labelled exploratory, and subgroup cells with `n < 100` are suppressed.
11. **Single dataset and single backbone.** Results generalise to this dataset and this
    configuration, not to real-estate valuation in general.
12. **Transfer argument is directional, not proved.** §0.7 is an expectation registered so
    it can be falsified, not a theorem.
13. **The pooled pool is cross-contaminated, structurally.** Every row of every `test_k` is
    seen by at least one other fold's `M_j` and 54–62% of test rows are another fold's
    calibration rows (§5.4). No calibration score is in-sample for its own model, but the
    pooled quantile is built from scores of models trained on rows that other folds then
    test. This is a property of the frozen structure, not a bug, and it is why §3.2's
    guarantee is withdrawn. **The practical consequence is bounded by nothing except the
    measured deviation**: the pooled-vs-reference comparison in §0.2.3 is the empirical
    answer, and it must be reported whichever way it goes.
14. **Pooled calibration is not 22,579 or 22,720 independent observations.** Only ~16,600
    distinct rows, `n_effective` ≈ 14,000, max multiplicity 4 (§0.14.3). Resolution and any
    confidence statement about `qhat` must use `n_effective`, not the entry count.
15. **Feature-contract risk is concentrated in three silent failure modes**: the 35-vs-38
    amenity set, lowercase vs strip `location`, and the V2.1-vs-V5 semantic mismatch
    (§0.13). All three produce ordinary-looking output. E4-L25 is the only defence, and it
    is a code-level defence, not a review-level one.
16. **`handle_unknown="ignore"` masks errors rather than raising.** In the
    `location_grouped` regime, unseen test localities legitimately encode as all-zero. That
    is intended — but it also means any encoding mistake is invisible in the metrics, which
    is exactly why rule 2 of §6.3 (encoder category membership) is a hard stop.
17. **The served model's in-sample residuals cannot distinguish the two ways the served model
    could be worse**: genuinely worse generalisation, versus better generalisation offset by an
    unseen-locality effect that shrinks the magnitude of its own error. Only `scale_v5`
    (in-sample, on the training distribution) and `scale_v5_serve` (in-sample, V5 feature
    transform) are available, and neither is a clean substitute for the served model's true
    error.
18. **The report is not a validation of the served artifact.** Nothing here certifies
    `models/valuation/final/**`, and a passing F1/F3 does not license deploying any interval
    built by these fold models.

---

## 20. Protected-state requirements

**Read-only**, verified unchanged before and after any Experiment 4 run:

- blockchain state (`.chain/state.json`) and all blockchain/token/escrow state;
- `chain-manifest.json`, `data/processed/millow_token_map.csv`, `src/config.json`
  (SHA-256 `e0b6c7d3c1eb56cad5d4d43d2e958b724cf7c74018e6c693139bec9d82610c33`);
- `contracts/**`;
- dataset `data/processed/MREID_property.csv` (SHA-256 `1be10465…`);
- V2.1 folds and artifacts (`artifacts/valuation/v2_1/**`, `models/valuation/v2_1/**`);
- **all Experiment 2 artifacts, models, protocol and results**
  (`artifacts/valuation/v2_2/**`, `models/valuation/v2_2/**`, `EXPERIMENT_2_*`);
- **all Experiment 3 artifacts and results** (`artifacts/valuation/v2_3/**`,
  `EXPERIMENT_3_*`);
- **the production V5 model** (`models/valuation/final/**`, `models/valuation/v5/**`);
- `backend/**`, `src/**`, `server/**`, `scripts/experiment_2/**`, `scripts/experiment_3/**`,
  `scripts/valuation_v2_1/**`;
- all 11 existing Git tags listed in §0.11.

**Experiment 4 must not touch blockchain state. No blockchain transaction is required.**

---

## 21. Implementation plan

Phased, mirroring the discipline of Experiments 1–3. **Nothing in this plan is executed
until this protocol is reviewed and frozen, and Phase 3 additionally requires explicit
authorisation to train.**

| Phase | Work | Output | Trains models? | Authorised now? |
|---|---|---|---|---|
| 0 | Author, review and freeze this protocol; record approvals and the no-result statement | frozen protocol + SHA-256 | No | **This document — awaiting review** |
| 1 | Pure library work: V5 feature driver (`create_features_training_time`) with **E4-L25 parity tests** against `train_final_valuation.create_features`; `DeploymentPointEstimates` type with no scoring-path accessor; fold driver reusing `experiment_2/conformal.py` read-only; unit tests for partition, feature-contract, point-estimate-binding and E4-L1 invariants | `scripts/experiment_4/**` + tests | **No** | Not yet |
| 2 | Load and verify datasets, frozen folds, nested calibration folds, production model hashes; **reproduce the §5.4 cross-fold disclosure table and the §0.14.3 multiplicity table**; verify the 38-column amenity set and 0-unknown encoder membership; write nothing outside `artifacts/valuation/v2_4/**` | verification report | **No** | Not yet |
| 2a | **STOP and report** Phase 1 + Phase 2 validation | report for review | No | — |
| 3 | Fold-wise training of 10 V5-config calibration instruments; out-of-fold scoring and pooled + reference quantiles | fold models, OOF predictions, quantiles | **Yes — 10 fits** | **NO — requires explicit authorisation** |
| 4 | Interval construction, coverage / width / interval-score / subgroup analysis, pooled-vs-reference comparison (§0.2.3), figures | results tables + figures | No (analysis) | Not yet |
| 5 | Manifest, deterministic re-run verification, protected-state check | verification report | No | Not yet |
| 6 | Review and freeze as a research checkpoint | tag (**only after approval**) | No | Not yet |

**No phase may consume `test_k` for tuning.** Phase 3 may not begin until Phase 1 and
Phase 2 have been reported and **training has been explicitly authorised in writing**.

**Phase 2 ordering constraint:** the §5.4 overlap table and the §0.14.3 multiplicity table
are verified **before** Phase 3, not reported afterwards. If they disagree with this
document, this document is wrong and must be corrected before any fit.

---

## 22. Pre-registration checklist

All items must be resolved and recorded before Phase 3 training.

- [ ] Protocol reviewed and frozen; SHA-256 recorded (§0 header).
- [ ] Relationship to Experiment 2 fixed as "independent, not a correction" (§0.1).
- [ ] `POOLED-HOLDOUT-CONFORMAL` construction and aggregation rule fixed; rejected
      aggregations and rejected names listed (§0.2, §0.2.1).
- [ ] Guarantee withdrawn for the pooled construction and the reason recorded (§3.2, §5.4).
- [ ] `HOLDOUT-REFERENCE` registered as a diagnostic, with its non-deployment role stated (§0.2.4).
- [ ] Method set closed and identical to Experiment 2 (§0.3).
- [ ] Nominal levels fixed at 0.80 / 0.90 / 0.95 (§0.4).
- [ ] Score and target space fixed at `log1p(price)` (§0.5).
- [ ] Usefulness criteria U1–U3 fixed, with the pre-registered U1 expectation (§0.6, §0.7).
- [ ] Dataset, outer-fold and nested-fold SHA-256 values recorded (§0.8).
- [ ] Production V5 model, preprocessor and metadata SHA-256 values recorded (§4.4).
- [ ] V5 model configuration fixed verbatim (§0.9).
- [ ] V5 **training-time** feature transform fixed; 38-column amenity set and strip-only
      `location` recorded; serving transform explicitly rejected as the reference (§0.13, §6.3).
- [ ] Two-frame design fixed: fold view vs V5 feature view, joined by `mreid_id` (§0.13.2).
- [ ] Mandated implementation order fixed; Phase 3 requires separate authorisation (§0.10).
- [ ] Protected-state list fixed (§0.11, §20).
- [ ] Research-discipline rules fixed, including terminology (§0.12).
- [ ] Primary (location-grouped) and secondary (random) roles fixed, both preserved (§12).
- [ ] Train / calibration / test provenance chain fixed from both frozen fold files (§4.3, §5.2).
- [ ] Per-fold partition sizes and partition-completeness facts fixed (§5.2, §5.4).
- [ ] Cross-fold overlap of the pooled pool quantified and disclosed (§5.4, §0.14.2).
- [ ] Pooled multiplicity and effective sample size recorded (§0.14.3).
- [ ] Point-estimate binding table fixed: which array feeds which metric, and that no reported
      number may use a production prediction (§6.1).
- [ ] Leakage controls E4-L1…L28 fixed, with E4-L1 (§15.1) as the critical control and
      E4-L25…L28 as the revision-2 additions (§15.2).
- [ ] Coverage, width, interval-score and subgroup metrics fixed (§9–§11, §13).
- [ ] Subgroup suppression rule and band edges fixed (§13).
- [ ] Failure gates F0–F10 and hard stop conditions fixed (§18).
- [ ] Reproducibility identifiers and manifest schema fixed (§16), including removal of
      `predict_property_value.py` from the runtime reuse table (§16.1).
- [ ] Analytical computational budget completed; measured timing deferred to Phase 3 (§17).
- [ ] Limitations acknowledged, especially the served-model transfer limitation (§19, §3).
- [ ] No-result-prediction statement retained verbatim (§0 header, §1).

---

## 23. Status / Next action

**Status: DRAFT — design only. Revision 2.**

No model has been trained. No fold has been regenerated. No quantile has been computed. No
interval exists. No new dataset, fold file or artifact exists. No results are claimed.
This document is the only file created. No existing research file, model, artifact,
protocol, dataset, fold, application file, blockchain state or Git ref is modified by
authoring it. The production V5 model was not trained, refit, retuned or replaced.

**What revision 2 changed** (all clarification, none execution): the method was named
`POOLED-HOLDOUT-CONFORMAL`; the coverage guarantee was withdrawn with reasons
(§3.2, §5.4); the cross-fold overlap was quantified (§5.4); the point-estimate binding was
made explicit (§6.1); the reference feature transform was corrected from the serving
transform to the training transform (§0.13, §6.3, §16.1); E4-L1 was rewritten as eight
specific prohibitions (§15.1) and E4-L25…L28 were added (§15.2).

**Next action:** review this protocol; resolve the pre-registration checklist (§22);
record the protocol SHA-256; then authorise Phase 1 (pure code, no training) and Phase 2
(verification, no training), report their validation, and only then — separately and
explicitly — authorise Phase 3 training.

---

## Appendix A — MILLOW research chain position

```
Experiment 1   model comparison / selection        FROZEN  v1.3.0-experiment1-catboost-backbone
     ↓
Experiment 2   conformal uncertainty (CatBoost)    FROZEN  v1.4.1-experiment2-results
     ↓
Experiment 3   spatial / property relationships     FROZEN  v1.6.0-experiment3-results
     ↓
Experiment 4   POOLED-HOLDOUT-CONFORMAL calibration of the V5 XGBoost
                                                  DRAFT — this document
     ↓
application integration of a conformal interval    NOT IMPLEMENTED, NOT AUTHORISED
     ↓
decision policy / risk layer                       NOT IMPLEMENTED
     ↓
blockchain / escrow integration                    NOT IMPLEMENTED, OUT OF SCOPE
```

Experiment 4 occupies the same **uncertainty-estimation layer** as Experiment 2, but aimed
at the production model rather than the research backbone. It does not establish that any
blockchain policy, escrow rule or on-chain execution is correct or desirable: valuation
uncertainty and blockchain execution remain separate layers, and Experiment 4 does not
integrate with the chain (`chain_touched: false` is required in the manifest).
