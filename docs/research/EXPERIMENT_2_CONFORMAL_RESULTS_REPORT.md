# Experiment 2 — Conformal Uncertainty for MILLOW Property Valuation

**Results Report**

| | |
|---|---|
| Status | Final. Experiment 2 is complete, audited, committed and tagged. |
| Protocol | `docs/research/EXPERIMENT_2_CONFORMAL_PROTOCOL_DRAFT.md` |
| Protocol SHA-256 | `dff5d84da558f0d1025e6b757547ab2cdfe5870192e900163f3ab2429822ae33` |
| Git commit | `77247ed2eacd7e411ee8312ab6fb1d610243d7c8` |
| Git tag | `v1.4.0-experiment2-conformal` |
| Numerical source of truth | `artifacts/valuation/v2_2/conformal/analysis/` (Phase 5 tables) |

This report is documentation only. It introduces no new statistical analysis, changes no
decision, and does not modify the Experiment 2 implementation, models or artifacts. All
numbers are read directly from the Phase 5 artifacts named in the text. Where an outcome
is affected by a recorded protocol ambiguity, it is stated as such and is **not** presented
as a definitive experimental finding.

---

## 1. Executive Summary

**Research objective.** Experiment 2 asks whether split conformal prediction can attach
statistically calibrated, usefully sharp interval estimates to the frozen MILLOW V2.1
property-valuation model, and how that calibration behaves when evaluation moves from a
random held-out fold to a **location-held-out distribution shift**. Point-accuracy
degradation under location shift had already been established in Experiment 1; Experiment 2
does not re-litigate it. It tests whether *interval* calibration degrades as well, and by
how much.

**Experimental design.** The V2.1 dataset, target, 48-feature contract, preprocessing
pipeline and 5-fold contract are inherited unchanged. Because the frozen 5-fold structure
provides no admissible out-of-sample calibration set (Section 3), a **nested 3-way split**
was derived inside each outer training fold: `T'_k` fits the model, `C_k` supplies
calibration residuals, and `test_k` is the frozen validation fold used only for scoring.
Ten nested CatBoost models were fitted (5 random folds, 5 location-grouped folds), one per
outer fold and regime.

**Frozen CatBoost backbone.** The hyperparameter configuration is taken verbatim from
Experiment 1 and never tuned: `iterations=700`, `learning_rate=0.04`, `depth=7`,
`l2_leaf_reg=2.0`, `random_seed=42`, `loss_function=RMSE`.

**Nested conformal calibration.** Calibration is strictly per outer fold. Conformal scores
and exact finite-sample quantiles are computed on `log1p(price)` residuals from `C_k` only,
with the frozen rule `k = ceil((n+1)(1−α))` and the `k`-th order statistic `s_(k)` (no
interpolation). Intervals are transformed to INR at presentation time only.

**Methods evaluated (closed set).** `A` (absolute residual split conformal), `B1`
(normalized by predicted log-price), `B3` (`σ̂ ≡ 1` control, required to reproduce `A`),
`C1` (Mondrian on `source_city`), `C2` (Mondrian on a predicted-price band), and `C3`
(group-mean absolute residual with a single group-level quantile).

**Regimes.** `random` is the approximately exchangeable positive-control regime;
`location_grouped` is a distribution-shift stress test in which calibration and test
localities are disjoint by construction.

**Major observed findings.**

- In the `random` regime, marginal coverage is close to nominal for all six methods at
  all three nominal levels; the largest absolute coverage error is `0.0092` (C3 at 90%).
- In the `location_grouped` regime, coverage moves modestly for `A`, `B1`, `B3` and `C2`,
  and more for `C3`. The largest shift-induced coverage errors for `A` are at 90%
  (`−0.0083`), 80% (`−0.0023`) and 95% (`−0.0022`).
- Conditional (subgroup) coverage is strongly non-uniform across predicted price bands and
  across cities, even in the random regime, with expensive bands under-covered and cheap
  bands over-covered. This is visible in the descriptive subgroup tables.
- **Sharpness is the binding constraint.** At the predefined operational gate
  (`median_relative_width ≤ 2.00` at 90%), every method fails in both regimes (`0/5` folds
  pass); the closest method, `C1` in the random regime, has a fold-mean median relative
  width of `2.30`.
- The F1/F2/F6 decision rows are **affected by a recorded protocol ambiguity** (Section 10)
  and must not be read as definitive. F3, F4, F5 and F7 are not affected by that ambiguity.

**Important limitations** include the absence of any distribution-free coverage guarantee
under location shift, a nested model that is not the frozen backbone, a five-fold variance
estimate, group-level (C3) calibration power that is an order of magnitude below row-level
power, subgroup power limits, and the F1/F2/F6 ambiguity. These are stated in full in
Section 12.

No method is described here as "best" or "superior". The results are comparative and
descriptive.

---

## 2. Research Questions

The following questions are preserved from the frozen protocol (§2). RQ1 is the
**negative control**: it must pass, or the implementation is not interpretable. RQ3 is the
headline.

| ID | Question |
|---|---|
| **RQ1** | In the random-CV regime, does split conformal attain nominal marginal coverage at 80/90/95%? |
| **RQ2** | Does normalized conformal improve sharpness over split conformal without degrading coverage? |
| **RQ3** | Does marginal coverage degrade under location-grouped evaluation relative to random CV, at matched nominal level? |
| **RQ4** | Is degradation under location shift monotone in nominal level (80 → 90 → 95)? |
| **RQ5** | Is coverage non-uniform across cities within a single regime and nominal level? |
| **RQ6** | Is coverage non-uniform across price bands within a single regime and nominal level? |
| **RQ7** | Are intervals sharp enough to be actionable, in absolute INR and in relative terms? |
| **RQ8** | Do residuals exhibit structure (non-normality, heavy tails, heteroscedasticity) that would undermine a symmetric absolute-residual score? |
| **RQ9** | Does any pre-declared location-aware strategy recover coverage relative to plain split conformal under location shift? |
| **RQ10** | Are observed deviations from nominal statistically distinguishable from Monte-Carlo noise at the achieved calibration size? |

**The two regimes answer different kinds of question.** The `random` regime is the
**positive control** and the approximately exchangeable setting in which the conformal
guarantee is expected to hold; a failure there indicates an implementation or conceptual
error rather than a limitation of conformal prediction. The `location_grouped` regime is a
**distribution-shift stress test** in which calibration and test localities are disjoint by
construction, so exchangeability is broken deliberately.

**No distribution-free conformal validity is claimed under location shift.** Under
location-grouped evaluation, coverage in that regime is *measured*, not guaranteed.
Throughout this report, location-grouped numbers are phrased as "observed coverage under
location shift".

---

## 3. Experimental Protocol

### 3.1 V2.1 dataset and protocol inheritance

Everything in this subsection is read-only input inherited from Experiment 1 and reused
verbatim.

| Element | Value |
|---|---|
| Source dataset | `data/processed/MREID_property.csv` |
| Rows after cleaning and de-duplication | **28,398** |
| Duplicates removed | 737 exact on `['source_city','location','area','no_of_bedrooms','price']` |
| Target | `price`, modelled in `log1p(price)` |
| Prediction space | `log1p(price)` |
| Presentation | INR via `max(expm1(·), 0)` |
| Feature count | **48** (11 numeric + 37 one-hot; pipeline inputs 46, the 2 frequency columns derived inside the pipeline) |
| Group key | `source_city + '__' + location` |
| Random state | 42 |
| Outer folds | 5 per regime |

The 48-feature contract excludes the target (`46 + 2 == 48`, and `'price'` is not an input
feature). Location-frequency features are fitted on training rows only; an unseen locality
receives exactly `0.0`. This is a structural property of the location-grouped regime, not
a defect.

### 3.2 Frozen CatBoost configuration

| Parameter | Value |
|---|---|
| `iterations` | 700 |
| `learning_rate` | 0.04 |
| `depth` | 7 |
| `l2_leaf_reg` | 2.0 |
| `random_seed` | 42 |
| `loss_function` | `RMSE` |

No hyperparameter was tuned anywhere in Experiment 2.

### 3.3 Folds

Five random folds (`KFold(shuffle=True)`) and five location-grouped folds
(`GroupKFold(shuffle=True)` over `source_city__location`) are inherited unchanged from the
frozen V2.1 fold file. In the random regime, 96.8–97.7% of validation rows come from a
locality also present in training. In the location-grouped regime, there are **0 shared
groups** in every fold and 100% unseen localities.

### 3.4 Nested `T'/C/test` construction

For each regime and outer fold `k`:

```
train_k  (frozen; never the test fold)
  ├── T'_k  ⊂ train_k   fits the nested CatBoost (frequency, scaler, one-hot, model)
  └── C_k   ⊂ train_k   produces calibration residuals only; never fitted on
test_k   = frozen valid_k, unchanged; used for nothing but scoring
```

Hard invariants, asserted at runtime and in tests: `T'_k`, `C_k` and `test_k` are pairwise
disjoint; `T'_k ∪ C_k = train_k` exactly; `test_k` is bit-identical to the frozen
`valid_k`. In the location-grouped regime the stricter group invariants also hold:
`group(T'_k) ∩ group(C_k) = ∅`, `group(T'_k) ∩ group(test_k) = ∅`, and
`group(C_k) ∩ group(test_k) = ∅`. This makes the calibration set structurally identical to
the test set under shift — both are localities the nested model has never seen.

### 3.5 Calibration fraction and group structure

The calibration fraction is pre-registered at `cal_frac = 0.20`. Realised sizes:

| Regime | Fold | `n_fit` | `n_cal` | `n_test` | Calibration groups |
|---|---|---|---|---|---|
| random | 1–5 | 18,174 / 18,174 / 18,174 / 18,175 / 18,175 | 4,544 (all folds) | 5,680 / 5,680 / 5,680 / 5,679 / 5,679 | 898 / 880 / 871 / 882 / 906 |
| location_grouped | 1 | 19,295 | 4,245 | 4,858 | 287 |
| location_grouped | 2 | 19,025 | 3,668 | 5,705 | 287 |
| location_grouped | 3 | 18,015 | 5,369 | 5,014 | 287 |
| location_grouped | 4 | 16,266 | 5,143 | 6,989 | 287 |
| location_grouped | 5 | 18,412 | 4,154 | 5,832 | 287 |

Random folds realise exactly 20.0% calibration. Location-grouped folds realise
approximately 16–23% because the inner split is group-based. The grouping key has 1,789
groups over 28,398 rows, with median group size **3** (min 1, max 685); 64.2% of groups
have at least 2 rows.

### 3.6 Nominal levels

Fixed at **0.80, 0.90, 0.95**, i.e. `α ∈ {0.20, 0.10, 0.05}`.

### 3.7 Why nested fitting was required

The frozen 5-fold contract provides only `train_k` and `valid_k`. Conformal calibration
requires residuals from rows that were neither fitted on nor scored. Three facts rule out
any admissible source from the frozen artifacts alone: (a) the only rows `model_k` never
saw are `valid_k`, the evaluation fold; (b) any other persisted model `model_j` (`j ≠ k`)
has seen 100% of `valid_k` in training, so its residuals there are in-sample; and (c)
Jackknife+ needs `n = 5` clean predictions per test point but only one exists, while LOO
would require 28,398 refits. Finally, in-sample residuals measured on the frozen model were
about 5–6% optimistically small, so calibrating on them would fabricate under-coverage. A
nested split is therefore scientifically necessary; it is *derived from* `train_k` and
written to a **new** file, leaving the frozen fold file untouched. The cost is that the
nested model is not the frozen Experiment 1 backbone, and its intervals are wider (Section
12).

---

## 4. Conformal Methods

Notation: `α = 1 − nominal_coverage`. All methods operate in `log1p(price)` space; INR
intervals are produced only at presentation time. No method outside the closed set
`{A, B1, B3, C1, C2, C3}` is evaluated. (`B2`, a k-NN localized scale, and `C4`, weighted
conformal under covariate shift, were considered and deliberately excluded at Phase 0;
they are not part of Experiment 2.)

### 4.1 Method A — absolute-residual split conformal

Score `s_i = |y_i − ŷ_i|` (log space); constant-width interval `[ŷ − q̂_α, ŷ + q̂_α]`.
Requires only exchangeability of calibration and test scores.

### 4.2 Method B1 — normalized residual using predicted log-price

Score `s_i = |y_i − ŷ_i| / ŷ_i`, with `ŷ` the frozen predictor's own log-price output.
This targets the measured heteroscedasticity (residual scale grows with predicted price).
`ŷ` cannot be zero or negative on this dataset, but the implementation floors it explicitly
and would record any activation in the manifest (none occurred). Interval:
`[ŷ − q̂_α·ŷ, ŷ + q̂_α·ŷ]` — heterogeneous width.

### 4.3 Method B3 — `σ = 1` control

Identical to A by construction and used as an implementation self-test: if normalized
conformal with `σ̂ ≡ 1` does not reproduce A to `1e-12`, the normalization code is wrong.
B3 is a control, not a competing method.

### 4.4 Method C1 — source-city Mondrian

Method A's score computed **within each `source_city` stratum**, giving one quantile per
city. Coverage is guaranteed conditional on the category under per-category exchangeability.
`source_city` is a feature and is assignable at prediction time. All six cities appear in
both `C_k` and `test_k` in every fold. Width is constant within a city and heterogeneous
across cities.

### 4.5 Method C2 — predicted-price-band Mondrian

The same construction with strata defined by **predicted** price tertiles (low / mid / high).
Cutpoints are computed on `T'_k` predictions and applied unchanged to `C_k` and `test_k`;
because the stratum depends only on `x`, C2 is assignable at prediction time. The
**true**-price band is used only for diagnostic reporting and never to assign a stratum.

### 4.6 Method C3 — group-mean absolute residual with one group-level quantile

For each calibration group `g` (a `source_city__location` locality),
`s_g = mean(|y_i − ŷ_i|)` over the group's calibration rows. For each
`(regime, fold, level)`, exactly **one** finite-sample quantile is taken over the group
scores `{s_g}`, so `n = number of calibration groups`. The test interval is
`[ŷ − q, ŷ + q]`. With 287 calibration groups in the location-grouped folds,
`k = ceil((287+1)·0.95) = 274` at 95%; the tail rests on ~14 groups. C3 is secondary by
construction. Per-group quantiles, max-residual aggregation and pooled fallbacks are
explicitly not implemented.

### 4.7 Exact finite-sample quantile rule

For sorted calibration scores `s_(1) ≤ … ≤ s_(n)` and miscoverage `α`:

```
k = ceil( (n + 1) · (1 − α) )
q̂_α = s_(k)      if k ≤ n
    = +inf       if k > n     (interval reported as unbounded)
```

`numpy.quantile` linear interpolation is **not** used; the exact order statistic is taken,
because the finite-sample guarantee is stated for the `k`-th order statistic. The `k > n`
branch is implemented and unit-tested, although it was unreachable at the realised sizes
(all methods reported `0` infinite and `0` fallback intervals).

### 4.8 Log-space construction and inverse transformation

Scores and quantiles are computed on `log1p(price)` residuals. A log-space interval
`[ŷ − q, ŷ + q]` is presented in INR as
`[max(expm1(ŷ − q), 0), max(expm1(ŷ + q), 0)]`. For homogeneous methods the row-level
relative width is, to the usual approximation, `2·sinh(q)`.

### 4.9 C1/C2 fallback behavior

A required C1/C2 stratum that is unseen in `C_k`, or whose calibration size is
insufficient, receives `lower = −inf`, `upper = +inf`, `fallback_used = True`, where
"insufficient" means only `k > n_stratum`. A pooled `C_k` quantile is **not** substituted;
an unbounded interval is the honest statement that no calibrated interval exists for that
stratum. In the realised run no stratum was insufficient and no fallback occurred.

---

## 5. Implementation and Reproducibility

| Phase | Output |
|---|---|
| Phase 2 | Calibration-fold artifact (`calibration_folds_v2_2.json`): nested `fit`/`calibrate`/`test` index sets per regime and outer fold, with digests and provenance back to the frozen V2.1 fold file. |
| Phase 3 | Ten nested CatBoost models (5 random, 5 location-grouped), each with a sidecar meta file. |
| Phase 4 | Conformal scores and intervals: `conformal_scoring_manifest.json`, `scores/`, `intervals/`. 1,022,328 intervals total (170,388 per method). |
| Phase 5 | Coverage / sharpness / subgroup / decision analysis: `fold_metrics.csv` (180), `summary_metrics.csv` (288), `regime_shift_coverage.csv` (72), `subgroup_metrics.csv` (2700), `decision_table.csv` (59), 12 figures (24 files). |

**Verification.**

- **233 tests passed** for the Experiment 2 suite (`python -m pytest scripts/experiment_2/ -q`).
- Deterministic reproduction was verified: fixed figure geometry/DPI, no random jitter, no
  bootstrap, PDF creation dates suppressed; re-running reproduces byte-identical tables and
  figures. The Phase 5 deterministic content digest is
  `984ba0a0abf70c5441bb46e8d66971cf653c41329b37ce98eb6450be6ba8d7ac`.
- Protected state was clean before and after (151/151 protected entries unchanged); the
  immutable/data/chain/model files and the four pre-existing tags were not touched.
- The analysis manifest records `models_trained: 0`, `folds_regenerated: false`,
  `protocol_modified: false`, `composite_score_computed: false`, `models_ranked: false`.

The durable provenance record includes the tag `v1.4.0-experiment2-conformal` (commit
`77247ed`) and the protocol SHA-256
`dff5d84da558f0d1025e6b757547ab2cdfe5870192e900163f3ab2429822ae33`, which is also embedded
in `analysis_manifest.json`.

---

## 6. Coverage Results

Values are the mean across the 5 outer folds with the sample SD (`ddof = 1`) across folds,
computed from `fold_metrics.csv` at each nominal level (the protocol's existing fold-mean /
sample-SD convention; no new pooling is introduced). "Cov err" is `empirical − nominal`.
"Mis-low"/"Mis-high" are the fold-mean lower/upper miscoverage rates.

### 6.1 Random regime (positive control)

| Method | Nominal | Coverage mean (SD) | Cov err | Mis-low | Mis-high |
|---|---|---|---|---|---|
| A | 0.80 | 0.79992 (0.01964) | −0.00008 | 0.09789 | 0.10219 |
| A | 0.90 | 0.90056 (0.01680) | +0.00056 | 0.03669 | 0.06275 |
| A | 0.95 | 0.95021 (0.00884) | +0.00021 | 0.01299 | 0.03680 |
| B1 | 0.80 | 0.80002 (0.01967) | +0.00002 | 0.09888 | 0.10110 |
| B1 | 0.90 | 0.89982 (0.01743) | −0.00018 | 0.03733 | 0.06286 |
| B1 | 0.95 | 0.95007 (0.00878) | +0.00007 | 0.01211 | 0.03782 |
| B3 | 0.80 | 0.79992 (0.01964) | −0.00008 | 0.09789 | 0.10219 |
| B3 | 0.90 | 0.90056 (0.01680) | +0.00056 | 0.03669 | 0.06275 |
| B3 | 0.95 | 0.95021 (0.00884) | +0.00021 | 0.01299 | 0.03680 |
| C1 | 0.80 | 0.80073 (0.01881) | +0.00073 | 0.09765 | 0.10163 |
| C1 | 0.90 | 0.90211 (0.01350) | +0.00211 | 0.03701 | 0.06088 |
| C1 | 0.95 | 0.95207 (0.01034) | +0.00207 | 0.01180 | 0.03613 |
| C2 | 0.80 | 0.80076 (0.01781) | +0.00076 | 0.10163 | 0.09761 |
| C2 | 0.90 | 0.90035 (0.01252) | +0.00035 | 0.03877 | 0.06088 |
| C2 | 0.95 | 0.95148 (0.00881) | +0.00148 | 0.00873 | 0.03979 |
| C3 | 0.80 | 0.80347 (0.02367) | +0.00347 | 0.09585 | 0.10068 |
| C3 | 0.90 | 0.89084 (0.02337) | −0.00916 | 0.04212 | 0.06705 |
| C3 | 0.95 | 0.94155 (0.01342) | −0.00845 | 0.01634 | 0.04212 |

### 6.2 Location-grouped regime (distribution-shift stress test)

| Method | Nominal | Coverage mean (SD) | Cov err | Mis-low | Mis-high |
|---|---|---|---|---|---|
| A | 0.80 | 0.79773 (0.02493) | −0.00227 | 0.11076 | 0.09151 |
| A | 0.90 | 0.89167 (0.01690) | −0.00833 | 0.05111 | 0.05722 |
| A | 0.95 | 0.94777 (0.01363) | −0.00223 | 0.01516 | 0.03707 |
| B1 | 0.80 | 0.79855 (0.02381) | −0.00145 | 0.11061 | 0.09084 |
| B1 | 0.90 | 0.89256 (0.01563) | −0.00744 | 0.05058 | 0.05686 |
| B1 | 0.95 | 0.94781 (0.01325) | −0.00219 | 0.01447 | 0.03771 |
| B3 | 0.80 | 0.79773 (0.02493) | −0.00227 | 0.11076 | 0.09151 |
| B3 | 0.90 | 0.89167 (0.01690) | −0.00833 | 0.05111 | 0.05722 |
| B3 | 0.95 | 0.94777 (0.01363) | −0.00223 | 0.01516 | 0.03707 |
| C1 | 0.80 | 0.79390 (0.02193) | −0.00610 | 0.11286 | 0.09325 |
| C1 | 0.90 | 0.90036 (0.02086) | +0.00036 | 0.03949 | 0.06014 |
| C1 | 0.95 | 0.95076 (0.01896) | +0.00076 | 0.00915 | 0.04008 |
| C2 | 0.80 | 0.80247 (0.02368) | +0.00247 | 0.11051 | 0.08702 |
| C2 | 0.90 | 0.90199 (0.00728) | +0.00199 | 0.04257 | 0.05544 |
| C2 | 0.95 | 0.95413 (0.01368) | +0.00413 | 0.00719 | 0.03868 |
| C3 | 0.80 | 0.78552 (0.03017) | −0.01448 | 0.11794 | 0.09654 |
| C3 | 0.90 | 0.88369 (0.01769) | −0.01631 | 0.05661 | 0.05970 |
| C3 | 0.95 | 0.94438 (0.02378) | −0.00562 | 0.01774 | 0.03787 |

The pooled, cross-fold coverage shown in `summary_metrics.csv` is a single descriptive
number per method and regime (summarised below); the level-specific figures above are the
ones used for all criterion evaluations.

| Method | Random pooled coverage | Location-grouped pooled coverage |
|---|---|---|
| A | 0.88356 | 0.87906 |
| B1 | 0.88330 | 0.87971 |
| B3 | 0.88356 | 0.87906 |
| C1 | 0.88497 | 0.88167 |
| C2 | 0.88420 | 0.88620 |
| C3 | 0.87862 | 0.87120 |

---

## 7. Sharpness and Interval Score

Sharpness is reported from `fold_metrics.csv` at the 90% level (fold mean). Absolute widths
are in INR; the interval score is on `log1p(price)` per Gneiting & Raftery (2007). "Bounded"
and "fallback" counts are across all methods, folds and levels.

| Method | Regime | Median width (INR) | Median relative width | Mean relative width | Interval score (log) |
|---|---|---|---|---|---|
| A | random | 17,156,152 | 2.4585 | 2.4585 | 3.0393 |
| B1 | random | 16,886,018 | 2.4197 | 2.4390 | 3.0165 |
| B3 | random | 17,156,152 | 2.4585 | 2.4585 | 3.0393 |
| C1 | random | 15,293,828 | 2.3044 | 2.4693 | 2.9328 |
| C2 | random | 16,544,136 | 2.3705 | 2.4989 | 2.9654 |
| C3 | random | 16,320,295 | 2.3391 | 2.3391 | 3.0464 |
| A | location_grouped | 20,337,493 | 2.7201 | 2.7201 | 3.1846 |
| B1 | location_grouped | 20,136,700 | 2.6902 | 2.7071 | 3.1598 |
| B3 | location_grouped | 20,337,493 | 2.7201 | 2.7201 | 3.1846 |
| C1 | location_grouped | 18,484,999 | 2.6257 | 2.7178 | 3.0191 |
| C2 | location_grouped | 21,013,908 | 2.7612 | 2.8345 | 3.0492 |
| C3 | location_grouped | 19,701,758 | 2.6259 | 2.6259 | 3.1922 |

For every method in both regimes, `n_infinite = 0` and `n_fallback = 0` across all folds and
levels: every interval was bounded and every stratum was estimable.

### 7.1 F3 usability gate (`median_relative_width ≤ 2.00` at 90%)

| Method | Random median relative width | Location-grouped median relative width | Outcome |
|---|---|---|---|
| A | 2.4585 | 2.7201 | not_usable |
| B1 | 2.4197 | 2.6902 | not_usable |
| B3 | 2.4585 | 2.7201 | not_usable |
| C1 | 2.3044 | 2.6257 | not_usable |
| C2 | 2.3705 | 2.7612 | not_usable |
| C3 | 2.3391 | 2.6259 | not_usable |

**The criterion was not met.** No method passes at 90% in either regime, and for every
method `0/5` individual folds pass. The narrowest 90% band, `C1` in the random regime, is
still `2.30`, above the `2.00` gate. This threshold is an **Experiment-specific operational
threshold** declared in Phase 0; it is not a universal real-estate industry standard, and
the protocol explicitly recorded in advance that the homogeneous methods were expected to
fail it.

---

## 8. Random vs Location-Grouped Shift

Paired fold-level quantities are taken from `regime_shift_coverage.csv`; the sign convention
is `delta = grouped − random`, so a negative coverage delta means worse observed coverage
under shift. Mean deltas:

| Method | Δcoverage @0.80 | Δcoverage @0.90 | Δcoverage @0.95 | Δmean width (log) @0.90 | Δinterval score @0.90 |
|---|---|---|---|---|---|
| A | −0.00219 | −0.00888 | −0.00244 | +0.16067 | +0.14537 |
| B1 | −0.00147 | −0.00726 | −0.00226 | +0.16552 | +0.14334 |
| B3 | −0.00219 | −0.00888 | −0.00244 | +0.16067 | +0.14537 |
| C1 | −0.00683 | −0.00174 | −0.00131 | +0.13386 | +0.08626 |
| C2 | +0.00171 | +0.00164 | +0.00266 | +0.18947 | +0.08374 |
| C3 | −0.01795 | −0.00715 | +0.00284 | +0.18098 | +0.14587 |

Observed changes are small in absolute terms and are comparable to the between-fold
variability (per-fold deltas are given in the artifact). For the pooled methods `A`, `B1`
and `B3`, coverage is slightly lower under shift at every level, and intervals are wider;
the width increase is consistent with the harder held-out-locality prediction setting
observed in Experiment 1; Experiment 2 does not separately identify its causal source. The location-aware variants behave differently: `C1` and `C2`
largely hold their coverage, and `C2` is marginally higher under shift at all three levels.
`C3`, with the smallest effective calibration sample, shows the largest coverage movement.

These are paired, measured differences at the achieved calibration size and the fixed five
folds. No predictive or generalization claim is made beyond the measured folds.

---

## 9. Subgroup Analysis

Subgroup coverage is reported in `subgroup_metrics.csv` for `source_city` (RQ5) and
`true_price_band` (RQ6, diagnostic), alongside the assignable `predicted_price_band` used by
C2. The protocol's suppression rule applies: cells with `n < 100` are flagged
`coverage_reliable = false` and are not treated as evidence. Across the 2,700 subgroup
cells, 180 were suppressed under this rule.

### 9.1 Source city (fold-mean coverage at 90%)

| City | A random | A grouped | C1 random | C1 grouped |
|---|---|---|---|---|
| bangalore | 0.9343 | 0.9372 | 0.9107 | 0.9010 |
| chennai | 0.9288 | 0.9394 | 0.8997 | 0.9142 |
| delhi | 0.8608 | 0.7887 | 0.8945 | 0.8752 |
| hyderabad | 0.9927 | 0.9945 | 0.9114 | 0.8970 |
| kolkata | 0.9181 | 0.9327 | 0.9013 | 0.9030 |
| mumbai | 0.8385 | 0.8099 | 0.8992 | 0.8990 |

For the pooled method `A`, city coverage is markedly non-uniform: under-covered in mumbai
and delhi, over-covered in hyderabad. The city-conditional method `C1` substantially flattens
this profile, bringing the observed city coverage into approximately the 0.875–0.914 range in the location-grouped regime.

### 9.2 Price band (fold-mean coverage at 90%, method `A`)

| Band | Random | Location-grouped |
|---|---|---|
| <0.5cr | 0.8962 | 0.8494 |
| 0.5–1cr | 0.9961 | 0.9992 |
| 1–2cr | 0.9667 | 0.9895 |
| 2–4cr | 0.6786 | 0.7334 |
| 4–10cr | 0.2723 | 0.1933 |
| ≥10cr | 0.1055 | 0.0430 (suppressed, n ≈ 43 < 100) |

The predicted-price-band variant `C2` assigns by `ŷ`, so its own band coverage is more even;
the true-price-band table above is diagnostic only. The dominant pattern is that a single
global score over-covers cheap properties and severely under-covers expensive ones — the
heteroscedasticity the normalized method `B1` and the band method `C2` were designed to
address. The `4–10cr` band is reliable (`n ≈ 177`) and remains strongly under-covered under
every method.

These are descriptive, exploratory measurements. Per §15.7 of the protocol no multiplicity
correction is applied, and the F6 decision outcome is reported separately in Section 10.

---

## 10. F0–F7 Decision Results

The audited decision table (`decision_table.csv`, 59 rows) is reproduced here exactly as
frozen. Counts are per criterion across all its scopes.

| Criterion | Outcome | Count | Interpretation |
|---|---|---|---|
| F0 | pass | 3 | Implementation validity gate passed. |
| F1 | fail | 6 | Negative control: **ambiguity-affected** (see below). |
| F2 | inconclusive | 1 | Headline falsifier: **ambiguity-affected** (see below). |
| F3 | not_usable | 12 | Every method fails the sharpness gate in both regimes. |
| F4 | not_triggered | 12 | Overcoverage falsifier not triggered. |
| F5 | descriptive | 12 | Monotonicity reported descriptively, no pass/fail. |
| F6 | uniform | 12 | Uniformity falsifier: **ambiguity-affected / vacuous** (see below). |
| F7 | improved | 1 | C1/C2 improvement falsifier. |

### 10.1 The F1/F2/F6 fold-aggregation ambiguity (binding caveat)

The frozen protocol does **not** explicitly specify whether the Clopper–Pearson (CP) screen
used by F1, F2 and F6 is evaluated **independently per fold with an any-significant rule**
or on **pooled observations across folds**. The implementation used the **per-fold
`any()`** interpretation.

Under that implementation:

- **F1 (negative control, RQ1, random regime), fail ×6.** The reported per-level absolute
  coverage errors are small — for `A` at 80/90/95%: `0.0129 / 0.0127 / 0.0068` — but the
  per-fold any-significant screen flags a fail for every method.
- **F2 (headline falsifier, RQ3), inconclusive ×1.** For `A` in the location-grouped
  regime the per-level absolute coverage errors are `0.0023 / 0.0083 / 0.0022`, none above
  the `0.05` rejection threshold, so the hypothesis is not rejected; the any-fold CP screen
  does not allow a clean "not rejected" either, hence inconclusive.
- **F6 (uniformity), uniform ×12, vacuous.** The screen becomes vacuous because its
  condition — a level at which pooled coverage is calibrated — is never satisfied under the
  per-fold interpretation, so no subgroup cell can be counted.

**Because the alternative pooled interpretation changes F1 and F2 and makes F6 testable
rather than vacuous, these 19 rows are ambiguity-affected and must not be presented as
definitive experimental findings.** The pooled interpretation would, for example, leave
F1 non-failing for most methods (only `C3` exceeds the tolerance at 90/95%) and would make
F2 read "not rejected". The protocol does **not** retroactively select the pooled
interpretation; the `any()` interpretation is not retroactively declared intended. This
limitation is recorded in protocol §15.10. No F1/F2/F6 value in `decision_table.csv` has been
changed, and none is reinterpreted here.

### 10.2 F4 wording issue

The F4 text is internally contradictory: it refers to `abs_coverage_error ≤ 0.05` while also
stating `coverage_error ≥ +0.05`, which cannot both hold. The decision is robust to this —
F4 is `not_triggered` in all 12 cells — and the issue is a documentation defect, not a
result-changing ambiguity.

### 10.3 F3/F7 aggregation note

The protocol does not fully specify how per-fold/per-level quantities aggregate for F3 and
F7. F3 is robust: `0/5` folds pass at 90% for every method, so any reasonable aggregation
yields `not_usable`. F7 is evaluated on the pre-registered rule `min(C1, C2) < A` using the
fold-mean absolute coverage error under location shift: `A = 0.01439`, `C1 = 0.01540`,
`C2 = 0.01252`; since `min(C1, C2) = 0.01252 < 0.01439`, F7 is `improved`. This is a
documented aggregation under-specification, not an ambiguity that changes the outcome.

---

## 11. F0 Resolution

F0 is the implementation-validity gate and passed in all three of its scopes. The resolution
of the constant-width / homogeneity ambiguity is recorded as **Resolution 8.2** (Phase 5
pre-interpretation addendum, dated 2026-10-02).

- **A and B3 are globally homogeneous**: their log-space width is the constant `2·q̂_α`, so
  their global relative SD should be zero up to serialization error. The global
  constant-width check therefore applies **only** to A and B3.
- **C1 and C2 are homogeneous within strata, heterogeneous across strata**: their pooled
  global SD is expected to be non-zero even for a correct implementation, so they are *not*
  members of the global homogeneous set.
- **A/B3 global tolerance:** `relative_sd = sd_width / max(|mean_width|, ε) ≤ 1e-6`. The
  tolerance is a serialization artifact of `%.10g` persistence, **not** a scientific
  threshold.
- **C1/C2 within-stratum tolerance:** the same `1e-6` relative tolerance, applied within
  each `source_city` stratum (C1) and each predicted-price-band stratum (C2).
- **A/B3 equivalence verified exactly:** `max|lower_A − lower_B3| = 0` and
  `max|upper_A − upper_B3| = 0` (within `1e-12`).
- **270 C1/C2 strata checked, 0 within-stratum violations.**

Observed F0 values from the decision table:

| F0 scope | Metric | Value | Threshold | Outcome |
|---|---|---|---|---|
| global A/B3 constant width | max relative SD width (log) | 3.0721220109009763e-09 | 1e-06 | pass |
| global B3 reproduces A | max abs interval difference | 0.0 | 1e-12 | pass |
| C1/C2 within-stratum invariance | max within-stratum relative SD width | 7.1797541639760385e-09 | 1e-06 | pass |

Resolution 8.2 does not alter any coverage, sharpness, subgroup, interval-score or
method-selection value; it only defines the scope of the F0 check.

---

## 12. Limitations

1. **No distribution-free guarantee under location shift.** Exchangeability fails by
   construction in the location-grouped regime. Coverage there is measured, not guaranteed,
   and is reported as "observed coverage under location shift" throughout.
2. **The calibrated model is not the frozen backbone.** The nested `T'_k` is ~80% of
   `train_k`, so the nested CatBoost is weaker in point accuracy than the Experiment 1
   backbone and its intervals are wider. Results generalise to "a CatBoost with this
   configuration fitted on ~80% of `train_k`", not to the Experiment 1 backbone.
3. **Five folds is a weak variance estimate.** A sample SD from 5 points has very wide
   uncertainty. Fold-level values are always reported alongside aggregates; regime
   comparisons rely on paired per-fold deltas and sign consistency, not SD magnitude.
4. **Rows within a locality are not independent.** Properties in one locality share
   characteristics and localities recur across random folds, so binomial confidence
   intervals — including the CP bounds used for RQ10 — are anti-conservative. Clustered /
   block bootstrap by locality was declared out of scope rather than silently omitted. Any
   "significant undercoverage" statement is qualified accordingly.
5. **Group-level C3 calibration size.** Median group size is 3; location-grouped folds have
   only 287 calibration groups, so the 95% group-level quantile rests on ~14 tail groups.
   C3 is secondary by construction and is reported with its group count.
6. **Subgroup power.** Subgroup cells with `n < 100` are suppressed; 180 cells were
   suppressed. Even reliable cells rest on small samples, and no multiplicity correction is
   applied (§15.7), so subgroup findings are exploratory and descriptive.
7. **Nested models differ from the exact Experiment 1 fitted models**, by design (point 2).
8. **Protocol ambiguity affecting F1/F2/F6.** As detailed in Section 10.1, these 19 rows are
   ambiguity-affected and are not definitive.
9. **F4 wording defect.** F4's threshold text is internally contradictory; the decision is
   robust, but the wording is a documented defect.
10. **F3/F7 aggregation under-specification.** The protocol does not fully specify how
    per-fold/per-level quantities aggregate for F3 and F7. Both outcomes are robust to the
    choices described in Section 10.3, but the under-specification is recorded rather than
    hidden.
11. **Single inner split per outer fold.** One deterministic split, not repeated
    resampling: coverage carries unquantified inner-split variability and there is no
    split-averaging to stabilise `q̂_α`.
12. **One backbone.** CatBoost only; nothing here generalises to other model families or to
    conformal prediction in general for this dataset.

---

## 13. What the Experiment Establishes

### 13.1 Directly observed findings

- Ten nested models were fitted and evaluated without leakage: `T'_k`, `C_k` and `test_k`
  are disjoint, and `test_k` equals the frozen `valid_k` exactly.
- In the random regime, marginal coverage was close to nominal at all three levels for all
  six methods; the largest absolute coverage error was `0.0092` (C3 at 90%).
- In the location-grouped regime, observed coverage moved only modestly for A/B1/B3 and C2,
  and more for C3; the largest shift-induced error for A was at 90% (`−0.0083`).
- Conditional coverage was strongly non-uniform by price band and by city. Expensive bands
  were under-covered and cheap bands over-covered under the pooled methods; `C1` flattened
  the city profile.
- Sharpness failed the pre-registered operational gate for every method in every fold at
  90%.
- All F0 implementation checks passed, and A/B3 equivalence was exact.

### 13.2 Interpretations (supported by the observed measurements)

- Under the current nested backbone, calibration and usability diverge: the intervals are
  close to nominally calibrated in aggregate at the achieved sizes but are wider than the
  gate the protocol set for actionability.
- The subgroup results show strong dependence of coverage on price level, consistent with
  heteroscedastic residual scale.
- The location-grouped stress test did not produce a large marginal coverage collapse at
  the achieved calibration size; the point-accuracy loss established in Experiment 1 is not
  mirrored by a comparably large loss of *marginal interval* coverage here.

### 13.3 Future hypotheses (not established here)

- Conditional calibration — particularly for expensive properties and specific cities —
  may require score/scale construction beyond a single global quantile.
- Reducing interval width to an actionable level may require a different modelling or
  calibration strategy than the methods in the Experiment 2 closed set.

### 13.4 What is explicitly not claimed

This experiment does **not** claim universal conformal validity, production readiness,
legal or regulatory assurance, market-wide applicability, or the superiority of any method.
The F1/F2/F6 results are not presented as definitive. The F3 failure is reported as an
outcome on an Experiment-specific threshold, not as an industry verdict.

---

## 14. Reproducibility Checkpoint

| Component | Evidence | Hash / identifier |
|---|---|---|
| Protocol document | `docs/research/EXPERIMENT_2_CONFORMAL_PROTOCOL_DRAFT.md` | `dff5d84da558f0d1025e6b757547ab2cdfe5870192e900163f3ab2429822ae33` |
| Calibration folds | `artifacts/valuation/v2_2/conformal/calibration_folds_v2_2.json` | `47b89d8b41e98301129860cbf253f8c30255e272ec79f8277da54bb73acc6ea9` |
| Nested-model manifest | `artifacts/valuation/v2_2/conformal/nested_models_manifest.json` | `b2072d0b53f7475437ddeffc50ff3fe1eaf592261c289aa48c40e37d435b3475` |
| Phase 4 scoring manifest | `artifacts/valuation/v2_2/conformal/conformal_scoring_manifest.json` | `144bc2184c6dfb617a8bc80d778408f0f946211da8fae4a2b8ea331ca9d571b1` |
| Phase 4 content digest | `analysis_manifest.phase4_inputs.deterministic_content_digest` | `933268a54f78dcc98736f880b36f1f6f2da919f4ae3dc3503829c9371fbbd9f8` |
| Phase 5 deterministic digest | `analysis_manifest.deterministic_content_digest` | `984ba0a0abf70c5441bb46e8d66971cf653c41329b37ce98eb6450be6ba8d7ac` |
| Phase 5 analysis manifest file | `artifacts/valuation/v2_2/conformal/analysis/analysis_manifest.json` | `d6f4053903c6f3955003fe18ca8b9be5ea708f751751b43f5c3a98f6ebd8cbbf` |
| Decision table | `.../analysis/decision_table.csv` | `6f0c38f00ff825ee4cbacfbe36396ccca17fc12c8255782f62cfa5bc53cad494` |
| Git commit | `77247ed2eacd7e411ee8312ab6fb1d610243d7c8` | — |
| Git tag | `v1.4.0-experiment2-conformal` | tag object `c475f8b8236105f685aa27bef2bb10d9295aad95` |

The Phase 5 manifest also records `models_trained: 0`, `folds_regenerated: false`,
`protocol_modified: false`, `dataset_modified: false`, `chain_touched: false`, and
`protected_state_clean: true`.

---

## 15. Relation to MILLOW

Experiment 2 occupies the uncertainty-estimation layer of MILLOW, between point valuation
and any downstream policy or execution layer:

```
valuation                 Experiment 1 — frozen CatBoost V2.1 point model
    ↓
uncertainty estimation    Experiment 2 — conformal interval calibration (this report)
    ↓
risk / decision layer     not implemented here
    ↓
future policy / escrow integration   not implemented here
```

The experiment characterises the uncertainty of the **valuation** layer. It establishes
what interval widths and coverage levels are achievable around the inherited point model
under both an exchangeable and a location-shifted evaluation. It says nothing about whether
any blockchain policy, escrow rule or on-chain execution is correct or desirable:
valuation uncertainty and blockchain execution remain separate layers. The experiment itself
does not prove that a blockchain policy works, and it does not integrate with the chain
(`chain_touched: false` in the manifest).

---

## 16. Future Work

The next planned direction is **Experiment 3: spatial / property-relationship modelling**.
The Experiment 2 subgroup results — strong price-band and city-conditional miscalibration,
and a median locality size of 3 — motivate models that exploit spatial or property-level
relationships rather than a single row-level or category-level score. Experiment 3 is noted
here as future work only; it was not implemented, and no Experiment 3 analysis is presented.

---

## 17. Final Conclusion

**What was implemented.** A nested conformal pipeline around the frozen MILLOW V2.1
CatBoost backbone: a derived three-way `T'/C/test` split per outer fold and regime, ten
nested models, six closed-set conformal methods (`A`, `B1`, `B3`, `C1`, `C2`, `C3`), exact
finite-sample quantiles in log space, INR presentation, and a Phase 5 coverage / sharpness /
subgroup / decision analysis with a 59-row decision table and 24 figure files.

**What was measured.** In the random positive-control regime, marginal coverage was close to
nominal at 80/90/95% for all methods (largest absolute error `0.0092`). Under
location-grouped shift, observed marginal coverage moved only modestly for most methods,
while conditional coverage by price band and city was strongly non-uniform. At the
pre-registered 90% sharpness gate, every method in every fold failed; the narrowest band was
`C1` at a median relative width of `2.30` against a `2.00` gate.

**What was reproducibly verified.** The Experiment 2 suite passes 233 tests; deterministic
reproduction yields byte-identical tables and figures; protected state is clean; and the
full provenance chain is recorded, culminating in commit `77247ed2eacd7e411ee8312ab6fb1d610243d7c8`
and tag `v1.4.0-experiment2-conformal` at protocol SHA
`dff5d84da558f0d1025e6b757547ab2cdfe5870192e900163f3ab2429822ae33`.

**What remains uncertain.** Whether the marginal coverage observed under location shift
would hold at larger calibration sizes; whether the F1/F2/F6 outcomes would differ under the
pooled Clopper–Pearson interpretation; conditional calibration for expensive properties and
specific cities; and whether any admissible method can reach an actionable width under the
Experiment-specific gate.

**Why the ambiguity-affected results are retained rather than reclassified.** F1/F2/F6 are
affected by an ambiguity in the frozen protocol that predates the results. Changing them
now — under either interpretation — would be a post-hoc reclassification of a pre-registered
decision. They are therefore retained exactly as audited, clearly labelled
ambiguity-affected, and explicitly excluded from definitive findings. The remaining
criteria (F0, F3, F4, F5, F7) are unaffected and stand as reported.
