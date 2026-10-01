# MILLOW RESEARCH EXPERIMENT 1
## Model-Selection Analysis and Valuation Backbone Recommendation

Companion documents:

- `EXPERIMENT_1_V2_1_BASELINE.md` — the frozen V2.1 protocol
- `EXPERIMENT_1_MODEL_COMPARISON.md` — the five-model run record

This document is analysis and reporting only. No model was trained, loaded,
re-scored, re-predicted or modified in producing it. No fold was regenerated.
No protocol constant was changed. Every number quoted below was computed at
run time from the persisted Experiment 1 artifacts by
`scripts/valuation_v2_1/model_comparison_analysis.py`, which hard-codes no
metric value.

---

## 1. Objective

Experiment 1 ran five valuation algorithms under one frozen,
leakage-controlled protocol to answer two questions:

1. How do the five algorithms compare under identical folds, identical
   features and identical evaluation?
2. Which of them should become the frozen valuation backbone for Experiment 2,
   conformal prediction?

The second question has to be answered from the evidence in the first, under
criteria fixed before the evidence was examined. XGBoost was the original
Experiment 1 anchor. That status is not treated as evidence, and the analysis
below does not assume it should carry forward.

**Headline outcome: the evidence does not support freezing XGBoost. It
supports freezing CatBoost.** The reasoning, including the strongest
counter-arguments, is in section 12.

---

## 2. Experimental protocol

Unchanged from `EXPERIMENT_1_V2_1_BASELINE.md`.

| Property | Value |
|---|---|
| Feature count | 48 |
| Feature composition | 3 basic + 6 row-wise + 2 train-only frequency + 2 categorical + 35 amenities |
| Removed for leakage | `city_frequency`, `amenity_known_count`, `amenity_unknown_count` |
| Retained leakage controls | `amenity_yes_count`, `amenity_fully_specified` provenance flag |
| Target | `log1p(price)`, scored after explicit INR inversion |
| Rows | 28,398 (29,135 raw, 737 duplicates removed, 0 invalid removed) |
| Preprocessing | `TrainOnlyFrequencyFeatures` then `ColumnTransformer(StandardScaler on 11 numerics, OneHotEncoder(handle_unknown="ignore") on 37)` then estimator, all inside the `Pipeline` |
| Frequency fallback | 0.0, fitted on training folds only |
| Random seed | 42 |
| Tuning | none; no grid search, no random search, no early stopping |
| Fold pooling | never; all statistics are fold-level |

Fold assignments were loaded from the committed contract
`artifacts/valuation/v2_1/cv_folds_v2_1.json`. `KFold` and `GroupKFold` were
never called during the comparison, so all five models saw byte-identical
train and validation indices.

| Regime | Digest | Train sizes | Valid sizes |
|---|---|---|---|
| Random CV | `1a9f896345492d60` | 22718, 22718, 22718, 22719, 22719 | 5680, 5680, 5680, 5679, 5679 |
| Location-Grouped CV | `4c1c331cc993dded` | 23540, 22693, 23384, 21409, 22566 | 4858, 5705, 5014, 6989, 5832 |

The location-grouped folds are deliberately uneven, because `GroupKFold` on
this location distribution cannot produce equal-sized groups.

---

## 3. Models evaluated

| Model | Family | Parameters | Trained in this comparison |
|---|---|---|---|
| Ridge | linear, closed form | `alpha=1.0` | yes |
| Random Forest | bagged trees | `n_estimators=700, max_depth=7, min_samples_leaf=5` | yes |
| LightGBM | gradient boosting | 700 rounds, lr 0.04, depth 7, 127 leaves | yes |
| CatBoost | gradient boosting | 700 iterations, lr 0.04, depth 7 | yes |
| XGBoost | gradient boosting | 700 rounds, lr 0.04, depth 7 | **no — read from the V2.1 anchor** |

All three boosting models were deliberately matched to the frozen XGBoost
anchor on the three capacity axes: 700 boosting rounds, learning rate 0.04,
maximum depth 7. LightGBM's `num_leaves=127` is `2**7 - 1`, so the depth cap
binds as it does for XGBoost rather than becoming separately tunable. The
comparison is therefore not confounded by one model receiving a larger budget.

Ridge is included as a linear reference point. It has no rounds, learning rate
or depth, so the matched axes are not applicable to it. This is stated rather
than hidden, and it is why Ridge is not treated as a capacity-matched
competitor.

One documented accommodation: CatBoost rejects scipy sparse matrices, so its
preprocessor uses `dense_output=True` while the others use
`dense_output=False`. This changes no fitted statistic.

---

## 4. Evaluation regimes

**Regime A, Random CV.** Five shuffled `KFold` splits. Each validation fold
contains locations that also appear in the training folds. This measures
in-distribution interpolation, and is the regime a naive single-split
evaluation would report.

**Regime B, Location-Grouped CV.** Five `GroupKFold` splits grouped on
`source_city__location`. Every validation fold consists of locations held out
entirely from training. This measures generalization to unseen locations and
is the regime that matches the research question about distribution shift.

The contrast between A and B is the substantive result of Experiment 1, not
an inconvenience to be minimised.

---

## 5. Metrics

Eight metrics, reported as fold-level mean plus or minus sample standard
deviation (`ddof=1`, n=5), never pooled across folds.

| Space | Metric | Direction | Meaning |
|---|---|---|---|
| INR | `MAE_INR` | lower better | typical rupee miss |
| INR | `RMSE_INR` | lower better | penalty-weighted miss, tail-sensitive |
| INR | `R2_INR` | higher better | variance explained in rupees |
| INR | `MAPE_percent` | lower better | mean percentage miss |
| INR | `MedAPE_percent` | lower better | median percentage miss |
| Log | `MAE_log` | lower better | typical miss in `log1p` space |
| Log | `RMSE_log` | lower better | penalty-weighted miss in `log1p` space |
| Log | `R2_log` | higher better | variance explained in `log1p` space |

`R2_log` is the protocol's headline goodness-of-fit measure because the target
is heavy-tailed (max/min price ratio 427) and INR-space R-squared is
compressed by a handful of luxury rows. The three designated primary
interpretation metrics are `R2_log`, `MAE_log` and `MedAPE_percent`.

**MAPE is reported but is not used as selection evidence.** It divides by the
true price, so it is dominated by cheap properties, and it is roughly 18
percentage points worse than MedAPE for every model in both regimes, which
means a small number of low-price rows are driving the mean. MedAPE asks the
same question with a median denominator statistic and is far better behaved
here.

---

## 6. Comparative results

### 6.1 Thesis summary table

Mean plus or minus sample SD, n=5. Full source:
`artifacts/valuation/v2_1/model_comparison/thesis_summary_table.csv`.

| Model | Random MAE (INR) | Grouped MAE (INR) | Random RMSE (INR) | Grouped RMSE (INR) | Random R2 | Grouped R2 | Random MedAPE (%) | Grouped MedAPE (%) | Random R2 (log) | Grouped R2 (log) |
|---|---|---|---|---|---|---|---|---|---|---|
| Ridge | 6,424,188 +/- 60,269 | 7,265,955 +/- 1,007,969 | 22,877,472 +/- 1,009,963 | 23,209,097 +/- 6,199,027 | 0.1006 +/- 0.0130 | 0.0618 +/- 0.0277 | 33.5493 +/- 0.4620 | 45.3847 +/- 1.5882 | 0.3296 +/- 0.0128 | 0.1690 +/- 0.0473 |
| Random Forest | 6,393,402 +/- 57,662 | 6,790,468 +/- 1,010,099 | 22,821,827 +/- 1,019,964 | 23,178,559 +/- 6,155,139 | 0.1050 +/- 0.0114 | 0.0638 +/- 0.0193 | 33.8880 +/- 0.7229 | 37.7671 +/- 2.5157 | 0.3308 +/- 0.0085 | 0.2576 +/- 0.0359 |
| LightGBM | 6,565,300 +/- 107,614 | 7,467,731 +/- 849,597 | 22,650,910 +/- 971,577 | 23,311,262 +/- 6,001,041 | 0.1182 +/- 0.0131 | 0.0502 +/- 0.0191 | 35.6561 +/- 0.6735 | 45.3190 +/- 3.4349 | 0.3397 +/- 0.0105 | 0.1878 +/- 0.0618 |
| CatBoost | 6,240,437 +/- 67,114 | 6,743,352 +/- 1,029,867 | 22,634,024 +/- 898,103 | 23,175,859 +/- 6,209,108 | 0.1193 +/- 0.0163 | 0.0650 +/- 0.0237 | 32.5243 +/- 0.7431 | 37.2936 +/- 2.3122 | 0.3640 +/- 0.0106 | 0.2677 +/- 0.0378 |
| XGBoost | 6,104,840 +/- 75,523 | 6,915,120 +/- 959,890 | 22,526,294 +/- 1,013,868 | 23,066,011 +/- 6,199,253 | 0.1280 +/- 0.0169 | 0.0739 +/- 0.0287 | 30.5240 +/- 0.5713 | 38.9338 +/- 2.7274 | 0.3723 +/- 0.0098 | 0.2340 +/- 0.0575 |

### 6.2 Per-metric ordering by regime

Full source:
`artifacts/valuation/v2_1/model_comparison/per_metric_ordering_by_regime.csv`.

This is a descriptive lookup table. It is **not** a ranking, it carries no
score, no weights and no win count, and it exists to make one factual point
auditable: the ordering is not the same in the two regimes.

| Regime | Metric | 1st | 2nd | 3rd |
|---|---|---|---|---|
| Random CV | MAE (INR) | XGBoost | CatBoost | Random Forest |
| Random CV | RMSE (INR) | XGBoost | CatBoost | LightGBM |
| Random CV | R2 | XGBoost | CatBoost | LightGBM |
| Random CV | MAPE (%) | XGBoost | CatBoost | Random Forest |
| Random CV | MedAPE (%) | XGBoost | CatBoost | Ridge |
| Random CV | MAE (log) | XGBoost | CatBoost | Ridge |
| Random CV | RMSE (log) | XGBoost | CatBoost | LightGBM |
| Random CV | R2 (log) | XGBoost | CatBoost | LightGBM |
| Location-Grouped CV | MAE (INR) | CatBoost | Random Forest | XGBoost |
| Location-Grouped CV | RMSE (INR) | XGBoost | CatBoost | Random Forest |
| Location-Grouped CV | R2 | XGBoost | CatBoost | Random Forest |
| Location-Grouped CV | MAPE (%) | CatBoost | Random Forest | XGBoost |
| Location-Grouped CV | MedAPE (%) | CatBoost | Random Forest | XGBoost |
| Location-Grouped CV | MAE (log) | CatBoost | Random Forest | XGBoost |
| Location-Grouped CV | RMSE (log) | CatBoost | Random Forest | XGBoost |
| Location-Grouped CV | R2 (log) | CatBoost | Random Forest | XGBoost |

XGBoost is first on all eight random-regime metrics and CatBoost is second on
all eight. Under location hold-out, CatBoost is first on six of eight and
XGBoost is first on two, both of them INR-space metrics discussed in 6.3.

### 6.3 Which metrics can actually separate these models

A metric is only useful for selection if the spread between models is larger
than the fold-to-fold wobble within a model. Define the discrimination ratio
as

```
discrimination ratio = (max model mean - min model mean)
                       / (mean of the five within-model fold SDs)
```

A ratio below 1 means the entire spread between the five models is smaller
than one model's fold-to-fold variation, so the ordering on that metric cannot
be trusted at n=5 however it is arranged. Full source:
`artifacts/valuation/v2_1/model_comparison/metric_discrimination.csv`.

| Regime | Metric | Best model | Spread | Mean fold SD | Ratio | Separates models? |
|---|---|---|---|---|---|---|
| Random CV | MAE (INR) | XGBoost | 460,461 | 73,636 | 6.25 | yes |
| Random CV | RMSE (INR) | XGBoost | 351,178 | 982,695 | 0.36 | **no** |
| Random CV | R2 | XGBoost | 0.0274 | 0.0141 | 1.94 | yes |
| Random CV | MAPE (%) | XGBoost | 4.17 | 1.05 | 3.98 | yes |
| Random CV | MedAPE (%) | XGBoost | 5.13 | 0.63 | 8.09 | yes |
| Random CV | MAE (log) | XGBoost | 0.0282 | 0.0064 | 4.42 | yes |
| Random CV | RMSE (log) | XGBoost | 0.0217 | 0.0078 | 2.80 | yes |
| Random CV | R2 (log) | XGBoost | 0.0427 | 0.0104 | 4.09 | yes |
| Location-Grouped CV | MAE (INR) | CatBoost | 724,380 | 971,484 | 0.75 | **no** |
| Location-Grouped CV | RMSE (INR) | XGBoost | 245,251 | 6,152,713 | 0.04 | **no** |
| Location-Grouped CV | R2 | XGBoost | 0.0237 | 0.0237 | 1.00 | **no** |
| Location-Grouped CV | MAPE (%) | CatBoost | 11.90 | 7.42 | 1.60 | yes |
| Location-Grouped CV | MedAPE (%) | CatBoost | 8.09 | 2.52 | 3.22 | yes |
| Location-Grouped CV | MAE (log) | CatBoost | 0.0565 | 0.0233 | 2.43 | yes |
| Location-Grouped CV | RMSE (log) | CatBoost | 0.0454 | 0.0234 | 1.94 | yes |
| Location-Grouped CV | R2 (log) | CatBoost | 0.0987 | 0.0481 | 2.05 | yes |

This table carries much of the argument in section 12, so it is worth being
explicit about what it says:

- Under location hold-out, the **only** two metrics on which XGBoost leads are
  `RMSE_INR` and `R2_INR`, and those are precisely the two metrics with the
  weakest discriminative power in that regime. The entire spread between all
  five models on grouped `RMSE_INR` is 245,251 rupees, which is **4% of a
  single model's fold SD** of 6.15 million. On grouped `R2_INR` the spread is
  0.0237 against a mean fold SD of 0.0237, a ratio of 1.00.
- Every metric that *can* separate the models under location hold-out —
  MedAPE, MAE (log), RMSE (log), R2 (log), MAPE — puts CatBoost first.
- The cause is visible in the per-fold results: grouped fold 3 has
  `RMSE_INR` near 32.5 million and fold 5 near 26 million for *every* model,
  while folds 1, 2 and 4 sit between 17.5 and 21 million. That 15-million
  spread is location heterogeneity in the validation sets, not model
  behaviour, and it swamps any between-model difference on a squared-error
  metric.
- The same applies to grouped `MAE_INR` (ratio 0.75), where CatBoost happens
  to lead anyway.

This is not a reason to discard INR-space metrics. It is a reason not to let
them decide the selection.

---

## 7. Random-split analysis

Under random CV the five models separate cleanly and the ordering is stable
across all eight metrics:

- XGBoost first on all eight; CatBoost second on all eight.
- XGBoost's margin over CatBoost is modest but consistent: `R2_log` 0.3723 vs
  0.3640 (+0.0083), `R2_INR` 0.1280 vs 0.1193 (+0.0087), `MAE_log` 0.4531 vs
  0.4658 (2.7% lower), `MedAPE` 30.52% vs 32.52% (2.00 pp lower), `MAE_INR`
  6,104,840 vs 6,240,437 (2.2% lower), `RMSE_INR` 22,526,294 vs 22,634,024
  (0.5% lower).
- Fold-to-fold variability is small for every model here: `R2_log` fold SD
  ranges from 0.0085 (Random Forest) to 0.0128 (Ridge).
- LightGBM is the only model that is neither first nor second on the
  discriminating metrics in this regime; it is last on `MAE_INR` and on
  `MedAPE`.

The random regime is a fair summary of interpolation ability. XGBoost is the
best interpolator of the five on this evidence, and it is the only regime in
which that claim rests on metrics strong enough to carry it.

---

## 8. Location-grouped analysis

Under location hold-out the picture changes:

- CatBoost is first on six of eight metrics, including all three
  protocol-primary metrics: `R2_log` 0.2677 vs XGBoost 0.2340 (+14.4%
  relative), `MAE_log` 0.5128 vs 0.5274 (2.8% lower), `MedAPE` 37.29% vs
  38.93% (1.64 pp lower).
- XGBoost is first on `RMSE_INR` (23,066,011 vs 23,175,859, a 0.5% margin)
  and `R2_INR` (0.0739 vs 0.0650), both non-discriminating metrics per 6.3.
- Random Forest is second on six of eight, never first, and never worse than
  third.
- Linear and shallow-leaf models degrade furthest: Ridge falls to `R2_log`
  0.1690 and `MedAPE` 45.38%, both worst in the regime. LightGBM, despite
  sharing XGBoost's exact capacity budget, is also near the bottom
  (`R2_log` 0.1878, `MedAPE` 45.32%). The LightGBM *configuration* is the
  plausible cause rather than the boosting paradigm itself, and Experiment 1
  cannot separate those two explanations because only one LightGBM
  configuration was tried.
- Absolute performance is low for every model: the best grouped `R2_log` is
  0.2677. Prices in locations never seen during training are largely
  unpredictable from these 48 features. That is a finding about the
  achievable ceiling, not a defect of any one model.

---

## 9. Generalization / degradation analysis

Change is computed as **Location-Grouped CV minus Random CV**, from persisted
means. Folds are *not* paired between regimes — the two regimes use different
splits with different validation sizes — so the change is an
independent-samples quantity and its interval uses the Welch–Satterthwaite
formula on the difference of fold means. No paired t-difference is used
anywhere. Full source:
`artifacts/valuation/v2_1/model_comparison/regime_shift_all_metrics.csv`.

### 9.1 Change on every required metric

| Model | d MAE (INR) | d RMSE (INR) | d R2 (INR) | d MedAPE (pp) | d MAE (log) | d RMSE (log) | d R2 (log) |
|---|---|---|---|---|---|---|---|
| Ridge | +841,767 (+13.1%) | +331,625 (+1.4%) | -0.0388 | +11.84 (+35.3%) | +0.0915 (+19.2%) | +0.0748 (+11.1%) | -0.1606 |
| Random Forest | +397,066 (+6.2%) | +356,732 (+1.6%) | -0.0412 | +3.88 (+11.4%) | +0.0357 (+7.4%) | +0.0348 (+5.2%) | -0.0732 |
| LightGBM | +902,431 (+13.7%) | +660,352 (+2.9%) | -0.0680 | +9.66 (+27.1%) | +0.0781 (+16.3%) | +0.0711 (+10.7%) | -0.1519 |
| CatBoost | +502,914 (+8.1%) | +541,835 (+2.4%) | -0.0544 | +4.77 (+14.7%) | +0.0470 (+10.1%) | +0.0468 (+7.2%) | -0.0963 |
| XGBoost | +810,280 (+13.3%) | +539,717 (+2.4%) | -0.0541 | +8.41 (+27.6%) | +0.0742 (+16.4%) | +0.0667 (+10.3%) | -0.1384 |

Relative change is given only for ratio-scale strictly positive quantities.
It is not reported for `R2_INR` or `R2_log`, which are not ratio scales.

### 9.2 Which changes are distinguishable from fold noise

95% Welch intervals on the change, n=5 per regime:

| Metric | Ridge | Random Forest | LightGBM | CatBoost | XGBoost |
|---|---|---|---|---|---|
| MAE (INR) | -0.41M to +2.09M | -0.86M to +1.65M | -0.15M to +1.95M | -0.77M to +1.78M | -0.38M to +2.00M |
| RMSE (INR) | -7.31M to +7.98M | -7.23M to +7.95M | -6.74M to +8.06M | -7.13M to +8.21M | -7.11M to +8.19M |
| R2 | [-0.073, -0.005] | [-0.065, -0.017] | [-0.093, -0.044] | [-0.085, -0.024] | [-0.090, -0.018] |
| MedAPE (pp) | [+9.89, +13.78] | [+0.80, +6.96] | [+5.44, +13.89] | [+1.95, +7.59] | [+5.06, +11.76] |
| MAE (log) | [+0.072, +0.111] | [+0.007, +0.065] | [+0.048, +0.109] | [+0.018, +0.076] | [+0.040, +0.109] |
| RMSE (log) | [+0.056, +0.094] | [+0.004, +0.066] | [+0.044, +0.099] | [+0.015, +0.079] | [+0.033, +0.100] |
| R2 (log) | [-0.219, -0.103] | [-0.117, -0.029] | [-0.228, -0.076] | [-0.143, -0.050] | [-0.209, -0.068] |

Two honest readings of this table:

1. The location-shift penalty is **detectable with confidence** for every
   model on `R2_INR`, `MedAPE`, `MAE_log`, `RMSE_log` and `R2_log`; none of
   those intervals includes zero.
2. The location-shift penalty in `MAE_INR` and `RMSE_INR` is **not
   distinguishable from zero at n=5** for any model, because the fold-to-fold
   standard deviation in those units (6.2M for grouped RMSE) is an order of
   magnitude larger than the shift itself. The point estimates are positive
   for every model and consistent with the other metrics; they simply cannot
   be resolved with five folds. Claiming a significant rupee-denominated shift
   here would overstate the evidence.

### 9.3 What the shift indicates about generalization

- The regime gap is real and universal: no model transfers to unseen
  locations at anything close to its in-distribution accuracy. Best grouped
  `R2_log` is 0.2677 against a best random `R2_log` of 0.3723.
- The gap is **model-dependent**, and it does not track the random-regime
  ordering. XGBoost had the best random accuracy and the third-largest
  `R2_log` degradation (-0.1384), behind Random Forest (-0.0732) and CatBoost
  (-0.0963). Random-regime skill did not buy location-transferable skill.
- CatBoost's degradation is roughly two-thirds of XGBoost's on the
  discriminating metrics: `R2_log` -0.0963 vs -0.1384, `MedAPE` +4.77 pp vs
  +8.41 pp, `MAE_log` +0.0470 vs +0.0742, `MAE_INR` +502,914 vs +810,280.
  On `RMSE_INR` and `R2_INR` the two are effectively tied (+541,835 vs
  +539,717; -0.0544 vs -0.0541).
- Random Forest is the most robust model in relative terms and also the
  weakest of the three tree ensembles on absolute accuracy in both regimes.
  Robustness and accuracy are genuinely separate axes here, and neither alone
  settles the choice.
- The experiment supports the statement that location identity carries
  information these features do not capture. It does **not** support any claim
  about *why* individual locations are harder, or about which feature families
  carry the location signal. That would need a per-location error
  decomposition, which was not part of Experiment 1.

---

## 10. Fold variability

Figure 7 plots the persisted per-fold `R2_log` values, not regenerated data.
Full source: `artifacts/valuation/v2_1/model_comparison/fold_stability.csv`.

| Model | Random fold SD | Grouped fold SD | SD inflation under location hold-out |
|---|---|---|---|
| Ridge | 0.0128 | 0.0473 | 3.70x |
| Random Forest | 0.0085 | 0.0359 | 4.20x |
| LightGBM | 0.0105 | 0.0618 | 5.88x |
| CatBoost | 0.0106 | 0.0378 | **3.59x** |
| XGBoost | 0.0098 | 0.0575 | **5.87x** |

Findings:

- Every model's fold-to-fold variability inflates by 3.6x to 5.9x when moving
  from random to location-held-out evaluation. No model has a per-fold
  estimate that is equally reproducible in both regimes.
- CatBoost and XGBoost are almost identical in the random regime (0.0106 vs
  0.0098) and then diverge sharply: CatBoost's grouped SD is 0.0378 against
  XGBoost's 0.0575, and the inflation factors are 3.59x versus 5.87x.
- XGBoost has the **second-largest** grouped fold variability of the five
  models. Random Forest (0.0359) and CatBoost (0.0378) are the two most stable
  under location hold-out; LightGBM (0.0618) is the least.
- The grouped per-fold `R2_log` values also show where the difference between
  CatBoost and XGBoost comes from. On fold 3 they are essentially identical
  (0.3091 vs 0.3096). The whole CatBoost advantage sits in folds 1, 2, 4 and
  5, and is largest on fold 4 (0.2413 vs 0.1669).
- Observed grouped `R2_log` ranges, taken as true extrema of the five persisted
  fold values rather than as a multiple of the SD:

  | Model | Random range | Grouped range |
  |---|---|---|
  | Ridge | 0.3101 – 0.3453 | 0.1299 – 0.2411 |
  | Random Forest | 0.3191 – 0.3387 | 0.2122 – 0.2991 |
  | LightGBM | 0.3222 – 0.3486 | 0.1236 – 0.2866 |
  | CatBoost | 0.3505 – 0.3732 | 0.2191 – 0.3091 |
  | XGBoost | 0.3592 – 0.3834 | 0.1669 – 0.3096 |

  CatBoost's worst grouped fold (0.2191) is still above the mean of all five
  models' worst grouped folds, and above XGBoost's mean grouped `R2_log` of
  0.2340 is not — XGBoost's floor of 0.1669 is pulled down by two folds while
  its ceiling of 0.3096 is the best in the regime. The models differ in the
  shape of their failure, not only in their average.
- Figure 6 shows why grouped `RMSE_INR` cannot be read naively: fold 3 sits
  near 32.5 million for all five models and fold 5 near 26 million, against
  17.5M to 21M for folds 1, 2 and 4.

---

## 11. Model-selection criteria

Six criteria, fixed before this analysis was performed, matching the
selection dimensions set out for Experiment 1. No weights were assigned, no
composite score was constructed, and no win counting was performed.

| # | Criterion | Evidence consulted | Status |
|---|---|---|---|
| A | Absolute valuation error (MAE, RMSE) | 6, 7, 8, 9 | discriminates in random CV; not in grouped CV |
| B | Relative valuation accuracy (MedAPE; MAPE with caveats) | 6.3, 8 | MAPE excluded as selection evidence; MedAPE discriminates in both regimes |
| C | Predictive performance (R2_INR, R2_log) | 6.3, 7, 8 | R2_log discriminates in both regimes; grouped R2_INR does not |
| D | Generalization to unseen locations | 9 | all models shift; magnitude is model-dependent and does not follow random-regime order |
| E | Stability (fold-to-fold SD) | 10 | CatBoost and Random Forest most stable under hold-out |
| F | Suitability as backbone for conformal prediction | 10, 12 | depends on a residual scale that is stable across locations |

### 11.1 Random-regime strengths

XGBoost. First on all eight metrics, with CatBoost the only close challenger,
and seven of the eight metrics discriminate the models strongly enough to
support that claim. This is a genuine and well-evidenced result, and it is
stated here without qualification.

### 11.2 Location-grouped strengths

CatBoost. First on six of eight metrics, on every metric that discriminates
the models in that regime, and on all three protocol-primary metrics. Random
Forest is the most shift-robust in relative terms and second on six of eight
grouped metrics.

### 11.3 Trade-offs

- XGBoost trades location-transferability for in-distribution accuracy, and the
  evidence for that trade is consistent: it wins the regime it is better at and
  loses the regime the research question is about.
- Random Forest trades absolute accuracy for robustness. It is the most
  shift-stable model and the weakest of the three ensembles on level, in both
  regimes. Choosing it would accept roughly 0.010 of grouped `R2_log` and
  0.004 of grouped `MAE_log` relative to CatBoost to gain a slightly larger
  `R2_log` degradation margin.
- CatBoost gives up the best random-regime numbers of the five. That is a real
  cost and is not minimised here.
- LightGBM matched XGBoost's capacity budget exactly and still came last on
  several discriminating metrics, which is a reminder that a single
  pre-declared configuration per family cannot isolate the family effect from
  the configuration effect.
- MAPE cannot be used to break any of these ties, for the reasons in section 5.
- Grouped `RMSE_INR` and grouped `R2_INR` cannot be used either, for the
  reasons in section 6.3.

---

## 12. Final model-selection justification

### 12.1 Recommendation

**CatBoost is recommended as the valuation backbone for Experiment 2,
conformal prediction. XGBoost is not recommended, and its Experiment 1 anchor
role is explicitly not part of the justification.**

### 12.2 The evidence that supports CatBoost

Every item below is a directly reported number from the persisted artifacts.

**On criterion C, predictive performance, under the regime that matches the
research question:** CatBoost leads all three protocol-primary metrics under
location hold-out — `R2_log` 0.2677 vs 0.2340, `MAE_log` 0.5128 vs 0.5274,
`MedAPE` 37.29% vs 38.93%.

**On criterion B, relative accuracy:** grouped `MedAPE` 37.29% vs 38.93%, and
grouped `MAPE` 59.35% vs 64.82%, on a metric whose grouped discrimination
ratio is 3.22 and 1.60 respectively — among the most reliable discriminators
available in that regime.

**On criterion D, generalization:** CatBoost's `R2_log` degradation is
-0.0963 against XGBoost's -0.1384, a 30% smaller loss of explanatory power
under location shift. Its `MedAPE` degradation is +4.77 pp against +8.41 pp,
a 43% smaller relative penalty. Its `MAE_log` degradation is +0.0470 against
+0.0742.

**On criterion E, stability:** CatBoost's grouped `R2_log` fold SD is 0.0378
against XGBoost's 0.0575. Measured as inflation from the random regime,
CatBoost's fold SD grows 3.59x while XGBoost's grows 5.87x. XGBoost has the
second-worst grouped fold variability of the five models.

**On the metrics where XGBoost does lead in the grouped regime:** those are
`RMSE_INR` (23,066,011 vs 23,175,859, a 0.5% margin) and `R2_INR` (0.0739 vs
0.0650). Their discrimination ratios are 0.04 and 1.00. The full
between-model spread on grouped `RMSE_INR` is 4% of a single model's fold
standard deviation. These two orderings cannot be distinguished from fold
noise at n=5, and they are the only grouped metrics where XGBoost leads.

### 12.3 The strongest argument against this recommendation

XGBoost is genuinely the best model in this experiment under random CV. It is
first on all eight metrics, and on seven of them the discrimination ratio
exceeds 1.9, so that result is not a fold-noise artefact. If Experiment 2
calibrated conformal intervals under random cross-validation only, and cared
nothing about unseen locations, XGBoost would be the correct backbone and
CatBoost's random-regime numbers (32.52% vs 30.52% MedAPE, 0.3640 vs 0.3723
`R2_log`) would be an acceptable price to pay.

The reason that does not apply is that the research question is explicitly
about distribution shift across locations, and conformal validity is a
statement about residual behaviour on exchangeable data from the deployment
distribution. A backbone whose residual scale inflates 5.87x when the
evaluation moves to unseen locations is a worse foundation for interval
guarantees than one whose residual scale inflates 3.59x, at a cost of
0.008 `R2_log` in-distribution.

### 12.4 Why this matters specifically for conformal prediction

Split or cross-conformal interval width at a given coverage level is driven by
a high quantile of the absolute residual. That quantile is only trustworthy if
the residual distribution is reasonably stable between the calibration data
and the test data. CatBoost's grouped fold-to-fold variability is the second
lowest of the five and its inflation factor is the lowest, so its residual
scale is the most reproducible of the three ensembles under the shift that
matters. XGBoost's grouped variability is 52% larger than CatBoost's
(0.0575 vs 0.0378).

This is a proxy argument, not a demonstration. Residual exchangeability is a
property that must be tested directly in Experiment 2, and it was not tested
here.

### 12.5 Alternative that was considered and rejected

**Random Forest** is the most shift-robust model in relative terms
(`R2_log` -0.0732) and has the lowest grouped fold SD (0.0359). It was not
selected because it is worse than CatBoost on every level metric in both
regimes — grouped `R2_log` 0.2576 vs 0.2677, grouped `MAE_log` 0.5171 vs
0.5128, grouped `MedAPE` 37.77% vs 37.29%, random `R2_log` 0.3308 vs 0.3640 —
and its robustness advantage is smaller than CatBoost's absolute accuracy
advantage. Trading measurable accuracy for a second-order robustness margin is
not supported by the evidence.

**LightGBM** and **Ridge** are excluded on the evidence without ambiguity:
LightGBM is last or near-last on several discriminating metrics in both
regimes despite an exactly matched capacity budget, and Ridge is last in the
grouped regime on every metric.

### 12.6 What additional analysis would be required to overturn this

Stated explicitly so the recommendation is falsifiable:

1. **A pre-declared cross-model significance test** under location hold-out.
   The Welch intervals in 9.2 test regime shift *within* each model, not
   CatBoost against XGBoost. A paired or unpaired cross-model test on the same
   folds, declared in advance, could reverse the ordering if the CatBoost
   advantage is within noise. At n=5 with these fold SDs, it plausibly is not
   resolvable, which is itself worth reporting.
2. **Repeated or nested cross-validation** to separate the model-family effect
   from the single-pre-declared-configuration effect. LightGBM's poor showing
   under an exactly matched budget shows this confound is real.
3. **Residual diagnostics** — heteroscedasticity by location, residual
   quantile stability across folds, and coverage of nominal split-conformal
   bands under location shift. This is the decisive evidence for Experiment 2
   and does not exist yet.
4. **A second seed.** Every result here is single-seed. With a seed sweep, the
   fold SDs reported would become seed-averaged quantities and the grouped
   orderings could move.

Until at least items 1 and 3 exist, this recommendation should be read as
"best-supported by available evidence", not "established".

---

## 13. Limitations

1. **One configuration per model family, no search.** The spread between
   families conflates the family with the particular arbitrary configuration
   chosen a priori. LightGBM's result is the clearest symptom.
2. **n=5 folds per regime.** Grouped `RMSE_INR` and `R2_INR` cannot separate
   the models at all (discrimination ratios 0.04 and 1.00), and the INR-denominated
   regime shift is not resolvable at this sample size.
3. **Single seed (42).** No seed variance is quantified.
4. **Two regimes only.** No intermediate shift level, and no
   leave-one-city-out design, so "how much shift" and "which kind of shift"
   cannot be separated.
5. **Absolute performance is low for every model.** Best grouped `R2_log` is
   0.2677. Conclusions are about relative model behaviour under a shared
   ceiling, not about achievable valuation accuracy.
6. **Residual behaviour was not studied.** Conformal suitability is inferred
   from error and stability aggregates, not measured.
7. **The 48-feature set is fixed.** No claim is made about feature selection,
   and the LightGBM result in particular should not be read as a verdict on
   LightGBM as a library.
8. **MAPE is reported but not used** for selection, for the reasons in
   section 5.
9. **No causal claims.** The experiment shows that accuracy falls when
   locations are held out, and that the size of that fall differs by model. It
   does not show why.
10. **Historical V2 remains non-comparable.** V2 and V2.1 differ in leakage
    control, duplicates, feature count and evaluation design; no
    better-or-worse claim is made between them.

---

## 14. Transition to Experiment 2

### 14.1 Backbone to freeze

**CatBoost**, with the exact pre-declared configuration already persisted at
`models/valuation/v2_1/model_comparison/CatBoost/`:

```
iterations         = 700
learning_rate      = 0.04
depth              = 7
l2_leaf_reg        = 2.0
random_seed        = 42
dense_output       = True   (CatBoost cannot consume scipy sparse input)
```

inside the frozen V2.1 `Pipeline`
(`TrainOnlyFrequencyFeatures` → `ColumnTransformer` → estimator). No
hyperparameter is to be changed, and no search is to be run, in Experiment 2.
Experiment 2 inherits the 48 features, the target, the preprocessing and the
fold contract unchanged.

### 14.2 Consequences of the switch that must be recorded

- The XGBoost anchor remains valid as a historical Experiment 1 record and is
  **not** to be retrained, deleted or overwritten.
- Any downstream comparison against the anchor must be stated as a
  cross-model comparison under the same folds, not as a before-and-after of the
  same model.
- If a deployed model exists that was fitted on the XGBoost anchor, it will
  need refitting on CatBoost. That is an implementation cost, and it is
  deliberately not treated as evidence either for or against the switch.

### 14.3 What Experiment 2 must establish before any claim of validity

1. **Residual exchangeability under location shift.** The whole point of
   freezing a backbone here is to test whether conformal coverage holds on
   held-out locations. If it does not, that is the finding, and no amount of
   backbone selection fixes it.
2. **Coverage versus nominal level** in both the random and the
   location-grouped regime, reported as a curve, not a single number.
3. **Interval width** at each nominal level, since the accuracy advantage of
   CatBoost should translate into tighter intervals at equal coverage — this
   is the direct payoff of the selection made above.
4. **Whether the residual scale differs by location group.** Section 10 shows
   fold-level variability inflates 3.6x to 5.9x; whether that is a location
   effect or a fold-size effect is not yet known.
5. **An explicit statement of what split conformal cannot fix.** Coverage
   guarantees are conditional on exchangeability; if locations differ
   systematically, marginal coverage can still be poor.

### 14.4 Boundary of this document

Nothing in this analysis was frozen, retrained or deployed. This document is a
recommendation for review. Selecting and freezing the backbone is a separate,
explicit step that has not been taken.

---

## Appendix A — Figures

All figures are at 200 DPI with vector PDF companions, produced
deterministically by `scripts/valuation_v2_1/model_comparison_analysis.py`
from persisted artifacts only.

| Figure | File | What it shows |
|---|---|---|
| 1 | `figures/01_mae_comparison.png` | Mean absolute error, both regimes, mean +/- SD |
| 2 | `figures/02_rmse_comparison.png` | Root mean squared error, both regimes |
| 3 | `figures/03_r2_comparison.png` | INR-space R-squared, both regimes |
| 4 | `figures/04_medape_comparison.png` | Median absolute percentage error, both regimes |
| 5 | `figures/05_r2_log_comparison.png` | Log-space R-squared, both regimes, the primary metric |
| 6 | `figures/06_location_generalization.png` | Random to location-held-out change: `R2_log` slope panel plus `MedAPE` change with Welch 95% intervals |
| 7 | `figures/07_fold_variability.png` | Box plots of the five persisted per-fold `R2_log` values per regime, with individual fold points |
| 8 | `figures/08_training_time.png` | Total fit and predict time; XGBoost absent because the anchor was not retrained |

## Appendix B — Generated tables

| Table | File |
|---|---|
| Thesis summary | `thesis_summary_table.csv` / `.md` |
| Generalization | `generalization_table.csv` / `.md` |
| Full shift, all metrics | `regime_shift_all_metrics.csv` / `.md` |
| Metric discriminative power | `metric_discrimination.csv` / `.md` |
| Fold stability | `fold_stability.csv` / `.md` |
| Per-metric ordering by regime | `per_metric_ordering_by_regime.csv` / `.md` |
| Machine-readable record | `analysis_results.json` |
| Run manifest with input and output SHA-256 | `analysis_manifest.json` |

All are under
`artifacts/valuation/v2_1/model_comparison/`.

## Appendix C — Reproduction

```
python -m scripts.valuation_v2_1.model_comparison_analysis
python -m pytest scripts/valuation_v2_1/test_model_comparison_analysis.py -q
```

The analysis script trains nothing, loads no model, regenerates no fold and
hard-codes no metric value. A structural test asserts that the module contains
no reference to any estimator class, `Pipeline` construction, `joblib`, or
`KFold`/`GroupKFold`, so it is incapable of retraining anything. Before
plotting, it re-derives all 80 summary statistics from the persisted per-fold
rows and aborts if any disagrees with the persisted summary table; the
observed maximum relative difference is below 1e-15.

## Appendix D — Scope compliance

| Constraint | Status |
|---|---|
| No model retrained, modified or loaded | honoured; analysis reads CSV/JSON only |
| V2.1 protocol unmodified | honoured |
| Persisted folds unmodified | honoured |
| Dataset unmodified | honoured |
| Historical V2 and V2.1 artifacts unmodified | honoured |
| `.chain/state.json` unmodified | honoured; no blockchain interaction of any kind |
| XGBoost anchor unchanged | honoured |
| No new experiment namespace | honoured; all output inside the existing `model_comparison` namespace |
| No Anvil, no live tests, no transactions | honoured |
| No commit, no tag | honoured; work left uncommitted for review |


