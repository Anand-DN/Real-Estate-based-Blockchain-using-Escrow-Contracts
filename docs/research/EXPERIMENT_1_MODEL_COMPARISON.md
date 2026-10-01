# MILLOW RESEARCH EXPERIMENT 1 - FIVE-MODEL BASELINE COMPARISON

Companion to `EXPERIMENT_1_V2_1_BASELINE.md`. The protocol is not redefined
here. This document records the completion of Experiment 1: the four
remaining baseline models trained under the frozen V2.1 protocol, and the
five-model comparison they enable.

**No model is declared the winner and no ranking is produced.** Section 8
explains why that is a deliberate design constraint of this protocol, not an
oversight.

---

## 1. What was run

| Item | Value |
|---|---|
| Protocol | V2.1, frozen, unchanged |
| Feature count | 48 |
| Dataset rows | 28,398 (29,135 raw, 737 duplicates removed, 0 invalid removed) |
| Target | `log1p(price)` |
| Regimes | A random 5-fold CV, B location-grouped 5-fold CV |
| Models compared | 5 |
| Models trained this run | 4 (Ridge, Random Forest, LightGBM, CatBoost) |
| Model **not** trained | XGBoost, read from the existing V2.1 anchor |
| Fold models persisted | 40 new + 10 pre-existing anchor = 50 |
| Hyperparameter search | none |
| Early stopping | none |
| Predictions pooled across folds | no |
| Post-hoc configuration change | none |
| Winner chosen | no |
| Models ranked | no |

The XGBoost anchor was deliberately **not** retrained. Its per-fold metrics
were read from `artifacts/valuation/v2_1/xgb_anchor_metrics_v2_1.json` and
joined into every comparison table below. Retraining it would have introduced
a second source of variance for no benefit and would have risked an
accidental configuration drift between the anchor and the comparison.

---

## 2. Pre-declared fixed configurations

All four configurations were written into
`scripts/valuation_v2_1/run_model_comparison.py` **before any model was
fitted**, printed at the start of the run, and persisted to
`artifacts/valuation/v2_1/model_comparison/model_configurations.json`.

### 2.1 Ridge

```
sklearn.linear_model.Ridge
    alpha            = 1.0
    fit_intercept    = True
    random_state     = 42
```

`alpha=1.0` is the scikit-learn documented default. It was fixed a priori,
not selected by search.

### 2.2 Random Forest

```
sklearn.ensemble.RandomForestRegressor
    n_estimators      = 700
    max_depth         = 7
    min_samples_leaf  = 5
    min_samples_split = 10
    random_state      = 42
    n_jobs            = -1
```

### 2.3 LightGBM

```
lightgbm.LGBMRegressor
    n_estimators      = 700
    learning_rate     = 0.04
    num_leaves        = 127
    max_depth         = 7
    min_child_samples = 20
    subsample         = 0.85
    subsample_freq    = 1
    colsample_bytree  = 0.85
    reg_alpha         = 0.1
    reg_lambda        = 2.0
    random_state      = 42
    n_jobs            = -1
    verbose           = -1
```

### 2.4 CatBoost

```
catboost.CatBoostRegressor
    iterations         = 700
    learning_rate      = 0.04
    depth              = 7
    l2_leaf_reg        = 2.0
    random_seed        = 42
    verbose            = False
    allow_writing_files = False
```

### 2.5 XGBoost (anchor, not retrained)

Recorded for comparability only, copied from the anchor run record:

```
xgboost.XGBRegressor
    objective         = reg:squarederror
    n_estimators      = 700
    max_depth         = 7
    learning_rate     = 0.04
    subsample         = 0.85
    colsample_bytree  = 0.85
    min_child_weight  = 5
    reg_alpha         = 0.1
    reg_lambda        = 2.0
    tree_method       = hist
    random_state      = 42
    n_jobs            = -1
    eval_metric       = rmse
```

### 2.6 The model-fairness constraint

The four new configurations were deliberately matched to the already-frozen
XGBoost anchor on the three axes that govern capacity, so that the comparison
is not confounded by one model simply receiving a larger budget:

| Capacity axis | LightGBM | CatBoost | XGBoost | Random Forest | Ridge |
|---|---|---|---|---|---|
| Boosting rounds | 700 | 700 | 700 | n/a (700 trees) | n/a |
| Learning rate | 0.04 | 0.04 | 0.04 | n/a | n/a |
| Max depth | 7 | 7 | 7 | 7 | n/a |

`num_leaves=127` is `2**7 - 1`, the maximum expressible at depth 7, so the
depth-7 cap binds for LightGBM exactly as it does for XGBoost rather than
becoming a separately tunable quantity. `l2_leaf_reg=2.0` matches the
anchor's `reg_lambda=2.0`. `min_samples_leaf=5` approximates the anchor's
`min_child_weight=5` leaf-size control.

Ridge is a closed-form convex model with no rounds, learning rate or depth,
so the matched axes are not applicable to it. That is stated rather than
papered over: Ridge is included as a linear reference point, not as a
capacity-matched competitor.

### 2.7 One documented model-specific accommodation

CatBoost rejects scipy sparse matrices, so its pipeline step uses
`dense_output=True` while the other four use `dense_output=False`. This
changes no fitted statistic and no feature value. It is an input-format
accommodation, not a preprocessing advantage, and is recorded here rather
than hidden.

---

## 3. Fold integrity

Fold assignments were loaded from the persisted, committed contract at
`artifacts/valuation/v2_1/cv_folds_v2_1.json`. `generate_folds`, `KFold` and
`GroupKFold` were never called. Every model saw byte-identical train and
validation indices.

| Regime | SHA-256 digest | Train sizes | Valid sizes |
|---|---|---|---|
| Random | `1a9f896345492d60...` | 22718, 22718, 22718, 22719, 22719 | 5680, 5680, 5680, 5679, 5679 |
| Location-grouped | `4c1c331cc993dded...` | 23540, 22693, 23384, 21409, 22566 | 4858, 5705, 5014, 6989, 5832 |

`cv_folds_v2_1.json` is byte-identical to its pre-experiment SHA-256
`989b79656b051b3b...`.

---

## 4. Regime A - random 5-fold CV

### 4.1 Per fold

| Model | Fold | MAE INR | RMSE INR | R2 INR | MAPE % | MedAPE % | MAE log | RMSE log | R2 log |
|---|---|---|---|---|---|---|---|---|---|
| Ridge | 1 | 6,395,965 | 21,876,247 | 0.1097 | 54.47 | 34.28 | 0.4854 | 0.6759 | 0.3273 |
| Ridge | 2 | 6,471,443 | 23,512,519 | 0.1029 | 53.58 | 33.21 | 0.4749 | 0.6718 | 0.3453 |
| Ridge | 3 | 6,466,252 | 24,322,878 | 0.1038 | 53.31 | 33.18 | 0.4709 | 0.6621 | 0.3312 |
| Ridge | 4 | 6,456,348 | 22,300,451 | 0.0779 | 52.67 | 33.35 | 0.4792 | 0.6809 | 0.3101 |
| Ridge | 5 | 6,330,934 | 22,375,265 | 0.1087 | 54.76 | 33.72 | 0.4783 | 0.6686 | 0.3342 |
| Random Forest | 1 | 6,342,360 | 22,039,889 | 0.0963 | 53.60 | 34.88 | 0.4894 | 0.6770 | 0.3251 |
| Random Forest | 2 | 6,485,950 | 23,397,179 | 0.1117 | 53.61 | 34.06 | 0.4826 | 0.6752 | 0.3387 |
| Random Forest | 3 | 6,370,316 | 24,350,975 | 0.1017 | 51.72 | 32.89 | 0.4702 | 0.6586 | 0.3382 |
| Random Forest | 4 | 6,410,897 | 22,105,919 | 0.0939 | 51.99 | 33.99 | 0.4830 | 0.6765 | 0.3191 |
| Random Forest | 5 | 6,357,484 | 22,215,172 | 0.1214 | 54.10 | 33.63 | 0.4816 | 0.6692 | 0.3330 |
| LightGBM | 1 | 6,403,117 | 21,708,006 | 0.1233 | 54.15 | 36.19 | 0.4843 | 0.6695 | 0.3399 |
| LightGBM | 2 | 6,677,627 | 23,246,651 | 0.1231 | 55.84 | 36.45 | 0.4829 | 0.6701 | 0.3486 |
| LightGBM | 3 | 6,582,606 | 24,055,026 | 0.1234 | 53.56 | 34.77 | 0.4709 | 0.6576 | 0.3403 |
| LightGBM | 4 | 6,638,904 | 22,092,792 | 0.0950 | 53.80 | 35.32 | 0.4825 | 0.6749 | 0.3222 |
| LightGBM | 5 | 6,524,248 | 22,152,077 | 0.1264 | 55.94 | 35.55 | 0.4787 | 0.6620 | 0.3473 |
| CatBoost | 1 | 6,234,521 | 21,862,439 | 0.1108 | 51.89 | 33.63 | 0.4763 | 0.6620 | 0.3547 |
| CatBoost | 2 | 6,332,913 | 23,230,133 | 0.1244 | 51.62 | 32.12 | 0.4665 | 0.6588 | 0.3704 |
| CatBoost | 3 | 6,198,397 | 23,921,365 | 0.1332 | 49.93 | 31.63 | 0.4542 | 0.6410 | 0.3732 |
| CatBoost | 4 | 6,275,937 | 22,090,242 | 0.0952 | 49.75 | 32.54 | 0.4667 | 0.6607 | 0.3505 |
| CatBoost | 5 | 6,160,420 | 22,065,941 | 0.1331 | 51.60 | 32.70 | 0.4652 | 0.6498 | 0.3711 |
| XGBoost (anchor) | 1 | 6,073,823 | 21,704,525 | 0.1236 | 51.21 | 31.34 | 0.4624 | 0.6564 | 0.3655 |
| XGBoost (anchor) | 2 | 6,196,552 | 23,196,493 | 0.1269 | 51.20 | 30.19 | 0.4534 | 0.6553 | 0.3771 |
| XGBoost (anchor) | 3 | 6,087,407 | 23,983,680 | 0.1286 | 49.38 | 29.95 | 0.4446 | 0.6392 | 0.3767 |
| XGBoost (anchor) | 4 | 6,161,316 | 21,946,489 | 0.1069 | 49.12 | 30.27 | 0.4539 | 0.6562 | 0.3592 |
| XGBoost (anchor) | 5 | 6,005,100 | 21,800,283 | 0.1539 | 51.53 | 30.87 | 0.4513 | 0.6434 | 0.3834 |

### 4.2 Mean and standard deviation (ddof=1, n=5, not pooled)

| Model | MAE INR | RMSE INR | R2 INR | MAPE % | MedAPE % | MAE log | RMSE log | R2 log |
|---|---|---|---|---|---|---|---|---|
| Ridge | 6,424,188 ± 60,269 | 22,877,472 ± 1,009,963 | 0.1006 ± 0.0130 | 53.76 ± 0.85 | 33.55 ± 0.46 | 0.4777 ± 0.0054 | 0.6719 ± 0.0071 | 0.3296 ± 0.0128 |
| Random Forest | 6,393,402 ± 57,662 | 22,821,827 ± 1,019,964 | 0.1050 ± 0.0114 | 53.01 ± 1.07 | 33.89 ± 0.72 | 0.4814 ± 0.0069 | 0.6713 ± 0.0077 | 0.3308 ± 0.0085 |
| LightGBM | 6,565,300 ± 107,614 | 22,650,910 ± 971,577 | 0.1182 ± 0.0131 | 54.66 ± 1.14 | 35.66 ± 0.67 | 0.4798 ± 0.0054 | 0.6668 ± 0.0069 | 0.3397 ± 0.0105 |
| CatBoost | 6,240,437 ± 67,114 | 22,634,024 ± 898,103 | 0.1193 ± 0.0163 | 50.96 ± 1.03 | 32.52 ± 0.74 | 0.4658 ± 0.0078 | 0.6545 ± 0.0089 | 0.3640 ± 0.0106 |
| XGBoost (anchor) | 6,104,840 ± 75,523 | 22,526,294 ± 1,013,868 | 0.1280 ± 0.0169 | 50.49 ± 1.14 | 30.52 ± 0.57 | 0.4531 ± 0.0064 | 0.6501 ± 0.0082 | 0.3723 ± 0.0098 |

---

## 5. Regime B - location-grouped 5-fold CV

### 5.1 Per fold

| Model | Fold | MAE INR | RMSE INR | R2 INR | MAPE % | MedAPE % | MAE log | RMSE log | R2 log |
|---|---|---|---|---|---|---|---|---|---|
| Ridge | 1 | 6,722,388 | 21,209,409 | 0.0446 | 65.77 | 43.86 | 0.5466 | 0.7255 | 0.1928 |
| Ridge | 2 | 6,402,643 | 18,557,589 | 0.1027 | 74.82 | 46.49 | 0.5671 | 0.7367 | 0.1299 |
| Ridge | 3 | 8,783,648 | 32,642,513 | 0.0309 | 61.01 | 44.05 | 0.5648 | 0.7517 | 0.2411 |
| Ridge | 4 | 6,613,180 | 17,621,752 | 0.0727 | 77.49 | 45.00 | 0.5790 | 0.7558 | 0.1368 |
| Ridge | 5 | 7,807,918 | 26,014,219 | 0.0579 | 77.17 | 47.53 | 0.5889 | 0.7636 | 0.1442 |
| Random Forest | 1 | 6,372,670 | 21,052,728 | 0.0587 | 57.74 | 36.09 | 0.5001 | 0.6936 | 0.2622 |
| Random Forest | 2 | 5,725,893 | 18,757,374 | 0.0833 | 57.98 | 34.50 | 0.4864 | 0.6688 | 0.2830 |
| Random Forest | 3 | 8,261,672 | 32,532,900 | 0.0374 | 53.04 | 37.95 | 0.5201 | 0.7224 | 0.2991 |
| Random Forest | 4 | 6,244,317 | 17,530,332 | 0.0823 | 65.61 | 39.69 | 0.5363 | 0.7132 | 0.2313 |
| Random Forest | 5 | 7,347,787 | 26,019,463 | 0.0575 | 68.04 | 40.61 | 0.5426 | 0.7326 | 0.2122 |
| LightGBM | 1 | 7,077,898 | 21,385,770 | 0.0287 | 65.29 | 44.81 | 0.5448 | 0.7290 | 0.1851 |
| LightGBM | 2 | 6,484,276 | 18,782,533 | 0.0809 | 70.56 | 44.75 | 0.5408 | 0.7096 | 0.1928 |
| LightGBM | 3 | 8,534,442 | 32,398,244 | 0.0453 | 58.21 | 40.15 | 0.5346 | 0.7289 | 0.2866 |
| LightGBM | 4 | 7,081,002 | 17,884,280 | 0.0449 | 79.16 | 47.99 | 0.5849 | 0.7615 | 0.1236 |
| LightGBM | 5 | 8,161,039 | 26,105,484 | 0.0513 | 77.06 | 48.89 | 0.5846 | 0.7606 | 0.1508 |
| CatBoost | 1 | 6,321,652 | 20,918,340 | 0.0707 | 56.82 | 36.55 | 0.4999 | 0.6896 | 0.2707 |
| CatBoost | 2 | 5,649,898 | 18,735,734 | 0.0854 | 56.45 | 34.02 | 0.4792 | 0.6616 | 0.2983 |
| CatBoost | 3 | 8,236,333 | 32,536,320 | 0.0372 | 52.18 | 36.85 | 0.5145 | 0.7173 | 0.3091 |
| CatBoost | 4 | 6,188,846 | 17,471,953 | 0.0884 | 64.45 | 39.37 | 0.5322 | 0.7085 | 0.2413 |
| CatBoost | 5 | 7,320,030 | 26,216,948 | 0.0432 | 66.84 | 39.67 | 0.5382 | 0.7294 | 0.2191 |
| XGBoost (anchor) | 1 | 6,421,620 | 21,037,519 | 0.0600 | 59.31 | 37.02 | 0.5070 | 0.6988 | 0.2511 |
| XGBoost (anchor) | 2 | 5,836,112 | 18,430,930 | 0.1149 | 62.76 | 36.94 | 0.5006 | 0.6814 | 0.2557 |
| XGBoost (anchor) | 3 | 8,228,930 | 32,505,300 | 0.0390 | 54.36 | 36.90 | 0.5143 | 0.7170 | 0.3096 |
| XGBoost (anchor) | 4 | 6,535,451 | 17,484,605 | 0.0871 | 75.11 | 42.27 | 0.5590 | 0.7425 | 0.1669 |
| XGBoost (anchor) | 5 | 7,553,485 | 25,871,702 | 0.0682 | 72.54 | 41.55 | 0.5560 | 0.7445 | 0.1865 |

### 5.2 Mean and standard deviation (ddof=1, n=5, not pooled)

| Model | MAE INR | RMSE INR | R2 INR | MAPE % | MedAPE % | MAE log | RMSE log | R2 log |
|---|---|---|---|---|---|---|---|---|
| Ridge | 7,265,955 ± 1,007,969 | 23,209,097 ± 6,199,027 | 0.0618 ± 0.0277 | 71.25 ± 7.44 | 45.38 ± 1.59 | 0.5693 ± 0.0160 | 0.7467 ± 0.0153 | 0.1690 ± 0.0473 |
| Random Forest | 6,790,468 ± 1,010,099 | 23,178,559 ± 6,155,139 | 0.0638 ± 0.0193 | 60.48 ± 6.18 | 37.77 ± 2.52 | 0.5171 ± 0.0238 | 0.7061 ± 0.0253 | 0.2576 ± 0.0359 |
| LightGBM | 7,467,731 ± 849,597 | 23,311,262 ± 6,001,041 | 0.0502 ± 0.0191 | 70.05 ± 8.59 | 45.32 ± 3.43 | 0.5579 ± 0.0247 | 0.7379 ± 0.0226 | 0.1878 ± 0.0618 |
| CatBoost | 6,743,352 ± 1,029,867 | 23,175,859 ± 6,209,108 | 0.0650 ± 0.0237 | 59.35 ± 6.09 | 37.29 ± 2.31 | 0.5128 ± 0.0241 | 0.7013 ± 0.0265 | 0.2677 ± 0.0378 |
| XGBoost (anchor) | 6,915,120 ± 959,890 | 23,066,011 ± 6,199,253 | 0.0739 ± 0.0287 | 64.82 ± 8.80 | 38.93 ± 2.73 | 0.5274 ± 0.0279 | 0.7168 ± 0.0274 | 0.2340 ± 0.0575 |

---

## 6. Summary statistics and fold counts

Every regime/model combination reports fold-level mean, sample standard
deviation with `ddof=1`, minimum, maximum and fold count. Full min/max values
are in `artifacts/valuation/v2_1/model_comparison/summary_metrics.csv`. Every
row carries `n_folds=5` and `std_ddof=1`.

No metric anywhere in this document is computed by pooling predictions across
folds. Pooling would let a single large fold dominate and would hide exactly
the fold-to-fold instability that the grouped regime is designed to expose.

---

## 7. Training time

Wall-clock seconds, summed over the 10 fold fits per model. Predict time is
reported separately and is not included in the fit figure.

| Model | Total fit (s) | Mean per fold (s) | Total predict (s) |
|---|---|---|---|
| Ridge | 11.622 | 1.16 | 0.670 |
| Random Forest | 727.399 | 72.74 | 6.773 |
| LightGBM | 124.869 | 12.49 | 3.547 |
| CatBoost | 233.610 | 23.36 | 2.444 |

Random Forest is by far the most expensive model at 727 s of fit time, which
is expected for 700 fully grown depth-7 trees fitted ten times on roughly
22,700 rows each. LightGBM is the cheapest boosting model at 125 s, and
Ridge is effectively free at 11.6 s across all ten folds.

These timings are reported for transparency about experiment cost. They were
not used to select, drop or reconfigure any model.

Per-fold timings are recorded in `training_time.csv`, and per-fold fit and
predict seconds are also stored on every row of `per_fold_metrics.csv`.

---

## 8. Why no ranking is produced

This protocol forbids declaring a winner, and the constraint is deliberate:

1. The five configurations were frozen a priori, but only one configuration
   per model family was tried. The spread between families therefore
   reflects both model family and this particular arbitrary configuration.
   Without a search, the experiment cannot separate the two.
2. Fold-level standard deviations overlap between adjacent models on most
   metrics. Overlapping `n=5` fold distributions with no significance
   testing do not support a defensible ordering.
3. Choosing a winner now, from these numbers, is precisely the
   result-driven selection the protocol prohibits. The next legitimate step
   is a separately pre-declared comparison with nested or repeated
   evaluation, not a ranking of this table.

What the comparison does support is the observation that all five models sit
in a narrow band under regime A and that the regime B numbers are materially
worse and far more variable for every model. That is a statement about the
evaluation regime, not about model choice.

---

## 9. Artifacts

New namespace only. Nothing under the historical V2 artifacts or the XGBoost
anchor namespace was written.

### Models

```
models/valuation/v2_1/model_comparison/
    Ridge/random_fold{1..5}.joblib
    Ridge/location_grouped_fold{1..5}.joblib
    RandomForest/random_fold{1..5}.joblib
    RandomForest/location_grouped_fold{1..5}.joblib
    LightGBM/random_fold{1..5}.joblib
    LightGBM/location_grouped_fold{1..5}.joblib
    CatBoost/random_fold{1..5}.joblib
    CatBoost/location_grouped_fold{1..5}.joblib
```

### Metrics and records

```
artifacts/valuation/v2_1/model_comparison/
    model_configurations.json   pre-declared configs, frozen before training
    per_fold_metrics.csv        50 rows, one per model x regime x fold
    per_fold_metrics.json       per-fold metrics plus full summary statistics
    summary_metrics.csv         mean, std, min, max, n, ddof per metric
    comparison_table.csv        wide table, one row per model per regime
    training_time.csv           per-fold and total fit/predict seconds
    environment.json            machine-independent environment record
    experiment_manifest.json    run manifest incl. safety assertions
    artifact_verification.json  re-load and re-predict verification result
```

### Reproduction

```
# Trains 4 models x 2 regimes x 5 folds. Reads the XGBoost anchor.
python -m scripts.valuation_v2_1.run_model_comparison

# Re-loads all 40 persisted models and re-predicts each validation fold.
python -m scripts.valuation_v2_1.verify_model_comparison

# Non-live protocol tests.
python -m pytest scripts/valuation_v2_1/test_protocol.py -q
```

---

## 10. Artifact verification

`scripts/valuation_v2_1/verify_model_comparison.py` independently re-loads
every persisted model from disk, re-predicts its own validation fold using the
exact persisted fold indices, and compares the recomputed metrics against the
recorded metrics.

| Check | Result |
|---|---|
| Persisted models re-loaded and re-predicted | 40 of 40 |
| XGBoost anchor rows referenced, not retrained | 10 |
| Worst absolute metric difference | 3.725e-09 |
| Metric mismatches beyond tolerance | 0 |
| Missing or failed models | 0 |

The artifacts on disk reproduce the reported numbers. Tolerances are
scale-appropriate per metric and are recorded in
`artifact_verification.json`.

---

## 11. Environment

| Package | Version |
|---|---|
| Python | 3.14.7 |
| pandas | 3.0.6 |
| numpy | 2.5.3 |
| scikit-learn | 1.9.1 |
| XGBoost | 3.4.1 |
| LightGBM | 4.7.0 |
| CatBoost | 1.2.10 |
| joblib | 1.6.0 |

`lightgbm` and `catboost` were absent when the V2.1 environment record was
first generated and were installed afterwards, unmodified from the versions
shown, solely to run this comparison. CatBoost pulled in `plotly` and
`graphviz` as transitive dependencies.

No pre-existing package was installed, upgraded or downgraded at any point.
`python`, `pandas`, `numpy`, `scikit-learn`, `xgboost` and `joblib` are at
exactly the versions recorded for the XGBoost anchor run, so the anchor and
this comparison were produced on an identical core stack.

The environment record is machine-independent: it stores the label `python`,
never an absolute interpreter path.

---

## 12. Non-equivalence with historical V2

The historical V2 reference (`artifacts/valuation/valuation_v2_metrics.csv`)
and V2.1 are **not directly comparable**. Frequency leakage was removed, 737
duplicates were removed, the feature set changed from 49 to 48 columns, and
evaluation changed from a single split to 5-fold cross-validation.

No better-or-worse claim is made between V2 and V2.1, and none between
XGBoost under the two protocols.

---

## 13. Safety and scope

No blockchain interaction occurred. `.chain/state.json`, the canonical token
map and the raw dataset were read only for verification and are byte-identical
to their pre-experiment SHA-256 values.

No live Anvil tests were run. No `*.live.test.js` file was executed. No
blanket `npm test` was run.

No Git commit and no Git tag were created for this comparison. This work is
intentionally left uncommitted pending review.
