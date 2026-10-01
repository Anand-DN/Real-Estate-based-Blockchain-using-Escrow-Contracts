### Per-metric ordering by regime (descriptive lookup, not a ranking)

| Regime | Metric | Metric key | Direction | position_1 | position_2 | position_3 | position_4 | position_5 |
|---|---|---|---|---|---|---|---|---|
| Random CV | MAE (INR) | MAE_INR | lower_is_better | XGBoost | CatBoost | Random Forest | Ridge | LightGBM |
| Random CV | RMSE (INR) | RMSE_INR | lower_is_better | XGBoost | CatBoost | LightGBM | Random Forest | Ridge |
| Random CV | R² | R2_INR | higher_is_better | XGBoost | CatBoost | LightGBM | Random Forest | Ridge |
| Random CV | MAPE (%) | MAPE_percent | lower_is_better | XGBoost | CatBoost | Random Forest | Ridge | LightGBM |
| Random CV | MedAPE (%) | MedAPE_percent | lower_is_better | XGBoost | CatBoost | Ridge | Random Forest | LightGBM |
| Random CV | MAE (log1p INR) | MAE_log | lower_is_better | XGBoost | CatBoost | Ridge | LightGBM | Random Forest |
| Random CV | RMSE (log1p INR) | RMSE_log | lower_is_better | XGBoost | CatBoost | LightGBM | Random Forest | Ridge |
| Random CV | R² (log space) | R2_log | higher_is_better | XGBoost | CatBoost | LightGBM | Random Forest | Ridge |
| Location-Grouped CV | MAE (INR) | MAE_INR | lower_is_better | CatBoost | Random Forest | XGBoost | Ridge | LightGBM |
| Location-Grouped CV | RMSE (INR) | RMSE_INR | lower_is_better | XGBoost | CatBoost | Random Forest | Ridge | LightGBM |
| Location-Grouped CV | R² | R2_INR | higher_is_better | XGBoost | CatBoost | Random Forest | Ridge | LightGBM |
| Location-Grouped CV | MAPE (%) | MAPE_percent | lower_is_better | CatBoost | Random Forest | XGBoost | LightGBM | Ridge |
| Location-Grouped CV | MedAPE (%) | MedAPE_percent | lower_is_better | CatBoost | Random Forest | XGBoost | LightGBM | Ridge |
| Location-Grouped CV | MAE (log1p INR) | MAE_log | lower_is_better | CatBoost | Random Forest | XGBoost | LightGBM | Ridge |
| Location-Grouped CV | RMSE (log1p INR) | RMSE_log | lower_is_better | CatBoost | Random Forest | XGBoost | LightGBM | Ridge |
| Location-Grouped CV | R² (log space) | R2_log | higher_is_better | CatBoost | Random Forest | XGBoost | LightGBM | Ridge |
