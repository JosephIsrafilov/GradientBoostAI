# GradientBoostAI — training a gradient boosting regressor

This project ships refreshed notebooks and a final pipeline built with `GradientBoostingRegressor`. The CSV files are kept unchanged.

## What is included
- The house-price model was switched to gradient boosting and evaluated on a holdout split.
- A baseline `RandomForestRegressor` is added for metric comparison.
- The notebook contains a metrics table, an actual-vs-predicted scatter plot, and a feature-importance bar plot.

## Files

| Path | Purpose |
| --- | --- |
| 01_pandas_playground.ipynb | Quick pandas tricks on `employees.csv` and a synthetic dataset. |
| 02_visualization_and_linear_models.ipynb | Visualizations and a simple linear regression on `Housing.csv` and `rost_ves.csv`. |
| 03_gradient_boosting_house_prices.ipynb | Full ML pipeline for `housing_az_sqm_azn.csv` with gradient boosting, metrics, and plots. |
| housing_az_sqm_azn.csv | Main housing-price dataset in AZN (original). |
| Housing.csv, employees.csv, rost_ves.csv | Extra datasets for examples. |

## Gradient Boosting training (notebook `03_gradient_boosting_house_prices.ipynb`)

| Step | Code block | What happens |
| --- | --- | --- |
| 1 | Import libraries and load `housing_az_sqm_azn.csv` | Read data and shuffle for stability. |
| 2 | Split features/target | `target = 'PriceAZN'`, `X = dataset.drop(target)`, `y = dataset[target]`. |
| 3 | Feature analysis | Summary stats for numerics and cardinality for categoricals. |
| 4 | Preprocessing | `ColumnTransformer`: numerics — median imputer + `StandardScaler`; categoricals — frequency imputer + `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`. |
| 5 | Models | `RandomForestRegressor` (baseline) and `GradientBoostingRegressor` (main). |
| 6 | Holdout evaluation | `train_test_split(test_size=0.2, random_state=13)` + MAE/RMSE/R2 for both models. |
| 7 | Cross-validation | `cross_val_score` on R2 for the final booster. |
| 8 | Prediction example | Price prediction for new listings (`Bedrooms`, `Bathrooms`, `Sqm`, `City`). |
| 9 | Feature importance | Top features from the baseline forest + horizontal bar plot. |
| 10 | Validation plot | Scatter of actual vs predicted prices for gradient boosting. |
| 11 | Metrics table | DataFrame with MAE/RMSE/R2 for both models. |

## Metrics (appear after running the notebook)

| Model | MAE | RMSE | R2 | CV R2 (mean ± std) |
| --- | --- | --- | --- | --- |
| RandomForestRegressor | printed in the notebook | printed in the notebook | printed in the notebook | — |
| GradientBoostingRegressor | printed in the notebook | printed in the notebook | printed in the notebook | printed after `cross_val_score` |

## Plots

| Location | Purpose |
| --- | --- |
| `02_visualization_and_linear_models.ipynb` | Histograms/boxplots/regressions for `Housing.csv`; regression line for `rost_ves.csv`. |
| `03_gradient_boosting_house_prices.ipynb` | Scatter of actual vs predicted prices (Gradient Boosting) and feature-importance bar plot (RandomForest). |

## How to reproduce training

1. Prepare a Python 3 environment with `pandas`, `seaborn`, `matplotlib`, `scikit-learn`, and `numpy`.
2. Open `03_gradient_boosting_house_prices.ipynb` in Jupyter or VS Code.
3. Run the cells sequentially: data loading → preprocessing → model training → evaluation → predictions → plots.
4. (Optional) Use `02_visualization_and_linear_models.ipynb` to build extra EDA plots.

After executing the notebooks you will see metrics tables and plots that reflect model quality and feature importance.
