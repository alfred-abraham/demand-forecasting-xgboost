# Demand Forecasting with XGBoost

A machine learning project for forecasting retail product demand using historical sales and business data. The project uses a time-aware modeling approach with lag features, rolling statistics, temporal cross-validation, and XGBoost.

The final model achieved an **RMSE of 12.39**, **MAE of 8.60**, and **R² of 0.921** on a chronologically held-out test period, reducing RMSE by approximately **77.7%** compared with a naive previous-demand baseline.

## Project Overview

Accurate demand forecasting can help retailers make better decisions around inventory planning, purchasing, promotions, and product availability.

The objective of this project was to build a machine learning model capable of predicting product demand from historical retail data while ensuring that the evaluation process reflects a realistic forecasting scenario.

Rather than randomly splitting observations into training and test sets, the project uses chronological splitting so that the model is trained on past data and evaluated on future, unseen observations.

The workflow includes:

- Data quality checks and exploratory analysis
- Calendar-based feature engineering
- Historical demand lag features
- Rolling demand statistics
- Chronological train/test splitting
- One-hot encoding of categorical variables
- XGBoost regression
- Hyperparameter tuning with `TimeSeriesSplit`
- Comparison against a naive forecasting baseline
- Residual analysis
- Feature importance analysis
- Product-level error analysis

## Dataset

The dataset contains **76,000 retail observations** covering the period from **January 2022 through January 2024**.

Available variables include:

- Date
- Store ID
- Product ID
- Product category
- Region
- Inventory level
- Units sold
- Units ordered
- Price
- Discount
- Weather condition
- Promotion status
- Competitor pricing
- Seasonality
- Epidemic indicator
- Demand

Initial data-quality checks found no missing values or duplicate rows in the source dataset.

After historical demand features were created, **75,400 observations** remained available for modeling. The reduction is expected because lag and rolling features require prior observations for each product.

## Feature Engineering

Several features were created to capture calendar patterns, pricing information, and historical demand behavior.

### Calendar Features

The date variable was transformed into:

- Year
- Month
- Day
- Day of week
- Week of year
- Quarter
- Weekend indicator

### Pricing Feature

A discounted-price variable was calculated from the original price and discount:

`Discounted Price = Price × (1 - Discount / 100)`

### Historical Demand Features

Demand lag variables were created using:

- 1 previous observation
- 7 previous observations
- 14 previous observations
- 30 previous observations

Rolling demand statistics were also calculated over:

- 7 observations
- 14 observations
- 30 observations

Both rolling means and rolling standard deviations were included.

All rolling calculations were shifted by one observation before being calculated. This prevents the current target value from being included in the predictors used to forecast that same observation and therefore helps prevent **target leakage**.

## Time-Aware Validation

A key consideration in this project was ensuring that model evaluation represented a real forecasting scenario.

A random train/test split could allow future observations to influence model development. Instead, the dataset was divided chronologically.

### Training Period

**January 7, 2022 – September 1, 2023**

60,300 observations

### Test Period

**September 2, 2023 – January 30, 2024**

15,100 observations

The final 20% of unique dates were reserved as the test period.

The test data remained separate during model development and hyperparameter tuning.

## Data Preprocessing

Categorical variables were one-hot encoded using scikit-learn's `OneHotEncoder`.

Categorical predictors included:

- Product ID
- Category
- Region
- Weather condition
- Seasonality

Numerical predictors were passed through without categorical encoding.

Preprocessing and XGBoost were combined into a scikit-learn `Pipeline` to ensure that transformations remained consistent throughout cross-validation and final evaluation.

## Model

The forecasting model uses **XGBoost regression** (`XGBRegressor`).

Hyperparameters were tuned using `RandomizedSearchCV` with **five-fold `TimeSeriesSplit` cross-validation**.

Unlike ordinary shuffled cross-validation, `TimeSeriesSplit` maintains temporal ordering during model validation.

The search evaluated **25 hyperparameter combinations**, resulting in 125 model fits.

The selected model produced a cross-validation RMSE of approximately:

**19.27**

The selected hyperparameters were:

| Hyperparameter | Value |
|---|---:|
| `n_estimators` | 500 |
| `max_depth` | 8 |
| `learning_rate` | 0.1 |
| `min_child_weight` | 1 |
| `subsample` | 1.0 |
| `colsample_bytree` | 0.7 |

## Model Performance

The final tuned model was evaluated on the chronologically held-out test period.

| Metric | XGBoost |
|---|---:|
| RMSE | **12.386** |
| MAE | **8.598** |
| R² | **0.921** |

An R² of approximately **0.92** indicates that the model explained a large proportion of the variation in demand within the holdout period.

RMSE and MAE were also used because they express forecast error directly in demand units.

## Baseline Comparison

Machine learning performance is more meaningful when compared with a simple forecasting strategy.

A naive baseline was therefore created using the rule:

> The next demand observation will equal the product's most recently observed demand.

This corresponds to the `Demand_Lag_1` feature.

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Naive previous-demand forecast | 55.459 | 43.369 | -0.580 |
| **XGBoost** | **12.386** | **8.598** | **0.921** |

The XGBoost model reduced RMSE by approximately:

**77.7% compared with the naive baseline.**

This result suggests that the model captured predictive structure beyond simply carrying the previous demand observation forward.

## Forecast Visualization

Actual and predicted demand were aggregated to the daily level during the test period.

This allows model performance to be visually inspected over time rather than relying exclusively on summary metrics.

The notebook includes an **Actual vs. Predicted Demand** visualization for the complete holdout period.

## Residual Analysis

Residuals were calculated as:

`Residual = Actual Demand - Predicted Demand`

The mean residual during the test period was approximately:

**-0.83 demand units**

This indicates a small average tendency toward overprediction.

The notebook also includes a residual plot to identify potential systematic error patterns, changing variance, and outliers.

## Feature Importance

XGBoost feature importance was examined after model training to better understand which variables contributed to the model's predictions.

Among the most important features in the fitted model were:

- Promotion
- Epidemic indicator
- Product category
- Seasonality
- Weather condition
- Product-specific indicators

Feature importance should be interpreted as **predictive importance rather than causal evidence**. A feature receiving high importance does not necessarily mean that changing that variable would cause demand to change.

## Product-Level Error Analysis

Overall forecasting metrics can hide differences between individual products.

For this reason, MAE and RMSE were also calculated separately for each product during the holdout period.

This analysis helps identify products that are more difficult to forecast and provides a starting point for future product-specific modeling or feature engineering.

## Key Results

The final forecasting pipeline achieved:

- **RMSE: 12.39**
- **MAE: 8.60**
- **R²: 0.921**
- **77.7% lower RMSE than the naive baseline**
- Time-aware model validation
- Leakage-conscious historical demand features

These results were measured on future observations that were not used during model training or hyperparameter selection.

## Limitations

Several limitations should be considered when interpreting the results.

First, the final evaluation uses one historical holdout period. Forecasting performance could differ during other periods or under changing market conditions.

Second, the baseline used in this project is intentionally simple. Future analysis could compare the model against stronger benchmarks such as seasonal-naive forecasts or moving-average models.

Third, some model inputs—such as promotions, inventory, pricing, weather, and competitor pricing—would need to be known or separately forecast when generating predictions for genuinely unknown future periods.

Finally, feature importance describes associations used by the predictive model and should not be interpreted as evidence of causal relationships.

## Future Improvements

Potential extensions to this project include:

- Walk-forward backtesting across multiple historical periods
- Seasonal-naive forecasting benchmarks
- Additional lag and rolling-window experiments
- SHAP-based model interpretation
- Forecast prediction intervals
- Multi-step forecasting
- Product-specific forecasting models
- Automated model retraining
- Streamlit deployment for interactive forecasting

## Technologies Used

- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- XGBoost
- Jupyter Notebook / Google Colab

