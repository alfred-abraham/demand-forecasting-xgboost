# Demand Forecasting with XGBoost

A machine learning project for forecasting retail product demand using historical sales and business data. This project uses a time-aware modeling approach with lag features, rolling statistics, temporal cross-validation, and XGBoost.

The final model achieved an **RMSE of 12.39**, **MAE of 8.60**, and **R² of 0.921** on a chronologically held-out test period, reducing RMSE by approximately **77.7%** compared with a naive previous-demand baseline.

---

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

---

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

Initial data-quality checks found:

- **Missing values:** 0
- **Duplicate rows:** 0

After historical demand features were created, **75,400 observations** remained available for modeling.

The reduction in observations is expected because lag and rolling features require prior observations for each product. The earliest observations for each product therefore do not have enough historical information to calculate all engineered features.

### Data Source

The dataset used in this project was obtained from the [DemandForecastingDataset repository](https://github.com/Onurbltc/DemandForecastingDataset) created by Onur Baltaci as part of his demand forecasting tutorial.

The dataset and original tutorial provided the starting point for this project. The forecasting workflow in this repository was subsequently expanded with time-aware validation, lag and rolling features, leakage prevention, time-series cross-validation, baseline comparison, and additional model diagnostics.

### Obtaining the Dataset

The dataset is **not redistributed in this repository**.

Download `demand_forecasting.csv` from the [original dataset repository](https://github.com/Onurbltc/DemandForecastingDataset) before running the notebook.

When using Google Colab, upload the downloaded CSV to the Colab environment. The notebook expects the dataset at:

    /content/demand_forecasting.csv

If the file is stored somewhere else, update the `DATA_PATH` variable in the notebook accordingly.

---

## Exploratory Data Analysis

The exploratory portion of the project examines demand over time and across product categories.

Daily demand was aggregated and visualized to identify broad temporal patterns. Average monthly demand was also compared across product categories to examine differences in demand behavior.

These exploratory relationships are treated as descriptive patterns rather than evidence of causal relationships.

---

## Feature Engineering

Several features were created to capture calendar patterns, pricing information, and historical demand behavior.

### Calendar Features

The original date variable was transformed into:

- Year
- Month
- Day
- Day of week
- Week of year
- Quarter
- Weekend indicator

These variables allow the model to learn recurring calendar-related patterns without using the raw date directly as a predictor.

### Discounted Price

A discounted-price variable was created using:

`Discounted Price = Price × (1 - Discount / 100)`

This combines the original product price and discount percentage into an estimate of the effective selling price.

### Historical Demand Features

Demand lag variables were created using:

- 1 previous observation
- 7 previous observations
- 14 previous observations
- 30 previous observations

These variables provide the model with information about previous demand for each product.

### Rolling Demand Features

Rolling demand statistics were calculated over:

- 7 observations
- 14 observations
- 30 observations

For each window, both the rolling mean and rolling standard deviation were calculated.

The rolling mean provides information about the recent level of demand, while the rolling standard deviation provides information about recent demand variability.

---

## Preventing Target Leakage

A major consideration when creating historical demand features is preventing information from the current observation from leaking into the predictors.

All rolling statistics were therefore calculated after shifting demand by one observation.

Conceptually, the model uses:

`Historical demand → Engineered features → Current demand prediction`

rather than:

`Current demand → Engineered feature → Current demand prediction`

This ensures that the target value being predicted is not included in the historical statistics used to predict it.

Because the longest historical feature requires 30 prior observations, the earliest observations for each product cannot contain every lag and rolling feature.

These rows were excluded from modeling rather than imputed because the missing values represent unavailable historical information rather than missing source data.

---

## Modeling Features

The final model uses a combination of categorical, numerical, calendar, pricing, and historical demand variables.

### Categorical Features

Categorical predictors include:

- Product ID
- Category
- Region
- Weather condition
- Seasonality

### Numerical and Engineered Features

Numerical predictors include variables such as:

- Price
- Discount
- Discounted price
- Inventory level
- Promotion
- Competitor pricing
- Epidemic indicator
- Year
- Month
- Day
- Day of week
- Week of year
- Quarter
- Weekend indicator
- Demand lag features
- Rolling demand means
- Rolling demand standard deviations

The raw `Date` variable is not passed directly to XGBoost. Instead, it is used to preserve chronological ordering, create calendar features, split the data, and visualize predictions.

---

## Time-Aware Train/Test Split

A key methodological consideration in this project was ensuring that model evaluation represented a realistic forecasting scenario.

A standard random train/test split could allow observations from later dates to appear in training while observations from earlier dates appear in testing.

That approach can be appropriate for many machine learning problems but is less representative of forecasting, where the goal is to use historical information to predict future observations.

Instead, the dataset was divided chronologically.

### Training Period

**January 7, 2022 – September 1, 2023**

**Training observations:** 60,300

### Test Period

**September 2, 2023 – January 30, 2024**

**Test observations:** 15,100

The final **20% of unique dates** were reserved as the test period.

Observations from the same date were kept together so that a date could not appear in both the training and test sets.

The test period remained separate during model development and hyperparameter tuning.

---

## Data Preprocessing

Categorical variables were encoded using scikit-learn's `OneHotEncoder`.

One-hot encoding was used rather than assigning arbitrary numerical ranks to nominal categories.

The preprocessing pipeline includes:

- One-hot encoding for categorical variables
- Pass-through treatment for numerical variables
- Integration with XGBoost using a scikit-learn `Pipeline`

Combining preprocessing and modeling in a pipeline helps ensure that the same transformations are consistently applied during cross-validation and final prediction.

---

## Model

The forecasting model uses **XGBoost regression** through `XGBRegressor`.

XGBoost is a gradient-boosted tree algorithm capable of modeling nonlinear relationships and interactions among predictors.

The model was configured using the squared-error regression objective.

---

## Hyperparameter Tuning

Model hyperparameters were tuned using `RandomizedSearchCV`.

The search evaluated **25 hyperparameter combinations**.

Each configuration was evaluated using **five-fold `TimeSeriesSplit` cross-validation**, resulting in:

**125 model fits**

Unlike ordinary shuffled cross-validation, `TimeSeriesSplit` maintains temporal ordering during validation.

This makes model selection more appropriate for forecasting because validation observations occur after the observations used to train each fold.

The model was selected using **Root Mean Squared Error (RMSE)**.

The best cross-validation RMSE was approximately:

**19.27**

### Selected Hyperparameters

| Hyperparameter | Selected Value |
|---|---:|
| `n_estimators` | 500 |
| `max_depth` | 8 |
| `learning_rate` | 0.1 |
| `min_child_weight` | 1 |
| `subsample` | 1.0 |
| `colsample_bytree` | 0.7 |

---

## Model Evaluation

After hyperparameter tuning, the selected model was evaluated on the chronologically held-out future test period.

Three evaluation metrics were used.

### Root Mean Squared Error (RMSE)

RMSE measures prediction error while placing additional weight on larger errors.

### Mean Absolute Error (MAE)

MAE represents the average absolute difference between predicted and actual demand.

### R²

R² measures how much of the variation in holdout demand is explained by the model relative to a constant-mean reference.

### XGBoost Results

| Metric | XGBoost |
|---|---:|
| RMSE | **12.386** |
| MAE | **8.598** |
| R² | **0.921** |

The model achieved an R² of approximately **0.92** on the future holdout period.

---

## Naive Forecasting Baseline

Machine learning performance is more informative when compared with a simple forecasting strategy.

A naive baseline was therefore created using the following rule:

> The next demand observation will equal the product's most recently observed demand.

This corresponds to the `Demand_Lag_1` feature.

### Baseline Results

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Naive previous-demand forecast | 55.459 | 43.369 | -0.580 |
| **XGBoost** | **12.386** | **8.598** | **0.921** |

The XGBoost model reduced RMSE by approximately **77.7%** relative to the naive previous-demand baseline.

This comparison provides evidence that the model captured useful predictive structure in this dataset beyond simply carrying the previous demand observation forward.

---

## Actual vs. Predicted Demand

Model performance was also evaluated visually.

Because the dataset contains multiple products for each date, actual and predicted demand were aggregated to the daily level during the test period.

The resulting visualization compares:

- Actual daily demand
- XGBoost predicted daily demand

This provides a visual assessment of whether the model follows the overall level and movement of demand throughout the future holdout period.

---

## Residual Analysis

Residuals were calculated as:

`Residual = Actual Demand - Predicted Demand`

The mean residual during the test period was approximately:

**-0.83 demand units**

A negative mean residual indicates a small average tendency toward overprediction.

The notebook also includes a residual plot comparing predicted demand with forecast errors.

Residual analysis helps identify:

- Systematic prediction errors
- Outliers
- Changes in forecast variance
- Potential patterns not captured by the model

A mean residual close to zero does not guarantee that all model errors are random, so the residual visualization should be considered alongside the numerical result.

---

## Feature Importance

XGBoost feature importance was examined after model training to understand which variables contributed most strongly to the fitted model's predictions.

Among the highest-ranked features were:

- Promotion
- Epidemic indicator
- Product category
- Seasonality
- Weather condition
- Product-specific indicators

In the fitted model, **Promotion** had the highest reported feature importance, followed by the **Epidemic** indicator and several product-category variables.

Feature importance should be interpreted carefully.

These values describe how the trained model used predictors to generate forecasts. They do **not** demonstrate that changing a feature would cause demand to change.

Correlated predictors may also distribute or concentrate feature importance in ways that make individual importance values difficult to interpret causally.

---

## Product-Level Error Analysis

Overall forecasting metrics can hide differences in performance across individual products.

For this reason, MAE and RMSE were calculated separately for each product during the test period.

The product-level analysis can help identify:

- Products that are relatively easy to forecast
- Products with higher prediction error
- Products that may contain unusual volatility
- Products that may benefit from additional feature engineering
- Opportunities for product-specific forecasting models

This type of error analysis is useful because a model with strong overall performance may still perform poorly for specific products.

---

## Key Results

The final forecasting pipeline achieved:

- **RMSE: 12.39**
- **MAE: 8.60**
- **R²: 0.921**
- **77.7% lower RMSE than the naive baseline**
- Chronological future holdout evaluation
- Time-series cross-validation
- Leakage-conscious historical demand features
- Product-level forecast error analysis

These results were measured on observations occurring after the training period and were not used during model selection.

---

## Key Methodological Improvements

This project began with a demand forecasting tutorial and was subsequently expanded into a more rigorous forecasting workflow.

Several methodological improvements were introduced.

### 1. Chronological Train/Test Splitting

Instead of randomly dividing observations, the model is trained on historical data and evaluated on future observations.

### 2. Time-Series Cross-Validation

`TimeSeriesSplit` is used during hyperparameter tuning so that temporal order is maintained during model selection.

### 3. Historical Demand Features

Lag and rolling features allow XGBoost to incorporate recent demand behavior.

### 4. Leakage Prevention

Historical features are shifted before rolling calculations so that the target value being predicted cannot enter its own predictors.

### 5. Improved Categorical Encoding

Nominal variables are one-hot encoded rather than converted into arbitrary integer rankings.

### 6. Baseline Comparison

The model is evaluated against a naive previous-demand forecast rather than interpreting performance metrics in isolation.

### 7. Multiple Evaluation Metrics

RMSE, MAE, and R² provide complementary perspectives on model performance.

### 8. Forecast Diagnostics

Actual-vs-predicted visualization, residual analysis, feature importance, and product-level error analysis provide additional insight beyond aggregate performance metrics.

---

## Limitations

Several limitations should be considered when interpreting the results.

### Single Final Holdout Period

The final evaluation uses one historical holdout period.

Although this provides a realistic future test, forecasting performance could differ during other periods or under changing market conditions.

### Simple Baseline

The naive baseline intentionally uses a simple previous-demand rule.

Future work could compare XGBoost against stronger forecasting benchmarks such as:

- Seasonal-naive forecasts
- Moving averages
- Exponential smoothing
- Traditional time-series models

### Future Predictor Availability

Some predictors used by the model may not automatically be known when forecasting genuinely unseen future dates.

Examples include:

- Promotions
- Inventory levels
- Product pricing
- Competitor pricing
- Weather conditions

A production forecasting system would need these variables to be known in advance, planned in advance, or separately forecast.

### Feature Importance Is Not Causality

Feature importance indicates which variables contributed to the model's predictions.

It does not establish causal relationships between those variables and demand.

### Structural Changes

Retail demand patterns can change because of factors not represented in historical data.

A production system would therefore require monitoring and periodic retraining.

---

## Future Improvements

Potential extensions to this project include:

- Walk-forward backtesting across multiple historical periods
- Seasonal-naive forecasting benchmarks
- Additional lag-window experiments
- Additional rolling-window features
- Exponentially weighted demand features
- Holiday and event features
- SHAP-based model interpretation
- Forecast prediction intervals
- Multi-step forecasting
- Product-specific forecasting models
- Store-product level forecasting
- Automated model retraining
- Model monitoring
- Streamlit deployment for interactive forecasting

---

## Technologies Used

### Programming Language

- Python

### Data Manipulation

- pandas
- NumPy

### Machine Learning

- scikit-learn
- XGBoost

### Visualization

- Matplotlib
- Seaborn

### Development Environment

- Jupyter Notebook
- Google Colab

---

## Python Skills Demonstrated

This project demonstrates practical experience with:

- Data loading and inspection
- Data quality validation
- Datetime manipulation
- Exploratory data analysis
- Feature engineering
- Grouped calculations
- Lag feature creation
- Rolling-window calculations
- Categorical encoding
- Machine learning pipelines
- Regression modeling
- Hyperparameter optimization
- Time-series cross-validation
- Model evaluation
- Baseline model comparison
- Residual diagnostics
- Feature importance
- Product-level error analysis
- Data visualization

---

## Data Science Concepts Demonstrated

The project also demonstrates understanding of:

- Demand forecasting
- Supervised machine learning
- Regression
- Time-aware validation
- Train/test leakage
- Target leakage
- Temporal feature engineering
- Forecast benchmarking
- Out-of-sample evaluation
- Model interpretation
- Error analysis
- Predictive vs. causal interpretation

---

## Installation

Clone the repository:

    git clone https://github.com/alfred-abraham/demand-forecasting-xgboost.git

Move into the project directory:

    cd demand-forecasting-xgboost

Install the required Python packages:

    pip install -r requirements.txt

Launch Jupyter Notebook:

    jupyter notebook

Then open:

`Demand_Forecasting_GitHub_Portfolio.ipynb`

Before running the notebook, download `demand_forecasting.csv` from the [original dataset repository](https://github.com/Onurbltc/DemandForecastingDataset).

---

## Google Colab

The project can also be run using Google Colab.

1. Download `demand_forecasting.csv` from the [original dataset repository](https://github.com/Onurbltc/DemandForecastingDataset).
2. Open `Demand_Forecasting_GitHub_Portfolio.ipynb` in Google Colab.
3. Upload `demand_forecasting.csv` to the Colab environment.
4. Run the notebook from top to bottom.

The notebook currently expects the dataset at:

`/content/demand_forecasting.csv`

If the dataset is stored somewhere else, update the `DATA_PATH` variable before running the notebook.

---

## Requirements

The primary Python dependencies are:

    numpy
    pandas
    matplotlib
    seaborn
    scikit-learn
    xgboost

These dependencies are also listed in `requirements.txt`.

---

## Reproducibility

The modeling workflow uses a fixed random state where applicable to improve reproducibility.

The complete preprocessing and model-training workflow is contained in the notebook, including:

1. Loading and inspecting the dataset
2. Sorting observations chronologically
3. Creating calendar features
4. Creating lag and rolling features
5. Removing observations without sufficient historical data
6. Creating the chronological train/test split
7. Building the preprocessing pipeline
8. Performing time-series cross-validation
9. Tuning XGBoost
10. Evaluating the final model
11. Comparing against the naive baseline
12. Analyzing residuals
13. Examining feature importance
14. Evaluating product-level forecast errors

---

## Project Background and Attribution

This project was initially developed while following Onur Baltaci's demand forecasting tutorial and using the dataset provided through his [DemandForecastingDataset repository](https://github.com/Onurbltc/DemandForecastingDataset).

The tutorial and accompanying dataset provided the foundation for the project.

I subsequently expanded the modeling workflow to place greater emphasis on forecasting methodology and model evaluation. The final version in this repository incorporates:

- Chronological model evaluation
- Time-series cross-validation
- Historical lag features
- Rolling demand statistics
- Target-leakage prevention
- One-hot categorical encoding
- Naive forecast benchmarking
- Multiple evaluation metrics
- Residual diagnostics
- Product-level error analysis
- More explicit discussion of model limitations

The goal of these additions was to move the project from a standard machine learning regression exercise toward a more defensible demand forecasting workflow while clearly acknowledging the tutorial and dataset that provided the original starting point.

---

## What I Learned

This project reinforced several important lessons about forecasting with machine learning.

One of the most important was that **forecasting requires different validation choices than ordinary regression**. A model intended to predict future observations should be evaluated using data that occurs after the training period.

The project also demonstrated the importance of historical feature engineering. Lag and rolling variables allow a tree-based model such as XGBoost to incorporate information about recent demand behavior.

Another important lesson was the need to prevent target leakage. Historical features must be constructed so that information from the observation being predicted does not enter the model indirectly.

Finally, comparing the model with a simple baseline provided much more context than reporting RMSE or R² alone. The comparison helped determine whether the additional modeling complexity actually produced useful predictive improvement.

---

## Portfolio Takeaway

This project demonstrates an end-to-end machine learning forecasting workflow with particular emphasis on:

**Temporal validation, leakage prevention, historical feature engineering, baseline comparison, reproducible preprocessing, hyperparameter tuning, and transparent model evaluation.**

The final XGBoost model achieved an **RMSE of 12.39**, **MAE of 8.60**, and **R² of 0.921** on the future holdout period and reduced RMSE by approximately **77.7%** compared with the naive previous-demand baseline.

---

## Author

**Alfred Abraham**

Data Analytics / Data Science Portfolio Project
