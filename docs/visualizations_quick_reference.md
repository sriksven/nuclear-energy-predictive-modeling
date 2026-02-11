# Data Visualizations - Quick Reference Guide

This document provides a visual overview of all plots generated in the Nuclear Energy Predictive Modeling project.

---

## 📊 Exploratory Data Analysis (EDA)

### Snapshot Data (2007 Reactor Data)

````carousel
![Correlation Matrix - Shows strong positive correlation between reactor capacity (MWe) and uranium demand](/Users/sriks/Documents/Projects/nuclear-energy-predictive-modeling/ml_pipeline/visualizations/eda/snapshot_correlation_matrix.png)
<!-- slide -->
![Top Uranium Consumers - Bar chart showing countries with highest uranium requirements in 2007](/Users/sriks/Documents/Projects/nuclear-energy-predictive-modeling/ml_pipeline/visualizations/eda/snapshot_top_consumers.png)
<!-- slide -->
![Pairplot Analysis - Scatter plot matrix showing relationships between all variables](/Users/sriks/Documents/Projects/nuclear-energy-predictive-modeling/ml_pipeline/visualizations/eda/snapshot_pairplot.png)
````

### Time-Series Data (2007-2025)

````carousel
![Historical Trend - Global uranium demand over 18 years showing steady upward trajectory](/Users/sriks/Documents/Projects/nuclear-energy-predictive-modeling/ml_pipeline/visualizations/eda/timeseries_trend.png)
<!-- slide -->
![Distribution Analysis - Histogram and density plot of uranium demand values](/Users/sriks/Documents/Projects/nuclear-energy-predictive-modeling/ml_pipeline/visualizations/eda/timeseries_distribution.png)
<!-- slide -->
![ACF/PACF Analysis - Autocorrelation plots used to determine SARIMA parameters](/Users/sriks/Documents/Projects/nuclear-energy-predictive-modeling/ml_pipeline/visualizations/eda/timeseries_acf_pacf.png)
````

---

## 🤖 Regression Models (Capacity → Demand)

### Model Performance & Validation

````carousel
![Model Comparison - Bar chart comparing R², MAE, and RMSE across all three models](/Users/sriks/Documents/Projects/nuclear-energy-predictive-modeling/ml_pipeline/regression_analysis/visualizations/model_comparison_metrics.png)
<!-- slide -->
![Residual Analysis - Scatter plot showing prediction errors are randomly distributed around zero](/Users/sriks/Documents/Projects/nuclear-energy-predictive-modeling/ml_pipeline/regression_analysis/visualizations/model_residuals.png)
<!-- slide -->
![Actual vs Predicted - Points cluster tightly around diagonal, demonstrating 95.73% R² accuracy](/Users/sriks/Documents/Projects/nuclear-energy-predictive-modeling/ml_pipeline/regression_analysis/visualizations/model_validation_plot.png)
<!-- slide -->
![Feature Importance - Reactor capacity (MWe) is the dominant predictor in Random Forest model](/Users/sriks/Documents/Projects/nuclear-energy-predictive-modeling/ml_pipeline/regression_analysis/visualizations/feature_importance.png)
````

### Future Projections

![Demand Projection - Scenario analysis showing potential 44% increase if all proposed reactors are built](/Users/sriks/Documents/Projects/nuclear-energy-predictive-modeling/ml_pipeline/regression_analysis/visualizations/demand_projection_plot.png)

---

## 📈 Time-Series Forecasting (SARIMA)

### Model Validation & Forecasting

````carousel
![Seasonal Decomposition - Breaks down time series into trend, seasonal, and residual components](/Users/sriks/Documents/Projects/nuclear-energy-predictive-modeling/ml_pipeline/time_series_forecast/visualizations/sarima_decomposition.png)
<!-- slide -->
![Backtest Validation - SARIMA forecast vs actual 2024-2025 data (MAE: 594 tonnes, <1% error)](/Users/sriks/Documents/Projects/nuclear-energy-predictive-modeling/ml_pipeline/time_series_forecast/visualizations/sarima_backtest.png)
<!-- slide -->
![5-Year Forecast - Predicts demand reaching ~71,700 tonnes by 2030 with confidence intervals](/Users/sriks/Documents/Projects/nuclear-energy-predictive-modeling/ml_pipeline/time_series_forecast/visualizations/sarima_forecast.png)
````

---

## 📋 Summary Statistics

### Regression Model Performance (Test Set)
| Model | R² Score | MAE (tonnes) | RMSE (tonnes) |
|-------|----------|--------------|---------------|
| **Random Forest** | **0.9573** | **353.45** | **701.73** |
| XGBoost | 0.9536 | 530.88 | 731.61 |
| Linear Regression | 0.9176 | 447.77 | 975.29 |

### SARIMA Model Performance (Backtest)
| Metric | Value | Interpretation |
|--------|-------|----------------|
| MAE | 594.51 tonnes | Average forecast error |
| RMSE | 741.33 tonnes | Error with penalty for outliers |
| **Relative Error** | **~0.8%** | Error as % of total demand |

---

## 🔑 Key Takeaways

1. **Strong Predictive Accuracy:** Regression models achieve 95.73% R² on test data
2. **Validated Forecasting:** SARIMA model achieves <1% error on 2024-2025 backtest
3. **Comprehensive Testing:** All models validated using train-test splits and backtesting
4. **Future Outlook:** Demand projected to reach ~71,700 tonnes by 2030
5. **Scenario Planning:** Potential 44% increase if all proposed reactors come online

---

**All visualizations are located in:**
- EDA: `ml_pipeline/visualizations/eda/`
- Regression: `ml_pipeline/regression_analysis/visualizations/`
- Time-Series: `ml_pipeline/time_series_forecast/visualizations/`
