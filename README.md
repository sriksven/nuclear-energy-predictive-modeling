# Nuclear Energy Predictive Modeling

This project analyzes historical and current nuclear reactor data to predict global uranium demand. It employs a dual-approach strategy: **Regression Analysis** for capacity-based demand prediction and **Time-Series Forecasting** (SARIMA) for temporal trend analysis.

## Key Features
-   **Automated Data Pipeline**: Scrapers to fetch "Snapshot" reactor data (2007) and "Historical" time-series data (2007-2025) from the World Nuclear Association.
-   **Exploratory Data Analysis (EDA)**: Automated generation of correlation matrices, trend plots, and seasonality checks.
-   **Ensemble Regression Modeling**: Comparitive analysis of **Linear Regression**, **Random Forest**, and **XGBoost** to model the relationship between Reactor Capacity (MWe) and Uranium Demand.
-   **Time-Series Forecasting**: **SARIMA** modeling with backtesting to forecast future global demand trends.

## Project Structure

```
nuclear-energy-predictive-modeling/
├── data/                               # Central Data Storage
│   ├── uranium_snapshot.csv            # Cleaned 2007 Reactor Data
│   └── uranium_time_series.csv         # Historical Demand Data (2007-2025)
│
├── data_pipeline/                      # Data Acquisition & EDA
│   ├── scraper_snapshot.py             # Scrapes snapshot data
│   ├── scrape_history.py               # Scrapes historical time-series
│   └── perform_eda.py                  # Generates EDA visualizations
│
├── ml_pipeline/                        # Machine Learning Pipelines
│   ├── regression_analysis/            # Snapshot Models
│   │   ├── train_linear_rf_xgboost.py  # Trains Regression Models
│   │   └── visualizations/             # Regression plots & metrics
│   │
│   └── time_series_forecast/           # Forecasting Models
│       ├── train_sarima.py             # Trains SARIMA Model
│       └── visualizations/             # Forecast plots & backtests
│
└── requirements.txt                    # Project Dependencies
```

## Setup & Installation

1.  **Clone the repository**:
    ```bash
    git clone <repo-url>
    cd nuclear-energy-predictive-modeling
    ```

2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Pipeline
Run the scrapers to populate the `data/` directory.
```bash
# Fetch Snapshot Data
python3 data_pipeline/scraper_snapshot.py

# Fetch Historical Time-Series
python3 data_pipeline/scrape_history.py

# Run Exploratory Data Analysis (EDA)
python3 data_pipeline/perform_eda.py
```

### 2. Regression Analysis (Capacity vs Demand)
Train and evaluate Linear Regression, Random Forest, and XGBoost models.
```bash
python3 ml_pipeline/regression_analysis/train_linear_rf_xgboost.py
```
*Outputs metrics to `ml_pipeline/regression_analysis/metrics_comparison.csv` and plots to `visualizations/`.*

### 3. Time-Series Forecasting (SARIMA)
Train the SARIMA model and generate a 5-year forecast.
```bash
python3 ml_pipeline/time_series_forecast/train_sarima.py
```
*Outputs forecast plots and backtest results to `ml_pipeline/time_series_forecast/visualizations/`.*

## Key Results

### Regression Performance (Test Set)
| Model | R² Score | MAE (tonnes) |
| :--- | :--- | :--- |
| **Random Forest** | **0.9573** | **353.45** |
| XGBoost | 0.9536 | 530.88 |
| Linear Regression | 0.9176 | 447.77 |

### Forecast (SARIMA)
-   **Backtest Accuracy**: MAE of ~594 tonnes (0.8% error) on 2024-2025 data.
-   **Trend**: Predicts a steady increase in demand, reaching **~71,700 tonnes by 2030**.