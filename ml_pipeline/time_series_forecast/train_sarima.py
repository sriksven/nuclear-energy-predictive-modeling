import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import os

def train_sarima():
    # 1. Load Data
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'uranium_time_series.csv')
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: {data_path} not found.")
        return
        
    vis_dir = os.path.join(os.path.dirname(__file__), 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # 2. Preprocessing
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    
    print("\n--- Raw Data Stats ---")
    print(df.describe())
    
    # Resample to Monthly 'MS' (Month Start) and Interpolate
    # Data is irregular (every few months), so we need to fill gaps
    # Linear interpolation is reasonable for demand trends
    y = df['Global_Uranium_Demand'].resample('MS').mean().interpolate(method='linear')
    
    print(f"\nResampled to Monthly: {len(y)} months")
    print(y.head())

    # 3. Seasonal Decomposition (Check for seasonality)
    # We assume annual seasonality (period=12)
    decomposition = seasonal_decompose(y, model='additive', period=12)
    
    plt.figure(figsize=(10, 8))
    decomposition.plot()
    
    save_path = os.path.join(vis_dir, 'sarima_decomposition.png')
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    
    # 4. Backtesting (Train/Test Split)
    # Split data: Train until end of 2023, Test 2024-2025
    split_date = '2024-01-01'
    train = y[:split_date]
    test = y[split_date:] # Real data for 2024-2025
    
    print(f"\n--- Backtesting (Train until {split_date}) ---")
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    
    model_bt = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                       enforce_stationarity=False, enforce_invertibility=False)
    results_bt = model_bt.fit(disp=False)
    
    # Forecast for test period
    pred_bt = results_bt.get_forecast(steps=len(test))
    pred_bt_mean = pred_bt.predicted_mean
    
    # Calculate Metrics
    mae = np.mean(np.abs(pred_bt_mean - test))
    rmse = np.sqrt(mean_squared_error(test, pred_bt_mean))
    r2 = 0 # Not typically used for time series but good for comparison
    # r2_score needs sklearn import
    
    print(f"Test MAE:  {mae:.2f}")
    print(f"Test RMSE: {rmse:.2f}")
    
    # Plot Backtest
    plt.figure(figsize=(10, 6))
    plt.plot(train.index, train, label='Train (History)')
    plt.plot(test.index, test, label='Test (Actual)')
    plt.plot(pred_bt_mean.index, pred_bt_mean, label='Forecast (Backtest)', color='red', linestyle='--')
    plt.title('SARIMA Backtesting Validation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(vis_dir, 'sarima_backtest.png')
    plt.savefig(save_path)
    print(f"Saved {save_path}")

    # Export Metrics
    metrics_data = [{
        'Model': 'SARIMA (1,1,1)x(1,1,1,12)',
        'MAE': mae,
        'RMSE': rmse
    }]
    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = os.path.join(os.path.dirname(__file__), 'sarima_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)

    # 5. Final Full Training & Forecast
    print("\nTraining Final Model on Full Data...")
    model = SARIMAX(y, 
                    order=(1, 1, 1), 
                    seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
                    
    results = model.fit(disp=False)
    
    # Forecast next 60 months (5 years)
    forecast_steps = 60
    pred_uc = results.get_forecast(steps=forecast_steps)
    pred_ci = pred_uc.conf_int()
    
    forecast_index = pd.date_range(y.index[-1], periods=forecast_steps+1, freq='MS')[1:]
    pred_mean = pred_uc.predicted_mean
    pred_mean.index = forecast_index
    pred_ci.index = forecast_index
    
    # Print Forecast
    print(f"\nForecast for next 5 years (Head):")
    print(pred_mean.head())
    
    # 6. Plot Forecast
    plt.figure(figsize=(12, 6))
    plt.plot(y, label='Observed (Interpolated)')
    plt.plot(pred_mean, label='Forecast', color='red')
    plt.fill_between(pred_ci.index,
                     pred_ci.iloc[:, 0],
                     pred_ci.iloc[:, 1], color='pink', alpha=0.3)
    
    plt.xlabel('Date')
    plt.ylabel('Global Uranium Demand (tonnes)')
    plt.title('Global Uranium Demand Forecast (SARIMA)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(vis_dir, 'sarima_forecast.png')
    plt.savefig(save_path)
    print(f"Saved {save_path}")

if __name__ == "__main__":
    train_sarima()
