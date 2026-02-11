import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Cache the data loading to prevent reloading on every interaction
@st.cache_data
def load_data():
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    # Load Snapshot Data (for Regression)
    snapshot_path = os.path.join(base_dir, 'uranium_snapshot.csv')
    df_snap = pd.read_csv(snapshot_path)
    
    # Load Time-Series Data (for Forecasting)
    ts_path = os.path.join(base_dir, 'uranium_time_series.csv')
    df_ts = pd.read_csv(ts_path)
    df_ts['Date'] = pd.to_datetime(df_ts['Date'])
    df_ts = df_ts.set_index('Date').sort_index()
    
    # Resample Time Series (Interpolated Month Start)
    df_ts_monthly = df_ts['Global_Uranium_Demand'].resample('MS').mean().interpolate(method='linear')
    
    return df_snap, df_ts_monthly

# Cache the model training (Retrain Random Forest on load)
@st.cache_resource
def load_rf_model(df_snap):
    # Filter for valid rows
    train_df = df_snap[df_snap['Uranium_Required_2007_tonnes'] > 0].copy()
    
    X = train_df[['Reactors_Operating_MWe', 'Reactors_Operating_No']]
    y = train_df['Uranium_Required_2007_tonnes']
    
    # Train Full Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

# Cache the SARIMA forecast (Retrain/Fit on load)
@st.cache_resource
def load_sarima_forecast(ts_data, periods=60):
    # Train SARIMA(1,1,1)x(1,1,1,12)
    model = SARIMAX(ts_data, 
                    order=(1, 1, 1), 
                    seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    results = model.fit(disp=False)
    
    # Forecast
    pred_uc = results.get_forecast(steps=periods)
    pred_mean = pred_uc.predicted_mean
    pred_ci = pred_uc.conf_int()
    
    # Fix indices for plotting
    forecast_index = pd.date_range(ts_data.index[-1], periods=periods+1, freq='MS')[1:]
    pred_mean.index = forecast_index
    pred_ci.index = forecast_index
    
    return pred_mean, pred_ci
