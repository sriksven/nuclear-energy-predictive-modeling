import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def setup_eda_dirs():
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'ml_pipeline', 'visualizations', 'eda')
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def perform_snapshot_eda(output_dir):
    print("\n--- Performing Snapshot Data EDA (2007) ---")
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'uranium_snapshot.csv')
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("Snapshot data not found.")
        return

    # Filter for relevant columns
    cols = ['Reactors_Operating_No', 'Reactors_Operating_MWe', 'Uranium_Required_2007_tonnes']
    subset = df[cols].dropna()

    # 1. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(subset.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix (Reactor stats vs Uranium Demand)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'snapshot_correlation_matrix.png'))
    print("Saved snapshot_correlation_matrix.png")

    # 2. Pairplot (Distributions & Relationships)
    sns.pairplot(subset, diag_kind='kde', height=2.5)
    plt.suptitle('Pairplot of Reactor Capacity and Uranium Demand', y=1.02)
    plt.savefig(os.path.join(output_dir, 'snapshot_pairplot.png'))
    print("Saved snapshot_pairplot.png")

    # 3. Top 10 Consumers (Bar Chart)
    top_consumers = df.nlargest(10, 'Uranium_Required_2007_tonnes')
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_consumers, x='Uranium_Required_2007_tonnes', y='Country', palette='viridis')
    plt.title('Top 10 Countries by Uranium Demand (2007)')
    plt.xlabel('Uranium Required (Tonnes)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'snapshot_top_consumers.png'))
    print("Saved snapshot_top_consumers.png")

def perform_timeseries_eda(output_dir):
    print("\n--- Performing Time-Series EDA (2007-2025) ---")
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'uranium_time_series.csv')
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("Time-series data not found.")
        return

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    
    # Interpolate for clean plotting
    y = df['Global_Uranium_Demand'].resample('MS').mean().interpolate(method='linear')

    # 1. Trend Analysis
    plt.figure(figsize=(12, 6))
    plt.plot(y, label='Global Uranium Demand', color='#2c3e50')
    # Rolling mean for trend
    plt.plot(y.rolling(window=12).mean(), label='12-Month Moving Average', color='#e74c3c', linestyle='--')
    plt.title('Global Uranium Demand Trend (2007-2025)')
    plt.xlabel('Year')
    plt.ylabel('Demand (Tonnes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'timeseries_trend.png'))
    print("Saved timeseries_trend.png")

    # 2. Distribution of Demand
    plt.figure(figsize=(8, 5))
    sns.histplot(y, kde=True, color='purple')
    plt.title('Distribution of Monthly Uranium Demand')
    plt.xlabel('Demand (Tonnes)')
    plt.savefig(os.path.join(output_dir, 'timeseries_distribution.png'))
    print("Saved timeseries_distribution.png")

    # 3. Autocorrelation Analysis (ACF & PACF)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plot_acf(y, ax=axes[0], lags=40)
    plot_pacf(y, ax=axes[1], lags=40)
    axes[0].set_title('Autocorrelation Function (ACF)')
    axes[1].set_title('Partial Autocorrelation Function (PACF)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timeseries_acf_pacf.png'))
    print("Saved timeseries_acf_pacf.png")

    # 4. Stationarity Test (ADF)
    print("\nAugmented Dickey-Fuller Test Results:")
    result = adfuller(y.dropna())
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print("Critical Values:")
    for key, value in result[4].items():
        print(f'\t{key}: {value:.4f}')
    
    if result[1] > 0.05:
        print("Interpretation: The time series is likely NON-STATIONARY (p > 0.05).")
    else:
        print("Interpretation: The time series is likely STATIONARY (p <= 0.05).")

if __name__ == "__main__":
    output_dir = setup_eda_dirs()
    perform_snapshot_eda(output_dir)
    perform_timeseries_eda(output_dir)
