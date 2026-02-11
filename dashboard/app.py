import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_loader import load_data, load_rf_model, load_sarima_forecast

# Page Config
st.set_page_config(
    page_title="Nuclear Energy Predictive Dashboard",
    page_icon="⚛️",
    layout="wide"
)

# Custom CSS for aesthetics
st.markdown("""
<style>
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #262730;
        border-radius: 5px;
        padding: 15px;
        text-align: center;
    }
    .metric-value {
        font_size: 2em;
        font-weight: bold;
        color: #00ff7f;
    }
</style>
""", unsafe_allow_html=True)

# Load Data
df_snap, df_ts_monthly = load_data()
rf_model = load_rf_model(df_snap)

# Title
st.title("⚛️ Nuclear Energy Predictive Modeling")
st.markdown("Explore global uranium demand trends, correlations, and future forecasts.")

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["Overview & EDA", "Scenario Simulator (Regression)", "Future Forecast (SARIMA)"])

# --- PAGE 1: OVERVIEW & EDA ---
if page == "Overview & EDA":
    st.header("Global Nuclear Overview (2007)")
    
    # Top Metrics
    total_demand = df_snap['Uranium_Required_2007_tonnes'].sum()
    total_reactors = df_snap['Reactors_Operating_No'].sum()
    total_capacity = df_snap['Reactors_Operating_MWe'].sum()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🌍 Total Uranium Demand", f"{total_demand:,.0f} tonnes")
    col2.metric("🏭 Total Reactors", f"{total_reactors:,.0f}")
    col3.metric("⚡ Total Capacity", f"{total_capacity/1000:,.1f} GW")
    
    st.divider()
    
    # Interactive Map
    st.subheader("Global Uranium Demand by Country")
    fig_map = px.choropleth(df_snap, locations="Country", locationmode="country names",
                            color="Uranium_Required_2007_tonnes",
                            hover_name="Country",
                            color_continuous_scale="Viridis",
                            title="Uranium Demand Heatmap")
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Correlation Heatmap (Simplified)
    st.subheader("Drivers of Demand")
    corr_cols = ['Reactors_Operating_MWe', 'Reactors_Operating_No', 'Uranium_Required_2007_tonnes']
    corr = df_snap[corr_cols].corr()
    
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
    st.plotly_chart(fig_corr, use_container_width=True)

# --- PAGE 2: SCENARIO SIMULATOR ---
elif page == "Scenario Simulator (Regression)":
    st.header("🛠️ Demand Simulator")
    st.markdown("""
    Use the sliders below to adjust global reactor capacity and see the predicted impact on Uranium demand.
    **Model Used: Random Forest Regressor**
    """)
    
    # Sidebar Sliders for Simulation
    st.sidebar.header("Scenario Inputs")
    
    # Base Values (Avg per country or Total? Let's do Total Global Simulation)
    # Since model is trained on Country level, we need to be careful.
    # Approach: We will simulate a "New Country" or "Global Addition".
    # Better Approach: Scale ALL countries by a percentage.
    
    growth_pct = st.sidebar.slider("Global Capacity Growth (%)", min_value=-20, max_value=100, value=0, step=5)
    
    # Apply Growth
    df_sim = df_snap.copy()
    df_sim['Reactors_Operating_MWe'] = df_sim['Reactors_Operating_MWe'] * (1 + growth_pct/100)
    # Assume reactor count grows proportionally? Or just keep it separate? 
    # Let's assume proportional growth for simplicity in this demo, or fixed efficiency.
    # Actually, let's just scale MWe since it's the dominant factor.
    
    # Predict New Demand
    X_sim = df_sim[['Reactors_Operating_MWe', 'Reactors_Operating_No']]
    predicted_demand_country = rf_model.predict(X_sim)
    total_pred_demand = predicted_demand_country.sum()
    
    # Display Results
    current_demand = df_snap['Uranium_Required_2007_tonnes'].sum()
    delta = total_pred_demand - current_demand
    
    col1, col2 = st.columns(2)
    col1.metric("Projected Global Demand", f"{total_pred_demand:,.0f} tonnes", delta=f"{delta:,.0f} tonnes")
    col2.metric("Capacity Variance", f"{growth_pct}%")
    
    # Plot Comparison
    scenarios =pd.DataFrame({
        'Scenario': ['Current Baseline', 'Simulated Scenario'],
        'Demand (Tonnes)': [current_demand, total_pred_demand]
    })
    
    fig_bar = px.bar(scenarios, x='Scenario', y='Demand (Tonnes)', color='Scenario',
                     color_discrete_sequence=['#3498db', '#e74c3c'])
    st.plotly_chart(fig_bar, use_container_width=True)

# --- PAGE 3: FUTURE FORECAST ---
elif page == "Future Forecast (SARIMA)":
    st.header("📈 Time-Series Forecast (2025-2030)")
    st.markdown("Predicting future demand trends based on historical data (2007-2025).")
    
    # Forecast Settings
    forecast_years = st.sidebar.slider("Forecast Horizon (Years)", 1, 10, 5)
    months = forecast_years * 12
    
    # Load Forecast
    with st.spinner("Training SARIMA Model..."):
        pred_mean, pred_ci = load_sarima_forecast(df_ts_monthly, periods=months)
    
    # Plotly Chart
    fig = go.Figure()
    
    # Historical Data
    fig.add_trace(go.Scatter(x=df_ts_monthly.index, y=df_ts_monthly, name="Historical Data",
                             line=dict(color='#2c3e50', width=2)))
    
    # Forecast Data
    fig.add_trace(go.Scatter(x=pred_mean.index, y=pred_mean, name="Forecast",
                             line=dict(color='#e74c3c', width=3)))
    
    # Confidence Interval
    fig.add_trace(go.Scatter(x=pred_ci.index, y=pred_ci.iloc[:, 1], mode='lines',
                             line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=pred_ci.index, y=pred_ci.iloc[:, 0], mode='lines',
                             fill='tonexty', fillcolor='rgba(231, 76, 60, 0.2)',
                             line=dict(width=0), name="95% Confidence Interval"))
    
    fig.update_layout(title="Global Uranium Demand Forecast",
                      xaxis_title="Date", yaxis_title="Demand (Tonnes)",
                      hovermode="x unified")
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Forecast Data")
    st.dataframe(pred_mean.reset_index().rename(columns={'index': 'Date', 0: 'Predicted Demand'}))
