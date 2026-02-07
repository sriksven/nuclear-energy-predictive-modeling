import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
try:
    import xgboost as xgb
except ImportError:
    xgb = None

def train_and_predict():
    # Load data
    try:
        # Path updated for modular structure (relative to script location)
        data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'uranium_snapshot.csv')
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: {data_path} not found. Run scraper_snapshot.py first.")
        return
        
    # Define absolute visualizations path
    vis_dir = os.path.join(os.path.dirname(__file__), 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    print("Loaded data structure:")
    print(df.head())
    
    # ... (rest of feature engineering logic remains same) ...
    # Feature Engineering
    train_df = df[df['Uranium_Required_2007_tonnes'] > 0].copy()
    
    X = train_df[['Reactors_Operating_MWe', 'Reactors_Operating_No']]
    y = train_df['Uranium_Required_2007_tonnes']
    
    print(f"\nTraining on {len(train_df)} countries.")
    
    # Split training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Function to calculate and print metrics
    def evaluate_model(model, X_test, y_test, name):
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        print(f"\n--- {name} ---")
        print(f"R2 Score: {r2:.4f}")
        print(f"MAE:      {mae:.2f}")
        print(f"RMSE:     {rmse:.2f}")
        return {'Model': name, 'R2': r2, 'MAE': mae, 'RMSE': rmse}, pred

    # --- MODEL 1: Linear Regression (Baseline) ---
    lr_model = LinearRegression(fit_intercept=False) 
    lr_model.fit(X_train, y_train)
    lr_metrics, lr_pred = evaluate_model(lr_model, X_test, y_test, "Linear Regression")
    
    # --- MODEL 2: Random Forest (Ensemble) ---
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_metrics, rf_pred = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # --- MODEL 3: XGBoost (if available) ---
    xgb_model = None
    xgb_metrics = None
    if xgb:
        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_metrics, xgb_pred = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    
    # Collect Metrics
    all_metrics = [lr_metrics, rf_metrics]
    if xgb_metrics:
        all_metrics.append(xgb_metrics)
        
    metrics_df = pd.DataFrame(all_metrics)
    # Save Metrics CSV
    metrics_path = os.path.join(os.path.dirname(__file__), 'metrics_comparison.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics to {metrics_path}")

    # Select Best Model based on R2 (or RMSE)
    best_model_name = metrics_df.loc[metrics_df['R2'].idxmax(), 'Model']
    print(f"\nSelected Best Model: {best_model_name}")
    
    if best_model_name == "Linear Regression":
        best_model = lr_model
    elif best_model_name == "Random Forest":
        best_model = rf_model
    else:
        best_model = xgb_model

    # --- COMPARISON PLOTS ---
    # Plot 1: Metrics Comparison (Grouped Bar Chart)
    metrics_long = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_long, x="Metric", y="Score", hue="Model", palette="viridis")
    plt.title("Model Performance Comparison (R2, MAE, RMSE)")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    save_path = os.path.join(vis_dir, 'model_comparison_metrics.png')
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    
    # Plot 2: Residuals Comparsion (Actual - Predicted)
    plt.figure(figsize=(12, 5))
    plt.scatter(y_test, y_test - lr_pred, label='Linear Regression', alpha=0.6)
    plt.scatter(y_test, y_test - rf_pred, label='Random Forest', alpha=0.6)
    if xgb:
        plt.scatter(y_test, y_test - xgb_pred, label='XGBoost', alpha=0.6)
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Actual Uranium Demand')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Residual Analysis (Test Set)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(vis_dir, 'model_residuals.png')
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    
    # Plot 3: Feature Importance (Random Forest)
    if best_model_name != "Linear Regression":
        plt.figure(figsize=(8, 4))
        feat_importances = pd.Series(best_model.feature_importances_, index=X.columns)
        feat_importances.style = 'seaborn' # deprecated but harmless, let's just plot
        feat_importances.nlargest(10).plot(kind='barh', color='#2ecc71')
        plt.title(f'Feature Importance ({best_model_name})')
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        save_path = os.path.join(vis_dir, 'feature_importance.png')
        plt.savefig(save_path)
        print(f"Saved {save_path}")
    
    # Plot 4: Actual vs Predicted (Best Model)
    y_full_pred = best_model.predict(X)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y, y=y_full_pred)
    plt.plot([0, y.max()], [0, y.max()], 'r--')
    plt.xlabel('Actual Uranium Required (tonnes)')
    plt.ylabel(f'Predicted ({best_model_name})')
    plt.title(f'Actual vs Predicted - {best_model_name}')
    
    save_path = os.path.join(vis_dir, 'model_validation_plot.png')
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    
    # --- PROJECTIONS ---
    
    # Scenario A: Under Construction
    X_building = df[['Reactors_Building_MWe', 'Reactors_Building_No']].copy()
    X_building.columns = ['Reactors_Operating_MWe', 'Reactors_Operating_No'] 
    df['Predicted_Demand_Building'] = best_model.predict(X_building)
    
    # Scenario B: Planned
    X_planned = df[['Reactors_Planned_MWe', 'Reactors_Planned_No']].copy()
    X_planned.columns = ['Reactors_Operating_MWe', 'Reactors_Operating_No']
    df['Predicted_Demand_Planned'] = best_model.predict(X_planned)
    
    # Scenario C: Proposed
    X_proposed = df[['Reactors_Proposed_MWe', 'Reactors_Proposed_No']].copy()
    X_proposed.columns = ['Reactors_Operating_MWe', 'Reactors_Operating_No']
    df['Predicted_Demand_Proposed'] = best_model.predict(X_proposed)
    
    # Summarize Global Demand Growth
    current_total = df['Uranium_Required_2007_tonnes'].sum()
    building_add = df['Predicted_Demand_Building'].sum()
    planned_add = df['Predicted_Demand_Planned'].sum()
    proposed_add = df['Predicted_Demand_Proposed'].sum()
    
    print("\n--- GLOBAL URANIUM DEMAND PROJECTIONS (Ensemble Model) ---")
    print(f"Current Global Demand (2007): {current_total:,.0f} tonnes")
    print(f"Additional Demand (Under Construction): +{building_add:,.0f} tonnes")
    print(f"Additional Demand (Planned): +{planned_add:,.0f} tonnes")
    print(f"Additional Demand (Proposed): +{proposed_add:,.0f} tonnes")
    
    # Plot Projections
    scenarios = ['Current', '+ Building', '+ Planned', '+ Proposed']
    totals = [
        current_total,
        current_total + building_add,
        current_total + building_add + planned_add,
        current_total + building_add + planned_add + proposed_add
    ]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(scenarios, totals, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f}',
                ha='center', va='bottom')
                
    plt.title(f'Projected Global Uranium Demand Growth ({best_model_name})')
    plt.ylabel('Total Uranium Demand (tonnes)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    save_path = os.path.join(vis_dir, 'demand_projection_plot.png')
    plt.savefig(save_path)
    print(f"Saved {save_path}")

if __name__ == "__main__":
    train_and_predict()
