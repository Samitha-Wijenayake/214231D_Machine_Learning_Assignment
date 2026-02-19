import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os

# Create plots directory if it doesn't exist
if not os.path.exists(os.path.join('outputs', 'explainability')):
    os.makedirs(os.path.join('outputs', 'explainability'))

def shap_analysis(data_path, model_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Preprocess Data (Same as training)
    drop_cols = ['Title', 'Price', 'Brand', 'Location', 'Location_Clean', 'Price_Normalized', 'Mileage_Normalized', 'Description', 'PublishedDate', 'Link', 'ImageURL']
    if 'Unnamed: 0' in df.columns:
        drop_cols.append('Unnamed: 0')
        
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Ensure all remaining columns are numeric (same logic as training)
    non_numeric = X.select_dtypes(include=['object']).columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)
        
    print(f"Features: {X.columns.tolist()}")
    
    # Load Model
    print(f"Loading model from {model_path}...")
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    
    # SHAP Explainer
    print("Computing SHAP values...")
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    
    # 1. Summary Plot
    print("Generating SHAP Summary Plot...")
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig(os.path.join('outputs', 'explainability', 'shap_summary.png'))
    plt.close()
    
    # 2. Bar Plot (Global Importance)
    print("Generating SHAP Bar Plot...")
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join('outputs', 'explainability', 'shap_bar.png'))
    plt.close()
    
    # 3. Dependence Plots for Top Features
    # Find top 2 features by mean absolute SHAP value
    mean_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_shap)[::-1]
    top_features = X.columns[top_indices[:2]]
    
    for feature in top_features:
        print(f"Generating SHAP Dependence Plot for {feature}...")
        plt.figure()
        shap.plots.scatter(shap_values[:, feature], show=False, color=shap_values)
        plt.title(f"SHAP Dependence Plot: {feature}")
        plt.tight_layout()
        plt.savefig(os.path.join('outputs', 'explainability', f'shap_dependence_{feature}.png'))
        plt.close()

    print("SHAP analysis complete. Plots saved to outputs/explainability/")

if __name__ == "__main__":
    shap_analysis('data/processed/vehicle_data_withprocessed.csv', os.path.join('outputs', 'model.json'))
