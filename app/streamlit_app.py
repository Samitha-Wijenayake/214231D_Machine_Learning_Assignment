"""
Sri Lanka Vehicle Price Predictor — Main Streamlit App.
Imports tab modules from app/ for a clean, modular structure.
"""
import os
import sys
import streamlit as st
import pandas as pd

# Ensure the project root is on the path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.helpers import load_resources
from app import tab_prediction, tab_performance, tab_explainability, tab_data_insights

# --- Page Config ---
st.set_page_config(page_title="Sri Lanka Vehicle Price Predictor", layout="wide", page_icon="🚗")

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header { font-size: 4rem !important; font-weight: 800 !important; color: #FFFFFF !important; margin-bottom: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px; padding: 20px; text-align: center;
        border: 1px solid #0f3460;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #00E676; }
    .metric-label { font-size: 0.85rem; color: #90A4AE; margin-top: 4px; }
    .price-box {
        background: linear-gradient(135deg, #004d40 0%, #00695c 100%);
        border-radius: 16px; padding: 30px; text-align: center;
        border: 2px solid #00E676; margin: 20px 0;
    }
    .price-value { font-size: 2.5rem; font-weight: 800; color: #fff; }
    .price-range { font-size: 1rem; color: #B2DFDB; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)

# --- Load Resources ---
model, mappings, feature_names, all_metrics = load_resources()
test_metrics = all_metrics.get("Test", {})

# --- Header ---
st.markdown('<p class="main-header">Sri Lanka Vehicle Price Predictor</p>', unsafe_allow_html=True)

# --- Metrics Row ---
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">RMSE</div>
        <div class="metric-value">Rs. {test_metrics.get('RMSE', 0):,.0f}</div>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">MAE</div>
        <div class="metric-value">Rs. {test_metrics.get('MAE', 0):,.0f}</div>
    </div>""", unsafe_allow_html=True)
with m3:
    r2 = test_metrics.get('R2', 0)
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">R² Score</div>
        <div class="metric-value">{r2:.4f}</div>
    </div>""", unsafe_allow_html=True)
with m4:
    n_samples = 0
    if os.path.exists("data/processed.csv"):
        n_samples = len(pd.read_csv("data/processed.csv"))
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Training Samples</div>
        <div class="metric-value">{n_samples:,}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# --- Tabs ---
tab_predict, tab_perf, tab_explain, tab_data = st.tabs([
    "🔍 Price Prediction", "📈 Model Performance", "🧠 Explainability", "📊 Data Insights"
])

with tab_predict:
    tab_prediction.render(model, mappings, feature_names, test_metrics)

with tab_perf:
    tab_performance.render(model, all_metrics)

with tab_explain:
    tab_explainability.render(model, feature_names)

with tab_data:
    tab_data_insights.render(mappings)
