"""
Shared helpers: resource loading, plotly layout, utility functions.
"""
import os
import json
import xgboost as xgb
import pandas as pd
import plotly.graph_objects as go


# --- Constants ---
CHART_HEIGHT = 450
HIDE_FEATS = {"Brand_Encoded", "Model_Encoded", "Mileage_Normalized", "Price_Normalized"}


# --- Plotly Helpers ---
def plotly_layout(**overrides):
    """Return a shared Plotly layout dict with transparent background and animations."""
    base = dict(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=CHART_HEIGHT, margin=dict(l=10, r=50, t=30, b=40),
        transition=dict(duration=800, easing='cubic-in-out'),
    )
    base.update(overrides)
    return base


def axis_style(title=""):
    """Return a shared axis style dict."""
    return dict(title=title, color='white', gridcolor='rgba(255,255,255,0.1)')


# --- Name Cleanup ---
def clean_name(name):
    """Remove Loc_ prefix and clean up feature names for display."""
    if name.startswith("Loc_"):
        return name[4:]
    return name


# --- Resource Loading ---
def load_resources():
    """Load model, mappings, feature names, and metrics from disk."""
    model = xgb.XGBRegressor()
    model_path = "models/xgboost_model.json"
    if os.path.exists(model_path):
        model.load_model(model_path)
    else:
        model = None

    mappings_path = "models/mappings.json"
    mappings = {}
    if os.path.exists(mappings_path):
        with open(mappings_path, "r") as f:
            mappings = json.load(f)

    features_path = "models/feature_names.json"
    feature_names = []
    if os.path.exists(features_path):
        with open(features_path, "r") as f:
            feature_names = json.load(f)["features"]

    metrics_path = "outputs/metrics.json"
    all_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            all_metrics = json.load(f)

    return model, mappings, feature_names, all_metrics


def has_real_options(mappings, col_name):
    """Check if a column has real options (not just 'Unknown')."""
    opts = mappings.get(col_name, [])
    return len(opts) > 0 and opts != ["Unknown"]


def get_options(mappings, col_name):
    """Get a list of options for a column from mappings."""
    return mappings.get(col_name, ["Unknown"])
