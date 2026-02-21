"""
Tab 3: Explainability — feature importance, SHAP summary, SHAP dependence.
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import shap
import plotly.graph_objects as go
from app.helpers import (
    clean_name, plotly_layout, axis_style, HIDE_FEATS,
)


def render(model, feature_names):
    """Render the explainability tab content."""
    st.subheader("Model Explainability")

    ex1, ex2 = st.columns(2)

    # --- Feature Importance ---
    with ex1:
        st.markdown("**Feature Importance (XGBoost)**")
        st.caption("Shows which features the model relies on most when making predictions.")
        if model is not None:
            importances = model.feature_importances_
            feat_imp = [(f, v) for f, v in zip(feature_names, importances) if f not in HIDE_FEATS]
            feat_imp.sort(key=lambda x: x[1])
            top_fi = feat_imp[-15:]
            names_fi = [clean_name(f) for f, v in top_fi]
            vals_fi = [v for f, v in top_fi]

            fig_fi = go.Figure()
            fig_fi.add_trace(go.Bar(
                y=names_fi, x=vals_fi, orientation='h',
                marker=dict(color=vals_fi, colorscale='Viridis', line=dict(width=0.5, color='white')),
                text=[f'{v:.3f}' for v in vals_fi], textposition='outside',
                textfont=dict(size=10, color='white'),
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
            ))
            fig_fi.update_layout(**plotly_layout(
                xaxis=axis_style("Importance Score"), yaxis=axis_style(),
            ))
            st.plotly_chart(fig_fi, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Model not found. Run `python src/train.py`.")

    # --- SHAP Summary ---
    with ex2:
        st.markdown("**SHAP Summary (Global)**")
        st.caption("Shows mean impact of each feature on model output. Longer bars = more influential features.")
        try:
            train_path = "data/train.csv"
            if os.path.exists(train_path) and model is not None:
                df_train = pd.read_csv(train_path)
                X_train = df_train.drop("price", axis=1)
                explainer_g = shap.TreeExplainer(model)
                shap_vals_g = explainer_g.shap_values(X_train.sample(min(300, len(X_train)), random_state=42))

                mean_abs = np.mean(np.abs(shap_vals_g), axis=0)
                feat_shap = [(f, v) for f, v in zip(feature_names, mean_abs) if f not in HIDE_FEATS]
                feat_shap.sort(key=lambda x: x[1])
                top_shap = feat_shap[-15:]
                s_names = [clean_name(f) for f, v in top_shap]
                s_vals = [v for f, v in top_shap]

                fig_shap = go.Figure()
                fig_shap.add_trace(go.Bar(
                    y=s_names, x=s_vals, orientation='h',
                    marker=dict(color=s_vals, colorscale='Plasma', line=dict(width=0.5, color='white')),
                    text=[f'{v:,.0f}' for v in s_vals], textposition='outside',
                    textfont=dict(size=10, color='white'),
                    hovertemplate='<b>%{y}</b><br>Mean |SHAP|: %{x:,.0f}<extra></extra>'
                ))
                fig_shap.update_layout(**plotly_layout(
                    xaxis=axis_style("Mean |SHAP Value|"), yaxis=axis_style(),
                ))
                st.plotly_chart(fig_shap, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("Training data not found. Run `python src/preprocess.py`.")
        except Exception as e:
            st.info(f"Could not compute SHAP summary. Error: {e}")

    st.markdown("---")

    # --- SHAP Dependence ---
    st.markdown("**SHAP Dependence Plot (Top Feature)**")
    st.caption("Shows how the most important feature's value affects the predicted price.")
    try:
        train_path = "data/train.csv"
        if os.path.exists(train_path) and model is not None:
            df_train = pd.read_csv(train_path)
            X_train = df_train.drop("price", axis=1)
            sample = X_train.sample(min(500, len(X_train)), random_state=42)
            explainer_d = shap.TreeExplainer(model)
            shap_vals_d = explainer_d.shap_values(sample)

            mean_abs_d = np.mean(np.abs(shap_vals_d), axis=0)
            feat_ranking = [(f, v, i) for i, (f, v) in enumerate(zip(feature_names, mean_abs_d)) if f not in HIDE_FEATS]
            feat_ranking.sort(key=lambda x: x[1], reverse=True)
            top_feat_name, _, top_idx = feat_ranking[0]

            feat_values = sample.iloc[:, top_idx].values
            shap_for_feat = shap_vals_d[:, top_idx]

            fig_dep = go.Figure()
            fig_dep.add_trace(go.Scatter(
                x=feat_values, y=shap_for_feat, mode='markers',
                marker=dict(
                    color=feat_values, colorscale='RdYlBu_r', size=6, opacity=0.7,
                    colorbar=dict(title=dict(text=clean_name(top_feat_name), font=dict(color='white')), tickfont=dict(color='white')),
                    line=dict(width=0.3, color='white')
                ),
                hovertemplate=f'<b>{clean_name(top_feat_name)}</b>: %{{x:,.1f}}<br>SHAP: %{{y:,.0f}}<extra></extra>'
            ))
            fig_dep.update_layout(**plotly_layout(
                xaxis=axis_style(clean_name(top_feat_name)), yaxis=axis_style("SHAP Value (Impact on Price)"),
                height=450,
            ))
            st.plotly_chart(fig_dep, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Training data not found. Run `python src/preprocess.py`.")
    except Exception as e:
        st.info(f"Could not compute dependence plot. Error: {e}")
