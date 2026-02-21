"""
Tab 2: Model Performance — metrics table, predicted vs actual, residuals.
"""
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from app.helpers import plotly_layout, axis_style


def render(model, all_metrics):
    """Render the model performance tab content."""
    st.subheader("Model Performance Across Splits")

    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics).T
        metrics_df.index.name = "Split"
        st.dataframe(metrics_df.style.format("{:,.2f}"), use_container_width=True)

    st.markdown("---")

    test_path = "data/test.csv"
    if os.path.exists(test_path) and model is not None:
        df_test = pd.read_csv(test_path)
        y_test = df_test["price"]
        X_test = df_test.drop("price", axis=1)
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred

        pc1, pc2 = st.columns(2)
        with pc1:
            st.markdown("**Predicted vs Actual**")
            st.caption("Points closer to the diagonal line indicate better accuracy.")
            fig_pva = go.Figure()
            fig_pva.add_trace(go.Scatter(
                x=y_test, y=y_pred, mode='markers',
                marker=dict(color='#42A5F5', size=5, opacity=0.6),
                hovertemplate='Actual: Rs. %{x:,.0f}<br>Predicted: Rs. %{y:,.0f}<extra></extra>'
            ))
            max_val = max(y_test.max(), y_pred.max())
            fig_pva.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val], mode='lines',
                line=dict(color='#FF5252', dash='dash', width=2), showlegend=False
            ))
            fig_pva.update_layout(**plotly_layout(
                xaxis=axis_style("Actual Price (Rs.)"), yaxis=axis_style("Predicted Price (Rs.)"),
            ))
            st.plotly_chart(fig_pva, use_container_width=True, config={'displayModeBar': False})

        with pc2:
            st.markdown("**Residual Distribution**")
            st.caption("A bell-shaped curve centered at zero means no systematic bias.")
            fig_res = go.Figure()
            fig_res.add_trace(go.Histogram(
                x=residuals, nbinsx=50,
                marker=dict(color='#AB47BC', line=dict(color='white', width=0.5)),
                hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>'
            ))
            fig_res.update_layout(**plotly_layout(
                xaxis=axis_style("Residual (Actual - Predicted)"), yaxis=axis_style("Count"),
            ))
            st.plotly_chart(fig_res, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("Test data or model not found. Run `python src/evaluate.py`.")
