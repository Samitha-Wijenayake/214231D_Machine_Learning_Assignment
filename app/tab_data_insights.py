"""
Tab 4: Data Insights — dataset overview, price distribution, brand & location charts.
"""
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from app.helpers import has_real_options, get_options, plotly_layout, axis_style


def render(mappings):
    """Render the data insights tab content."""
    st.subheader("Dataset Overview")

    dataset_path = "data/processed.csv"
    if os.path.exists(dataset_path):
        df_full = pd.read_csv(dataset_path)

        di1, di2, di3 = st.columns(3)
        di1.metric("Total Records", f"{len(df_full):,}")
        di2.metric("Features", f"{len(df_full.columns) - 1}")
        di3.metric("Target", "price")

        st.markdown("---")

        ch1, ch2 = st.columns(2)

        # --- Price Distribution ---
        with ch1:
            st.markdown("**Price Distribution**")
            st.caption("Histogram of vehicle prices in the dataset.")
            fig_pd = go.Figure()
            fig_pd.add_trace(go.Histogram(
                x=df_full["price"].dropna(), nbinsx=50,
                marker=dict(color='#1E88E5', line=dict(color='white', width=0.5)),
                hovertemplate='Price: %{x}<br>Count: %{y}<extra></extra>'
            ))
            fig_pd.update_layout(**plotly_layout(
                xaxis=axis_style("Price (Rs.)"), yaxis=axis_style("Count"),
            ))
            st.plotly_chart(fig_pd, use_container_width=True, config={'displayModeBar': False})

        # --- Top Brands ---
        with ch2:
            if "Brand" in df_full.columns and has_real_options(mappings, "Brand"):
                st.markdown("**Top 15 Brands by Average Price**")
                st.caption("Average price for the most common vehicle brands.")
                brand_map = {i: name for i, name in enumerate(get_options(mappings, "Brand"))}
                df_full["Brand_Name"] = df_full["Brand"].map(brand_map)
                top_brands = df_full.groupby("Brand_Name")["price"].agg(["mean", "count"]).sort_values("count", ascending=False).head(15).sort_values("mean", ascending=True)

                fig_br = go.Figure()
                fig_br.add_trace(go.Bar(
                    y=top_brands.index, x=top_brands["mean"], orientation='h',
                    marker=dict(color=top_brands["mean"], colorscale='Emrld', line=dict(width=0.5, color='white')),
                    text=[f'Rs. {v:,.0f}' for v in top_brands["mean"]], textposition='outside',
                    textfont=dict(size=10, color='white'),
                    hovertemplate='<b>%{y}</b><br>Avg Price: Rs. %{x:,.0f}<extra></extra>'
                ))
                fig_br.update_layout(**plotly_layout(
                    xaxis=axis_style("Average Price (Rs.)"), yaxis=axis_style(),
                ))
                st.plotly_chart(fig_br, use_container_width=True, config={'displayModeBar': False})

        st.markdown("---")

        # --- Price by Location ---
        if has_real_options(mappings, "Location") and "Location" in df_full.columns:
            st.markdown("**Price by Location**")
            st.caption("Average vehicle price across different districts in Sri Lanka.")
            loc_map = {i: name for i, name in enumerate(get_options(mappings, "Location"))}
            df_full["Location_Name"] = df_full["Location"].map(loc_map)
            loc_prices = df_full.groupby("Location_Name")["price"].mean().sort_values(ascending=True)

            fig_loc = go.Figure()
            fig_loc.add_trace(go.Bar(
                y=loc_prices.index, x=loc_prices.values, orientation='h',
                marker=dict(color=loc_prices.values, colorscale='Sunset', line=dict(width=0.5, color='white')),
                text=[f'Rs. {v:,.0f}' for v in loc_prices.values], textposition='outside',
                textfont=dict(size=10, color='white'),
                hovertemplate='<b>%{y}</b><br>Avg Price: Rs. %{x:,.0f}<extra></extra>'
            ))
            fig_loc.update_layout(**plotly_layout(
                xaxis=axis_style("Average Price (Rs.)"), yaxis=axis_style(),
                height=500,
            ))
            st.plotly_chart(fig_loc, use_container_width=True, config={'displayModeBar': False})
    else:
        st.warning("Processed dataset not found. Run `python src/preprocess.py` first.")
