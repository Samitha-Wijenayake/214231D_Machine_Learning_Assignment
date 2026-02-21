"""
Tab 1: Price Prediction — input form, prediction, SHAP local explanation.
"""
import streamlit as st
import pandas as pd
import shap
import plotly.graph_objects as go
from app.helpers import (
    has_real_options, get_options, clean_name,
    plotly_layout, axis_style, HIDE_FEATS,
)


def render(model, mappings, feature_names, test_metrics):
    """Render the prediction tab content."""
    input_data = {}
    location = None

    # --- Vehicle Identity ---
    st.markdown("##### Vehicle Identity")
    vi1, vi2, vi3 = st.columns(3)
    with vi1:
        if has_real_options(mappings, "Brand"):
            brand = st.selectbox("Brand", options=get_options(mappings, "Brand"), help="Select the vehicle manufacturer")
            input_data["Brand"] = get_options(mappings, "Brand").index(brand)
    with vi2:
        if has_real_options(mappings, "Model"):
            model_name = st.selectbox("Model", options=get_options(mappings, "Model"), help="Select the vehicle model")
            input_data["Model"] = get_options(mappings, "Model").index(model_name)
    with vi3:
        if "Year" in feature_names:
            year = st.number_input("Year", min_value=1950, max_value=2026, value=2018, help="Year of manufacture")
            input_data["Year"] = int(year)

    # --- Specifications ---
    st.markdown("##### Specifications")
    sp1, sp2, sp3 = st.columns(3)
    with sp1:
        if "Mileage" in feature_names:
            mileage = st.number_input("Mileage (km)", min_value=0, max_value=1000000, value=50000, step=5000, help="Total distance driven")
            input_data["Mileage"] = float(mileage)
    with sp2:
        if "EngineCapacity" in feature_names:
            engine_cap = st.number_input("Engine (cc)", min_value=0, max_value=10000, value=1500, step=100, help="Engine capacity in cubic centimeters")
            input_data["EngineCapacity"] = float(engine_cap)
    with sp3:
        if has_real_options(mappings, "Transmission"):
            transmission = st.selectbox("Transmission", options=get_options(mappings, "Transmission"))
            input_data["Transmission"] = get_options(mappings, "Transmission").index(transmission)
        if has_real_options(mappings, "FuelType"):
            fuel_type = st.selectbox("Fuel Type", options=get_options(mappings, "FuelType"))
            input_data["FuelType"] = get_options(mappings, "FuelType").index(fuel_type)

    # --- Other Details ---
    st.markdown("##### Other Details")
    od1, od2, od3 = st.columns(3)
    with od1:
        if has_real_options(mappings, "Location"):
            location = st.selectbox("Location", options=get_options(mappings, "Location"), help="District where the vehicle is listed")
            input_data["Location"] = get_options(mappings, "Location").index(location)
    with od2:
        if has_real_options(mappings, "Condition"):
            condition = st.selectbox("Condition", options=get_options(mappings, "Condition"))
            input_data["Condition"] = get_options(mappings, "Condition").index(condition)

    st.markdown("")
    if st.button("Predict Price", type="primary", use_container_width=True):
        if model is None or not feature_names:
            st.error("Model or feature mapping not found. Please train the model first.")
        else:
            with st.spinner("Calculating prediction..."):
                ordered_input = []
                for feat in feature_names:
                    if feat == "Brand_Encoded":
                        ordered_input.append(input_data.get("Brand", 0))
                    elif feat == "Model_Encoded":
                        ordered_input.append(input_data.get("Model", 0))
                    elif feat.startswith("Loc_"):
                        loc_name = feat.replace("Loc_", "")
                        ordered_input.append(1.0 if location and loc_name == location else 0.0)
                    elif feat == "Price_Normalized":
                        ordered_input.append(0.0)
                    elif feat == "Mileage_Normalized":
                        ordered_input.append(input_data.get("Mileage", 0) / 100000.0)
                    else:
                        ordered_input.append(input_data.get(feat, 0))

                df_input = pd.DataFrame([ordered_input], columns=feature_names)
                predicted_price = model.predict(df_input)[0]
                mae = test_metrics.get("MAE", 0)
                low = max(0, predicted_price - mae)
                high = predicted_price + mae

            st.markdown(f"""<div class="price-box">
                <div class="price-value">Rs. {predicted_price:,.2f}</div>
                <div class="price-range">Estimated range: Rs. {low:,.0f} — Rs. {high:,.0f}</div>
            </div>""", unsafe_allow_html=True)

            # --- SHAP Local Explanation ---
            with st.expander("What influenced this price?", expanded=True):
                st.markdown(
                    "The chart below shows which features had the biggest impact on this prediction. "
                    "**Green bars** pushed the price **up**, **red bars** pulled it **down**."
                )
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer(df_input)
                    vals = shap_values[0].values
                    feats = shap_values[0].feature_names if shap_values[0].feature_names is not None else feature_names

                    filtered = [(f, v) for f, v in zip(feats, vals) if f not in HIDE_FEATS]
                    filtered.sort(key=lambda x: abs(x[1]))
                    top = filtered[-10:]

                    names = [clean_name(f) for f, v in top]
                    values = [v for f, v in top]
                    colors = ['#00E676' if v > 0 else '#FF5252' for v in values]

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        y=names, x=values, orientation='h',
                        marker=dict(color=colors, line=dict(width=1, color='rgba(255,255,255,0.3)')),
                        text=[f'Rs. {v:,.0f}' for v in values],
                        textposition='outside', textfont=dict(size=11, color='white'),
                        hovertemplate='<b>%{y}</b><br>Impact: Rs. %{x:,.0f}<extra></extra>'
                    ))
                    fig.update_layout(**plotly_layout(
                        title=dict(text="Top Feature Impacts", font=dict(size=16, color='white')),
                        xaxis=axis_style("Impact on Price (Rs.)"), yaxis=axis_style(),
                        height=400, margin=dict(l=10, r=40, t=50, b=40),
                    ))
                    fig.update_xaxes(zeroline=True, zerolinecolor='rgba(255,255,255,0.3)')
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                except Exception as e:
                    st.error(f"Could not generate SHAP explanation. Run `python src/explain.py` first.\nError: {e}")
