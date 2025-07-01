import streamlit as st
import joblib
import numpy as np
import pandas as pd
from utils.preprocessor import preprocess_input

# Load model and scaler
model = joblib.load('models/LR_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Page config
st.set_page_config(page_title="GHG Emission Predictor", page_icon="🌍", layout="centered")

# 🎨 Custom CSS — no more white container
st.markdown("""
    <style>
    html, body, [class*="stApp"] {
        background: linear-gradient(to right, #E3F2FD, #F0F9FF);
        font-family: 'Segoe UI', sans-serif;
    }
    .title-block {
        background: #1A73E8;
        color: white;
        text-align: center;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 30px;
    }
    .output-block {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00C9A7;
    }
    </style>
""", unsafe_allow_html=True)

# 🌍 Title block
st.markdown("<div class='title-block'><h1>🌱 GHG Emission Predictor</h1><p>Estimate Supply Chain Emission Factors using DQ Metrics</p></div>", unsafe_allow_html=True)

# 📥 Input form — no extra input container
with st.form("prediction_form"):
    st.subheader("📝 Input Parameters")
    col1, col2 = st.columns(2)

    with col1:
        substance = st.selectbox("🌫️ Substance", ['carbon dioxide', 'methane', 'nitrous oxide', 'other GHGs'])
        unit = st.selectbox("⚖️ Unit", ['kg/2018 USD, purchaser price', 'kg CO2e/2018 USD, purchaser price'])
        source = st.selectbox("🏭 Source", ['Commodity', 'Industry'])
        supply_wo_margin = st.number_input("🚚 Emission Factors (Without Margins)", min_value=0.0, format="%.4f")
        margin = st.number_input("📈 Margins", min_value=0.0, format="%.4f")

    with col2:
        dq_reliability = st.slider("✅ DQ Reliability", 0.0, 1.0, 0.5)
        dq_temporal = st.slider("🕒 DQ Temporal Correlation", 0.0, 1.0, 0.5)
        dq_geo = st.slider("🗺️ DQ Geographical Correlation", 0.0, 1.0, 0.5)
        dq_tech = st.slider("⚙️ DQ Technological Correlation", 0.0, 1.0, 0.5)
        dq_data = st.slider("📊 DQ Data Collection", 0.0, 1.0, 0.5)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        submit = st.form_submit_button("🔍 Predict")

# 🎯 Prediction output
if submit:
    input_data = {
        'Substance': substance,
        'Unit': unit,
        'Supply Chain Emission Factors without Margins': supply_wo_margin,
        'Margins of Supply Chain Emission Factors': margin,
        'DQ ReliabilityScore of Factors without Margins': dq_reliability,
        'DQ TemporalCorrelation of Factors without Margins': dq_temporal,
        'DQ GeographicalCorrelation of Factors without Margins': dq_geo,
        'DQ TechnologicalCorrelation of Factors without Margins': dq_tech,
        'DQ DataCollection of Factors without Margins': dq_data,
        'Source': source
    }

    input_df = preprocess_input(pd.DataFrame([input_data]))
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.markdown("<div class='output-block'>", unsafe_allow_html=True)
    st.markdown(f"🎯 **Predicted Supply Chain Emission Factor with Margin:** `{prediction:.4f}` kg CO2e/2018 USD")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    🔹 This is a predictive result based on your inputs.  
    🔹 For verified emissions data, consult LCA databases or government portals.
    """)