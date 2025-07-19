import streamlit as st
import joblib
import numpy as np

# Load model, scaler, and feature columns
model = joblib.load('credit_default_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('feature_columns.pkl')

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")
st.title("💳 Credit Risk Default Predictor")

st.markdown("Enter customer details to check if they are likely to default on credit card payment.")

# Collect user input
user_input = []
for feature in features:
    value = st.number_input(f"Enter {feature}", step=1.0, format="%.2f")
    user_input.append(value)

if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0][prediction]

    if prediction == 1:
        st.error(f"⚠️ High Risk: Likely to Default (Confidence: {proba:.2%})")
    else:
        st.success(f"✅ Low Risk: Not Likely to Default (Confidence: {proba:.2%})")
