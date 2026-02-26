import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==============================
# Load Model and Columns
# ==============================

with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

st.set_page_config(page_title="Fraud Detection App", layout="wide")

st.title("💳 Credit Card Fraud Detection System")
st.write("Fill all transaction details below to predict fraud.")

# ==============================
# Create Dynamic Input Fields
# ==============================

input_data = {}

for col in model_columns:

    # If numeric column
    if col.lower().startswith(("amount", "old", "new", "balance", "step")):
        input_data[col] = st.number_input(f"{col}", value=0.0)

    # Otherwise treat as categorical
    else:
        input_data[col] = st.text_input(f"{col}")

# ==============================
# Convert to DataFrame (Correct Order)
# ==============================

input_df = pd.DataFrame([input_data])
input_df = input_df[model_columns]  # ensure correct order

# ==============================
# Prediction
# ==============================

if st.button("Predict Fraud"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")

    st.write(f"Fraud Probability: {probability:.4f}")