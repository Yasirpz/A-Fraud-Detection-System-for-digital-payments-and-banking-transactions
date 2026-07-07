import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# 1. Page Configuration
st.set_page_config(page_title="Fraud Detection AI", page_icon="🛡️", layout="wide")

st.title("🛡️ Fraud Detection System")
st.write("Enter the transaction details below to evaluate the risk score in real-time.")

# 2. Load the Trained Model
@st.cache_resource  # Caches the model so it doesn't reload on every button click
def load_model():
    model_path = Path("models/fraud_model.joblib")
    if not model_path.exists():
        model_path = Path("fraud_model.joblib") # Fallback
    return joblib.load(model_path)

try:
    pipeline = load_model()
except FileNotFoundError:
    st.error("🚨 Model not found. Please ensure 'fraud_model.joblib' exists.")
    st.stop()

# 3. Build the Input Form
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction Details")
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=250.0)
        
        # Using a dropdown for the categorical feature
        merchant = st.selectbox("Merchant Category", [
            "crypto_exchange", "online_retail", "travel", 
            "electronics", "grocery", "restaurant", "other"
        ])
        
        hour = st.slider("Hour of Day", min_value=0, max_value=23, value=14)
        distance = st.number_input("Distance from Home (km)", min_value=0.0, value=15.0)

    with col2:
        st.subheader("Security & Behavior")
        account_age = st.number_input("Account Age (days)", min_value=0, value=365)
        velocity = st.number_input("Txn Velocity (Last 1h)", min_value=0, value=1)
        behavioral_score = st.slider("Behavioral Anomaly Score", min_value=0.0, max_value=1.0, value=0.1)
        
        # Radios for boolean/binary features
        geo_mismatch = st.radio("Geo/IP Mismatch?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)
        new_device = st.radio("New Device Fingerprint?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)

# 4. Prediction Logic
st.markdown("---")
if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
    
    # Map the UI inputs to the exact pandas format your model expects
    input_data = pd.DataFrame([{
        "transaction_amount": amount,
        "hour_of_day": hour,
        "geo_ip_mismatch": geo_mismatch,
        "new_device_fingerprint": new_device,
        "behavioral_anomaly_score": behavioral_score,
        "txn_velocity_last_1h": velocity,
        "distance_from_home_km": distance,
        "account_age_days": account_age,
        "merchant_category": merchant
    }])
    
    with st.spinner("Analyzing risk factors..."):
        # Get probability of the "fraud" class (class 1)
        prob = pipeline.predict_proba(input_data)[0][1]
        
        st.subheader("Analysis Result")
        
        # Threshold logic from your predict.py
        if prob >= 0.8:
            st.error(f"🚫 **Action: BLOCK + ALERT** (Risk Score: {prob * 100:.1f}%)")
            st.write("Transaction blocked due to critical risk factors.")
        elif prob >= 0.5:
            st.warning(f"⚠️ **Action: HOLD FOR REVIEW** (Risk Score: {prob * 100:.1f}%)")
            st.write("Flagged for manual review by the fraud team.")
        else:
            st.success(f"✅ **Action: APPROVE** (Risk Score: {prob * 100:.1f}%)")
            st.write("Transaction looks legitimate.")