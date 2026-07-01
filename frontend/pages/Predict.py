"""
Transaction evaluation and prediction.
"""
import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api import api_client
from config import render_header

render_header("🔍 Real-Time Predictor", "Evaluate transaction properties for fraud risks")

left, right = st.columns([2, 1])

with left:
    st.subheader("Transaction Input parameters")
    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            trans_date_time = st.text_input("Transaction Time", "2020-06-21 12:14:30")
            cc_num = st.number_input("CC Number", value=123456789012, step=1)
            merchant = st.text_input("Merchant", "fraud_test_store")
            category = st.text_input("Category", "grocery_pos")
            amt = st.number_input("Amount ($)", value=150.0, step=0.1)
            first = st.text_input("First Name", "John")
            last = st.text_input("Last Name", "Doe")
            gender = st.selectbox("Gender", ["M", "F"])
            street = st.text_input("Street", "123 Test St")
            city = st.text_input("City", "Dallas")
        with c2:
            state = st.text_input("State", "TX")
            zip_code = st.number_input("ZIP Code", value=75201, step=1)
            lat = st.number_input("Lat", value=32.7767, format="%.4f")
            long = st.number_input("Long", value=-96.7970, format="%.4f")
            city_pop = st.number_input("City Pop", value=1300000, step=1)
            job = st.text_input("Job", "Software Engineer")
            dob = st.text_input("DOB", "1988-04-12")
            trans_num = st.text_input("Transaction Number", "dummy_num_1234")
            unix_time = st.number_input("Unix Time", value=1592741670, step=1)
            merch_lat = st.number_input("Merchant Lat", value=32.7800, format="%.4f")
            merch_long = st.number_input("Merchant Long", value=-96.8000, format="%.4f")
            
        submit = st.form_submit_button("Submit Transaction")

with right:
    st.subheader("Model Decision Outcome")
    if submit:
        transaction = {
            "trans_date_trans_time": trans_date_time,
            "cc_num": cc_num,
            "merchant": merchant,
            "category": category,
            "amt": amt,
            "first": first,
            "last": last,
            "gender": gender,
            "street": street,
            "city": city,
            "state": state,
            "zip": zip_code,
            "lat": lat,
            "long": long,
            "city_pop": city_pop,
            "job": job,
            "dob": dob,
            "trans_num": trans_num,
            "unix_time": unix_time,
            "merch_lat": merch_lat,
            "merch_long": merch_long,
        }
        
        try:
            with st.spinner("Scoring..."):
                res = api_client.predict_single(transaction)
                
            is_fraud = res.get("is_fraud", 0)
            prob = res.get("probability", 0.0)
            threshold = res.get("threshold_used", 0.5)
            latency = res.get("latency_ms", 0.0)
            model_ver = res.get("model_version", "unknown")
            
            if is_fraud == 1:
                st.error("🚨 FRAUD DETECTED")
                risk_level = "CRITICAL"
            else:
                st.success("✅ LEGITIMATE")
                risk_level = "LOW"
                
            st.markdown(f"""
            *   **Probability**: `{prob:.6f}`
            *   **Risk Level**: `{risk_level}`
            *   **Decision Threshold**: `{threshold:.4f}`
            *   **Backend Inference Latency**: `{latency:.2f} ms`
            *   **Model Build Version**: `{model_ver}`
            """)
        except Exception as e:
            st.error(f"Inference Connection Error: {e}")
    else:
        st.info("Input transaction details and click Submit to run predictions.")
