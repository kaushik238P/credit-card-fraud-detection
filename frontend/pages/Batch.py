"""
Batch JSON Predictor.
"""
import json
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api import api_client
from config import render_header

render_header("📁 Batch Prediction", "Process multiple transactions simultaneously via JSON payload")

uploaded_file = st.file_uploader("Upload JSON Transaction list", type=["json"])

if uploaded_file is not None:
    try:
        data = json.load(uploaded_file)
        if not isinstance(data, list):
            st.error("Invalid format: JSON file must contain a list of records.")
        else:
            st.success(f"Parsed {len(data)} transactions successfully.")
            run_batch = st.button("Execute Batch Prediction")
            
            if run_batch:
                try:
                    with st.spinner("Processing batch..."):
                        response = api_client.predict_batch(data)
                        
                    predictions = response.get("predictions", [])
                    total = response.get("batch_size", len(predictions))
                    
                    frauds = sum(1 for p in predictions if p.get("is_fraud") == 1)
                    legit = total - frauds
                    avg_prob = sum(p.get("probability", 0.0) for p in predictions) / total if total > 0 else 0.0
                    
                    # 1. Summary Cards
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Records", str(total))
                    col2.metric("Fraud Cases", str(frauds))
                    col3.metric("Legitimate Cases", str(legit))
                    col4.metric("Avg Fraud Prob", f"{avg_prob:.4f}")
                    
                    # 2. Results Dataframe
                    st.subheader("Prediction Outputs Table")
                    results_df = pd.DataFrame(predictions)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # 3. Download button
                    json_str = json.dumps(response, indent=2)
                    st.download_button(
                        label="📥 Download Output Predictions JSON",
                        data=json_str,
                        file_name="batch_predictions.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Batch prediction invocation failed: {e}")
    except Exception as parse_err:
        st.error(f"Failed to read upload file: {parse_err}")
