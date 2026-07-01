"""
Dashboard Home Page.
"""
import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api import api_client
from config import render_header, render_status_badge

render_header("🏠 Dashboard Overview", "Real-Time Model Performance & Infrastructure Monitoring")

# 1. Probe statuses
api_status = "Unhealthy"
mlflow_status = "Disconnected"
threshold = "0.4047"
features_count = 33

try:
    liveness = api_client.get_liveness()
    readiness = api_client.get_readiness()
    if liveness.get("status") == "healthy" and readiness.get("status") == "ready":
        api_status = "Healthy"
except Exception:
    pass

try:
    meta = api_client.get_metadata()
    threshold = f"{meta.get('threshold', 0.404694):.4f}"
    run_id = meta.get("mlflow_run_id")
    if run_id and run_id != "local":
        mlflow_status = "Connected"
    else:
        mlflow_status = "Offline Mode"
except Exception:
    pass

# 2. Render metric grid
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Production Model", "CatBoost")
with col2:
    st.metric("Inference API Service", api_status)
with col3:
    st.metric("MLflow Model Registry", mlflow_status)

st.markdown("---")

# 3. System Overview & Quick Actions
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("📋 System Specifications")
    st.markdown(f"""
    *   **Inference Dataset Profile**: Credit Card Fraud synthetic transaction streams
    *   **Inference Input Features**: {features_count} preprocessed parameters (temporal, distance, frequency)
    *   **Classification Optimized Decision Threshold**: `{threshold}` (MAX_F1 Strategy)
    *   **Model Pipeline Code Version**: `v1.0.0`
    """)

with c2:
    st.subheader("⚡ Quick Actions")
    st.markdown("""
    Go to **Predict** page to run single transaction scoring.
    
    Go to **Batch Prediction** to run CSV-free json arrays.
    """)
