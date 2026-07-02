"""
System Health and diagnostics monitor.
"""
import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api import api_client
from config import render_header, render_status_badge

render_header("🖥️ System Diagnostics", "Monitor inference infrastructure status and model metadata")

# Probe Backend
liveness_text = "OFFLINE"
readiness_text = "OFFLINE"
model_loaded = "NO"
model_type = "CatBoost"
model_version = "v1.0.0"
pipeline_version = "v1.0.0"
threshold = "0.4047"
feature_count = 33
# mlflow_status = "Offline Mode"

try:
    liveness = api_client.get_liveness()
    liveness_text = liveness.get("status", "unhealthy")
except Exception:
    pass

try:
    readiness = api_client.get_readiness()
    readiness_text = "ready" if readiness.get("status") == "ready" else "unready"
    model_loaded = "YES" if readiness.get("model_loaded") else "NO"
except Exception:
    pass

try:
    model_info = api_client.get_model_info()
    model_type = model_info.get("model_type", "CatBoost")
    model_version = model_info.get("model_version", "v1.0.0")
    feature_count = model_info.get("feature_count", 33)
except Exception:
    pass

try:
    meta = api_client.get_metadata()
    pipeline_version = meta.get("training_version", "v1.0.0")
    threshold = f"{meta.get('threshold', 0.404694):.4f}"
    run_id = meta.get("mlflow_run_id")
    if run_id and run_id != "local":
        mlflow_status = f"Connected (ID: {run_id})"
    else:
        mlflow_status = "Local Artifact Mode (Offline)"
except Exception:
    pass

# Status Badges
st.subheader("Infrastructure Health Status")
c1, c2, c3 = st.columns(3)
with c1:
    render_status_badge("API Liveness", liveness_text)
with c2:
    render_status_badge("API Readiness", readiness_text)
with c3:
    render_status_badge("Model Loaded", "online" if model_loaded == "YES" else "offline")

st.markdown("---")

# Details Table
st.subheader("System Configuration Details")
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    *   **Production Model**: `{model_type}`
    *   **Model Version**: `{model_version}`
    *   **Pipeline Version**: `{pipeline_version}`
    *   **Model Decision Threshold**: `{threshold}`
    """)
with col2:
    st.markdown(f"""
    *   **Feature Schema Dimensions**: `{feature_count} features`
    *   **MLflow Status**: `{mlflow_status}`
    *   **API Base URL**: `{api_client.base_url}`
    """)
