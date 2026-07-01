"""
About Page describing the fraud detection project stack.
"""
import sys
import streamlit as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import render_header

render_header("ℹ️ About the Project", "Technical specifications and system design of the Fraud Detection System")

st.subheader("Project Overview")
st.write("""
This system is an enterprise-grade credit card fraud detection pipeline. It ingests card transaction streams,
validates transaction data formats, runs real-time feature engineering, preprocesses features, and scores the result
using a champion machine learning classifier.
""")

st.subheader("Core Technology Stack")
st.markdown("""
*   **Python 3.12**: Implementation language.
*   **CatBoost**: Champion gradient boosted decision tree classifier model.
*   **FastAPI**: Lightweight HTTP REST API inference layer.
*   **Streamlit**: Professional dashboard user interface layer.
*   **MLflow**: Model lifecycle tracking and model registry server.
*   **Docker / Qdrant**: Production deployment containerization and vector embedding placeholders.
""")

st.subheader("System Features")
st.markdown("""
*   **Dynamic Decision Thresholding**: Automatically optimizes threshold classifications to maximize F1 scores.
*   **Robust Preloading**: Caches model parameters and preprocessing assets on startup to avoid overhead.
*   **Clean Separation of Concerns**: Ingestion, Preprocessing, Feature Engineering, and Inference reside in decoupled, single-responsibility modules.
""")

st.markdown("---")
st.subheader("Links & Documentation")
st.markdown("""
*   [GitHub Repository](https://github.com/)
*   [API Documentation (Swagger UI)](http://localhost:8000/docs)
""")
