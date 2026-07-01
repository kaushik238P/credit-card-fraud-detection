"""
Streamlit main navigation and routing hub.
"""
import streamlit as st
import sys
from pathlib import Path

# Add frontend directory to path to enable direct imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load styling
with open("frontend/assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Define pages
dashboard = st.Page("pages/Home.py", title="Dashboard", icon="🏠", default=True)
predict = st.Page("pages/Predict.py", title="Predict", icon="🔍")
batch = st.Page("pages/Batch.py", title="Batch Prediction", icon="📁")
system = st.Page("pages/System.py", title="System Health", icon="🖥️")
about = st.Page("pages/About.py", title="About", icon="ℹ️")

pg = st.navigation({
    "Menu": [dashboard, predict, batch, system, about]
})

from config import render_sidebar_footer
render_sidebar_footer()

pg.run()
