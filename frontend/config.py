"""
Streamlit frontend configuration and shared UI components.
"""
import os
import streamlit as st

API_URL = os.environ.get("FRAUD_BACKEND_URL", "http://localhost:8000")
API_KEY = os.environ.get("FRAUD_API_KEY", "test_api_key_123")


def render_sidebar_footer() -> None:
    """Renders unified enterprise navigation footer in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏷️ System Info")
    st.sidebar.info("**Production Model**: CatBoost\n\n**Pipeline Version**: v1.0.0")


def render_header(title: str, subtitle: str) -> None:
    """Renders professional header block."""
    st.title(title)
    st.markdown(f"*{subtitle}*")
    st.markdown("---")


def render_status_badge(label: str, status_val: str) -> None:
    """Renders badge representation for statuses."""
    status_lower = status_val.lower()
    if "healthy" in status_lower or "online" in status_lower or "ready" in status_lower or "connected" in status_lower:
        badge_cls = "status-healthy"
    elif "unhealthy" in status_lower or "offline" in status_lower or "error" in status_lower:
        badge_cls = "status-unhealthy"
    else:
        badge_cls = "status-partial"
        
    st.markdown(
        f'<span class="status-badge {badge_cls}">{label}: {status_val.upper()}</span>',
        unsafe_allow_html=True
    )
