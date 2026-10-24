"""NeuroLab Streamlit application entry point."""

import streamlit as st

from src.neurolab.dashboard import render_dashboard
from src.neurolab.navigation import NAVIGATION, render_placeholder
from src.neurolab.theme import apply_theme


st.set_page_config(
    page_title="NeuroLab | Deep Neural Network Playground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

with st.sidebar:
    st.markdown('<div class="brand-mark">N</div>', unsafe_allow_html=True)
    st.markdown("### NeuroLab")
    st.caption("Build. Train. Visualize. Understand.")
    st.markdown('<div class="sidebar-label">WORKSPACE</div>', unsafe_allow_html=True)
    selected = st.radio(
        "Navigate NeuroLab",
        NAVIGATION,
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="sidebar-footer"><span class="status-dot"></span> Foundation in development</div>',
        unsafe_allow_html=True,
    )

if selected == "Dashboard":
    render_dashboard()
else:
    render_placeholder(selected)
