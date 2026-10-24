"""NeuroLab Streamlit application entry point."""

import streamlit as st

from src.neurolab.dashboard import render_dashboard
from src.neurolab.activation_playground import render_activation_playground
from src.neurolab.backpropagation_explorer import render_backpropagation_explorer
from src.neurolab.loss_playground import render_loss_playground
from src.neurolab.network_playground import render_network_playground
from src.neurolab.optimizer_lab import render_optimizer_lab
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
elif selected == "Activation Functions":
    render_activation_playground()
elif selected == "Loss Functions":
    render_loss_playground()
elif selected == "Network Playground":
    render_network_playground()
elif selected == "Optimizer Lab":
    render_optimizer_lab()
elif selected == "Backpropagation Explorer":
    render_backpropagation_explorer()
else:
    render_placeholder(selected)
