"""NeuroLab Streamlit application entry point."""

import streamlit as st

from src.neurolab.dashboard import render_dashboard
from src.neurolab.activation_playground import render_activation_playground
from src.neurolab.backpropagation_explorer import render_backpropagation_explorer
from src.neurolab.loss_playground import render_loss_playground
from src.neurolab.learn_dnn import render_learn_dnn
from src.neurolab.network_playground import render_network_playground
from src.neurolab.optimizer_lab import render_optimizer_lab
from src.neurolab.experiments_reports import render_experiments_reports
from src.neurolab.navigation import NAVIGATION, render_placeholder
from src.neurolab.theme import apply_theme


st.set_page_config(
    page_title="NeuroLab | Deep Neural Network Playground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()


def _select_navigation(section: str) -> None:
    st.session_state.navigation_selection = section


with st.sidebar:
    st.markdown('<div class="brand-mark">N</div>', unsafe_allow_html=True)
    st.markdown("### NeuroLab")
    st.caption("Build. Train. Visualize. Understand.")
    st.markdown('<div class="sidebar-label">WORKSPACE</div>', unsafe_allow_html=True)
    if "navigation_selection" not in st.session_state:
        st.session_state.navigation_selection = NAVIGATION[0]

    navigation_icons = ("⌂", "⬡", "ƒ", "∿", "↗", "⇠", "▤", "✧")
    for section, icon in zip(NAVIGATION, navigation_icons):
        st.button(
            f"{icon}   {section}",
            key=f"nav_{section.lower().replace(' ', '_').replace('&', 'and')}",
            type="primary" if section == st.session_state.navigation_selection else "secondary",
            use_container_width=True,
            on_click=_select_navigation,
            args=(section,),
        )
    selected = st.session_state.navigation_selection
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
elif selected == "Experiments & Reports":
    render_experiments_reports()
elif selected == "Learn DNN":
    render_learn_dnn()
else:
    render_placeholder(selected)
