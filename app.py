"""NeuroLab Streamlit application entry point."""

import streamlit as st

from src.neurolab.dashboard import render_dashboard
from src.neurolab.dataset_explorer import render_dataset_explorer
from src.neurolab.experiment_wizard import render_experiment_wizard
from src.neurolab.model_manager import render_model_manager
from src.neurolab.math_calculator import render_math_calculator
from src.neurolab.quiz_center import render_quiz_center
from src.neurolab.activation_playground import render_activation_playground
from src.neurolab.backpropagation_explorer import render_backpropagation_explorer
from src.neurolab.loss_playground import render_loss_playground
from src.neurolab.learn_dnn import render_learn_dnn
from src.neurolab.network_playground import render_network_playground
from src.neurolab.network_visualizer import render_network_visualizer
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
    st.markdown(
        """<div class="brand-lockup">
            <div class="brand-mark" aria-label="NeuroLab neural network logo">
                <svg viewBox="0 0 48 48" role="img" aria-hidden="true">
                    <path d="M12 14 23 24 36 12M12 14l1 20 10-10m0 0 12 12m-12-12 13-12" />
                    <circle class="node node-blue" cx="12" cy="14" r="4" />
                    <circle class="node node-cyan" cx="36" cy="12" r="4" />
                    <circle class="node node-violet" cx="23" cy="24" r="4.5" />
                    <circle class="node node-cyan" cx="13" cy="34" r="4" />
                    <circle class="node node-blue" cx="36" cy="36" r="4" />
                </svg>
            </div>
            <div class="brand-copy">
                <div class="brand-name">NeuroLab</div>
                <div class="brand-tagline">Build. Train. Visualize. Understand.</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-label">WORKSPACE</div>', unsafe_allow_html=True)
    if "navigation_selection" not in st.session_state:
        st.session_state.navigation_selection = NAVIGATION[0]

    navigation_icons = ("⌂", "⬡", "▦", "✦", "⇄", "∑", "✎", "◉", "ƒ", "∿", "↗", "⇠", "▤", "✧")
    for section, icon in zip(NAVIGATION, navigation_icons):
        st.button(
            f"{icon}   {section}",
            key=f"nav_{section.lower().replace(' ', '_').replace('&', 'and')}",
            type="primary" if section == st.session_state.navigation_selection else "secondary",
            width="stretch",
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
elif selected == "Network Visualizer":
    render_network_visualizer()
elif selected == "Dataset Explorer":
    render_dataset_explorer()
elif selected == "Experiment Wizard":
    render_experiment_wizard()
elif selected == "Model Manager":
    render_model_manager()
elif selected == "Math Calculator":
    render_math_calculator()
elif selected == "Quiz & Challenge Center":
    render_quiz_center()
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
