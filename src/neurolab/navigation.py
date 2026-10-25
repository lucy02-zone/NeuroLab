"""Navigation labels and placeholder rendering for the application shell."""

import streamlit as st


NAVIGATION = (
    "Dashboard",
    "Network Playground",
    "Dataset Explorer",
    "Experiment Wizard",
    "Model Manager",
    "Network Visualizer",
    "Activation Functions",
    "Loss Functions",
    "Optimizer Lab",
    "Backpropagation Explorer",
    "Experiments & Reports",
    "Learn DNN",
)


def render_placeholder(section: str) -> None:
    """Render a clear placeholder for a section that is not implemented yet."""
    st.markdown('<div class="eyebrow">NEUROLAB WORKSPACE</div>', unsafe_allow_html=True)
    st.title(section)
    st.markdown(
        f'<section class="placeholder-card"><div class="placeholder-icon">✦</div>'
        f'<h2>Coming Soon</h2><p>{section} is part of the NeuroLab roadmap. '
        "This space will be ready as the project grows.</p></section>",
        unsafe_allow_html=True,
    )
