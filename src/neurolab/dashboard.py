"""Dashboard content and module launchpad for the NeuroLab application shell."""

import re

import streamlit as st

from .navigation import NAVIGATION


MODULES = {
    "Network Playground": ("Configure a network and train it on a synthetic classification task.", "◉", "blue"),
    "Dataset Explorer": ("Inspect data shape, class balance, noise, and validation splits.", "▦", "cyan"),
    "Network Visualizer": ("See how neurons connect across the layers in your architecture.", "⌘", "violet"),
    "Activation Functions": ("Compare activation curves and inspect their derivatives.", "ƒ", "blue"),
    "Loss Functions": ("Explore how prediction errors become numerical loss values.", "∿", "cyan"),
    "Experiment Wizard": ("Set up, train, evaluate, and save a reproducible experiment.", "✦", "violet"),
    "Optimizer Lab": ("Compare SGD, Momentum, and Adam using matched experiments.", "⟳", "blue"),
    "Model Manager": ("Export, import, inspect, and run predictions with model files.", "⇄", "cyan"),
    "Experiments & Reports": ("Review saved experiment configurations and results.", "▤", "violet"),
    "Math Calculator": ("Work through forward propagation, loss, gradients, and updates.", "∑", "blue"),
    "Backpropagation Explorer": ("Trace gradients backward through a neural network.", "∇", "cyan"),
    "Learn DNN": ("Build understanding with explanations and interactive lessons.", "✧", "violet"),
    "Quiz & Challenge Center": ("Check your understanding with practice and timed quizzes.", "✎", "blue"),
}

GROUPS = (
    ("BUILD & EXPLORE", "Shape a network and understand its inputs.", (
        "Network Playground", "Dataset Explorer", "Network Visualizer",
        "Activation Functions", "Loss Functions",
    )),
    ("TRAIN & COMPARE", "Run experiments, compare optimizers, and manage models.", (
        "Experiment Wizard", "Optimizer Lab", "Model Manager", "Experiments & Reports",
    )),
    ("LEARN & PRACTICE", "Explore the mathematics and test your understanding.", (
        "Math Calculator", "Backpropagation Explorer", "Learn DNN", "Quiz & Challenge Center",
    )),
)


def _open_module(name: str) -> None:
    """Navigate to a NeuroLab page selected from the dashboard."""
    st.session_state["navigation_selection"] = name


def _render_module_group(label: str, description: str, names: tuple[str, ...], start: int) -> int:
    st.markdown(
        f'<div class="dashboard-group-heading"><div><span>{label}</span>'
        f'<p>{description}</p></div></div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(3, gap="medium")
    for offset, name in enumerate(names):
        description_text, icon, accent = MODULES[name]
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        with columns[offset % 3]:
            with st.container(border=True, key=f"dashboard_module_{slug}"):
                st.markdown(
                    f'<div class="dashboard-module-card {accent}">'
                    f'<div class="feature-top"><span class="feature-icon">{icon}</span>'
                    f'<span class="feature-number">{start + offset:02d}</span></div>'
                    f'<h3>{name}</h3><p>{description_text}</p></div>',
                    unsafe_allow_html=True,
                )
                st.button(
                    "Open section  →", key=f"dashboard_open_{slug}",
                    type="secondary", width="stretch", on_click=_open_module, args=(name,),
                )
    return start + len(names)


def render_dashboard() -> None:
    """Render the NeuroLab overview and clickable launchpad for every module."""
    st.markdown('<div class="eyebrow">DEEP LEARNING, MADE VISIBLE</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero"><div class="hero-copy"><div class="hero-kicker">YOUR DNN LEARNING STUDIO</div>'
        '<h1>NeuroLab<span class="hero-period">.</span></h1>'
        '<p class="hero-tagline">Build. Train. Visualize. Understand.</p>'
        '<p class="hero-description">A hands-on space to explore what happens inside a neural network. '
        'Build intuition for the mathematics, one concept and experiment at a time.</p>'
        '<div class="hero-chip"><span class="status-dot"></span> Project foundation in progress</div>'
        '</div><div class="hero-art"><div class="orbit orbit-one"></div><div class="orbit orbit-two"></div>'
        '<div class="neuron n1"></div><div class="neuron n2"></div><div class="neuron n3"></div>'
        '<div class="neuron n4"></div><div class="neuron n5"></div><div class="neuron n6"></div>'
        '<div class="neuron n7"></div><div class="neuron n8"></div><div class="neuron n9"></div>'
        '<div class="neuron n10"></div><div class="neuron n11"></div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-heading"><div><div class="eyebrow">YOUR LEARNING TOOLKIT</div>'
        '<h2>Go straight to a NeuroLab tool</h2></div>'
        '<span class="section-note">All modules · one click away</span></div>',
        unsafe_allow_html=True,
    )
    if set(MODULES) != set(NAVIGATION) - {"Dashboard"}:
        st.error("Dashboard shortcuts and application navigation are out of sync.")
        return

    number = 1
    for label, description, names in GROUPS:
        number = _render_module_group(label, description, names, number)

    left, right = st.columns([1.15, 0.85], gap="medium")
    with left:
        st.markdown(
            '<section class="info-panel"><div class="eyebrow">PROJECT STATUS</div><h2>Learning workspace ready</h2>'
            '<p>Explore the NumPy network engine, interactive math pages, synthetic datasets, model training, '
            'evaluation, experiment reports, model files, and knowledge challenges. Training runs only when you start one.</p>'
            '<div class="progress-caption"><span><span class="status-dot"></span>Core modules available</span>'
            '<span>Progress stays in your session where applicable</span></div></section>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            '<section class="info-panel stack-panel"><div class="eyebrow">BUILT WITH</div><h2>Technology stack</h2>'
            '<div class="stack-list"><span>Python</span><span>NumPy</span><span>Streamlit</span>'
            '<span>Matplotlib</span><span>scikit-learn</span><span>pytest</span></div></section>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="page-footer">NEUROLAB <span>•</span> Learn by making the invisible visible.</div>',
                unsafe_allow_html=True)
