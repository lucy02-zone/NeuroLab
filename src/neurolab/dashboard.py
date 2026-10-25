"""Dashboard content for the NeuroLab application shell."""

import streamlit as st


FEATURES = (
    ("01", "Network Playground", "Shape a network by choosing hidden layers and neurons.", "◉", "blue"),
    ("02", "Dataset Playground", "Explore classic classification datasets such as XOR, moons, and circles.", "⌘", "cyan"),
    ("03", "Training Visualization", "Follow loss, model behavior, and decision boundaries as learning unfolds.", "↗", "violet"),
    ("04", "Optimizer Lab", "Compare the ideas behind SGD, Momentum, and Adam.", "⟳", "blue"),
    ("05", "Backpropagation Explorer", "Trace how gradients connect predictions to weight updates.", "∇", "cyan"),
    ("06", "Experiments & Reports", "Review configurations and take experiment findings with you.", "▤", "violet"),
)


def render_dashboard() -> None:
    """Render the project overview and planned capabilities."""
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

    st.markdown('<div class="section-heading"><div><div class="eyebrow">THE LEARNING TOOLKIT</div>'
                '<h2>Explore the building blocks</h2></div><span class="section-note">Interactive modules</span></div>',
                unsafe_allow_html=True)
    columns = st.columns(3, gap="medium")
    for index, (number, title, description, icon, accent) in enumerate(FEATURES):
        with columns[index % 3]:
            st.markdown(
                f'<article class="feature-card {accent}"><div class="feature-top"><span class="feature-icon">{icon}</span>'
                f'<span class="feature-number">{number}</span></div><h3>{title}</h3>'
                f'<p>{description}</p><div class="card-link">Explore concept <span>↗</span></div></article>',
                unsafe_allow_html=True,
            )

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
