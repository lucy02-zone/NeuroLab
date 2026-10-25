"""Interactive educational page for common activation functions."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.neurolab.models import activations
from .visualization import GRID, style_axis


INFO = {
    "Sigmoid": ("sigmoid", "(0, 1)", "Binary classification output units; maps a logit to a probability-like value."),
    "Tanh": ("tanh", "(-1, 1)", "Hidden layers when centered outputs are useful; saturates for large magnitudes."),
    "ReLU": ("relu", "[0, ∞)", "A common hidden-layer choice; simple and efficient, though negative inputs become zero."),
    "Leaky ReLU": ("leaky_relu", "(-∞, ∞)", "Hidden layers; keeps a small gradient for negative inputs."),
    "Softmax": ("softmax", "(0, 1), sums to 1", "Multiclass output probabilities. It acts on a vector of logits, not each logit independently."),
}

CURVES = {
    "Sigmoid": activations.sigmoid,
    "Tanh": activations.tanh,
    "ReLU": activations.relu,
    "Leaky ReLU": activations.leaky_relu,
}
DERIVATIVES = {
    "Sigmoid": activations.sigmoid_derivative,
    "Tanh": activations.tanh_derivative,
    "ReLU": activations.relu_derivative,
    "Leaky ReLU": activations.leaky_relu_derivative,
}


def render_activation_playground() -> None:
    """Render activation, derivative, comparison, and point inspection views."""
    st.markdown('<div class="eyebrow">MATHEMATICAL BUILDING BLOCKS</div>', unsafe_allow_html=True)
    st.markdown(
        """<section class="hero playground-hero activation-hero">
            <div class="hero-copy">
                <div class="hero-kicker">MATHEMATICAL BUILDING BLOCKS</div>
                <h1>Activation Functions<span class="hero-period">.</span></h1>
                <div class="hero-tagline">See how neurons transform their inputs.</div>
                <p class="hero-description">Compare nonlinear functions, inspect their derivatives, and explore the output at any input value.</p>
                <div class="hero-chip"><span class="status-dot"></span>Interactive curves · Mathematical explanations</div>
            </div>
            <div class="hero-art" aria-hidden="true">
                <span class="orbit orbit-one"></span><span class="orbit orbit-two"></span>
                <span class="neuron n1"></span><span class="neuron n2"></span><span class="neuron n3"></span>
                <span class="neuron n4"></span><span class="neuron n5"></span><span class="neuron n6"></span>
                <span class="neuron n7"></span><span class="neuron n8"></span><span class="neuron n9"></span>
            </div>
        </section>""",
        unsafe_allow_html=True,
    )

    with st.container(border=True, key="activation_controls"):
        control_col, range_col, point_col = st.columns([1.2, 1.3, 1.2], gap="medium")
        with control_col:
            selected = st.selectbox("Activation function", tuple(INFO), key="activation_function_select")
        with range_col:
            input_range = st.slider("Input range (±)", min_value=2, max_value=12, value=6, key="activation_input_range")
        with point_col:
            point = st.number_input("Inspect input x", value=0.5, step=0.1, format="%.2f", key="activation_input_point")

    alpha = 0.01
    if selected == "Leaky ReLU":
        alpha = st.slider("Negative slope α", min_value=0.01, max_value=0.50, value=0.10, step=0.01, key="activation_leaky_alpha")

    x_values = np.linspace(-float(input_range), float(input_range), 400)
    function_key, output_range, use_case = INFO[selected]
    if selected == "Softmax":
        # Plot the probability assigned to the first class for logits [x, 0, -x].
        curve_y = activations.softmax(np.column_stack((x_values, np.zeros_like(x_values), -x_values)), axis=1)[:, 0]
        point_output = activations.softmax(np.array([point, 0.0, -point]))
        selected_label = "Class 1 probability for logits [x, 0, -x]"
        selected_derivative = None
    else:
        function = CURVES[selected]
        derivative = DERIVATIVES[selected]
        if selected == "Leaky ReLU":
            curve_y = function(x_values, alpha=alpha)
            point_output = function(point, alpha=alpha)
            derivative_y = derivative(x_values, alpha=alpha)
            derivative_output = derivative(point, alpha=alpha)
        else:
            curve_y = function(x_values)
            point_output = function(point)
            derivative_y = derivative(x_values)
            derivative_output = derivative(point)
        selected_label = f"{selected} output"
        selected_derivative = derivative_y

    plot_col, derivative_col = st.columns(2, gap="medium")
    with plot_col:
        figure, axis = plt.subplots(figsize=(7.0, 4.0), facecolor="#F3F6FA")
        style_axis(axis, selected_label, "light")
        axis.axhline(0, color=GRID, linewidth=0.8)
        axis.axvline(0, color=GRID, linewidth=0.8)
        color = "#397F8B" if selected in ("Softmax", "Leaky ReLU") else "#2F5F83"
        axis.plot(x_values, curve_y, color=color, linewidth=2.6)
        axis.scatter([point], [point_output[0] if selected == "Softmax" else point_output], color=VIOLET, edgecolors="#edf3ff", s=48, zorder=4)
        axis.set_xlabel("Input x")
        axis.set_ylabel("Output")
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)
    with derivative_col:
        if selected_derivative is None:
            st.markdown(
                '<section class="info-panel" style="height:100%;display:flex;flex-direction:column;justify-content:center">'
                '<div class="eyebrow">DERIVATIVE VIEW</div><h2>Softmax uses a Jacobian</h2>'
                '<p>Softmax couples all class probabilities. Its derivative is a matrix of partial derivatives, '
                'so it is not shown here as an elementwise curve.</p></section>',
                unsafe_allow_html=True,
            )
        else:
            figure, axis = plt.subplots(figsize=(7.0, 4.0), facecolor="#F3F6FA")
            style_axis(axis, f"Derivative of {selected}", "light")
            axis.axhline(0, color=GRID, linewidth=0.8)
            axis.axvline(0, color=GRID, linewidth=0.8)
            axis.plot(x_values, selected_derivative, color="#8B79CA", linewidth=2.6)
            axis.scatter([point], [derivative_output], color="#49B7C4", edgecolors="#FFFFFF", s=48, zorder=4)
            axis.set_xlabel("Input x")
            axis.set_ylabel("Derivative")
            figure.tight_layout()
            st.pyplot(figure, width="stretch")
            plt.close(figure)

    result_col, explanation_col = st.columns([0.8, 1.2], gap="medium")
    with result_col:
        st.markdown('<section class="info-panel"><div class="eyebrow">AT THIS INPUT</div><h2>Numerical output</h2>', unsafe_allow_html=True)
        if selected == "Softmax":
            st.code(f"logits [x, 0, -x] = [{point:.2f}, 0, {-point:.2f}]\nprobabilities = {np.array2string(point_output, precision=4)}", language="text")
        else:
            st.metric(label=f"{selected}(x)", value=f"{float(point_output):.6f}")
        st.markdown("</section>", unsafe_allow_html=True)
    with explanation_col:
        st.markdown(
            f'<section class="info-panel"><div class="eyebrow">FUNCTION GUIDE</div><h2>{selected}</h2>'
            f'<p><b style="color:#4bd4e8">Output range:</b> {output_range}</p>'
            f'<p>{use_case}</p></section>',
            unsafe_allow_html=True,
        )

    with st.expander("Compare activation curves"):
        figure, axis = plt.subplots(figsize=(11.0, 4.8), facecolor="#F3F6FA")
        style_axis(axis, "Elementwise activation comparison", "light")
        axis.axhline(0, color=GRID, linewidth=0.8)
        curve_colors = ("#2F5F83", "#397F8B", "#8B79CA", "#5B8F75")
        for (name, function), color in zip(CURVES.items(), curve_colors):
            values = function(x_values, alpha=alpha) if name == "Leaky ReLU" else function(x_values)
            axis.plot(x_values, values, linewidth=2.1, color=color, label=name)
        axis.set_xlabel("Input x")
        axis.set_ylabel("Output")
        axis.legend(facecolor="#FFFFFF", edgecolor="#C8D9E6", labelcolor="#2F4156", ncol=4)
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)
        st.caption("Softmax is excluded from this scalar comparison because it transforms a vector of logits together.")
