"""Interactive page for learning how common loss functions behave."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.neurolab.models import losses
from .visualization import style_axis


LOSS_NAMES = ("Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "Binary Cross-Entropy", "Categorical Cross-Entropy")
LOSS_INFO = {
    LOSS_NAMES[0]: ("Regression", r"L = \frac{1}{N}\sum_i (y_i-\hat{y}_i)^2", "Penalizes larger errors more strongly because the differences are squared."),
    LOSS_NAMES[1]: ("Regression", r"L = \frac{1}{N}\sum_i |y_i-\hat{y}_i|", "Measures average absolute error and is less influenced by large outliers than MSE."),
    LOSS_NAMES[2]: ("Binary classification", r"L = -\frac{1}{N}\sum_i [y_i\log(p_i)+(1-y_i)\log(1-p_i)]", "Scores predicted probabilities for binary labels; confident incorrect predictions receive a large penalty."),
    LOSS_NAMES[3]: ("Multiclass classification", r"L = -\frac{1}{N}\sum_i\sum_c y_{ic}\log(p_{ic})", "Scores a probability distribution against one-hot class targets; class losses are summed per sample."),
}


def _plot(values: np.ndarray, curves: dict[str, np.ndarray], title: str, target: float | None = None) -> None:
    figure, axis = plt.subplots(figsize=(9.5, 4.2), facecolor="#F3F6FA")
    style_axis(axis, title, "light")
    curve_colors = ("#2F5F83", "#397F8B", "#8B79CA", "#5B8F75")
    for (label, curve), color in zip(curves.items(), curve_colors):
        axis.plot(values, curve, linewidth=2.4, color=color, label=label)
    if target is not None:
        axis.axvline(target, color="#8B79CA", linestyle="--", linewidth=1.2, label="Target")
    axis.set_xlabel("Prediction" if target is not None else "Predicted probability")
    axis.set_ylabel("Loss")
    axis.legend(facecolor="#FFFFFF", edgecolor="#C8D9E6", labelcolor="#2F4156")
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def render_loss_playground() -> None:
    """Render interactive loss explanations, numerical examples, and curves."""
    with st.container(key="loss_playground_page"):
        _render_loss_content()


def _render_loss_content() -> None:
    st.markdown(
        """<section class="hero playground-hero loss-hero">
            <div class="hero-copy">
                <div class="hero-kicker">MEASURING PREDICTION ERROR</div>
                <h1>Loss Functions<span class="hero-period">.</span></h1>
                <div class="hero-tagline">Turn prediction errors into a learning signal.</div>
                <p class="hero-description">Compare regression and classification losses with formulas, live examples, and real loss curves.</p>
                <div class="hero-chip"><span class="status-dot"></span>Interactive calculations · Clear use cases</div>
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

    with st.container(border=True, key="loss_intro_panel"):
        selected = st.selectbox("Choose a loss", LOSS_NAMES, key="loss_playground_select")
        task_type, formula, explanation = LOSS_INFO[selected]
        st.markdown(f'<span class="loss-task-badge">{task_type}</span>', unsafe_allow_html=True)
        st.latex(formula)
        st.markdown(f'<p class="loss-explanation">{explanation}</p>', unsafe_allow_html=True)

    if selected in LOSS_NAMES[:2]:
        target_col, prediction_col = st.columns(2, gap="medium")
        with target_col:
            target = st.number_input("Target value", value=1.0, min_value=-10.0, max_value=10.0, step=0.1)
        with prediction_col:
            prediction = st.number_input("Prediction value", value=0.25, min_value=-10.0, max_value=10.0, step=0.1)
        y = np.array([target])
        p = np.array([prediction])
        calculator = losses.mean_squared_error if selected == LOSS_NAMES[0] else losses.mean_absolute_error
        calculated_loss = calculator(y, p)
        st.caption("Regression example: target and prediction are real-valued; one scalar pair is shown.")
        x_values = np.linspace(target - 3.0, target + 3.0, 400)
        curves = {
            "MSE": (x_values - target) ** 2,
            "MAE": np.abs(x_values - target),
        }
        comparison_label = "Regression loss comparison (same target)"
        graph_values = x_values
        target_marker = target
        _plot(graph_values, curves, comparison_label, target_marker)
    elif selected == LOSS_NAMES[2]:
        target_col, prediction_col = st.columns(2, gap="medium")
        with target_col:
            target = st.selectbox("Binary target", (0.0, 1.0), format_func=lambda value: str(int(value)))
        with prediction_col:
            prediction = st.slider("Predicted probability", 0.001, 0.999, 0.75, 0.001)
        y = np.array([target])
        p = np.array([prediction])
        calculated_loss = losses.binary_cross_entropy(y, p)
        st.caption("Binary classification example: target is 0 or 1 and prediction is a probability in (0, 1).")
        x_values = np.linspace(0.001, 0.999, 400)
        curves = {
            "BCE, target 1": -np.log(x_values),
            "BCE, target 0": -np.log1p(-x_values),
        }
        _plot(x_values, curves, "Binary cross-entropy for both target classes")
    else:
        classes_col, target_col, prediction_col = st.columns([0.8, 1.1, 1.3], gap="medium")
        with classes_col:
            class_count = st.selectbox("Number of classes", (2, 3, 4, 5), index=2)
        with target_col:
            target_class = st.selectbox("Target class", tuple(range(class_count)), format_func=lambda value: f"Class {value}")
        with prediction_col:
            target_probability = st.slider("Probability assigned to target class", 0.01, 0.99, 0.70, 0.01)
        y = np.zeros((1, class_count), dtype=float)
        y[0, target_class] = 1.0
        p = np.full((1, class_count), (1.0 - target_probability) / (class_count - 1))
        p[0, target_class] = target_probability
        calculated_loss = losses.categorical_cross_entropy(y, p)
        st.caption("Multiclass example: one-hot target and a valid class-probability vector that sums to 1.")
        x_values = np.linspace(0.01, 0.99, 400)
        cce_curve = -np.log(x_values)
        bce_curve = -np.log(x_values)
        _plot(x_values, {"Categorical CE (true class)": cce_curve, "Binary CE (positive target)": bce_curve},
              "Classification loss versus probability on the true class")

    display_cols = st.columns([0.75, 1.25], gap="medium")
    with display_cols[0]:
        st.metric("Calculated loss", f"{calculated_loss:.6f}")
    with display_cols[1]:
        st.markdown('<section class="info-panel"><div class="eyebrow">NUMERICAL EXAMPLE</div><h3>Targets and predictions</h3>', unsafe_allow_html=True)
        st.code(f"target = {np.array2string(y, precision=4)}\nprediction = {np.array2string(p, precision=4)}", language="text")
        st.markdown("</section>", unsafe_allow_html=True)

    with st.expander("Compare suitable loss functions"):
        if selected in LOSS_NAMES[:2]:
            st.markdown("MSE and MAE are compared for the selected regression target.")
            st.caption("At a perfect prediction, both losses are zero. MSE grows quadratically; MAE grows linearly.")
        else:
            st.markdown("Classification losses are shown against confidence assigned to the true class.")
            st.caption("Categorical cross-entropy extends the positive-class binary formula by summing over classes.")
