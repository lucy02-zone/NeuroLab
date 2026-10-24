"""Interactive Streamlit lab for fair optimizer comparisons."""

import matplotlib.pyplot as plt
import streamlit as st

from .datasets import make_classification_dataset
from .experiment_ui import render_save_experiment_control
from .optimizer_comparison import OPTIMIZERS, OptimizerComparison, run_optimizer_comparison
from .visualization import (
    plot_dataset_preview,
    plot_decision_boundary,
    plot_optimizer_final_accuracy,
    plot_optimizer_history,
)


DATASETS = {
    "XOR": "xor",
    "Two interleaving moons": "moons",
    "Concentric circles": "circles",
    "Spiral classification": "spiral",
}
ACTIVATIONS = {"ReLU": "relu", "Leaky ReLU": "leaky_relu", "Sigmoid": "sigmoid", "Tanh": "tanh"}
RESULT_KEY = "optimizer_lab_result"
RUN_COUNT_KEY = "optimizer_lab_run_count"


@st.cache_data(show_spinner=False)
def _cached_dataset(name: str, samples: int, noise: float, seed: int):
    return make_classification_dataset(name, n_samples=samples, noise=noise, random_seed=seed)


def _show_figure(figure: plt.Figure) -> None:
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def _summary_rows(comparison: OptimizerComparison) -> list[dict[str, str]]:
    rows = []
    for name, result in comparison.runs.items():
        history = result.history
        row = {
            "Optimizer": name,
            "Training loss": f"{history['training_loss'][-1]:.4f}" if history["training_loss"] else "—",
            "Validation loss": f"{history['validation_loss'][-1]:.4f}" if history.get("validation_loss") else "—",
            "Epochs completed": str(result.epochs_completed),
        }
        row["Training accuracy"] = f"{history['training_accuracy'][-1]:.1%}" if history.get("training_accuracy") else "—"
        row["Validation accuracy"] = f"{history['validation_accuracy'][-1]:.1%}" if history.get("validation_accuracy") else "—"
        rows.append(row)
    return rows


def render_optimizer_lab() -> None:
    """Render experiment configuration and real optimizer comparison results."""
    st.markdown('<div class="eyebrow">COMPARE PARAMETER UPDATE STRATEGIES</div>', unsafe_allow_html=True)
    st.title("Optimizer Lab")
    st.markdown("Run the same neural network experiment with independent SGD, Momentum, and Adam optimizers.")

    st.markdown("### 1 · Experiment setup")
    data_columns = st.columns([0.85, 1.15], gap="large")
    with data_columns[0]:
        dataset_label = st.selectbox("Synthetic dataset", tuple(DATASETS), key="optimizer_dataset")
        sample_count = st.slider("Number of samples", 100, 1000, 300, 50, key="optimizer_samples")
        noise = st.slider("Dataset noise", 0.0, 0.5, 0.08, 0.01, key="optimizer_noise")
        seed = int(st.number_input("Random seed", 0, 999_999, 21, 1, key="optimizer_seed"))
        validation_fraction = st.slider("Validation split", 0.1, 0.4, 0.2, 0.05, key="optimizer_validation")
    features, labels = _cached_dataset(DATASETS[dataset_label], sample_count, noise, seed)
    with data_columns[1]:
        _show_figure(plot_dataset_preview(features, labels, f"{dataset_label} · {sample_count} samples"))

    st.markdown("### 2 · Shared network and training settings")
    shared = st.columns(5, gap="medium")
    with shared[0]:
        layer_count = st.slider("Hidden layers", 0, 5, 2, key="optimizer_layers")
    sizes = []
    for index in range(layer_count):
        sizes.append(st.number_input(f"Layer {index + 1} neurons", 1, 32, 8, 1, key=f"optimizer_layer_{index}"))
    with shared[1]:
        activation_label = st.selectbox("Activation", tuple(ACTIVATIONS), key="optimizer_activation")
    with shared[2]:
        epochs = st.number_input("Epochs", 1, 1000, 100, 10, key="optimizer_epochs")
    with shared[3]:
        batch_size = st.number_input("Batch size", 1, sample_count, min(32, sample_count), 1, key="optimizer_batch")
    with shared[4]:
        learning_rate = st.number_input("Learning rate", 0.0001, 1.0, 0.01, 0.001, format="%.4f", key="optimizer_lr")
    st.caption("All runs use binary cross-entropy and the sigmoid binary-classification output. The same stratified split, initialization seed, and mini-batch order are used for each optimizer.")

    st.markdown("### 3 · Optimizer configuration")
    selected = st.multiselect("Optimizers to compare", OPTIMIZERS, default=list(OPTIMIZERS), key="optimizer_selection")
    parameter_columns = st.columns(3, gap="medium")
    momentum = 0.9
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    with parameter_columns[0]:
        if "SGD" in selected:
            st.info("**SGD** updates each parameter directly: θ ← θ − η∇L.")
    with parameter_columns[1]:
        if "Momentum" in selected:
            momentum = st.slider("Momentum coefficient", 0.0, 0.99, 0.9, 0.01, key="optimizer_momentum")
            st.caption("Accumulates a velocity from recent gradients to smooth updates.")
    with parameter_columns[2]:
        if "Adam" in selected:
            beta1 = st.number_input("Adam β₁", 0.0, 0.999, 0.9, 0.01, format="%.3f", key="optimizer_beta1")
            beta2 = st.number_input("Adam β₂", 0.0, 0.9999, 0.999, 0.001, format="%.4f", key="optimizer_beta2")
            epsilon = st.number_input("Adam ε", 1e-10, 1e-3, 1e-8, format="%.1e", key="optimizer_epsilon")
            st.caption("Tracks bias-corrected first and second gradient moments.")

    st.markdown("### 4 · Comparison controls")
    run_clicked = st.button("Run Comparison", type="primary", use_container_width=True, key="optimizer_run")
    if run_clicked:
        if not selected:
            st.warning("Select at least one optimizer to run a comparison.")
        else:
            try:
                with st.spinner("Running independent optimizer experiments…"):
                    comparison = run_optimizer_comparison(
                        features,
                        labels,
                        selected,
                        hidden_layer_sizes=tuple(int(value) for value in sizes),
                        hidden_activation=ACTIVATIONS[activation_label],
                        epochs=int(epochs),
                        batch_size=int(batch_size),
                        learning_rate=float(learning_rate),
                        validation_fraction=float(validation_fraction),
                        random_seed=seed,
                        momentum=float(momentum),
                        beta1=float(beta1),
                        beta2=float(beta2),
                        optimizer_epsilon=float(epsilon),
                    )
                run_number = int(st.session_state.get(RUN_COUNT_KEY, 0)) + 1
                st.session_state[RUN_COUNT_KEY] = run_number
                st.session_state[RESULT_KEY] = {
                    "comparison": comparison,
                    "run_number": run_number,
                    "dataset_name": dataset_label,
                    "selected": tuple(selected),
                    "configuration": {
                        "epochs": int(epochs), "batch_size": int(batch_size), "learning_rate": float(learning_rate),
                        "hidden_layer_sizes": tuple(int(value) for value in sizes), "activation": activation_label,
                        "hidden_activation": ACTIVATIONS[activation_label], "seed": seed,
                        "loss": "binary_cross_entropy", "output_activation": "sigmoid",
                        "epochs_completed": {name: result.epochs_completed for name, result in comparison.runs.items()},
                    },
                    "dataset_configuration": {
                        "n_samples": int(sample_count), "noise": float(noise),
                        "validation_fraction": float(validation_fraction), "data_seed": seed,
                        "split_seed": seed, "standardize": True,
                    },
                }
                st.success(f"Comparison run {run_number} completed.")
            except (ValueError, FloatingPointError, RuntimeError) as error:
                st.error(f"Comparison could not be completed: {error}")
            except Exception as error:
                st.error(f"Unexpected comparison error: {error}")

    saved = st.session_state.get(RESULT_KEY)
    st.markdown("### 5 · Loss and validation plots")
    if saved is None:
        st.info("Configure the experiment and select **Run Comparison** to record real optimizer results.")
        return

    comparison: OptimizerComparison = saved["comparison"]
    st.caption(f"Showing run {saved['run_number']} · {saved['dataset_name']} · {', '.join(saved['selected'])}")
    _show_figure(plot_optimizer_history(comparison.runs, "training_loss", "Training loss by epoch"))
    _show_figure(plot_optimizer_history(comparison.runs, "validation_loss", "Validation loss by epoch"))

    st.markdown("### 6 · Training summary")
    st.table(_summary_rows(comparison))
    _show_figure(plot_optimizer_final_accuracy(comparison.runs))

    st.markdown("### 7 · Decision boundary comparison")
    boundary_optimizer = st.selectbox("Optimizer model", tuple(comparison.runs), key="optimizer_boundary")
    result = comparison.runs[boundary_optimizer]
    boundary = plot_decision_boundary(
        result.network, comparison.features, comparison.labels, scaler=comparison.split.scaler
    )
    boundary.axes[0].set_title(f"{boundary_optimizer} · learned decision boundary", loc="left", color="#eaf1ff", fontsize=12, pad=12)
    _show_figure(boundary)
    save_candidates = {}
    for name, result in comparison.runs.items():
        run_config = {
            **saved["configuration"],
            "optimizer": name,
            "learning_rate": float(result.optimizer.learning_rate),
            "batch_size": saved["configuration"]["batch_size"],
            "epochs_requested": saved["configuration"]["epochs"],
            "epochs_completed": result.epochs_completed,
            "optimizer_config": _optimizer_configuration(result.optimizer),
        }
        save_candidates[name] = {
            "dataset": saved["dataset_name"],
            "dataset_configuration": saved["dataset_configuration"],
            "configuration": run_config,
            "history": result.history,
        }
    render_save_experiment_control(save_candidates, key="optimizer_lab")


def _optimizer_configuration(optimizer: object) -> dict[str, float | int | str]:
    """Describe the selected optimizer without persisting its mutable state."""
    values: dict[str, float | int | str] = {
        "type": type(optimizer).__name__, "learning_rate": float(optimizer.learning_rate)
    }
    for attribute in ("momentum", "beta1", "beta2", "epsilon"):
        if hasattr(optimizer, attribute):
            values[attribute] = getattr(optimizer, attribute)
    return values
