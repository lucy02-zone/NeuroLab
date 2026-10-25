"""Streamlit interface for configuring and training a small neural network."""

import matplotlib.pyplot as plt
import streamlit as st

from src.neurolab.datasets import make_classification_dataset, split_classification_data
from src.neurolab.experiment_ui import render_save_experiment_control
from src.neurolab.training import TrainingResult, train_model
from src.neurolab.visualization import (
    plot_dataset_preview,
    plot_decision_boundary,
    plot_loss_history,
    plot_network_architecture,
)


DATASETS = {
    "XOR": "xor",
    "Two interleaving moons": "moons",
    "Concentric circles": "circles",
    "Spiral classification": "spiral",
}
ACTIVATIONS = {
    "ReLU": "relu",
    "Leaky ReLU": "leaky_relu",
    "Sigmoid": "sigmoid",
    "Tanh": "tanh",
}
RESULT_KEY = "network_playground_result"
RUN_COUNT_KEY = "network_playground_run_count"


@st.cache_data(show_spinner=False)
def _cached_dataset(name: str, samples: int, noise: float, seed: int,
                    class_1_fraction: float | None = None, circle_factor: float = 0.5,
                    spiral_turns: float = 2.0):
    return make_classification_dataset(
        name, n_samples=samples, noise=noise, random_seed=seed,
        class_1_fraction=class_1_fraction, circle_factor=circle_factor, spiral_turns=spiral_turns,
    )


def _show_figure(figure: plt.Figure) -> None:
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def _render_metrics(result: TrainingResult) -> None:
    history = result.history
    if not history["epoch"]:
        st.info("No epoch completed in this run. Adjust the configuration and train again.")
        return
    metric_columns = st.columns(5)
    final_training_loss = history["training_loss"][-1]
    metric_columns[0].metric("Training loss", f"{final_training_loss:.4f}")
    if "validation_loss" in history:
        metric_columns[1].metric("Validation loss", f"{history['validation_loss'][-1]:.4f}")
    else:
        metric_columns[1].metric("Validation loss", "—")
    if "training_accuracy" in history:
        metric_columns[2].metric("Training accuracy", f"{history['training_accuracy'][-1]:.1%}")
        metric_columns[3].metric("Validation accuracy", f"{history['validation_accuracy'][-1]:.1%}")
    else:
        metric_columns[2].metric("Training accuracy", "Not applicable")
        metric_columns[3].metric("Validation accuracy", "Not applicable")
    metric_columns[4].metric("Epochs completed", str(result.epochs_completed))


def render_network_playground() -> None:
    """Render dataset controls, architecture, training action, and results."""
    st.markdown('<div class="eyebrow">CONFIGURE A NETWORK EXPERIMENT</div>', unsafe_allow_html=True)
    st.title("Network Playground")
    st.markdown("Choose a synthetic classification problem, shape a network, and train it with NeuroLab's NumPy engine.")

    st.markdown("### 1 · Dataset")
    data_controls, data_preview = st.columns([0.85, 1.15], gap="large")
    with data_controls:
        dataset_label = st.selectbox("Synthetic dataset", tuple(DATASETS), key="play_dataset")
        sample_count = st.slider("Number of samples", 40, 2000, 300, 20, key="play_samples")
        noise = st.slider("Dataset noise", 0.0, 0.5, 0.08, 0.01, key="play_noise")
        random_seed = st.number_input("Random seed", min_value=0, max_value=999_999, value=21, step=1, key="play_seed")
        class_balance = st.selectbox(
            "Class balance", ("Generator default", *(f"{share}% class 1" for share in range(10, 95, 5))),
            key="play_class_balance", help="Set a target class share or use each generator's default distribution.",
        )
        class_1_fraction = None if class_balance == "Generator default" else int(class_balance.split("%", 1)[0]) / 100
        circle_factor = 0.5
        spiral_turns = 2.0
        if dataset_label == "Concentric circles":
            circle_factor = st.slider("Inner-circle radius factor", 0.15, 0.85, 0.5, 0.05, key="play_circle_factor")
        if dataset_label == "Spiral classification":
            spiral_turns = st.slider("Spiral turns", 1.0, 4.0, 2.0, 0.25, key="play_spiral_turns")
        validation_fraction = st.slider(
            "Validation split", 0.1, 0.4, 0.2, 0.05, key="play_validation_fraction",
            help="The split is stratified. Standardization is fitted on training samples only.",
        )
    features, labels = _cached_dataset(
        DATASETS[dataset_label], sample_count, noise, int(random_seed),
        class_1_fraction, float(circle_factor), float(spiral_turns),
    )
    with data_preview:
        preview_figure = plot_dataset_preview(features, labels, f"{dataset_label} · {sample_count} samples")
        _show_figure(preview_figure)

    st.markdown("### 2 · Network architecture")
    architecture_controls, architecture_diagram = st.columns([0.75, 1.25], gap="large")
    with architecture_controls:
        hidden_layer_count = st.slider("Hidden layers", 0, 5, 2, key="play_hidden_layer_count")
        hidden_sizes = []
        for index in range(hidden_layer_count):
            hidden_sizes.append(
                st.number_input(
                    f"Layer {index + 1} neurons", min_value=1, max_value=32, value=8, step=1,
                    key=f"play_hidden_size_{index}",
                )
            )
        activation_label = st.selectbox("Hidden-layer activation", tuple(ACTIVATIONS), key="play_activation")
        st.selectbox(
            "Output activation", ("Sigmoid · binary classification",), disabled=True,
            help="The current model uses a sigmoid output. Softmax output is not implemented yet.",
            key="play_output_activation",
        )
    with architecture_diagram:
        architecture_figure = plot_network_architecture(
            hidden_sizes, input_size=2, output_size=1, hidden_activation=ACTIVATIONS[activation_label]
        )
        _show_figure(architecture_figure)
        st.caption("Every neuron in one layer connects to every neuron in the next layer.")

    st.markdown("### 3 · Training configuration")
    config_columns = st.columns(4, gap="medium")
    with config_columns[0]:
        epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=100, step=10, key="play_epochs")
    with config_columns[1]:
        learning_rate = st.number_input(
            "Learning rate", min_value=0.0001, max_value=1.0, value=0.01, step=0.001,
            format="%.4f", key="play_learning_rate",
        )
    with config_columns[2]:
        batch_size = st.number_input(
            "Batch size", min_value=1, max_value=sample_count, value=min(32, sample_count), step=1,
            key="play_batch_size",
        )
    with config_columns[3]:
        optimizer_label = st.selectbox("Optimizer", ("SGD", "Momentum", "Adam"), index=2, key="play_optimizer")

    loss_label = st.selectbox(
        "Loss function", ("Binary cross-entropy",), key="play_loss",
        help="The playground datasets are binary classification tasks and use the model's sigmoid output. "
             "MSE and MAE are available in the training engine for regression; categorical cross-entropy needs a softmax head.",
    )
    st.caption("Binary classification uses a 0.5 probability threshold. Current output activation: sigmoid.")

    train_clicked = st.button("Train Network", type="primary", width="stretch", key="play_train_button")
    if train_clicked:
        try:
            # Dataset and scaler are prepared only after an explicit training action.
            split = split_classification_data(
                features,
                labels,
                validation_fraction=float(validation_fraction),
                random_seed=int(random_seed),
                standardize=True,
            )
            with st.spinner("Training the configured network…"):
                result = train_model(
                    split.X_train,
                    split.y_train,
                    hidden_layer_sizes=tuple(int(size) for size in hidden_sizes),
                    hidden_activation=ACTIVATIONS[activation_label],
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    optimizer=optimizer_label.lower(),
                    learning_rate=float(learning_rate),
                    loss="binary_cross_entropy",
                    validation_data=split.validation_data,
                    random_seed=int(random_seed),
                )
            run_number = int(st.session_state.get(RUN_COUNT_KEY, 0)) + 1
            st.session_state[RUN_COUNT_KEY] = run_number
            st.session_state[RESULT_KEY] = {
                "result": result,
                "features": features.copy(),
                "labels": labels.copy(),
                "scaler": split.scaler,
                "dataset_name": dataset_label,
                "run_number": run_number,
                "config": {
                    "hidden_layer_sizes": tuple(int(size) for size in hidden_sizes),
                    "hidden_activation": activation_label,
                    "optimizer": optimizer_label,
                    "epochs_requested": int(epochs),
                    "learning_rate": float(learning_rate),
                    "batch_size": int(batch_size),
                    "validation_fraction": float(validation_fraction),
                    "seed": int(random_seed),
                    "loss": loss_label,
                    "epochs_completed": result.epochs_completed,
                    "output_activation": "sigmoid",
                    "optimizer_config": _optimizer_configuration(result.optimizer),
                },
                "dataset_configuration": {
                    "n_samples": int(sample_count),
                    "noise": float(noise),
                    "class_1_fraction": class_1_fraction,
                    "circle_factor": float(circle_factor),
                    "spiral_turns": float(spiral_turns),
                    "validation_fraction": float(validation_fraction),
                    "data_seed": int(random_seed),
                    "split_seed": int(random_seed),
                    "standardize": True,
                },
            }
            st.success(f"Training run {run_number} completed.")
        except (ValueError, FloatingPointError, RuntimeError) as error:
            st.error(f"Training could not be completed: {error}")
        except Exception as error:  # Surface unexpected failures without losing prior results.
            st.error(f"Unexpected training error: {error}")

    saved_run = st.session_state.get(RESULT_KEY)
    if saved_run is None:
        st.markdown("### 4 · Training results")
        st.info("The dataset preview and architecture are ready. Select **Train Network** to create real training results.")
        st.markdown("### 5 · Loss curves")
        st.caption("Loss curves will appear after a training run.")
        st.markdown("### 6 · Decision boundary")
        st.caption("A decision boundary requires a trained model; no predictions are shown yet.")
        return

    result: TrainingResult = saved_run["result"]
    st.markdown("### 4 · Training results")
    st.caption(
        f"Showing run {saved_run['run_number']} · {saved_run['dataset_name']} · "
        f"{saved_run['config']['optimizer']} · {saved_run['config']['epochs_requested']} requested epochs"
    )
    if result.stopped_early:
        st.warning("Training stopped early at a mini-batch boundary.")
    _render_metrics(result)

    st.markdown("### 5 · Loss curves")
    loss_figure = plot_loss_history(result.history)
    _show_figure(loss_figure)

    st.markdown("### 6 · Decision boundary")
    boundary_figure = plot_decision_boundary(
        result.network,
        saved_run["features"],
        saved_run["labels"],
        scaler=saved_run["scaler"],
    )
    _show_figure(boundary_figure)

    config = saved_run["config"]
    save_configuration = {
        **config,
        "hidden_activation": config["hidden_activation"],
        "optimizer": config["optimizer"],
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "epochs_requested": config["epochs_requested"],
        "loss": "binary_cross_entropy",
        "random_seed": config["seed"],
    }
    render_save_experiment_control({
        f"Run {saved_run['run_number']}": {
            "dataset": saved_run["dataset_name"],
            "dataset_configuration": saved_run["dataset_configuration"],
            "configuration": save_configuration,
            "history": result.history,
        }
    }, key="network_playground")


def _optimizer_configuration(optimizer: object) -> dict[str, float | int | str]:
    """Describe optimizer settings without persisting optimizer state or objects."""
    values: dict[str, float | int | str] = {"type": type(optimizer).__name__,
                                            "learning_rate": float(optimizer.learning_rate)}
    for attribute in ("momentum", "beta1", "beta2", "epsilon"):
        if hasattr(optimizer, attribute):
            values[attribute] = getattr(optimizer, attribute)
    return values
