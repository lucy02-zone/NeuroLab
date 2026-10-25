"""Guided end-to-end experiment workflow built from NeuroLab's shared services."""

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from .datasets import make_classification_dataset, split_classification_data
from .experiment_store import ExperimentStore
from .training import TrainingResult, train_model
from .visualization import plot_dataset_preview, plot_decision_boundary, plot_loss_history


STATE_KEY = "experiment_wizard"
RESULT_KEY = "experiment_wizard_result"
STEP_NAMES = (
    "Experiment details", "Dataset", "Network", "Training", "Review",
    "Train", "Evaluate", "Save",
)
DATASETS = {
    "XOR": "xor", "Two interleaving moons": "moons",
    "Concentric circles": "circles", "Spiral classification": "spiral",
}
ACTIVATIONS = {"ReLU": "relu", "Leaky ReLU": "leaky_relu", "Sigmoid": "sigmoid", "Tanh": "tanh"}
OPTIMIZERS = ("SGD", "Momentum", "Adam")


def new_wizard_state() -> dict[str, Any]:
    """Return fresh default state for one browser session's wizard."""
    return {
        "step": 0, "confirmed": False, "saved_id": None,
        "details": {"name": "", "description": "", "notes": "", "seed": 21},
        "dataset": {"name": "XOR", "samples": 300, "noise": 0.08,
                    "class_1_fraction": 0.5, "circle_factor": 0.5, "spiral_turns": 2.0,
                    "validation_fraction": 0.2},
        "network": {"hidden_layers": 2, "hidden_layer_sizes": [8, 8],
                    "hidden_activation": "ReLU", "output_activation": "Sigmoid · binary classification",
                    "loss": "Binary cross-entropy"},
        "training": {"optimizer": "Adam", "learning_rate": 0.01, "epochs": 100,
                     "batch_size": 32, "momentum": 0.9, "beta1": 0.9,
                     "beta2": 0.999, "epsilon": 1e-8},
    }


def validate_wizard_step(step_number: int, config: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate the fields needed to leave one configuration step (1–4)."""
    errors: list[str] = []
    if step_number == 1:
        details = config.get("details", {})
        if not str(details.get("name", "")).strip():
            errors.append("Enter an experiment name before continuing.")
        if not 0 <= int(details.get("seed", -1)) <= 999_999:
            errors.append("Random seed must be between 0 and 999999.")
    elif step_number == 2:
        data = config.get("dataset", {})
        if data.get("name") not in DATASETS:
            errors.append("Select one of the supported datasets.")
        if not isinstance(data.get("samples"), (int, np.integer)) or not 40 <= data["samples"] <= 2000:
            errors.append("Sample count must be between 40 and 2000.")
        if not np.isfinite(data.get("noise", np.nan)) or not 0 <= data["noise"] <= 0.5:
            errors.append("Noise must be between 0 and 0.5.")
        if not np.isfinite(data.get("class_1_fraction", np.nan)) or not 0.1 <= data["class_1_fraction"] <= 0.9:
            errors.append("Class 1 share must be between 10% and 90%.")
        if not np.isfinite(data.get("validation_fraction", np.nan)) or not 0.1 <= data["validation_fraction"] <= 0.4:
            errors.append("Validation split must be between 10% and 40%.")
    elif step_number == 3:
        network = config.get("network", {})
        layers = network.get("hidden_layer_sizes", [])
        if len(layers) != network.get("hidden_layers"):
            errors.append("Provide a neuron count for every hidden layer.")
        if any(isinstance(width, bool) or not isinstance(width, (int, np.integer)) or width < 1 for width in layers):
            errors.append("Each hidden layer must contain at least one neuron.")
        if network.get("hidden_activation") not in ACTIVATIONS:
            errors.append("Choose a supported hidden-layer activation.")
        if network.get("output_activation") != "Sigmoid · binary classification" or network.get("loss") != "Binary cross-entropy":
            errors.append("The supported binary output requires sigmoid with binary cross-entropy.")
    elif step_number == 4:
        training = config.get("training", {})
        if training.get("optimizer") not in OPTIMIZERS:
            errors.append("Choose SGD, Momentum, or Adam.")
        if not np.isfinite(training.get("learning_rate", np.nan)) or not 0.0001 <= training["learning_rate"] <= 1.0:
            errors.append("Learning rate must be between 0.0001 and 1.0.")
        if not isinstance(training.get("epochs"), (int, np.integer)) or not 1 <= training["epochs"] <= 1000:
            errors.append("Epoch count must be between 1 and 1000.")
        if not isinstance(training.get("batch_size"), (int, np.integer)) or not 1 <= training["batch_size"] <= config.get("dataset", {}).get("samples", 0):
            errors.append("Batch size must be positive and no larger than the dataset.")
        for name, low, high in (("momentum", 0.0, 0.99), ("beta1", 0.0, 0.999), ("beta2", 0.0, 0.9999), ("epsilon", 1e-10, 1e-3)):
            value = training.get(name, low)
            if not np.isfinite(value) or not low <= value <= high:
                errors.append(f"{name} is outside its supported range.")
    else:
        errors.append("Wizard step must be between 1 and 4.")
    return tuple(errors)


def wizard_next(state: dict[str, Any], config: Mapping[str, Any]) -> tuple[str, ...]:
    """Advance through configuration only when the current step is valid."""
    step = int(state.get("step", 0))
    if step >= 4:
        return ("Use the review confirmation to authorize training.",)
    errors = validate_wizard_step(step + 1, config)
    if not errors:
        state["step"] = step + 1
    return errors


def wizard_previous(state: dict[str, Any]) -> None:
    """Move one screen back without discarding any configured values."""
    state["step"] = max(0, int(state.get("step", 0)) - 1)


def training_is_authorized(state: Mapping[str, Any]) -> bool:
    """Training is available only after review confirmation and on its step."""
    return bool(state.get("confirmed")) and int(state.get("step", -1)) == 5


def _dataset(configuration: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    data = configuration["dataset"]
    return make_classification_dataset(
        DATASETS[data["name"]], n_samples=int(data["samples"]), noise=float(data["noise"]),
        random_seed=int(configuration["details"]["seed"]),
        class_1_fraction=float(data["class_1_fraction"]), circle_factor=float(data["circle_factor"]),
        spiral_turns=float(data["spiral_turns"]),
    )


def _render_dataset_summary(configuration: Mapping[str, Any], *, preview: bool = True) -> tuple[np.ndarray, np.ndarray]:
    features, labels = _dataset(configuration)
    data = configuration["dataset"]
    counts = np.bincount(labels, minlength=2)
    columns = st.columns(4)
    columns[0].metric("Samples", f"{len(labels):,}")
    columns[1].metric("Class 0", f"{counts[0]:,}")
    columns[2].metric("Class 1", f"{counts[1]:,}")
    columns[3].metric("Features", str(features.shape[1]))
    st.caption(f"Actual class shares: {counts[0] / len(labels):.1%} / {counts[1] / len(labels):.1%}.")
    if preview:
        figure = plot_dataset_preview(features, labels, f"{data['name']} · {len(labels)} samples")
        st.pyplot(figure, width="stretch")
        plt.close(figure)
    return features, labels


def _collect_widget_values(state: dict[str, Any]) -> None:
    """Copy visible widget values into the wizard's durable config object."""
    details, data, network, training = (state[k] for k in ("details", "dataset", "network", "training"))
    details.update({"name": st.session_state.get("wizard_name", details["name"]),
                    "description": st.session_state.get("wizard_description", details["description"]),
                    "notes": st.session_state.get("wizard_notes", details["notes"]),
                    "seed": int(st.session_state.get("wizard_seed", details["seed"]))})
    data.update({"name": st.session_state.get("wizard_dataset", data["name"]),
                 "samples": int(st.session_state.get("wizard_samples", data["samples"])),
                 "noise": float(st.session_state.get("wizard_noise", data["noise"])),
                 "class_1_fraction": int(st.session_state.get("wizard_class_balance", 50)) / 100,
                 "validation_fraction": float(st.session_state.get("wizard_validation", data["validation_fraction"]))})
    data["circle_factor"] = float(st.session_state.get("wizard_circle_factor", data["circle_factor"]))
    data["spiral_turns"] = float(st.session_state.get("wizard_spiral_turns", data["spiral_turns"]))
    layer_count = int(st.session_state.get("wizard_layers", network["hidden_layers"]))
    network["hidden_layers"] = layer_count
    network["hidden_layer_sizes"] = [int(st.session_state.get(f"wizard_layer_{i}", 8)) for i in range(layer_count)]
    network["hidden_activation"] = st.session_state.get("wizard_activation", network["hidden_activation"])
    training.update({
        "optimizer": st.session_state.get("wizard_optimizer", training["optimizer"]),
        "learning_rate": float(st.session_state.get("wizard_learning_rate", training["learning_rate"])),
        "epochs": int(st.session_state.get("wizard_epochs", training["epochs"])),
        "batch_size": int(st.session_state.get("wizard_batch_size", training["batch_size"])),
        "momentum": float(st.session_state.get("wizard_momentum", training["momentum"])),
        "beta1": float(st.session_state.get("wizard_beta1", training["beta1"])),
        "beta2": float(st.session_state.get("wizard_beta2", training["beta2"])),
        "epsilon": float(st.session_state.get("wizard_epsilon", training["epsilon"])),
    })


def _summary_table(config: Mapping[str, Any]) -> list[dict[str, str]]:
    details, data, network, training = (config[key] for key in ("details", "dataset", "network", "training"))
    architecture = "Input (2) → " + (" → ".join(map(str, network["hidden_layer_sizes"])) + " → " if network["hidden_layer_sizes"] else "") + "Output (1)"
    return [
        {"Section": "Experiment", "Setting": "Name", "Value": details["name"]},
        {"Section": "Experiment", "Setting": "Seed", "Value": str(details["seed"])},
        {"Section": "Dataset", "Setting": "Dataset / samples", "Value": f"{data['name']} / {data['samples']}"},
        {"Section": "Dataset", "Setting": "Noise / class 1 share", "Value": f"{data['noise']:.2f} / {data['class_1_fraction']:.0%}"},
        {"Section": "Dataset", "Setting": "Validation split", "Value": f"{data['validation_fraction']:.0%}"},
        {"Section": "Network", "Setting": "Architecture", "Value": architecture},
        {"Section": "Network", "Setting": "Activations / loss", "Value": f"{network['hidden_activation']} / sigmoid / {network['loss']}"},
        {"Section": "Training", "Setting": "Optimizer / learning rate", "Value": f"{training['optimizer']} / {training['learning_rate']:.4g}"},
        {"Section": "Training", "Setting": "Epochs / batch", "Value": f"{training['epochs']} / {training['batch_size']}"},
    ]


def render_experiment_wizard() -> None:
    """Render the persistent, explicitly gated experiment workflow."""
    with st.container(key="experiment_wizard_page"):
        _render_experiment_wizard_content()


def _render_experiment_wizard_content() -> None:
    st.markdown(
        """<section class="hero playground-hero">
            <div class="hero-copy">
                <div class="hero-kicker">EXPERIMENT STUDIO · DESIGN A RUN</div>
                <h1>Experiment Wizard<span class="hero-period">.</span></h1>
                <div class="hero-tagline">Plan carefully. Train deliberately. Learn from the result.</div>
                <p class="hero-description">Configure a reproducible experiment, review every setting, then explicitly start training and save its results.</p>
                <div class="hero-chip"><span class="status-dot"></span>Training starts only after your confirmation</div>
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

    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = new_wizard_state()
    state = st.session_state[STATE_KEY]
    reset_key = "wizard_reset_requested"
    reset_columns = st.columns([0.8, 0.2])
    with reset_columns[1]:
        if st.button("Reset wizard", key="wizard_reset"):
            st.session_state[reset_key] = True
    if st.session_state.get(reset_key):
        st.warning("Reset all wizard settings and remove its unsaved run from this session?")
        confirm, cancel = st.columns(2)
        if confirm.button("Confirm reset", type="primary", key="wizard_reset_confirm"):
            for key in list(st.session_state):
                if key.startswith("wizard_") or key in (STATE_KEY, RESULT_KEY):
                    del st.session_state[key]
            st.rerun()
        if cancel.button("Keep this experiment", key="wizard_reset_cancel"):
            st.session_state[reset_key] = False
            st.rerun()

    step = int(state["step"])
    st.caption(f"Step {step + 1} of {len(STEP_NAMES)} · {STEP_NAMES[step]}")
    st.progress((step + 1) / len(STEP_NAMES), text=f"{STEP_NAMES[step]} · {step + 1}/{len(STEP_NAMES)}")
    st.markdown(
        f'<div class="wizard-step-heading"><span>STEP {step + 1:02d}</span><h2>{STEP_NAMES[step]}</h2>'
        f'<div class="wizard-step-count">{step + 1} / {len(STEP_NAMES)}</div></div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True, key="wizard_step_panel"):
        details, data, network, training = (state[key] for key in ("details", "dataset", "network", "training"))
        if step == 0:
            st.write("Name the run and record enough context to reproduce or explain it later.")
            st.text_input("Experiment name *", value=details["name"], key="wizard_name")
            st.text_area("Short description (optional)", value=details["description"], key="wizard_description")
            note_cols = st.columns(2)
            with note_cols[0]:
                st.text_area("Experiment notes", value=details["notes"], key="wizard_notes")
            with note_cols[1]:
                st.number_input("Random seed", 0, 999_999, int(details["seed"]), 1, key="wizard_seed")
        elif step == 1:
            data_cols = st.columns([0.85, 1.15], gap="large")
            with data_cols[0]:
                dataset_name = st.selectbox("Synthetic dataset", tuple(DATASETS), index=tuple(DATASETS).index(data["name"]), key="wizard_dataset")
                samples = st.slider("Number of samples", 40, 2000, int(data["samples"]), 20, key="wizard_samples")
                noise = st.slider("Noise level", 0.0, 0.5, float(data["noise"]), 0.01, key="wizard_noise")
                st.select_slider("Class 1 share", options=tuple(range(10, 95, 5)), value=int(round(data["class_1_fraction"] * 100)), format_func=lambda value: f"{value}%", key="wizard_class_balance")
                if dataset_name == "Concentric circles":
                    st.slider("Inner-circle radius factor", 0.15, 0.85, float(data["circle_factor"]), 0.05, key="wizard_circle_factor")
                if dataset_name == "Spiral classification":
                    st.slider("Spiral turns", 1.0, 4.0, float(data["spiral_turns"]), 0.25, key="wizard_spiral_turns")
                st.slider("Validation split", 0.1, 0.4, float(data["validation_fraction"]), 0.05, key="wizard_validation")
            with data_cols[1]:
                _collect_widget_values(state)
                _render_dataset_summary(state)
        elif step == 2:
            st.select_slider("Hidden layers", options=tuple(range(0, 6)), value=int(network["hidden_layers"]), key="wizard_layers")
            layer_count = int(st.session_state.get("wizard_layers", network["hidden_layers"]))
            widths = []
            for index in range(layer_count):
                default = network["hidden_layer_sizes"][index] if index < len(network["hidden_layer_sizes"]) else 8
                widths.append(st.number_input(f"Layer {index + 1} neurons", 1, 64, int(default), 1, key=f"wizard_layer_{index}"))
            if layer_count != len(network["hidden_layer_sizes"]):
                network["hidden_layer_sizes"] = [int(value) for value in widths]
                network["hidden_layers"] = layer_count
            st.selectbox("Hidden-layer activation", tuple(ACTIVATIONS), index=tuple(ACTIVATIONS).index(network["hidden_activation"]), key="wizard_activation")
            output_cols = st.columns(2)
            output_cols[0].selectbox("Output activation", ("Sigmoid · binary classification",), disabled=True, key="wizard_output_activation")
            output_cols[1].selectbox("Loss function", ("Binary cross-entropy",), disabled=True, key="wizard_loss")
            st.info("The current training engine supports binary classification with one sigmoid output and binary cross-entropy.")
            st.markdown("**Architecture summary**")
            st.write("Input (2) → " + (" → ".join(map(str, widths)) + " → " if widths else "") + "Output (1)")
        elif step == 3:
            training_cols = st.columns(3)
            with training_cols[0]:
                st.selectbox("Optimizer", OPTIMIZERS, index=OPTIMIZERS.index(training["optimizer"]), key="wizard_optimizer")
                st.number_input("Learning rate", 0.0001, 1.0, float(training["learning_rate"]), 0.001, format="%.4f", key="wizard_learning_rate")
            with training_cols[1]:
                st.number_input("Epochs", 1, 1000, int(training["epochs"]), 10, key="wizard_epochs")
                st.number_input("Batch size", 1, int(data["samples"]), min(int(training["batch_size"]), int(data["samples"])), 1, key="wizard_batch_size")
            with training_cols[2]:
                selected_optimizer = st.session_state.get("wizard_optimizer", training["optimizer"])
                if selected_optimizer == "Momentum":
                    st.slider("Momentum coefficient", 0.0, 0.99, float(training["momentum"]), 0.01, key="wizard_momentum")
                elif selected_optimizer == "Adam":
                    st.number_input("Adam β₁", 0.0, 0.999, float(training["beta1"]), 0.01, format="%.3f", key="wizard_beta1")
                    st.number_input("Adam β₂", 0.0, 0.9999, float(training["beta2"]), 0.001, format="%.4f", key="wizard_beta2")
                    st.number_input("Adam ε", 1e-10, 1e-3, float(training["epsilon"]), format="%.1e", key="wizard_epsilon")
            st.caption("Training and validation are stratified. Standardization is fitted on the training split only.")
        elif step == 4:
            _collect_widget_values(state)
            st.write("Review all settings. Returning to earlier steps keeps your values. Confirming moves to a separate training screen; it does not start training yet.")
            st.dataframe(_summary_table(state), hide_index=True, width="stretch")
            if details["description"]:
                st.markdown(f"**Description:** {details['description']}")
            if details["notes"]:
                st.markdown(f"**Notes:** {details['notes']}")
        elif step == 5:
            if not training_is_authorized(state):
                st.warning("Review and confirm the configuration before training.")
            else:
                result_entry = st.session_state.get(RESULT_KEY)
                if result_entry:
                    st.success(f"Training completed: {result_entry['result'].epochs_completed} epochs. No training runs again during ordinary reruns.")
                    if st.button("Continue to evaluation", type="primary", key="wizard_to_evaluate"):
                        state["step"] = 6
                        st.rerun()
                elif st.button("Start training", type="primary", key="wizard_train"):
                    try:
                        with st.spinner(f"Training {training['optimizer']} for up to {training['epochs']} epochs…"):
                            features, labels = _dataset(state)
                            split = split_classification_data(
                                features, labels, validation_fraction=float(data["validation_fraction"]),
                                random_seed=int(details["seed"]), standardize=True,
                            )
                            optimizer_name = str(training["optimizer"]).lower()
                            result = train_model(
                                split.X_train, split.y_train,
                                hidden_layer_sizes=tuple(int(size) for size in network["hidden_layer_sizes"]),
                                hidden_activation=ACTIVATIONS[network["hidden_activation"]],
                                epochs=int(training["epochs"]), batch_size=int(training["batch_size"]),
                                optimizer=optimizer_name, learning_rate=float(training["learning_rate"]),
                                momentum=float(training["momentum"]), beta1=float(training["beta1"]),
                                beta2=float(training["beta2"]), optimizer_epsilon=float(training["epsilon"]),
                                loss="binary_cross_entropy", validation_data=split.validation_data,
                                random_seed=int(details["seed"]),
                            )
                        st.session_state[RESULT_KEY] = {"result": result, "features": features, "labels": labels, "split": split}
                        state["step"] = 6
                        st.rerun()
                    except (ValueError, FloatingPointError, RuntimeError) as error:
                        st.error(f"Training could not be completed: {error}")
                    except Exception as error:
                        st.error(f"Unexpected training error: {error}")
        elif step == 6:
            entry = st.session_state.get(RESULT_KEY)
            if entry is None:
                state["step"] = 5
                st.warning("Train the configured model before viewing evaluation results.")
            else:
                result: TrainingResult = entry["result"]
                history = result.history
                if not history["epoch"]:
                    st.warning("The model did not complete an epoch, so no evaluation metrics are available.")
                else:
                    last = -1
                    metric_cols = st.columns(4)
                    metric_cols[0].metric("Training loss", f"{history['training_loss'][last]:.5f}")
                    metric_cols[1].metric("Validation loss", f"{history['validation_loss'][last]:.5f}")
                    metric_cols[2].metric("Training accuracy", f"{history['training_accuracy'][last]:.1%}")
                    metric_cols[3].metric("Validation accuracy", f"{history['validation_accuracy'][last]:.1%}")
                    st.caption(f"Completed epochs: {result.epochs_completed} · Binary prediction threshold: 0.5")
                    loss_figure = plot_loss_history(history)
                    st.pyplot(loss_figure, width="stretch")
                    plt.close(loss_figure)
                    split = entry["split"]
                    probabilities = result.network.forward(split.X_validation)[:, 0]
                    predicted = (probabilities >= 0.5).astype(int)
                    actual = np.asarray(split.y_validation).reshape(-1).astype(int)
                    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
                    st.markdown("#### Validation confusion matrix")
                    st.dataframe([
                        {"Actual \\ Predicted": "Class 0", "Class 0": int(matrix[0, 0]), "Class 1": int(matrix[0, 1])},
                        {"Actual \\ Predicted": "Class 1", "Class 0": int(matrix[1, 0]), "Class 1": int(matrix[1, 1])},
                    ], hide_index=True, width="stretch")
                    report = classification_report(actual, predicted, labels=[0, 1], output_dict=True, zero_division=0)
                    st.dataframe([
                        {"Class": key, "Precision": report[key]["precision"], "Recall": report[key]["recall"], "F1": report[key]["f1-score"], "Support": int(report[key]["support"])}
                        for key in ("0", "1")
                    ], hide_index=True, width="stretch")
                    boundary = plot_decision_boundary(
                        result.network, entry["features"], entry["labels"], scaler=split.scaler,
                    )
                    st.pyplot(boundary, width="stretch")
                    plt.close(boundary)
        else:
            entry = st.session_state.get(RESULT_KEY)
            if entry is None:
                state["step"] = 5
                st.warning("Complete training and evaluation before saving this experiment.")
            elif state.get("saved_id") is not None:
                st.success(f"This experiment is already saved as record #{state['saved_id']}; revisiting this step will not create another record.")
                if st.button("Open Experiments & Reports", key="wizard_open_reports"):
                    st.session_state["navigation_selection"] = "Experiments & Reports"
                    st.rerun()
            else:
                st.write("Save the completed configuration and recorded epoch history to the existing local experiment store.")
                if st.button("Save completed experiment", type="primary", key="wizard_save"):
                    try:
                        notes_parts = []
                        if details["description"].strip():
                            notes_parts.append("Description: " + details["description"].strip())
                        if details["notes"].strip():
                            notes_parts.append("Notes: " + details["notes"].strip())
                        data_configuration = {
                            **data, "data_seed": int(details["seed"]),
                            "split_seed": int(details["seed"]), "standardize": True,
                        }
                        training_configuration = {
                            **network, **training,
                            "hidden_layer_sizes": list(network["hidden_layer_sizes"]),
                            "hidden_activation": ACTIVATIONS[network["hidden_activation"]],
                            "optimizer": training["optimizer"], "loss": "binary_cross_entropy",
                            "output_activation": "sigmoid", "random_seed": int(details["seed"]),
                            "epochs_completed": entry["result"].epochs_completed,
                        }
                        record_id = ExperimentStore().save(
                            name=details["name"], dataset=data["name"],
                            dataset_configuration=data_configuration,
                            configuration=training_configuration,
                            history=entry["result"].history, notes="\n\n".join(notes_parts),
                        )
                        state["saved_id"] = record_id
                        st.success(f"Saved experiment #{record_id}.")
                    except (TypeError, ValueError, OSError) as error:
                        st.error(f"Experiment could not be saved: {error}")

    _collect_widget_values(state)
    st.divider()
    back_col, nav_col = st.columns([1, 1])
    if step > 0 and back_col.button("← Back", key=f"wizard_back_{step}"):
        wizard_previous(state)
        st.rerun()
    if step < 4 and nav_col.button("Next →", type="primary", key=f"wizard_next_{step}"):
        errors = wizard_next(state, state)
        if errors:
            for error in errors:
                st.error(error)
        else:
            st.rerun()
    elif step == 4 and nav_col.button("Confirm configuration", type="primary", key="wizard_confirm"):
        errors = validate_wizard_step(1, state) + validate_wizard_step(2, state) + validate_wizard_step(3, state) + validate_wizard_step(4, state)
        if errors:
            for error in errors:
                st.error(error)
        else:
            state["confirmed"] = True
            state["step"] = 5
            # A newly confirmed configuration is a new run and may be saved separately.
            st.session_state.pop(RESULT_KEY, None)
            state["saved_id"] = None
            st.rerun()
    elif step == 5 and st.session_state.get(RESULT_KEY) and nav_col.button("Continue to evaluation →", type="primary", key="wizard_eval_bottom"):
        state["step"] = 6
        st.rerun()
    elif step == 6 and st.session_state.get(RESULT_KEY) and nav_col.button("Continue to save →", type="primary", key="wizard_save_bottom"):
        state["step"] = 7
        st.rerun()
    elif step == 7 and st.session_state.get(RESULT_KEY) and state.get("saved_id") is not None and nav_col.button("Finish", type="primary", key="wizard_finish"):
        st.session_state["navigation_selection"] = "Experiments & Reports"
        st.rerun()
