"""Interactive, staged inspection of the existing NumPy backpropagation engine."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .models import Adam, NeuralNetwork, SGD, SGDMomentum
from .models.losses import binary_cross_entropy
from .visualization import BLUE, CYAN, GRID, MUTED, NAVY, TEXT, VIOLET


ACTIVATIONS = {"ReLU": "relu", "Leaky ReLU": "leaky_relu", "Sigmoid": "sigmoid", "Tanh": "tanh"}
OPTIMIZERS = ("SGD", "Momentum", "Adam")
STATE_KEY = "backpropagation_explorer_state"
EXAMPLE_INPUTS = np.array(
    [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]],
    dtype=float,
)
EXAMPLE_TARGETS = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=int)


def plot_backpropagation_network(network: NeuralNetwork, inputs: np.ndarray) -> plt.Figure:
    """Draw neurons and signed, weight-scaled connections from the live model."""
    if not network.activations:
        raise ValueError("Run forward propagation before drawing live neuron values.")
    widths = network.layer_sizes
    largest = max(widths)
    figure, axis = plt.subplots(figsize=(max(7, len(widths) * 2.1), max(4.2, largest * 0.72)), facecolor=NAVY)
    axis.set_facecolor(NAVY)
    x_positions = np.linspace(0.08, 0.92, len(widths))
    positions: list[list[tuple[float, float]]] = []
    for x_pos, width in zip(x_positions, widths):
        ys = np.linspace(0.15, 0.85, width) if width > 1 else np.array([0.5])
        positions.append([(float(x_pos), float(y)) for y in ys])
    max_weight = max(float(np.max(np.abs(weight))) for weight in network.weights)
    for layer, weights in enumerate(network.weights):
        for target_index, (x2, y2) in enumerate(positions[layer + 1]):
            for source_index, (x1, y1) in enumerate(positions[layer]):
                weight = float(weights[target_index, source_index])
                positive = weight >= 0
                axis.plot(
                    [x1, x2], [y1, y2], color=CYAN if positive else VIOLET,
                    linewidth=0.45 + 1.35 * abs(weight) / max(max_weight, 1e-12),
                    alpha=0.65, zorder=1,
                )
    for layer_index, layer_positions in enumerate(positions):
        values = inputs if layer_index == 0 else network.activations[layer_index].reshape(-1)
        color = BLUE if layer_index == 0 else CYAN if layer_index == len(widths) - 1 else VIOLET
        for neuron_index, (x_pos, y_pos) in enumerate(layer_positions):
            axis.scatter([x_pos], [y_pos], s=620, color=color, edgecolors=TEXT, linewidths=0.9, zorder=3)
            axis.text(x_pos, y_pos, f"{values[neuron_index]:.3f}", color=("#0b1020" if color == VIOLET else "#edf3ff"), fontsize=8,
                      weight="bold", ha="center", va="center", zorder=4)
            axis.text(x_pos, y_pos - 0.095, f"{['Input', 'Hidden', 'Output'][0 if layer_index == 0 else 2 if layer_index == len(widths)-1 else 1]} {neuron_index + 1}",
                      color=TEXT, fontsize=7, ha="center", va="top")
    for index, (x_pos, width) in enumerate(zip(x_positions, widths)):
        label = "Input" if index == 0 else "Output · sigmoid" if index == len(widths) - 1 else f"Hidden · {network.hidden_activation}"
        axis.text(x_pos, 0.99, f"{label}\n{width} neuron{'s' if width != 1 else ''}",
                  color=TEXT, fontsize=8, ha="center", va="top")
    axis.text(0.03, 0.02, "+ cyan = positive weight", color=CYAN, fontsize=8, ha="left")
    axis.text(0.53, 0.02, "− violet = negative weight", color=VIOLET, fontsize=8, ha="left")
    axis.set_xlim(0, 1)
    axis.set_ylim(-0.08, 1.13)
    axis.axis("off")
    figure.tight_layout()
    return figure


def _optimizer(name: str, learning_rate: float, momentum: float, beta1: float, beta2: float, epsilon: float):
    if name == "SGD":
        return SGD(learning_rate)
    if name == "Momentum":
        return SGDMomentum(learning_rate, momentum=momentum)
    return Adam(learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)


def _copy_parameters(network: NeuralNetwork) -> tuple[list[np.ndarray], list[np.ndarray]]:
    return ([value.copy() for value in network.weights], [value.copy() for value in network.biases])


def _parameter_table(network: NeuralNetwork, gradients: dict[str, object] | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for layer_index, (weights, biases) in enumerate(zip(network.weights, network.biases), start=1):
        weight_gradients = gradients["weight_gradients"][layer_index - 1] if gradients else None
        bias_gradients = gradients["bias_gradients"][layer_index - 1] if gradients else None
        for neuron_index, bias in enumerate(biases[:, 0]):
            rows.append({
                "Layer": str(layer_index), "Parameter": f"b{layer_index}[{neuron_index + 1}]",
                "Value": f"{bias:+.6f}",
                "Gradient": f"{bias_gradients[neuron_index, 0]:+.6f}" if bias_gradients is not None else "—",
            })
            for input_index, weight in enumerate(weights[neuron_index]):
                rows.append({
                    "Layer": str(layer_index), "Parameter": f"W{layer_index}[{neuron_index + 1},{input_index + 1}]",
                    "Value": f"{weight:+.6f}",
                    "Gradient": f"{weight_gradients[neuron_index, input_index]:+.6f}" if weight_gradients is not None else "—",
                })
    return rows


def _render_forward_values(network: NeuralNetwork, target: float, loss: float | None) -> None:
    st.markdown("#### Weighted sums and activations")
    rows = [{"Layer": "Input", "Neuron": str(index + 1), "Input / weighted sum": f"{value:.6f}", "Activation output": f"{value:.6f}"}
            for index, value in enumerate(network.activations[0][:, 0])]
    for layer_index, (z_values, activations) in enumerate(zip(network.weighted_sums, network.activations[1:]), start=1):
        layer_name = "Output · sigmoid" if layer_index == len(network.weights) else f"Layer {layer_index} · {network.hidden_activation}"
        rows.extend({
            "Layer": layer_name, "Neuron": str(neuron_index + 1),
            "Input / weighted sum": f"{z_values[neuron_index, 0]:.6f}",
            "Activation output": f"{activations[neuron_index, 0]:.6f}",
        } for neuron_index in range(z_values.shape[0]))
    st.table(rows)
    prediction = float(network.activations[-1][0, 0])
    metrics = st.columns(3)
    metrics[0].metric("Prediction", f"{prediction:.6f}")
    metrics[1].metric("Target", f"{target:.1f}")
    metrics[2].metric("Binary cross-entropy", f"{loss:.6f}" if loss is not None else "Calculate next")


def _render_gradient_details(gradients: dict[str, object]) -> None:
    st.markdown("#### Chain rule values · sample-major shapes")
    st.caption("Each row is one sample and each column is one neuron. For this single-sample explorer the first dimension is 1.")
    for index, (dlda, derivative, dz) in enumerate(zip(
        gradients["activation_gradients"], gradients["activation_derivatives"], gradients["preactivation_gradients"]
    ), start=1):
        label = "Output" if index == len(gradients["preactivation_gradients"]) else f"Hidden layer {index}"
        with st.expander(f"{label} · dL/dA, f′(Z), dL/dZ", expanded=True):
            st.write(f"**dL/dA** · shape `{dlda.shape}`")
            st.code(np.array2string(dlda, precision=6, floatmode="fixed"))
            st.write(f"**Activation derivative f′(Z)** · shape `{derivative.shape}`")
            st.code(np.array2string(derivative, precision=6, floatmode="fixed"))
            st.write(f"**dL/dZ = dL/dA ⊙ f′(Z)** · shape `{dz.shape}`")
            st.code(np.array2string(dz, precision=6, floatmode="fixed"))
    st.markdown("#### Parameter gradients")
    st.caption("For each layer: dW = dZ @ A_previousᵀ and db = sum(dZ over samples). The loss derivative already includes averaging.")
    st.table(_parameter_table(st.session_state[STATE_KEY]["network"], gradients))
    figure, axis = plt.subplots(figsize=(7.5, 3.2), facecolor=NAVY)
    axis.set_facecolor(NAVY)
    layer_labels = [f"Layer {index + 1}" for index in range(len(gradients["weight_norms"]))]
    x = np.arange(len(layer_labels))
    axis.bar(x - 0.18, gradients["weight_norms"], 0.36, color=BLUE, label="||dW||₂")
    axis.bar(x + 0.18, gradients["bias_norms"], 0.36, color=VIOLET, label="||db||₂")
    axis.set_xticks(x, layer_labels, color=MUTED)
    axis.set_ylabel("Gradient L2 norm", color=MUTED)
    axis.tick_params(colors=MUTED)
    axis.set_title("Gradient magnitude by layer", color=TEXT, loc="left")
    for spine in axis.spines.values():
        spine.set_color(GRID)
    axis.grid(axis="y", color=GRID, alpha=0.55)
    axis.legend(facecolor="#111a2c", edgecolor=GRID, labelcolor=TEXT)
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def render_backpropagation_explorer() -> None:
    """Render the step-by-step forward, gradient, and parameter update lesson."""
    st.markdown('<div class="eyebrow">FOLLOW THE CHAIN RULE THROUGH A NETWORK</div>', unsafe_allow_html=True)
    st.title("Backpropagation Explorer")
    st.markdown("Inspect one real forward and backward pass, then explicitly apply a single optimizer update.")

    st.markdown("### 1 · Network configuration")
    columns = st.columns(5, gap="medium")
    with columns[0]:
        hidden_count = st.selectbox("Hidden layers", (1, 2), key="backprop_hidden_count")
    sizes = []
    for index in range(hidden_count):
        sizes.append(st.number_input(f"Layer {index + 1} neurons", 1, 6, 3, 1, key=f"backprop_width_{index}"))
    with columns[1]:
        activation_label = st.selectbox("Hidden activation", tuple(ACTIVATIONS), key="backprop_activation")
    with columns[2]:
        seed = int(st.number_input("Initialization seed", 0, 999_999, 13, 1, key="backprop_seed"))
    with columns[3]:
        optimizer_name = st.selectbox("Update optimizer", OPTIMIZERS, key="backprop_optimizer")
    with columns[4]:
        learning_rate = st.number_input("Learning rate", 0.0001, 1.0, 0.05, 0.001, format="%.4f", key="backprop_lr")

    momentum, beta1, beta2, epsilon = 0.9, 0.9, 0.999, 1e-8
    if optimizer_name == "Momentum":
        momentum = st.slider("Momentum coefficient", 0.0, 0.99, 0.9, 0.01, key="backprop_momentum")
    elif optimizer_name == "Adam":
        params = st.columns(3)
        beta1 = params[0].number_input("Adam β₁", 0.0, 0.999, 0.9, 0.01, format="%.3f", key="backprop_beta1")
        beta2 = params[1].number_input("Adam β₂", 0.0, 0.9999, 0.999, 0.001, format="%.4f", key="backprop_beta2")
        epsilon = params[2].number_input("Adam ε", 1e-10, 1e-3, 1e-8, format="%.1e", key="backprop_epsilon")

    st.markdown("### 2 · Input and target")
    sample_idx = st.selectbox(
        "Deterministic XOR example row",
        range(len(EXAMPLE_INPUTS)),
        format_func=lambda index: f"Row {index + 1}: ({EXAMPLE_INPUTS[index, 0]:+.1f}, {EXAMPLE_INPUTS[index, 1]:+.1f}) → {EXAMPLE_TARGETS[index]}",
        key="backprop_example_row",
    )
    sample = EXAMPLE_INPUTS[sample_idx]
    input_columns = st.columns(3)
    x1 = input_columns[0].number_input("Input x₁", value=float(sample[0]), step=0.1, format="%.3f", key="backprop_x1")
    x2 = input_columns[1].number_input("Input x₂", value=float(sample[1]), step=0.1, format="%.3f", key="backprop_x2")
    target = input_columns[2].number_input("Binary target y", min_value=0.0, max_value=1.0,
                                            value=float(EXAMPLE_TARGETS[sample_idx]), step=1.0, key="backprop_target")
    st.caption("The sample selector provides a small deterministic XOR set. The feature and target fields can be edited directly.")

    init_clicked = st.button("Initialize Network", type="primary", key="backprop_initialize")
    if init_clicked:
        try:
            network = NeuralNetwork(2, tuple(int(size) for size in sizes), 1, random_seed=seed,
                                    hidden_activation=ACTIVATIONS[activation_label])
            st.session_state[STATE_KEY] = {
                "network": network,
                "optimizer": _optimizer(optimizer_name, float(learning_rate), float(momentum), float(beta1), float(beta2), float(epsilon)),
                "config": {"hidden_layer_sizes": tuple(int(size) for size in sizes), "activation": activation_label,
                           "seed": seed, "optimizer": optimizer_name, "learning_rate": float(learning_rate)},
                "stage": "initialized",
                "gradients": None,
                "loss_value": None,
                "before_update": None,
                "after_update": None,
            }
            st.success("Network initialized with deterministic parameters. No parameters have been updated.")
        except (ValueError, FloatingPointError) as error:
            st.error(f"Network could not be initialized: {error}")

    state = st.session_state.get(STATE_KEY)
    if state is None:
        st.info("Configure the network and sample, then initialize it to begin the walkthrough.")
        return

    st.caption(f"Active network · {state['config']['hidden_layer_sizes']} hidden widths · {state['config']['activation']} · seed {state['config']['seed']} · {state['config']['optimizer']}")
    network: NeuralNetwork = state["network"]
    st.markdown("### 3 · Forward propagation and loss")
    st.latex(r"Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]},\qquad A^{[l]} = f(Z^{[l]})")
    forward_label = "Run Forward Propagation Again" if state["stage"] == "updated" else "Run Forward Propagation"
    if st.button(forward_label, key="backprop_forward"):
        if state["stage"] == "updated" and state.get("sample_inputs") is not None:
            values = state["sample_inputs"].copy()
            observed_target = float(state["sample_target"])
        else:
            values = np.array([x1, x2], dtype=float)
            observed_target = float(target)
        try:
            prediction = network.forward(values)
            state.update({"stage": "forward", "sample_inputs": values, "sample_target": observed_target,
                          "prediction_before_update": float(prediction[0, 0]), "loss_value": None,
                          "gradients": None})
        except (ValueError, FloatingPointError, RuntimeError) as error:
            st.error(f"Forward propagation could not be completed: {error}")

    if state["stage"] in ("forward", "loss", "gradients", "updated_forward") and network.activations:
        _render_forward_values(network, float(state["sample_target"]), state.get("loss_value"))
        _show = plot_backpropagation_network(network, state["sample_inputs"])
        st.pyplot(_show, width="stretch")
        plt.close(_show)
        st.markdown("#### Weights and biases")
        st.caption("The sign is printed explicitly. In the diagram, cyan edges are positive and violet edges are negative; thicker edges have larger absolute weights.")
        st.table(_parameter_table(network))

    st.markdown("### 4 · Calculate loss")
    loss_ready = state["stage"] == "forward" and network.activations
    if st.button("Calculate Loss", disabled=not loss_ready, key="backprop_loss"):
        try:
            state["loss_value"] = binary_cross_entropy(
                np.array([[state["sample_target"]]]), network.activations[-1].T
            )
            state["loss_before_update"] = state["loss_value"]
            state["stage"] = "loss"
        except (ValueError, FloatingPointError) as error:
            st.error(f"Loss could not be calculated: {error}")

    st.markdown("### 5 · Backpropagation")
    st.markdown("The chain rule moves the loss signal backward. Each layer multiplies the signal from the next layer by its weights, then by the local activation slope.")
    st.latex(r"\frac{\partial L}{\partial A^{[l]}} = (W^{[l+1]})^T\frac{\partial L}{\partial Z^{[l+1]}},\quad "
             r"\frac{\partial L}{\partial Z^{[l]}} = \frac{\partial L}{\partial A^{[l]}}\odot f'(Z^{[l]})")
    st.latex(r"\frac{\partial L}{\partial W^{[l]}} = \frac{\partial L}{\partial Z^{[l]}}(A^{[l-1]})^T,\quad "
             r"\frac{\partial L}{\partial b^{[l]}} = \sum_{samples}\frac{\partial L}{\partial Z^{[l]}}")
    st.caption("For binary cross-entropy with a sigmoid output, the engine uses the equivalent simplified output gradient dL/dZ = (prediction − target) / number of output elements.")
    gradient_ready = state["stage"] == "loss" and network.activations
    if st.button("Calculate Gradients", disabled=not gradient_ready, key="backprop_gradients"):
        try:
            # backward computes and stores gradients only. It does not update parameters.
            state["gradients"] = network.backward(np.array([state["sample_target"]]), loss="binary_cross_entropy")
            state["stage"] = "gradients"
        except (ValueError, FloatingPointError, RuntimeError) as error:
            st.error(f"Gradients could not be calculated: {error}")
    if state["stage"] in ("gradients", "updated", "updated_forward") and state.get("gradients") is not None:
        if state["stage"] in ("updated", "updated_forward"):
            st.caption("These are the gradients from the pre-update weights. They are retained for comparison and were not recomputed after the update.")
        _render_gradient_details(state["gradients"])

    st.markdown("### 6 · Apply one optimizer update")
    update_ready = state["stage"] == "gradients" and state.get("gradients") is not None
    if st.button("Apply One Update", disabled=not update_ready, key="backprop_update"):
        try:
            state["before_update"] = {
                "weights": [value.copy() for value in network.weights],
                "biases": [value.copy() for value in network.biases],
                "prediction": state["prediction_before_update"],
                "loss": state["loss_before_update"],
            }
            network.apply_gradients(state["optimizer"], state["gradients"])
            state["after_update"] = _copy_parameters(network)
            state["stage"] = "updated"
            st.success("One explicit optimizer update was applied. Run forward propagation again to evaluate the updated network.")
        except (ValueError, FloatingPointError, RuntimeError, TypeError) as error:
            st.error(f"Parameter update could not be applied: {error}")

    if state.get("before_update") is not None:
        st.markdown("### 7 · Before-and-after parameters")
        before = state["before_update"]
        after = state.get("after_update")
        if after is not None:
            rows = []
            for layer_index, (old_w, new_w, old_b, new_b) in enumerate(zip(
                before["weights"], after[0], before["biases"], after[1]
            ), start=1):
                for neuron_index in range(old_b.shape[0]):
                    rows.append({"Layer": str(layer_index), "Parameter": f"b{layer_index}[{neuron_index+1}]",
                                 "Before": f"{old_b[neuron_index, 0]:+.6f}", "After": f"{new_b[neuron_index, 0]:+.6f}"})
                    for input_index in range(old_w.shape[1]):
                        rows.append({"Layer": str(layer_index), "Parameter": f"W{layer_index}[{neuron_index+1},{input_index+1}]",
                                     "Before": f"{old_w[neuron_index, input_index]:+.6f}",
                                     "After": f"{new_w[neuron_index, input_index]:+.6f}"})
            st.table(rows)
            st.markdown("### 8 · Re-evaluate after the update")
            if state["stage"] != "updated_forward" and st.button("Evaluate Updated Network", key="backprop_evaluate_after"):
                try:
                    prediction = network.forward(state["sample_inputs"])
                    loss = binary_cross_entropy(np.array([[state["sample_target"]]]), prediction)
                    state.update({"stage": "updated_forward", "prediction_after_update": float(prediction[0, 0]),
                                  "loss_after_update": loss})
                except (ValueError, FloatingPointError, RuntimeError) as error:
                    st.error(f"Updated network evaluation failed: {error}")
            if state["stage"] == "updated_forward":
                metric_columns = st.columns(2)
                metric_columns[0].metric("Prediction after update", f"{state['prediction_after_update']:.6f}",
                                         delta=f"{state['prediction_after_update'] - before['prediction']:+.6f}")
                metric_columns[1].metric("Loss after update", f"{state['loss_after_update']:.6f}",
                                         delta=f"{state['loss_after_update'] - before['loss']:+.6f}", delta_color="inverse")
