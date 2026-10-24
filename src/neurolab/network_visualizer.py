"""Interactive architecture and forward-pass inspection for NeuroLab networks."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from .models import NeuralNetwork
from .visualization import BLUE, CYAN, GRID, MUTED, NAVY, TEXT, VIOLET


NEW_NETWORK_KEY = "network_visualizer_new_network"
TRACE_KEY = "network_visualizer_forward_trace"
MAX_DIAGRAM_NEURONS = 8
ACTIVATIONS = {"ReLU": "relu", "Leaky ReLU": "leaky_relu", "Sigmoid": "sigmoid", "Tanh": "tanh"}


def architecture_summary(network: NeuralNetwork) -> list[dict[str, object]]:
    """Describe every layer using the live network's architecture and parameters."""
    layer_count = len(network.layer_sizes)
    result: list[dict[str, object]] = []
    for index, neuron_count in enumerate(network.layer_sizes):
        if index == 0:
            result.append({
                "index": index, "type": "Input", "neurons": neuron_count,
                "activation": "Input values", "weight_shape": None, "bias_shape": None,
            })
        else:
            is_output = index == layer_count - 1
            result.append({
                "index": index,
                "type": "Output" if is_output else "Hidden",
                "neurons": neuron_count,
                "activation": "Sigmoid" if is_output else network.hidden_activation,
                "weight_shape": tuple(network.weights[index - 1].shape),
                "bias_shape": tuple(network.biases[index - 1].shape),
            })
    return result


def inspect_forward_pass(network: NeuralNetwork, sample: np.ndarray) -> dict[str, object]:
    """Run one actual sample and copy its input, Z values, activations, and output."""
    values = np.asarray(sample, dtype=float)
    if values.ndim == 2:
        if values.shape[0] != 1:
            raise ValueError("Forward inspection accepts exactly one sample at a time.")
        values = values[0]
    if values.ndim != 1 or values.shape[0] != network.input_feature_count:
        raise ValueError(
            f"sample must have shape ({network.input_feature_count},), got {values.shape}."
        )
    predictions = network.forward(values)
    if not network.activations or len(network.weighted_sums) != len(network.weights):
        raise RuntimeError("The network did not expose a complete forward-pass cache.")
    return {
        "input": network.activations[0][:, 0].copy(),
        "weighted_sums": tuple(value[:, 0].copy() for value in network.weighted_sums),
        "activations": tuple(value[:, 0].copy() for value in network.activations[1:]),
        "prediction": predictions[0].copy(),
    }


def plot_network_diagram(
    network: NeuralNetwork,
    *,
    trace: dict[str, object] | None = None,
    highlighted_layer: int | None = None,
    max_neurons: int = MAX_DIAGRAM_NEURONS,
) -> tuple[plt.Figure, tuple[tuple[int, int], ...]]:
    """Draw live weights as signed connections, capping visible nodes per layer."""
    if isinstance(max_neurons, bool) or not isinstance(max_neurons, int) or max_neurons < 1:
        raise ValueError("max_neurons must be a positive integer.")
    widths = network.layer_sizes
    shown = tuple(min(width, max_neurons) for width in widths)
    simplified = tuple((index, width) for index, (width, visible) in enumerate(zip(widths, shown)) if width > visible)
    figure, axis = plt.subplots(
        figsize=(max(7.5, len(widths) * 2.2), max(4.5, max(shown) * 0.72)),
        facecolor=NAVY,
    )
    axis.set_facecolor(NAVY)
    x_positions = np.linspace(0.08, 0.92, len(widths))
    positions: list[list[tuple[float, float]]] = []
    for x_pos, count in zip(x_positions, shown):
        ys = np.linspace(0.18, 0.82, count) if count > 1 else np.array([0.5])
        positions.append([(float(x_pos), float(y)) for y in ys])

    all_weights = [np.abs(weight[:shown[i + 1], :shown[i]]) for i, weight in enumerate(network.weights)]
    max_weight = max((float(np.max(value)) for value in all_weights if value.size), default=1.0)
    max_weight = max(max_weight, 1e-12)
    for layer_index, weights in enumerate(network.weights):
        visible_weights = weights[:shown[layer_index + 1], :shown[layer_index]]
        for target_index, (x2, y2) in enumerate(positions[layer_index + 1]):
            for source_index, (x1, y1) in enumerate(positions[layer_index]):
                weight = float(visible_weights[target_index, source_index])
                positive = weight >= 0
                axis.plot(
                    [x1, x2], [y1, y2],
                    color=CYAN if positive else VIOLET,
                    linestyle="-" if positive else "--",
                    linewidth=0.55 + 1.8 * abs(weight) / max_weight,
                    alpha=0.72, zorder=1,
                )

    trace_activations: Sequence[np.ndarray] | None = None
    if trace is not None:
        trace_activations = trace["activations"]
    for layer_index, layer_positions in enumerate(positions):
        layer_values = None
        if trace is not None:
            layer_values = trace["input"] if layer_index == 0 else trace_activations[layer_index - 1]
        is_highlighted = layer_index == highlighted_layer
        for neuron_index, (x_pos, y_pos) in enumerate(layer_positions):
            if is_highlighted:
                face, text = BLUE, "#ffffff"
            elif layer_index == 0:
                face, text = "#eee8ef", TEXT
            else:
                face, text = "#f4f5f8", TEXT
            axis.add_patch(Circle(
                (x_pos, y_pos), 0.037, facecolor=face,
                edgecolor=BLUE if is_highlighted else CYAN if layer_index == len(widths) - 1 else VIOLET,
                linewidth=2.1 if is_highlighted else 1.3, zorder=3,
            ))
            label = f"{layer_values[neuron_index]:.2f}" if layer_values is not None else f"{neuron_index + 1}"
            axis.text(x_pos, y_pos, label, color=text, fontsize=7.5, weight="bold",
                      ha="center", va="center", zorder=4)

    layer_rows = architecture_summary(network)
    for index, (x_pos, count, layer) in enumerate(zip(x_positions, shown, layer_rows)):
        label = f"L{index} · {layer['type']}\n{layer['activation']}"
        axis.text(x_pos, 0.04, label, color=TEXT, fontsize=8, ha="center", va="top")
        axis.text(x_pos, 0.92, f"{count}/{layer['neurons']} neurons",
                  color=MUTED, fontsize=7, ha="center", va="bottom")
    legend = (
        Line2D([0], [0], color=CYAN, linewidth=2, linestyle="-", label="Positive weight"),
        Line2D([0], [0], color=VIOLET, linewidth=2, linestyle="--", label="Negative weight"),
    )
    axis.legend(handles=legend, loc="upper center", ncol=2, frameon=False,
                labelcolor=TEXT, fontsize=8, bbox_to_anchor=(0.5, 1.08))
    axis.set_xlim(0, 1)
    axis.set_ylim(-0.02, 1.02)
    axis.axis("off")
    figure.tight_layout()
    return figure, simplified


def _trained_network_options() -> dict[str, dict[str, object]]:
    """Collect existing model objects and their matching sample/scaler context."""
    options: dict[str, dict[str, object]] = {}
    saved = st.session_state.get("network_playground_result")
    if saved:
        result = saved["result"]
        options[f"Network Playground · run {saved['run_number']} · {saved['dataset_name']}"] = {
            "network": result.network, "trained": True,
            "features": saved.get("features"), "labels": saved.get("labels"),
            "scaler": saved.get("scaler"), "description": "Trained in Network Playground",
        }
    optimizer_saved = st.session_state.get("optimizer_lab_result")
    if optimizer_saved:
        comparison = optimizer_saved["comparison"]
        for name, result in comparison.runs.items():
            options[f"Optimizer Lab · {name} · run {optimizer_saved['run_number']}"] = {
                "network": result.network, "trained": True,
                "features": comparison.features, "labels": comparison.labels,
                "scaler": comparison.split.scaler, "description": f"Trained with {name}",
            }
    backprop_saved = st.session_state.get("backpropagation_explorer_state")
    if backprop_saved and isinstance(backprop_saved.get("network"), NeuralNetwork):
        options["Backpropagation Explorer · current model"] = {
            "network": backprop_saved["network"], "trained": backprop_saved.get("stage") == "updated",
            "features": None, "labels": None, "scaler": None,
            "description": f"Backpropagation Explorer · {backprop_saved.get('stage', 'initialized')}",
        }
    return options


def _render_new_model() -> NeuralNetwork | None:
    """Render a compact initializer for an untrained inspectable network."""
    columns = st.columns(4, gap="medium")
    with columns[0]:
        input_count = st.number_input("Input features", 1, 8, 2, 1, key="visualizer_input_count")
    with columns[1]:
        layer_count = st.number_input("Hidden layers", 0, 5, 2, 1, key="visualizer_layer_count")
    with columns[2]:
        activation_label = st.selectbox("Hidden activation", tuple(ACTIVATIONS), key="visualizer_activation")
    with columns[3]:
        seed = st.number_input("Initialization seed", 0, 999_999, 17, 1, key="visualizer_seed")
    widths = []
    for index in range(int(layer_count)):
        widths.append(st.number_input(
            f"Hidden layer {index + 1} neurons", min_value=1, max_value=32, value=6, step=1,
            key=f"visualizer_width_{index}",
        ))
    output_count = st.number_input("Output neurons", 1, 8, 1, 1, key="visualizer_output_count")
    st.caption("This model has reproducible random parameters and is untrained. Its output is a forward-pass demonstration, not an accuracy result.")
    if st.button("Initialize Network", type="primary", key="visualizer_initialize"):
        try:
            network = NeuralNetwork(
                int(input_count), tuple(int(width) for width in widths), int(output_count),
                random_seed=int(seed), hidden_activation=ACTIVATIONS[activation_label],
            )
            st.session_state[NEW_NETWORK_KEY] = network
            st.session_state[TRACE_KEY] = None
            st.success("Initialized an untrained network with the selected architecture.")
            return network
        except ValueError as error:
            st.error(f"Network could not be initialized: {error}")
    return st.session_state.get(NEW_NETWORK_KEY)


def render_network_visualizer() -> None:
    """Render architecture, real parameter, and cached forward-pass inspection."""
    st.markdown('<div class="eyebrow">INSPECT NETWORK STRUCTURE AND INFORMATION FLOW</div>', unsafe_allow_html=True)
    st.title("Network Visualizer")
    st.markdown("Inspect a configured network, run one sample through its actual parameters, and follow each layer's values.")

    models = _trained_network_options()
    source_labels = ("New initialized network", *models.keys())
    source_columns = st.columns([1.0, 2.0], gap="large")
    with source_columns[0]:
        source = st.selectbox("Model source", source_labels, key="visualizer_model_source")
    model_info = models.get(source)

    if model_info is None:
        network = _render_new_model()
        trained = False
        source_description = "Untrained · newly initialized parameters"
        features = labels = scaler = None
    else:
        network = model_info["network"]
        trained = bool(model_info["trained"])
        source_description = str(model_info["description"])
        features, labels, scaler = model_info["features"], model_info["labels"], model_info["scaler"]
        with source_columns[1]:
            st.info(f"Using the existing model from {source_description}. No training is triggered here.")

    if network is None:
        st.info("Choose an architecture and select **Initialize Network** to inspect its real parameters.")
        return

    summary = architecture_summary(network)
    with source_columns[1]:
        st.caption(("Trained model" if trained else "Untrained model") + f" · {network.layer_sizes} · {network.hidden_activation} hidden · sigmoid output")
    st.markdown("### 1 · Architecture overview")
    st.dataframe(
        [
            {
                "Layer": f"L{layer['index']} · {layer['type']}",
                "Neurons": layer["neurons"],
                "Activation": layer["activation"],
                "Weight shape": str(layer["weight_shape"]) if layer["weight_shape"] else "—",
                "Bias shape": str(layer["bias_shape"]) if layer["bias_shape"] else "—",
            }
            for layer in summary
        ],
        hide_index=True,
        width="stretch",
    )

    st.markdown("### 2 · Network diagram")
    if network.activations and len(network.weighted_sums) == len(network.weights):
        latest = {
            "input": network.activations[0][:, 0].copy(),
            "weighted_sums": tuple(value[:, 0].copy() for value in network.weighted_sums),
            "activations": tuple(value[:, 0].copy() for value in network.activations[1:]),
            "prediction": network.activations[-1][:, 0].copy(),
        }
    else:
        latest = None
    step_options = tuple(range(len(network.layer_sizes)))
    if st.session_state.get("visualizer_step") not in step_options:
        st.session_state["visualizer_step"] = step_options[0]
    step = st.select_slider(
        "Forward-pass step",
        options=step_options,
        value=0,
        format_func=lambda index: "Input" if index == 0 else f"Layer {index} · {'Output' if index == len(network.layer_sizes) - 1 else 'Hidden'}",
        key="visualizer_step",
    )
    figure, simplified = plot_network_diagram(
        network, trace=latest, highlighted_layer=step, max_neurons=MAX_DIAGRAM_NEURONS
    )
    st.pyplot(figure, width="stretch")
    plt.close(figure)
    if simplified:
        st.caption(
            f"Diagram simplified for readability: showing at most {MAX_DIAGRAM_NEURONS} neurons in "
            + ", ".join(f"L{index} ({MAX_DIAGRAM_NEURONS}/{width})" for index, width in simplified)
            + ". All listed parameter values remain actual."
        )
    st.caption("Solid cyan-plum connections are positive; dashed violet connections are negative. Line width reflects absolute weight magnitude.")

    st.markdown("### 3 · Input and forward propagation")
    sample_columns = st.columns([1.1, 1.0], gap="large")
    with sample_columns[0]:
        use_data = features is not None and len(features) > 0
        input_mode = st.radio(
            "Input source",
            ("Enter a sample", "Choose a dataset sample") if use_data else ("Enter a sample",),
            horizontal=True,
            key="visualizer_input_mode",
        )
        if input_mode == "Choose a dataset sample" and use_data:
            row_index = st.number_input("Dataset row", 1, len(features), 1, 1, key="visualizer_dataset_row") - 1
            raw_sample = np.asarray(features[int(row_index)], dtype=float)
            sample = scaler.transform(raw_sample.reshape(1, -1))[0] if scaler is not None else raw_sample
            st.caption(f"Actual dataset row {int(row_index) + 1}; model input is transformed with the saved training scaler when present.")
            st.code(f"Raw features: {np.array2string(raw_sample, precision=4)}\nModel input:  {np.array2string(sample, precision=4)}")
        else:
            input_columns = st.columns(min(network.input_feature_count, 4))
            values = []
            for index in range(network.input_feature_count):
                with input_columns[index % len(input_columns)]:
                    values.append(st.number_input(
                        f"x{index + 1}", value=0.0, step=0.1, format="%.3f",
                        key=f"visualizer_input_{index}",
                    ))
            sample = np.asarray(values, dtype=float)
        if st.button("Run Forward Propagation", type="primary", use_container_width=True, key="visualizer_forward"):
            try:
                trace = inspect_forward_pass(network, sample)
                st.session_state[TRACE_KEY] = {"network": network, "trace": trace, "input": sample.copy()}
            except (ValueError, FloatingPointError, RuntimeError) as error:
                st.error(f"Forward propagation failed: {error}")
    saved_trace = st.session_state.get(TRACE_KEY)
    trace = saved_trace["trace"] if saved_trace and saved_trace.get("network") is network else None
    if trace is None:
        st.info("Run forward propagation to reveal weighted sums, activations, and the final output. Parameters below are available now.")
    else:
        with sample_columns[1]:
            st.markdown("#### Final output")
            output = np.asarray(trace["prediction"])
            for index, value in enumerate(output):
                st.metric(f"Output {index + 1} · sigmoid", f"{float(value):.6f}")
            st.caption("Values come from this model's current weights and biases. They are not trained accuracy scores.")

        st.markdown("### 4 · Layer-by-layer stepper")
        layer_index = min(step, len(network.weights)) - 1
        if layer_index >= 0:
            _render_layer_step(network, trace, layer_index)

    st.markdown("### 5 · Layer inspector")
    parameterized_layers = list(range(1, len(network.layer_sizes)))
    if st.session_state.get("visualizer_inspect_layer") not in parameterized_layers:
        st.session_state["visualizer_inspect_layer"] = parameterized_layers[0]
    selected_layer = st.selectbox(
        "Select a layer",
        parameterized_layers,
        format_func=lambda index: f"Layer {index} · {'Output' if index == len(network.layer_sizes) - 1 else 'Hidden'}",
        key="visualizer_inspect_layer",
    )
    _render_layer_inspector(network, trace, selected_layer)

    st.markdown("### 6 · Weight and bias inspector")
    _render_connection_inspector(network, selected_layer)


def _render_layer_step(network: NeuralNetwork, trace: dict[str, object], layer_index: int) -> None:
    """Show one cached Z/A pair and the input activation for a selected layer."""
    weights = network.weights[layer_index]
    biases = network.biases[layer_index]
    previous = trace["input"] if layer_index == 0 else trace["activations"][layer_index - 1]
    z_values = trace["weighted_sums"][layer_index]
    activations = trace["activations"][layer_index]
    st.latex(r"z_j^{[l]} = \sum_i W_{ji}^{[l]} a_i^{[l-1]} + b_j^{[l]}, \qquad a_j^{[l]} = f(z_j^{[l]})")
    st.caption(
        f"Layer {layer_index + 1}: input activation shape {previous.shape} · "
        f"W shape {weights.shape} · b shape {biases.shape}"
    )
    rows = []
    for neuron in range(len(z_values)):
        rows.append({
            "Neuron": f"L{layer_index + 1}:N{neuron + 1}",
            "Weighted sum z": f"{z_values[neuron]:+.6f}",
            "Activation a": f"{activations[neuron]:.6f}",
            "Bias": f"{biases[neuron, 0]:+.6f}",
        })
    st.dataframe(rows, hide_index=True, width="stretch")
    if previous.size <= 12:
        st.code(f"a(previous) = {np.array2string(previous, precision=5)}")



def _render_layer_inspector(
    network: NeuralNetwork, trace: dict[str, object] | None, layer_index: int
) -> None:
    weights = network.weights[layer_index - 1]
    biases = network.biases[layer_index - 1]
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.caption(f"W{layer_index} · shape {weights.shape}")
        st.dataframe(weights, width="stretch")
    with col2:
        st.caption(f"b{layer_index} · shape {biases.shape}")
        st.dataframe(biases, width="stretch")
    if trace is None:
        st.caption("Run a forward pass to inspect this layer's cached weighted sums and activations.")
        return
    z_values = trace["weighted_sums"][layer_index - 1]
    activations = trace["activations"][layer_index - 1]
    st.caption("Cached values from the selected single-sample forward pass.")
    st.code(
        f"z{layer_index} = {np.array2string(z_values, precision=6)}\n"
        f"a{layer_index} = {np.array2string(activations, precision=6)}"
    )


def _render_connection_inspector(network: NeuralNetwork, layer_index: int) -> None:
    weights = network.weights[layer_index - 1]
    biases = network.biases[layer_index - 1]
    source_col, target_col, value_col = st.columns([1, 1, 2], gap="medium")
    with source_col:
        source = st.selectbox(
            "Source neuron",
            range(weights.shape[1]),
            format_func=lambda index: f"L{layer_index - 1}:N{index + 1}",
            key="visualizer_source_neuron",
        )
    with target_col:
        target = st.selectbox(
            "Destination neuron",
            range(weights.shape[0]),
            format_func=lambda index: f"L{layer_index}:N{index + 1}",
            key="visualizer_target_neuron",
        )
    weight = float(weights[target, source])
    gradient_value = None
    gradients = getattr(network, "gradients", None)
    if isinstance(gradients, dict):
        gradient_layers = gradients.get("weight_gradients", [])
        if len(gradient_layers) >= layer_index:
            gradient_value = float(gradient_layers[layer_index - 1][target, source])
    with value_col:
        st.metric("Weight", f"{weight:+.6f}")
        st.metric("Last gradient · pre-update", f"{gradient_value:+.6f}" if gradient_value is not None else "Not available")
    st.caption(f"Bias for destination neuron L{layer_index}:N{target + 1} = {biases[target, 0]:+.6f}")
    if gradient_value is not None:
        st.caption("Gradient is the last recorded value, calculated before its corresponding parameter update.")
