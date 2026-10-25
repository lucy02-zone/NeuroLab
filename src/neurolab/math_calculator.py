"""Interactive numerical calculator for NeuroLab's existing DNN mathematics."""

from copy import deepcopy
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from .models import Adam, NeuralNetwork, SGD, SGDMomentum
from .models import activations, losses


ACTIVATIONS = {
    "Sigmoid": activations.sigmoid,
    "Tanh": activations.tanh,
    "ReLU": activations.relu,
    "Leaky ReLU": activations.leaky_relu,
}
LOSS_FUNCTIONS = {
    "Mean Squared Error": losses.mean_squared_error,
    "Mean Absolute Error": losses.mean_absolute_error,
    "Binary Cross-Entropy": losses.binary_cross_entropy,
    "Categorical Cross-Entropy": losses.categorical_cross_entropy,
}
GRADIENT_LOSSES = {
    "Mean Squared Error": "mean_squared_error",
    "Mean Absolute Error": "mean_absolute_error",
    "Binary Cross-Entropy": "binary_cross_entropy",
}
PARAMETERS_KEY = "mathcalc_manual_parameters"
REVISION_KEY = "mathcalc_parameter_revision"
OPTIMIZER_STATE_KEY = "mathcalc_optimizer_states"
GRADIENT_RESULT_KEY = "mathcalc_gradient_result"
GUIDE_STEPS = (
    "Enter values", "Calculate weighted sums", "Apply activations", "Calculate loss",
    "Inspect gradients", "Preview an update", "Apply an update",
)
GUIDE_TEXT = (
    "Start in Single Neuron or Forward Propagation and enter the input activations, weights, and biases.",
    "Inspect each layer's weighted sum using Z = W A + b, with rows for destination neurons.",
    "Compare each weighted sum with the activation output; hidden layers use your selected function and the output uses sigmoid.",
    "Enter matching target and prediction values in Loss Calculator and inspect the substituted loss terms.",
    "Use Gradient Calculator to trace dL/dA, activation derivatives, dZ, dW, db, and the propagated input gradient.",
    "In Parameter Updates, preview SGD, Momentum, or Adam on one selected parameter without changing it.",
    "Choose Apply update to change only the calculator's manual parameter copy; trained models remain untouched.",
)


def parse_numeric_vector(text: str, name: str = "values") -> np.ndarray:
    """Parse a comma-separated vector and reject empty or non-finite entries."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Enter at least one numeric {name} value.")
    try:
        values = np.asarray([float(part.strip()) for part in text.split(",")], dtype=float)
    except ValueError as error:
        raise ValueError(f"Every {name} entry must be numeric and comma-separated.") from error
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name.capitalize()} must contain finite numeric values.")
    return values


def calculate_single_neuron(
    inputs: np.ndarray | Sequence[float],
    weights: np.ndarray | Sequence[float],
    bias: float,
    activation: str,
    leaky_alpha: float = 0.01,
) -> dict[str, Any]:
    """Calculate a single neuron's weighted sum and activation with trace terms."""
    x = np.asarray(inputs, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.ndim != 1 or w.ndim != 1 or x.size == 0 or x.shape != w.shape:
        raise ValueError("Inputs and weights must be non-empty vectors with the same length.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(w)) or not np.isfinite(bias):
        raise ValueError("Inputs, weights, and bias must contain only finite values.")
    if activation not in ACTIVATIONS:
        raise ValueError(f"Activation must be one of {tuple(ACTIVATIONS)}.")
    if activation == "Leaky ReLU" and (not np.isfinite(leaky_alpha) or leaky_alpha < 0):
        raise ValueError("Leaky ReLU alpha must be finite and non-negative.")
    products = x * w
    running_totals = np.cumsum(products)
    z = float(np.dot(w, x) + bias)
    function = ACTIVATIONS[activation]
    output = function(z, alpha=leaky_alpha) if activation == "Leaky ReLU" else function(z)
    return {
        "inputs": x.copy(), "weights": w.copy(), "products": products,
        "running_totals": running_totals, "bias": float(bias),
        "weighted_sum": z, "activation": activation, "output": float(output),
    }


def manual_forward_propagation(
    inputs: np.ndarray | Sequence[float],
    weights: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
    hidden_activation: str = "ReLU",
) -> dict[str, Any]:
    """Run a small manually parameterized network using ``Z = W @ A + b``.

    Each hidden layer uses the selected activation. The final layer uses the
    existing sigmoid output convention from :class:`NeuralNetwork`.
    """
    activation_names = {"ReLU": "relu", "Leaky ReLU": "leaky_relu", "Sigmoid": "sigmoid", "Tanh": "tanh"}
    if hidden_activation not in activation_names:
        raise ValueError(f"Unsupported hidden activation {hidden_activation!r}.")
    current = np.asarray(inputs, dtype=float)
    if current.ndim == 2 and current.shape[1] == 1:
        current = current[:, 0]
    if current.ndim != 1 or current.size == 0 or not np.all(np.isfinite(current)):
        raise ValueError("Network input must be a non-empty finite feature vector.")
    if not weights or len(weights) != len(biases):
        raise ValueError("Provide matching non-empty weight and bias arrays for every layer.")
    function = {
        "relu": activations.relu, "leaky_relu": activations.leaky_relu,
        "sigmoid": activations.sigmoid, "tanh": activations.tanh,
    }[activation_names[hidden_activation]]
    calculations: list[dict[str, np.ndarray | str]] = []
    for index, (raw_weight, raw_bias) in enumerate(zip(weights, biases)):
        weight = np.asarray(raw_weight, dtype=float)
        bias = np.asarray(raw_bias, dtype=float)
        if weight.ndim != 2 or weight.shape[1] != current.size:
            raise ValueError(f"Layer {index + 1} weight shape must be (neurons, {current.size}); got {weight.shape}.")
        if bias.ndim == 1:
            bias = bias.reshape(-1, 1)
        if bias.shape != (weight.shape[0], 1):
            raise ValueError(f"Layer {index + 1} bias shape must be ({weight.shape[0]}, 1); got {bias.shape}.")
        if not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias)):
            raise ValueError(f"Layer {index + 1} weights and biases must be finite.")
        z = weight @ current.reshape(-1, 1) + bias
        layer_activation = "sigmoid" if index == len(weights) - 1 else activation_names[hidden_activation]
        if layer_activation == "sigmoid":
            next_activation = np.asarray(activations.sigmoid(z), dtype=float)
        elif layer_activation == "leaky_relu":
            next_activation = np.asarray(activations.leaky_relu(z), dtype=float)
        else:
            next_activation = np.asarray(function(z), dtype=float)
        calculations.append({
            "layer": f"Output" if index == len(weights) - 1 else f"Hidden layer {index + 1}",
            "activation_name": layer_activation,
            "input_activation": current.copy(),
            "weights": weight.copy(),
            "bias": bias[:, 0].copy(),
            "weighted_sum": z[:, 0].copy(),
            "output_activation": next_activation[:, 0].copy(),
        })
        current = next_activation[:, 0]
    return {"layers": calculations, "prediction": current.copy()}


def calculate_loss_details(loss_name: str, targets: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    """Calculate a supported loss with its actual scalar result and trace terms."""
    if loss_name not in LOSS_FUNCTIONS:
        raise ValueError(f"Loss must be one of {tuple(LOSS_FUNCTIONS)}.")
    y, p = np.asarray(targets, dtype=float), np.asarray(predictions, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"Targets and predictions must have matching shapes; got {y.shape} and {p.shape}.")
    scalar_target = y.ndim == 0
    if scalar_target:
        y, p = y.reshape(1), p.reshape(1)
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(p)):
        raise ValueError("Targets and predictions must be finite.")
    loss_value = LOSS_FUNCTIONS[loss_name](y, p)
    if loss_name == "Mean Squared Error":
        terms = np.square(p - y)
        formula = "L = mean((target − prediction)²)"
        substituted = "mean(" + ", ".join(f"({a:g} − {b:g})²" for a, b in zip(y.flat, p.flat)) + ")"
    elif loss_name == "Mean Absolute Error":
        terms = np.abs(p - y)
        formula = "L = mean(|target − prediction|)"
        substituted = "mean(" + ", ".join(f"|{a:g} − {b:g}|" for a, b in zip(y.flat, p.flat)) + ")"
    elif loss_name == "Binary Cross-Entropy":
        clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
        terms = -(y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped))
        formula = "L = −mean(y log(p) + (1−y) log(1−p))"
        substituted = "−mean(" + ", ".join(f"{a:g}·log({b:g}) + (1−{a:g})·log(1−{b:g})" for a, b in zip(y.flat, clipped.flat)) + ")"
    else:
        y_rows = y[None, :] if y.ndim == 1 else y
        p_rows = p[None, :] if p.ndim == 1 else p
        clipped = np.clip(p_rows, 1e-12, 1.0)
        terms = -np.sum(y_rows * np.log(clipped), axis=1)
        formula = "L = −mean over samples (sum over classes yᵢ log(pᵢ))"
        rows = []
        for target_row, prediction_row in zip(y_rows, clipped):
            rows.append("−(" + " + ".join(
                f"{target_value:g}·log({prediction_value:g})"
                for target_value, prediction_value in zip(target_row, prediction_row)
            ) + ")")
        substituted = "mean(" + ", ".join(rows) + ")"
    return {"loss": float(loss_value), "formula": formula, "substituted": substituted,
            "terms": np.asarray(terms), "targets": y, "predictions": p}


def calculate_manual_gradients(
    inputs: np.ndarray | Sequence[float], targets: np.ndarray | Sequence[float],
    weights: Sequence[np.ndarray], biases: Sequence[np.ndarray],
    hidden_activation: str, loss_name: str,
) -> dict[str, Any]:
    """Use the existing NeuralNetwork forward/backward methods on manual values."""
    loss_key = GRADIENT_LOSSES.get(loss_name)
    if loss_key is None:
        raise ValueError("Categorical Cross-Entropy gradients are not supported by the current sigmoid-output network.")
    values = np.asarray(inputs, dtype=float)
    if values.ndim != 1:
        raise ValueError("Gradient inspection currently accepts one feature vector at a time.")
    if not weights:
        raise ValueError("At least one layer is required for gradient calculation.")
    hidden_sizes = tuple(np.asarray(weight).shape[0] for weight in weights[:-1])
    output_size = np.asarray(weights[-1]).shape[0]
    hidden_key = {"ReLU": "relu", "Leaky ReLU": "leaky_relu", "Sigmoid": "sigmoid", "Tanh": "tanh"}.get(hidden_activation)
    if hidden_key is None:
        raise ValueError(f"Unsupported hidden activation {hidden_activation!r}.")
    network = NeuralNetwork(values.size, hidden_sizes, output_size, random_seed=0, hidden_activation=hidden_key)
    if len(network.weights) != len(weights) or len(network.biases) != len(biases):
        raise ValueError("Manual parameter arrays do not match the selected network architecture.")
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        weight_values = np.asarray(weight, dtype=float)
        bias_values = np.asarray(bias, dtype=float)
        if weight_values.shape != network.weights[index].shape or bias_values.shape != network.biases[index].shape:
            raise ValueError(f"Layer {index + 1} manual parameters do not match expected shapes.")
        network.weights[index] = weight_values.copy()
        network.biases[index] = bias_values.copy()
    prediction = network.forward(values)
    target_values = np.asarray(targets, dtype=float)
    if target_values.ndim == 0:
        target_values = target_values.reshape(1, 1)
    elif target_values.ndim == 1:
        target_values = target_values.reshape(1, -1)
    gradients = network.backward(target_values, loss=loss_key)
    first_delta = np.asarray(gradients["preactivation_gradients"][0], dtype=float).T
    input_gradient = network.weights[0].T @ first_delta
    return {"network": network, "prediction": prediction, "gradients": gradients,
            "input_gradient": input_gradient[:, 0].copy(), "targets": target_values}


def _default_parameters(input_count: int, hidden_sizes: Sequence[int]) -> dict[str, list[np.ndarray]]:
    widths = (input_count, *hidden_sizes, 1)
    weights = [np.full((out_width, in_width), 0.2, dtype=float)
               for in_width, out_width in zip(widths[:-1], widths[1:])]
    biases = [np.full((out_width, 1), 0.1, dtype=float) for out_width in widths[1:]]
    return {"weights": weights, "biases": biases}


def _format_array(values: np.ndarray) -> str:
    return np.array2string(np.asarray(values), precision=5, suppress_small=True)


def _make_optimizer(name: str, learning_rate: float, momentum: float, beta1: float, beta2: float, epsilon: float):
    if name == "SGD":
        return SGD(learning_rate)
    if name == "Momentum":
        return SGDMomentum(learning_rate, momentum=momentum)
    return Adam(learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)


def preview_parameter_update(value: float, gradient: float, optimizer_name: str,
                             learning_rate: float, momentum: float = 0.9,
                             beta1: float = 0.9, beta2: float = 0.999,
                             epsilon: float = 1e-8, optimizer_state: Any = None) -> tuple[float, Any]:
    """Preview a scalar update using existing optimizer classes without mutating inputs."""
    if not np.isfinite(value) or not np.isfinite(gradient):
        raise ValueError("Parameter and gradient must be finite numbers.")
    optimizer = deepcopy(optimizer_state) if optimizer_state is not None else _make_optimizer(
        optimizer_name, learning_rate, momentum, beta1, beta2, epsilon
    )
    parameter = np.array([[float(value)]], dtype=float)
    optimizer.step([parameter], [np.array([[float(gradient)]], dtype=float)])
    return float(parameter[0, 0]), optimizer


def apply_calculator_update(weights: Sequence[np.ndarray], biases: Sequence[np.ndarray],
                            parameter_kind: str, layer_index: int, row: int, column: int,
                            gradient: float, optimizer: Any) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Apply one optimizer step to a copied parameter set for explicit UI commits."""
    updated_weights = [np.asarray(value, dtype=float).copy() for value in weights]
    updated_biases = [np.asarray(value, dtype=float).copy() for value in biases]
    target = updated_weights if parameter_kind == "Weight" else updated_biases if parameter_kind == "Bias" else None
    if target is None:
        raise ValueError("parameter_kind must be 'Weight' or 'Bias'.")
    if isinstance(layer_index, bool) or not isinstance(layer_index, (int, np.integer)) or not 0 <= layer_index < len(target):
        raise ValueError("layer_index does not identify an available layer.")
    parameter = target[layer_index]
    if parameter.ndim != 2 or not (0 <= row < parameter.shape[0]) or not (0 <= column < parameter.shape[1]):
        raise ValueError("Selected parameter row or column is outside the matrix bounds.")
    if not np.isfinite(gradient):
        raise ValueError("Gradient must be finite.")
    optimizer.step([parameter[row:row + 1, column:column + 1]], [np.array([[gradient]], dtype=float)])
    return updated_weights, updated_biases


def _optimizer_state_id(name: str, layer: int, kind: str, row: int, column: int,
                        learning_rate: float, momentum: float, beta1: float, beta2: float, epsilon: float) -> str:
    return f"{name}:{layer}:{kind}:{row}:{column}:{learning_rate:.12g}:{momentum:.12g}:{beta1:.12g}:{beta2:.12g}:{epsilon:.12g}"


def render_math_calculator() -> None:
    """Render the step-guided math lab with editable parameters and real calculations."""
    st.markdown('<div class="eyebrow">CALCULATE · TRACE · UNDERSTAND</div>', unsafe_allow_html=True)
    st.title("Neural Network Math Calculator")
    st.markdown("Enter values and follow the same operations NeuroLab uses for forward propagation, loss, gradients, and parameter updates.")

    if "mathcalc_guide_step" not in st.session_state:
        st.session_state.mathcalc_guide_step = 0
    guide_step = int(st.session_state.mathcalc_guide_step)
    st.progress((guide_step + 1) / len(GUIDE_STEPS), text=f"Guided sequence · Step {guide_step + 1} of {len(GUIDE_STEPS)}: {GUIDE_STEPS[guide_step]}")
    st.info(GUIDE_TEXT[guide_step])
    step_cols = st.columns([1, 5, 1])
    if step_cols[0].button("← Back", key="mathcalc_guide_back", disabled=guide_step == 0):
        st.session_state.mathcalc_guide_step = max(0, guide_step - 1)
        st.rerun()
    step_cols[1].caption("Enter the example once, then use the sections below to inspect each mathematical stage. Values remain in the page while you move through the sequence.")
    if step_cols[2].button("Next →", key="mathcalc_guide_next", disabled=guide_step == len(GUIDE_STEPS) - 1):
        st.session_state.mathcalc_guide_step = min(len(GUIDE_STEPS) - 1, guide_step + 1)
        st.rerun()

    tabs = st.tabs(("Single Neuron", "Forward Propagation", "Loss Calculator", "Gradient Calculator", "Parameter Updates"))
    with tabs[0]:
        _render_single_neuron()
    with tabs[1]:
        forward = _render_forward_propagation()
    with tabs[2]:
        _render_loss_calculator(forward)
    with tabs[3]:
        gradient_result = _render_gradient_calculator(forward)
    with tabs[4]:
        _render_parameter_updates(forward, gradient_result)


def _render_single_neuron() -> None:
    st.markdown("Enter one input and weight per feature. Each product is accumulated before adding the bias.")
    controls = st.columns([1, 1, 0.7, 1])
    with controls[0]:
        inputs_text = st.text_input("Input values x", "1.0, -2.0", key="mathcalc_neuron_inputs")
    with controls[1]:
        weights_text = st.text_input("Weights w", "0.5, 0.25", key="mathcalc_neuron_weights")
    with controls[2]:
        bias = st.number_input("Bias b", value=0.1, format="%.5f", key="mathcalc_neuron_bias")
    with controls[3]:
        activation = st.selectbox("Activation function", tuple(ACTIVATIONS), key="mathcalc_neuron_activation")
    alpha = 0.01
    if activation == "Leaky ReLU":
        alpha = st.number_input("Leaky ReLU α", 0.0, 1.0, 0.01, 0.01, key="mathcalc_neuron_alpha")
    try:
        result = calculate_single_neuron(parse_numeric_vector(inputs_text, "input"),
                                         parse_numeric_vector(weights_text, "weight"), float(bias), activation, float(alpha))
        st.latex(r"z = \sum_i w_i x_i + b")
        for index, (x_value, weight, product, running) in enumerate(zip(
            result["inputs"], result["weights"], result["products"], result["running_totals"]
        ), start=1):
            st.write(f"{index}.  {weight:.5g} × {x_value:.5g} = {product:.5g}; running sum = {running:.5g}")
        st.write(f"Add bias: {result['running_totals'][-1]:.5g} + ({result['bias']:.5g}) = **z = {result['weighted_sum']:.7g}**")
        st.latex(r"a = f(z)")
        st.metric("Activation output", f"{result['output']:.7g}")
    except ValueError as error:
        st.error(str(error))


def _render_forward_propagation() -> dict[str, Any] | None:
    st.markdown("For each layer, values are stored as vectors and the weight matrix has shape (neurons, incoming features). The final layer uses the engine's sigmoid output.")
    controls = st.columns(4)
    with controls[0]:
        input_count = st.number_input("Input features", 1, 4, 2, 1, key="mathcalc_input_count")
    with controls[1]:
        hidden_count = st.selectbox("Hidden layers", (1, 2), key="mathcalc_hidden_count")
    hidden_sizes: list[int] = []
    for index in range(int(hidden_count)):
        with controls[index + 2]:
            hidden_sizes.append(st.number_input(f"Layer {index + 1} neurons", 1, 5, 2, 1, key=f"mathcalc_hidden_size_{index}"))
    activation = st.selectbox("Hidden-layer activation", tuple(ACTIVATIONS), key="mathcalc_forward_activation")
    inputs_text = st.text_input("Input activation vector A⁽⁰⁾", ", ".join(["1.0"] * int(input_count)), key="mathcalc_forward_inputs")
    sizes = (int(input_count), *map(int, hidden_sizes), 1)
    signature = tuple(sizes)
    stored = st.session_state.get(PARAMETERS_KEY)
    if not isinstance(stored, dict) or stored.get("signature") != signature:
        stored = {"signature": signature, **_default_parameters(int(input_count), hidden_sizes)}
        st.session_state[PARAMETERS_KEY] = stored
        st.session_state[REVISION_KEY] = int(st.session_state.get(REVISION_KEY, 0)) + 1
        st.session_state[OPTIMIZER_STATE_KEY] = {}
    revision = int(st.session_state.get(REVISION_KEY, 0))
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    st.markdown("#### Editable manual parameters")
    st.caption("Rows in each weight table are destination neurons; columns are the incoming activations. Bias vectors contain one value per destination neuron.")
    for index, (in_width, out_width) in enumerate(zip(sizes[:-1], sizes[1:])):
        weight_key = f"mathcalc_weight_{revision}_{index}"
        bias_key = f"mathcalc_bias_{revision}_{index}"
        left, right = st.columns([1.35, 0.65], gap="large")
        with left:
            st.caption(f"Layer {index + 1} weights · shape ({out_width}, {in_width})")
            initial_weights = stored["weights"][index]
            edited_weights = st.data_editor(
                pd.DataFrame(initial_weights, columns=[f"A{i + 1}" for i in range(in_width)],
                             index=[f"neuron {i + 1}" for i in range(out_width)]),
                key=weight_key, hide_index=False, num_rows="fixed", width="stretch",
            )
        with right:
            st.caption(f"Layer {index + 1} bias · shape ({out_width},)")
            edited_bias = st.data_editor(
                pd.DataFrame(stored["biases"][index], columns=["b"], index=[f"neuron {i + 1}" for i in range(out_width)]),
                key=bias_key, hide_index=False, num_rows="fixed", width="stretch",
            )
        try:
            weights.append(edited_weights.to_numpy(dtype=float))
            biases.append(edited_bias.to_numpy(dtype=float))
        except (TypeError, ValueError) as error:
            st.error(f"Layer {index + 1} parameters must be numeric: {error}")
    if len(weights) != len(sizes) - 1:
        return None
    manually_changed = any(not np.array_equal(old, new) for old, new in zip(stored["weights"] + stored["biases"], weights + biases))
    st.session_state[PARAMETERS_KEY] = {"signature": signature,
                                        "weights": [value.copy() for value in weights],
                                        "biases": [value.copy() for value in biases]}
    if manually_changed:
        st.session_state[OPTIMIZER_STATE_KEY] = {}
    try:
        inputs = parse_numeric_vector(inputs_text, "input feature")
        if inputs.size != input_count:
            raise ValueError(f"Input vector has {inputs.size} values; the configured network expects {input_count}.")
        result = manual_forward_propagation(inputs, weights, biases, activation)
        st.session_state["mathcalc_forward_cache"] = {"inputs": inputs.copy(), "weights": weights, "biases": biases,
                                                       "activation": activation, "prediction": result["prediction"].copy()}
        for layer in result["layers"]:
            st.markdown(f"#### {layer['layer']} · activation: {layer['activation_name']}")
            st.write(f"Input activation A, shape ({len(layer['input_activation'])},):", _format_array(layer["input_activation"]))
            st.write(f"Weight matrix W, shape {layer['weights'].shape}:\n\n{_format_array(layer['weights'])}")
            st.write(f"Bias vector b, shape ({len(layer['bias'])},):", _format_array(layer["bias"]))
            st.latex(r"Z = W A + b")
            st.write(f"Weighted sum Z, shape ({len(layer['weighted_sum'])},):", _format_array(layer["weighted_sum"]))
            st.write(f"Activated output A, shape ({len(layer['output_activation'])},):", _format_array(layer["output_activation"]))
        st.success("Forward pass completed with the entered parameters.")
        return {**result, "inputs": inputs, "weights": weights, "biases": biases, "activation": activation}
    except ValueError as error:
        st.error(str(error))
        return None


def _render_loss_calculator(forward: dict[str, Any] | None) -> None:
    st.markdown("Targets and predictions must have the same shape. Categorical cross-entropy additionally needs one-hot targets and probability rows summing to one.")
    columns = st.columns(3)
    with columns[0]:
        loss_name = st.selectbox("Loss function", tuple(LOSS_FUNCTIONS), key="mathcalc_loss_name")
    with columns[1]:
        target_text = st.text_input("Target values", "1.0, 0.0", key="mathcalc_loss_targets")
    with columns[2]:
        prediction_text = st.text_input("Predicted values", "0.7, 0.3", key="mathcalc_loss_predictions")
    use_forward = st.checkbox("Use the current manual network output", value=False, key="mathcalc_loss_use_forward")
    try:
        targets = parse_numeric_vector(target_text, "target")
        predictions = forward["prediction"] if use_forward and forward is not None else parse_numeric_vector(prediction_text, "prediction")
        if use_forward and forward is None:
            raise ValueError("Complete a valid forward pass before using its output here.")
        if loss_name == "Categorical Cross-Entropy":
            targets = targets.reshape(1, -1)
            predictions = np.asarray(predictions, dtype=float).reshape(1, -1)
        result = calculate_loss_details(loss_name, targets, predictions)
        st.latex(result["formula"])
        st.write("Substituted values:", result["substituted"])
        st.write("Intermediate loss terms:", _format_array(result["terms"]))
        st.metric("Calculated loss", f"{result['loss']:.8g}")
    except ValueError as error:
        st.error(str(error))


def _render_gradient_calculator(forward: dict[str, Any] | None) -> dict[str, Any] | None:
    st.markdown("The calculation below calls NeuroLab's existing `NeuralNetwork.backward()` method. It reports gradients only; it does not update any parameter.")
    if forward is None:
        st.info("Complete the manual forward pass first.")
        st.session_state.pop(GRADIENT_RESULT_KEY, None)
        return None
    cols = st.columns(3)
    with cols[0]:
        target = st.number_input("Binary target", 0.0, 1.0, 1.0, 0.1, key="mathcalc_gradient_target")
    with cols[1]:
        loss_name = st.selectbox("Gradient loss", tuple(GRADIENT_LOSSES), key="mathcalc_gradient_loss")
    with cols[2]:
        st.metric("Prediction", f"{float(forward['prediction'][0]):.7g}")
    try:
        result = calculate_manual_gradients(
            forward["inputs"], np.array([target]), forward["weights"], forward["biases"],
            forward["activation"], loss_name,
        )
        gradients = result["gradients"]
        network = result["network"]
        predictions = result["prediction"]
        if loss_name == "Mean Squared Error":
            loss_derivative = losses.mean_squared_error_derivative(result["targets"], predictions)
        elif loss_name == "Mean Absolute Error":
            loss_derivative = losses.mean_absolute_error_derivative(result["targets"], predictions)
        else:
            loss_derivative = losses.binary_cross_entropy_derivative(result["targets"], predictions)
        st.write("Loss derivative with respect to prediction, dL/dA:", _format_array(loss_derivative))
        for index, (derivative, d_z, d_weight, d_bias) in enumerate(zip(
            gradients["activation_derivatives"], gradients["preactivation_gradients"],
            gradients["weight_gradients"], gradients["bias_gradients"],
        ), start=1):
            st.markdown(f"#### Layer {index + 1}")
            st.write(f"Activation derivative f′(Z), shape {derivative.shape}:", _format_array(derivative))
            st.write(f"Pre-activation gradient dZ, shape {d_z.shape}:", _format_array(d_z))
            st.write(f"Weight gradient dW, shape {d_weight.shape}:", _format_array(d_weight))
            st.write(f"Bias gradient db, shape {d_bias.shape}:", _format_array(d_bias))
        for index, activation_gradient in enumerate(gradients["activation_gradients"][:-1], start=1):
            st.write(f"Gradient propagated to hidden activation A⁽{index}⁾, dL/dA shape {activation_gradient.shape}:", _format_array(activation_gradient))
        st.write("Gradient propagated to the input vector, dL/dA⁽⁰⁾:", _format_array(result["input_gradient"]))
        st.caption(f"Loss used for these gradients: {gradients['loss']:.8g}. These values are gradients, not updated parameters.")
        st.session_state[GRADIENT_RESULT_KEY] = result
        return result
    except (ValueError, RuntimeError, FloatingPointError) as error:
        st.error(f"Gradients could not be calculated: {error}")
        st.session_state.pop(GRADIENT_RESULT_KEY, None)
        return None


def _render_parameter_updates(forward: dict[str, Any] | None, gradient_result: dict[str, Any] | None) -> None:
    st.markdown("Choose one parameter to inspect. Previewing leaves it unchanged; **Apply update** changes only this calculator's manual parameter copy, never a trained model.")
    if forward is None or gradient_result is None:
        st.info("Complete a manual forward pass and gradient calculation first.")
        return
    network = gradient_result["network"]
    controls = st.columns(5)
    with controls[0]:
        optimizer_name = st.selectbox("Optimizer", ("SGD", "Momentum", "Adam"), key="mathcalc_optimizer")
    with controls[1]:
        learning_rate = st.number_input("Learning rate η", 0.0001, 1.0, 0.01, 0.001, format="%.4f", key="mathcalc_learning_rate")
    with controls[2]:
        momentum = st.number_input("Momentum β", 0.0, 0.99, 0.9, 0.01, key="mathcalc_update_momentum") if optimizer_name == "Momentum" else 0.9
        if optimizer_name == "Adam":
            beta1 = st.number_input("Adam β₁", 0.0, 0.999, 0.9, 0.01, key="mathcalc_update_beta1")
        else:
            beta1 = 0.9
    with controls[3]:
        if optimizer_name == "Adam":
            beta2 = st.number_input("Adam β₂", 0.0, 0.9999, 0.999, 0.001, key="mathcalc_update_beta2")
            epsilon = st.number_input("Adam ε", 1e-10, 1e-3, 1e-8, format="%.1e", key="mathcalc_update_epsilon")
        else:
            beta2, epsilon = 0.999, 1e-8
    with controls[4]:
        parameter_kind = st.selectbox("Parameter", ("Weight", "Bias"), key="mathcalc_parameter_kind")

    layer_choices = tuple(range(len(network.weights)))
    layer_index = st.selectbox("Layer index", layer_choices, format_func=lambda value: f"Layer {value + 1}", key="mathcalc_parameter_layer")
    parameter = network.weights[layer_index] if parameter_kind == "Weight" else network.biases[layer_index]
    gradient_array = gradient_result["gradients"]["weight_gradients"][layer_index] if parameter_kind == "Weight" else gradient_result["gradients"]["bias_gradients"][layer_index]
    index_cols = st.columns(2 if parameter_kind == "Weight" else 1)
    row = index_cols[0].selectbox("Neuron / row", tuple(range(parameter.shape[0])), format_func=lambda value: str(value + 1), key="mathcalc_parameter_row")
    if parameter_kind == "Weight":
        column = index_cols[1].selectbox("Input / column", tuple(range(parameter.shape[1])), format_func=lambda value: str(value + 1), key="mathcalc_parameter_column")
    else:
        column = 0
    original = float(parameter[row, column])
    gradient = float(gradient_array[row, column])
    state_id = _optimizer_state_id(optimizer_name, layer_index, parameter_kind, row, column,
                                   float(learning_rate), float(momentum), float(beta1), float(beta2), float(epsilon))
    states = st.session_state.setdefault(OPTIMIZER_STATE_KEY, {})
    optimizer = states.get(state_id)
    if optimizer is None:
        optimizer = _make_optimizer(optimizer_name, float(learning_rate), float(momentum), float(beta1), float(beta2), float(epsilon))
    updated, preview_optimizer = preview_parameter_update(
        original, gradient, optimizer_name, float(learning_rate), float(momentum),
        float(beta1), float(beta2), float(epsilon), optimizer,
    )
    if optimizer_name == "SGD":
        st.latex(r"\theta_{new} = \theta - \eta g")
        st.write(f"{original:.8g} − {learning_rate:.8g} × ({gradient:.8g}) = **{updated:.8g}**")
    elif optimizer_name == "Momentum":
        old_velocity = float(optimizer.velocities[0][0, 0]) if optimizer.velocities else 0.0
        new_velocity = float(preview_optimizer.velocities[0][0, 0])
        st.latex(r"v_t = \beta v_{t-1} - \eta g, \quad \theta_{new}=\theta+v_t")
        st.write(f"vₜ = {momentum:.5g} × ({old_velocity:.8g}) − {learning_rate:.5g} × ({gradient:.8g}) = {new_velocity:.8g}")
        st.write(f"{original:.8g} + ({new_velocity:.8g}) = **{updated:.8g}**")
    else:
        st.latex(r"\hat m_t=\frac{m_t}{1-\beta_1^t},\quad \hat v_t=\frac{v_t}{1-\beta_2^t},\quad \theta_{new}=\theta-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}")
        step = optimizer.time_step + 1
        old_first = float(optimizer.first_moments[0][0, 0]) if optimizer.first_moments else 0.0
        old_second = float(optimizer.second_moments[0][0, 0]) if optimizer.second_moments else 0.0
        moment = float(preview_optimizer.first_moments[0][0, 0])
        second_moment = float(preview_optimizer.second_moments[0][0, 0])
        corrected_moment = moment / (1 - float(beta1) ** step)
        corrected_second = second_moment / (1 - float(beta2) ** step)
        st.write(f"Step t={step}: mₜ = {beta1:.5g}×({old_first:.8g}) + (1−{beta1:.5g})×({gradient:.8g}) = {moment:.8g}")
        st.write(f"vₜ = {beta2:.5g}×({old_second:.8g}) + (1−{beta2:.5g})×({gradient:.8g})² = {second_moment:.8g}")
        st.write(f"Bias corrected: m̂ₜ = {corrected_moment:.8g}; v̂ₜ = {corrected_second:.8g}")
        st.write(f"{original:.8g} − {learning_rate:.6g}×{corrected_moment:.8g}/(√{corrected_second:.8g}+{epsilon:.2g}) = **{updated:.8g}**")
    st.write(f"Gradient: {gradient:.8g} · Parameter before update: {original:.8g} · Learning rate: {learning_rate:.8g}")
    st.metric("Previewed parameter value", f"{updated:.8g}")
    if st.button("Apply update to calculator copy", type="primary", key="mathcalc_apply_update"):
        try:
            updated_weights, updated_biases = apply_calculator_update(
                forward["weights"], forward["biases"], parameter_kind, layer_index, row, column,
                gradient, optimizer,
            )
            states[state_id] = optimizer
            stored = st.session_state[PARAMETERS_KEY]
            stored["weights"] = updated_weights
            stored["biases"] = updated_biases
            st.session_state[PARAMETERS_KEY] = stored
            st.session_state[REVISION_KEY] = int(st.session_state.get(REVISION_KEY, 0)) + 1
            st.session_state.pop(GRADIENT_RESULT_KEY, None)
            st.success("Update applied to the calculator's local parameters. Trained models remain unchanged.")
            st.rerun()
        except (ValueError, FloatingPointError) as error:
            st.error(f"Parameter update could not be applied: {error}")
