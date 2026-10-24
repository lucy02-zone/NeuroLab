"""Finite-difference checks for NeuroLab's analytical backpropagation."""

from collections.abc import Sequence

import numpy as np

from . import losses
from .neural_network import NeuralNetwork


_LOSS_FUNCTIONS = {
    "binary_cross_entropy": losses.binary_cross_entropy,
    "bce": losses.binary_cross_entropy,
    "mean_squared_error": losses.mean_squared_error,
    "mse": losses.mean_squared_error,
    "mean_absolute_error": losses.mean_absolute_error,
    "mae": losses.mean_absolute_error,
}


def check_gradients(
    network: NeuralNetwork,
    inputs: np.ndarray,
    targets: np.ndarray,
    loss: str = "binary_cross_entropy",
    epsilon: float = 1e-5,
    tolerance: float = 1e-5,
) -> dict[str, object]:
    """Compare every parameter gradient with a centered finite difference.

    Parameters are restored in a ``finally`` block even if an evaluation
    fails. The network's caches are refreshed with the original inputs after
    checking. Relative differences are calculated elementwise as
    ``abs(a-n) / max(abs(a)+abs(n), 1e-12)``.
    """
    if loss not in _LOSS_FUNCTIONS:
        raise ValueError(f"Unsupported gradient-check loss {loss!r}.")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a finite positive number.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be a finite positive number.")

    network.forward(inputs)
    targets_array = network._prepare_targets(targets)
    analytical = network.backward(targets_array, loss=loss)
    analytical_weights = [gradient.copy() for gradient in analytical["weight_gradients"]]
    analytical_biases = [gradient.copy() for gradient in analytical["bias_gradients"]]
    parameters: Sequence[np.ndarray] = (*network.weights, *network.biases)
    analytical_parameters: Sequence[np.ndarray] = (*analytical_weights, *analytical_biases)
    original_parameters = [parameter.copy() for parameter in parameters]
    numerical_parameters = [np.zeros_like(parameter) for parameter in parameters]
    evaluate_loss = _LOSS_FUNCTIONS[loss]

    def objective() -> float:
        prediction = network.forward(inputs)
        return evaluate_loss(targets_array, prediction)

    try:
        for parameter, numerical in zip(parameters, numerical_parameters):
            for index in np.ndindex(parameter.shape):
                original_value = parameter[index]
                parameter[index] = original_value + epsilon
                plus_loss = objective()
                parameter[index] = original_value - epsilon
                minus_loss = objective()
                parameter[index] = original_value
                numerical[index] = (plus_loss - minus_loss) / (2.0 * epsilon)
    finally:
        for parameter, original in zip(parameters, original_parameters):
            parameter[...] = original
        network.forward(inputs)

    differences = [
        np.abs(analytical_value - numerical_value)
        / np.maximum(np.abs(analytical_value) + np.abs(numerical_value), 1e-12)
        for analytical_value, numerical_value in zip(analytical_parameters, numerical_parameters)
    ]
    max_relative_error = max(float(np.max(value)) for value in differences)
    return {
        "passed": max_relative_error <= tolerance,
        "tolerance": float(tolerance),
        "max_relative_error": max_relative_error,
        "weight_gradients": analytical_weights,
        "bias_gradients": analytical_biases,
        "numerical_weight_gradients": numerical_parameters[: len(network.weights)],
        "numerical_bias_gradients": numerical_parameters[len(network.weights) :],
    }
