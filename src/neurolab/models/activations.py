"""NumPy activation functions and elementwise derivatives.

Softmax is provided separately because its derivative is a matrix Jacobian,
not an elementwise function. It is intended for multiclass output layers.
"""

import numpy as np


def _return_scalar_if_scalar(original: object, result: np.ndarray) -> np.ndarray | float:
    """Keep scalar calls scalar while preserving array shapes."""
    return float(result) if np.ndim(original) == 0 else result


def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    r"""Compute ``1 / (1 + exp(-x))`` with a stable positive/negative split."""
    source = np.asarray(values, dtype=float)
    result = np.empty_like(source)
    positive = source >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-source[positive]))
    exp_values = np.exp(source[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return _return_scalar_if_scalar(values, result)


def sigmoid_derivative(
    values: np.ndarray | float,
    activation_output: np.ndarray | float | None = None,
) -> np.ndarray | float:
    r"""Compute ``sigmoid(x) * (1 - sigmoid(x))``.

    Pass ``activation_output`` to reuse a sigmoid result computed in a prior
    forward pass; otherwise ``values`` is treated as the pre-activation input.
    """
    output = sigmoid(values) if activation_output is None else np.asarray(activation_output, dtype=float)
    result = np.asarray(output) * (1.0 - np.asarray(output))
    return _return_scalar_if_scalar(values if activation_output is None else activation_output, result)


def tanh(values: np.ndarray | float) -> np.ndarray | float:
    """Compute the hyperbolic tangent elementwise."""
    result = np.tanh(np.asarray(values, dtype=float))
    return _return_scalar_if_scalar(values, result)


def tanh_derivative(
    values: np.ndarray | float,
    activation_output: np.ndarray | float | None = None,
) -> np.ndarray | float:
    r"""Compute ``1 - tanh(x)^2``, optionally reusing the tanh output."""
    output = tanh(values) if activation_output is None else np.asarray(activation_output, dtype=float)
    result = 1.0 - np.square(output)
    return _return_scalar_if_scalar(values if activation_output is None else activation_output, np.asarray(result))


def relu(values: np.ndarray | float) -> np.ndarray | float:
    r"""Compute ``max(0, x)`` elementwise."""
    result = np.maximum(0.0, np.asarray(values, dtype=float))
    return _return_scalar_if_scalar(values, result)


def relu_derivative(values: np.ndarray | float) -> np.ndarray | float:
    """Return 1 for positive inputs and 0 at or below zero (including 0)."""
    result = (np.asarray(values, dtype=float) > 0.0).astype(float)
    return _return_scalar_if_scalar(values, result)


def leaky_relu(values: np.ndarray | float, alpha: float = 0.01) -> np.ndarray | float:
    r"""Compute ``max(alpha * x, x)`` elementwise."""
    _validate_alpha(alpha)
    source = np.asarray(values, dtype=float)
    result = np.maximum(alpha * source, source)
    return _return_scalar_if_scalar(values, result)


def leaky_relu_derivative(values: np.ndarray | float, alpha: float = 0.01) -> np.ndarray | float:
    """Return 1 for positive inputs and ``alpha`` for zero or negative inputs."""
    _validate_alpha(alpha)
    source = np.asarray(values, dtype=float)
    result = np.where(source > 0.0, 1.0, alpha)
    return _return_scalar_if_scalar(values, result)


def _validate_alpha(alpha: float) -> None:
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be a finite non-negative number.")


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray | float:
    r"""Convert logits to probabilities along an axis using max subtraction.

    For a vector ``z``, this computes ``exp(z_i - max(z)) / sum_j exp(z_j -
    max(z))``. Subtracting the maximum leaves probabilities unchanged while
    preventing overflow. This function intentionally does not define an
    elementwise derivative: the full softmax derivative is a Jacobian matrix.
    """
    source = np.asarray(values, dtype=float)
    shifted = source - np.max(source, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    result = exp_values / np.sum(exp_values, axis=axis, keepdims=True)
    return _return_scalar_if_scalar(values, result)
