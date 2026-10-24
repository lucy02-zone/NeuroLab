"""NumPy loss functions and prediction-space derivatives.

Regression losses average over every target/prediction element. Binary
cross-entropy also averages over every element, while categorical
cross-entropy sums over classes per sample and then averages over samples.
The derivatives follow those same reductions and are with respect to
predictions/probabilities only; they do not include output activation
derivatives.
"""

import numpy as np


def mean_squared_error(targets: np.ndarray, predictions: np.ndarray) -> float:
    r"""Mean of ``(target - prediction)^2`` over all elements."""
    y, p = _matching_arrays(targets, predictions)
    return float(np.mean(np.square(p - y)))


def mean_absolute_error(targets: np.ndarray, predictions: np.ndarray) -> float:
    r"""Mean of ``abs(target - prediction)`` over all elements."""
    y, p = _matching_arrays(targets, predictions)
    return float(np.mean(np.abs(p - y)))


def binary_cross_entropy(
    targets: np.ndarray,
    predictions: np.ndarray,
    epsilon: float = 1e-12,
) -> float:
    r"""Binary cross-entropy averaged over all elements.

    Targets and predictions must have matching shapes and values in [0, 1].
    Predictions are clipped to ``[epsilon, 1-epsilon]`` before logarithms.
    """
    y, p = _matching_arrays(targets, predictions)
    _validate_unit_interval(y, "targets")
    _validate_unit_interval(p, "predictions")
    epsilon = _validate_epsilon(epsilon)
    p = np.clip(p, epsilon, 1.0 - epsilon)
    terms = y * np.log(p) + (1.0 - y) * np.log1p(-p)
    return float(-np.mean(terms))


def categorical_cross_entropy(
    targets: np.ndarray,
    predictions: np.ndarray,
    epsilon: float = 1e-12,
) -> float:
    r"""Categorical cross-entropy for one-hot targets and class probabilities.

    Both arrays have shape ``(n_samples, n_classes)`` (a single sample may be
    a 1D class vector). Each row of targets must be one-hot and each
    probability row must sum to one. Class losses are summed per sample, then
    averaged over samples. Probabilities are clipped before logarithms.
    """
    y, p = _matching_arrays(targets, predictions)
    if y.ndim == 1:
        y, p = y[None, :], p[None, :]
    if y.ndim != 2 or y.shape[0] == 0 or y.shape[1] < 2:
        raise ValueError("Categorical inputs must have shape (n_samples, n_classes) with at least two classes.")
    _validate_unit_interval(y, "targets")
    _validate_unit_interval(p, "predictions")
    if not np.allclose(y.sum(axis=1), 1.0) or not np.all(np.isclose(y, 0.0) | np.isclose(y, 1.0)):
        raise ValueError("Categorical targets must be one-hot encoded with one active class per sample.")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-7):
        raise ValueError("Each categorical prediction row must sum to 1.")
    epsilon = _validate_epsilon(epsilon)
    p = np.clip(p, epsilon, 1.0)
    return float(-np.mean(np.sum(y * np.log(p), axis=1)))


def mean_squared_error_derivative(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    r"""Derivative of global elementwise-mean MSE: ``2*(p-y)/n_elements``."""
    y, p = _matching_arrays(targets, predictions)
    return 2.0 * (p - y) / p.size


def mean_absolute_error_derivative(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    r"""Subgradient of mean absolute error: ``sign(p-y)/n_elements``.

    At exact equality this chooses the valid subgradient zero.
    """
    y, p = _matching_arrays(targets, predictions)
    return np.sign(p - y) / p.size


def binary_cross_entropy_derivative(
    targets: np.ndarray,
    predictions: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    r"""Derivative of mean BCE with respect to predictions only.

    Uses ``(p-y)/(p*(1-p)*n_elements)`` on the clipped probability values.
    No sigmoid derivative is included.
    """
    y, p = _matching_arrays(targets, predictions)
    _validate_unit_interval(y, "targets")
    _validate_unit_interval(p, "predictions")
    epsilon = _validate_epsilon(epsilon)
    p = np.clip(p, epsilon, 1.0 - epsilon)
    return (p - y) / (p * (1.0 - p) * p.size)


def categorical_cross_entropy_derivative(
    targets: np.ndarray,
    predictions: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    r"""Derivative of mean categorical CE w.r.t. probabilities: ``-y/p/n``.

    The averaging is over samples; class derivatives are not divided by the
    class count. No softmax derivative is included.
    """
    y, p = _matching_arrays(targets, predictions)
    if y.ndim == 1:
        y, p = y[None, :], p[None, :]
    if y.ndim != 2 or y.shape[1] < 2:
        raise ValueError("Categorical inputs must have shape (n_samples, n_classes) with at least two classes.")
    _validate_unit_interval(y, "targets")
    _validate_unit_interval(p, "predictions")
    if not np.allclose(y.sum(axis=1), 1.0) or not np.all(np.isclose(y, 0.0) | np.isclose(y, 1.0)):
        raise ValueError("Categorical targets must be one-hot encoded with one active class per sample.")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-7):
        raise ValueError("Each categorical prediction row must sum to 1.")
    epsilon = _validate_epsilon(epsilon)
    p = np.clip(p, epsilon, 1.0)
    return -y / p / y.shape[0]


# Short aliases are convenient in examples and common ML notation.
mse = mean_squared_error
mae = mean_absolute_error
mse_derivative = mean_squared_error_derivative
mae_derivative = mean_absolute_error_derivative


def _matching_arrays(targets: np.ndarray, predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(targets, dtype=float)
    p = np.asarray(predictions, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"Targets and predictions must have matching shapes; got {y.shape} and {p.shape}.")
    if y.size == 0:
        raise ValueError("Targets and predictions must not be empty.")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(p)):
        raise ValueError("Targets and predictions must contain only finite values.")
    return y, p


def _validate_unit_interval(values: np.ndarray, name: str) -> None:
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{name} must contain values in [0, 1].")


def _validate_epsilon(epsilon: float) -> float:
    if not np.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be finite and between 0 and 0.5.")
    return float(epsilon)
