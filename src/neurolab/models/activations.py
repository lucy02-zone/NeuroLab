"""Activation functions used by NeuroLab's NumPy models."""

import numpy as np


def relu(values: np.ndarray) -> np.ndarray:
    """Apply the rectified linear unit elementwise."""
    return np.maximum(0.0, values)


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Apply sigmoid without overflowing for large-magnitude inputs."""
    result = np.empty_like(values, dtype=float)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result
