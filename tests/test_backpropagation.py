"""Tests for analytical gradients and finite-difference checking."""

import numpy as np
import pytest

from src.neurolab.models import NeuralNetwork
from src.neurolab.models.gradient_check import check_gradients


def test_backward_produces_finite_gradients_with_parameter_shapes_without_updating():
    network = NeuralNetwork(2, [3, 2], 1, random_seed=22)
    inputs = np.array([[0.2, 0.8], [0.7, 0.1], [0.4, 0.6]])
    targets = np.array([[1.0], [0.0], [1.0]])
    network.forward(inputs)
    before_weights = [weight.copy() for weight in network.weights]
    before_biases = [bias.copy() for bias in network.biases]

    gradients = network.backward(targets)

    assert len(gradients["weight_gradients"]) == len(network.weights)
    assert len(gradients["bias_gradients"]) == len(network.biases)
    for parameter, gradient in zip(network.weights, gradients["weight_gradients"]):
        assert gradient.shape == parameter.shape
        assert np.isfinite(gradient).all()
    for parameter, gradient in zip(network.biases, gradients["bias_gradients"]):
        assert gradient.shape == parameter.shape
        assert np.isfinite(gradient).all()
    assert all(norm >= 0 and np.isfinite(norm) for norm in gradients["weight_norms"])
    assert all(norm >= 0 and np.isfinite(norm) for norm in gradients["bias_norms"])
    for old, current in zip(before_weights, network.weights):
        np.testing.assert_array_equal(current, old)
    for old, current in zip(before_biases, network.biases):
        np.testing.assert_array_equal(current, old)


@pytest.mark.parametrize("hidden_activation", ["relu", "leaky_relu", "sigmoid", "tanh"])
def test_backward_supports_hidden_activation_derivatives(hidden_activation):
    network = NeuralNetwork(2, [3, 2], 1, random_seed=23, hidden_activation=hidden_activation)
    network.forward(np.array([[0.2, 0.8], [0.7, 0.1]]))

    gradients = network.backward(np.array([1.0, 0.0]), loss="mse")

    assert all(np.isfinite(value).all() for value in gradients["weight_gradients"])
    assert all(np.isfinite(value).all() for value in gradients["bias_gradients"])


def test_gradient_check_matches_finite_differences_and_restores_parameters():
    network = NeuralNetwork(2, [2], 1, random_seed=31)
    inputs = np.array([[0.15, 0.35], [0.65, 0.25]])
    targets = np.array([[1.0], [0.0]])
    original_weights = [weight.copy() for weight in network.weights]
    original_biases = [bias.copy() for bias in network.biases]

    result = check_gradients(network, inputs, targets, epsilon=1e-5, tolerance=1e-6)

    assert result["passed"]
    assert result["max_relative_error"] < 1e-6
    for old, current in zip(original_weights, network.weights):
        np.testing.assert_array_equal(current, old)
    for old, current in zip(original_biases, network.biases):
        np.testing.assert_array_equal(current, old)


def test_gradient_check_supports_regression_loss():
    network = NeuralNetwork(1, [2], 1, random_seed=32)
    inputs = np.array([[-0.3], [0.4]])
    targets = np.array([[0.2], [0.8]])

    result = check_gradients(network, inputs, targets, loss="mse", tolerance=1e-6)

    assert result["passed"]


def test_backward_requires_forward_and_rejects_categorical_output_loss():
    network = NeuralNetwork(2, [2], 1, random_seed=33)
    with pytest.raises(RuntimeError, match="forward"):
        network.backward(np.array([[1.0]]))

    network.forward(np.array([[0.2, 0.8]]))
    with pytest.raises(ValueError, match="Unsupported loss"):
        network.backward(np.array([[1.0]]), loss="categorical_cross_entropy")


def test_backward_rejects_target_shape_mismatch():
    network = NeuralNetwork(2, [2], 1, random_seed=34)
    network.forward(np.array([[0.2, 0.8], [0.7, 0.1]]))

    with pytest.raises(ValueError, match="Targets must have shape"):
        network.backward(np.array([[1.0]]))
