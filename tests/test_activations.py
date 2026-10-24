"""Numerical checks for activation functions and their derivatives."""

import numpy as np
import pytest

from src.neurolab.models.activations import (
    leaky_relu,
    leaky_relu_derivative,
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    softmax,
    tanh,
    tanh_derivative,
)


def test_activation_functions_match_known_values_and_preserve_shapes():
    inputs = np.array([-1.0, 0.0, 1.0])

    np.testing.assert_allclose(sigmoid(inputs), [1 / (1 + np.e), 0.5, 1 / (1 + np.exp(-1))])
    np.testing.assert_allclose(tanh(inputs), np.tanh(inputs))
    np.testing.assert_array_equal(relu(inputs), [0.0, 0.0, 1.0])
    np.testing.assert_allclose(leaky_relu(inputs, alpha=0.1), [-0.1, 0.0, 1.0])
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert np.shape(tanh(inputs)) == inputs.shape


def test_elementwise_derivatives_at_representative_inputs():
    inputs = np.array([-1.0, 0.0, 1.0])

    np.testing.assert_allclose(sigmoid_derivative(inputs), [0.19661193, 0.25, 0.19661193], rtol=1e-6)
    np.testing.assert_allclose(tanh_derivative(inputs), 1.0 - np.tanh(inputs) ** 2)
    np.testing.assert_array_equal(relu_derivative(inputs), [0.0, 0.0, 1.0])
    np.testing.assert_allclose(leaky_relu_derivative(inputs, alpha=0.1), [0.1, 0.1, 1.0])
    np.testing.assert_allclose(sigmoid_derivative(inputs, sigmoid(inputs)), sigmoid_derivative(inputs))
    np.testing.assert_allclose(tanh_derivative(inputs, tanh(inputs)), tanh_derivative(inputs))


def test_softmax_probabilities_sum_to_one_and_support_batch_axis():
    probabilities = softmax(np.array([1.0, 2.0, 3.0]))
    batch_probabilities = softmax(np.array([[1.0, 2.0], [3.0, 4.0]]), axis=1)

    assert probabilities.shape == (3,)
    assert probabilities.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(batch_probabilities.sum(axis=1), [1.0, 1.0])


def test_softmax_and_sigmoid_are_finite_for_extreme_values():
    assert np.isfinite(sigmoid(np.array([-1000.0, 1000.0]))).all()
    probabilities = softmax(np.array([10000.0, 9999.0, -10000.0]))
    assert np.isfinite(probabilities).all()
    assert probabilities.sum() == pytest.approx(1.0)


def test_leaky_relu_alpha_must_be_nonnegative_and_finite():
    with pytest.raises(ValueError, match="alpha"):
        leaky_relu(np.array([-1.0]), alpha=-0.1)


def test_network_can_select_hidden_activation_without_changing_sigmoid_output():
    from src.neurolab.models import NeuralNetwork

    network = NeuralNetwork(1, [1], 1, random_seed=1, hidden_activation="tanh")
    network.weights[0][:] = 1.0
    network.weights[1][:] = 1.0
    prediction = network.forward(np.array([[1.0]]))

    np.testing.assert_allclose(network.activations[1], np.tanh([[1.0]]))
    np.testing.assert_allclose(prediction, 1.0 / (1.0 + np.exp(-np.tanh([[1.0]]))))


def test_network_rejects_unknown_hidden_activation():
    from src.neurolab.models import NeuralNetwork

    with pytest.raises(ValueError, match="hidden_activation"):
        NeuralNetwork(2, [3], 1, hidden_activation="softmax")
