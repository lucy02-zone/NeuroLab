"""Tests for forward propagation and parameter initialization."""

import numpy as np
import pytest

from src.neurolab.models import NeuralNetwork


def test_network_initializes_expected_parameter_shapes():
    network = NeuralNetwork(2, [4], 1, random_seed=1)

    assert [weight.shape for weight in network.weights] == [(4, 2), (1, 4)]
    assert [bias.shape for bias in network.biases] == [(4, 1), (1, 1)]


def test_network_supports_multiple_hidden_layers():
    network = NeuralNetwork(3, [5, 4, 2], 1, random_seed=2)

    assert [weight.shape for weight in network.weights] == [
        (5, 3), (4, 5), (2, 4), (1, 2)
    ]


def test_forward_batch_returns_expected_shape_and_finite_predictions():
    network = NeuralNetwork(2, [4], 1, random_seed=3)
    predictions = network.forward(np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]))

    assert predictions.shape == (3, 1)
    assert np.isfinite(predictions).all()


def test_forward_accepts_a_single_sample_and_keeps_batch_shape():
    network = NeuralNetwork(2, [4], 1, random_seed=4)

    assert network.forward(np.array([0.25, 0.75])).shape == (1, 1)


def test_forward_caches_layer_sums_and_activations():
    network = NeuralNetwork(2, [4], 1, random_seed=5)
    network.forward(np.array([[0.2, 0.8], [0.7, 0.3]]))

    assert [value.shape for value in network.weighted_sums] == [(4, 2), (1, 2)]
    assert [value.shape for value in network.activations] == [(2, 2), (4, 2), (1, 2)]


def test_seed_reproduces_initial_parameters():
    first = NeuralNetwork(2, [4, 3], 1, random_seed=17)
    second = NeuralNetwork(2, [4, 3], 1, random_seed=17)

    for left, right in zip(first.weights + first.biases, second.weights + second.biases):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("bad_input", [np.array([1.0]), np.ones((2, 3)), np.ones((2, 2, 1))])
def test_forward_rejects_invalid_input_dimensions(bad_input):
    network = NeuralNetwork(2, [4], 1, random_seed=6)

    with pytest.raises(ValueError, match="Expected|Inputs must"):
        network.forward(bad_input)


def test_network_rejects_nonpositive_layer_width():
    with pytest.raises(ValueError, match="positive integer"):
        NeuralNetwork(2, [0], 1)


def test_sigmoid_remains_finite_for_extreme_values():
    network = NeuralNetwork(1, [], 1, random_seed=8)
    network.weights[0][:] = 1.0
    result = network.forward(np.array([[1000.0], [-1000.0]]))

    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result, np.array([[1.0], [0.0]]))
