"""Deterministic checks for parameter optimizers and single-step training."""

import numpy as np
import pytest

from src.neurolab.models import Adam, Momentum, NeuralNetwork, SGD
from src.neurolab.models.losses import binary_cross_entropy


def test_sgd_updates_each_parameter_in_gradient_descent_direction():
    parameters = [np.array([1.0, -2.0]), np.array([[0.5]])]
    gradients = [np.array([2.0, -4.0]), np.array([[3.0]])]
    optimizer = SGD(learning_rate=0.1)

    optimizer.step(parameters, gradients)

    np.testing.assert_allclose(parameters[0], [0.8, -1.6])
    np.testing.assert_allclose(parameters[1], [[0.2]])


def test_momentum_accumulates_velocity_over_updates():
    parameter = np.array([1.0])
    optimizer = Momentum(learning_rate=0.1, momentum=0.9)

    optimizer.step([parameter], [np.array([0.5])])
    assert parameter[0] == pytest.approx(0.95)
    np.testing.assert_allclose(optimizer.velocities[0], [-0.05])
    optimizer.step([parameter], [np.array([0.25])])

    assert parameter[0] == pytest.approx(0.88)
    np.testing.assert_allclose(optimizer.velocities[0], [-0.07])


def test_adam_first_step_uses_bias_corrected_moments():
    parameter = np.array([1.0])
    gradient = np.array([2.0])
    optimizer = Adam(learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8)

    optimizer.step([parameter], [gradient])

    assert optimizer.time_step == 1
    np.testing.assert_allclose(optimizer.first_moments[0], [0.2])
    np.testing.assert_allclose(optimizer.second_moments[0], [0.004])
    assert parameter[0] == pytest.approx(1.0 - 0.1 * 2.0 / (2.0 + 1e-8))


def test_optimizer_instances_keep_independent_states():
    parameter_a = np.array([1.0])
    parameter_b = np.array([1.0])
    first = Momentum(learning_rate=0.1, momentum=0.9)
    second = Momentum(learning_rate=0.1, momentum=0.9)

    first.step([parameter_a], [np.array([1.0])])
    second.step([parameter_b], [np.array([0.0])])

    np.testing.assert_allclose(first.velocities[0], [-0.1])
    np.testing.assert_allclose(second.velocities[0], [0.0])
    assert first.velocities[0] is not second.velocities[0]


@pytest.mark.parametrize("optimizer_type", [SGD, Momentum, Adam])
def test_optimizer_rejects_invalid_learning_rate(optimizer_type):
    with pytest.raises(ValueError, match="learning_rate"):
        optimizer_type(learning_rate=0.0)


def test_optimizer_rejects_bad_gradients_without_partially_updating_parameters():
    first = np.array([1.0])
    second = np.array([2.0])
    optimizer = SGD(learning_rate=0.1)

    with pytest.raises(ValueError, match="shape"):
        optimizer.step([first, second], [np.array([0.5]), np.array([1.0, 2.0])])
    np.testing.assert_array_equal(first, [1.0])
    np.testing.assert_array_equal(second, [2.0])


def test_nonfinite_candidate_update_is_rejected_transactionally():
    parameter = np.array([1.0])
    optimizer = SGD(learning_rate=1e308)

    with pytest.raises(ValueError, match="non-finite"):
        optimizer.step([parameter], [np.array([1e308])])
    np.testing.assert_array_equal(parameter, [1.0])


def test_training_step_returns_preupdate_loss_updates_parameters_and_keeps_finite_gradients():
    network = NeuralNetwork(1, [], 1, random_seed=42)
    inputs = np.array([[0.0], [1.0]])
    targets = np.array([[1.0], [0.0]])
    predictions = network.forward(inputs)
    expected_loss = binary_cross_entropy(targets, predictions)
    old_parameters = [parameter.copy() for parameter in network.trainable_parameters]

    returned_loss = network.train_step(inputs, targets, SGD(learning_rate=0.1))

    assert returned_loss == pytest.approx(expected_loss)
    assert any(not np.array_equal(old, new) for old, new in zip(old_parameters, network.trainable_parameters))
    assert np.isfinite(returned_loss)
    assert all(np.isfinite(gradient).all() for gradient in network.gradients["weight_gradients"])
    assert all(np.isfinite(gradient).all() for gradient in network.gradients["bias_gradients"])
    assert network.activations == []  # updated weights invalidate the previous forward cache
    with pytest.raises(RuntimeError, match="backward"):
        network.apply_gradients(SGD(learning_rate=0.1))
