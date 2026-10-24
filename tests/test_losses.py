"""Value, validation, stability, and derivative tests for loss functions."""

import numpy as np
import pytest

from src.neurolab.models.losses import (
    binary_cross_entropy,
    binary_cross_entropy_derivative,
    categorical_cross_entropy,
    categorical_cross_entropy_derivative,
    mae_derivative,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_error_derivative,
)


def test_regression_losses_match_known_values_and_perfect_predictions():
    targets = np.array([1.0, 2.0])
    predictions = np.array([0.0, 4.0])

    assert mean_squared_error(targets, predictions) == pytest.approx(2.5)
    assert mean_absolute_error(targets, predictions) == pytest.approx(1.5)
    assert mean_squared_error(targets, targets) == pytest.approx(0.0)
    assert mean_absolute_error(targets, targets) == pytest.approx(0.0)


def test_classification_losses_match_known_values_and_perfect_predictions():
    bce = binary_cross_entropy(np.array([1.0, 0.0]), np.array([0.8, 0.25]))
    expected_bce = -(np.log(0.8) + np.log(0.75)) / 2
    assert bce == pytest.approx(expected_bce)

    targets = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    predictions = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    assert categorical_cross_entropy(targets, predictions) == pytest.approx(-(np.log(0.7) + np.log(0.8)) / 2)
    assert categorical_cross_entropy(targets, targets) == pytest.approx(0.0)
    assert binary_cross_entropy(np.array([1.0]), np.array([1.0])) < 1e-10


def test_loss_derivatives_match_known_examples_and_reductions():
    targets = np.array([1.0, 2.0])
    predictions = np.array([0.0, 4.0])
    np.testing.assert_allclose(mean_squared_error_derivative(targets, predictions), [-1.0, 2.0])
    np.testing.assert_allclose(mae_derivative(targets, predictions), [-0.5, 0.5])

    y_binary = np.array([1.0, 0.0])
    p_binary = np.array([0.8, 0.25])
    expected = (p_binary - y_binary) / (p_binary * (1.0 - p_binary) * 2)
    np.testing.assert_allclose(binary_cross_entropy_derivative(y_binary, p_binary), expected)

    y_multi = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    p_multi = np.array([[0.2, 0.5, 0.3], [0.6, 0.1, 0.3]])
    expected_multi = -y_multi / p_multi / 2
    np.testing.assert_allclose(categorical_cross_entropy_derivative(y_multi, p_multi), expected_multi)


def test_cross_entropy_is_finite_for_boundary_probabilities():
    assert np.isfinite(binary_cross_entropy(np.array([1.0, 0.0]), np.array([0.0, 1.0])))
    y = np.array([[1.0, 0.0, 0.0]])
    p = np.array([[0.0, 1.0, 0.0]])
    assert np.isfinite(categorical_cross_entropy(y, p))
    assert np.isfinite(binary_cross_entropy_derivative(np.array([1.0]), np.array([0.0]))).all()
    assert np.isfinite(categorical_cross_entropy_derivative(y, p)).all()


def test_loss_derivatives_match_finite_differences():
    step = 1e-6
    y = np.array([0.3, 1.2])
    p = np.array([0.7, 0.8])
    numerical_mse = np.array([
        (mean_squared_error(y, p + np.eye(2)[i] * step) - mean_squared_error(y, p - np.eye(2)[i] * step)) / (2 * step)
        for i in range(2)
    ])
    np.testing.assert_allclose(mean_squared_error_derivative(y, p), numerical_mse, atol=1e-7)
    numerical_mae = np.array([
        (mean_absolute_error(y, p + np.eye(2)[i] * step) - mean_absolute_error(y, p - np.eye(2)[i] * step)) / (2 * step)
        for i in range(2)
    ])
    np.testing.assert_allclose(mae_derivative(y, p), numerical_mae, atol=1e-7)

    y_binary = np.array([1.0, 0.0])
    p_binary = np.array([0.7, 0.3])
    numerical_bce = np.array([
        (binary_cross_entropy(y_binary, p_binary + np.eye(2)[i] * step) - binary_cross_entropy(y_binary, p_binary - np.eye(2)[i] * step)) / (2 * step)
        for i in range(2)
    ])
    np.testing.assert_allclose(binary_cross_entropy_derivative(y_binary, p_binary), numerical_bce, atol=1e-7)

    y_multi = np.array([[1.0, 0.0, 0.0]])
    p_multi = np.array([[0.5, 0.3, 0.2]])
    direction = np.array([[1.0, -1.0, 0.0]])  # stays on the probability simplex
    numerical_directional = (
        categorical_cross_entropy(y_multi, p_multi + step * direction)
        - categorical_cross_entropy(y_multi, p_multi - step * direction)
    ) / (2 * step)
    analytical_directional = np.sum(categorical_cross_entropy_derivative(y_multi, p_multi) * direction)
    assert analytical_directional == pytest.approx(numerical_directional, abs=1e-7)


def test_shape_and_value_validation_errors_are_clear():
    with pytest.raises(ValueError, match="matching shapes"):
        mean_squared_error(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        binary_cross_entropy(np.array([1.0]), np.array([1.2]))
    with pytest.raises(ValueError, match="one-hot"):
        categorical_cross_entropy(np.array([[0.5, 0.5]]), np.array([[0.5, 0.5]]))
    with pytest.raises(ValueError, match="sum to 1"):
        categorical_cross_entropy(np.array([[1.0, 0.0]]), np.array([[0.2, 0.2]]))
