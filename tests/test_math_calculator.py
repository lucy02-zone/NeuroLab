"""Reusable arithmetic and UI checks for the Neural Network Math Calculator."""

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.neurolab.math_calculator import (
    calculate_loss_details,
    calculate_manual_gradients,
    calculate_single_neuron,
    apply_calculator_update,
    manual_forward_propagation,
    parse_numeric_vector,
    preview_parameter_update,
)
from src.neurolab.models import Adam, NeuralNetwork, SGD, SGDMomentum
from src.neurolab.models import activations, losses


def test_single_neuron_weighted_sum_trace_and_activation_use_existing_functions():
    result = calculate_single_neuron([2.0, -1.0], [0.5, 2.0], 0.5, "Tanh")
    assert result["products"].tolist() == [1.0, -2.0]
    assert result["running_totals"].tolist() == [1.0, -1.0]
    assert result["weighted_sum"] == -0.5
    assert result["output"] == pytest.approx(activations.tanh(-0.5))


def test_single_neuron_rejects_mismatched_vectors_and_nonfinite_values():
    with pytest.raises(ValueError, match="same length"):
        calculate_single_neuron([1.0, 2.0], [1.0], 0.0, "ReLU")
    with pytest.raises(ValueError, match="finite"):
        calculate_single_neuron([np.inf], [1.0], 0.0, "ReLU")
    with pytest.raises(ValueError, match="numeric"):
        parse_numeric_vector("1.0, bad", "input")


def test_manual_forward_propagation_matches_direct_numpy_calculation():
    inputs = np.array([1.0, 2.0])
    weights = [np.array([[1.0, 2.0], [-1.0, 1.0]]), np.array([[0.2, 0.3]])]
    biases = [np.array([[0.5], [0.0]]), np.array([[0.1]])]
    result = manual_forward_propagation(inputs, weights, biases, "ReLU")
    hidden = np.maximum(0.0, weights[0] @ inputs.reshape(-1, 1) + biases[0])
    expected_logits = weights[1] @ hidden + biases[1]
    expected_output = 1.0 / (1.0 + np.exp(-expected_logits))
    np.testing.assert_allclose(result["layers"][0]["weighted_sum"], [5.5, 1.0])
    np.testing.assert_allclose(result["prediction"], expected_output[:, 0])
    assert result["layers"][0]["weights"].shape == (2, 2)
    assert result["layers"][1]["bias"].shape == (1,)


@pytest.mark.parametrize("name,function,targets,predictions", [
    ("Mean Squared Error", losses.mean_squared_error, [1.0, 0.0], [0.7, 0.3]),
    ("Mean Absolute Error", losses.mean_absolute_error, [1.0, 0.0], [0.7, 0.3]),
    ("Binary Cross-Entropy", losses.binary_cross_entropy, [1.0, 0.0], [0.7, 0.3]),
    ("Categorical Cross-Entropy", losses.categorical_cross_entropy, [[1.0, 0.0]], [[0.7, 0.3]]),
])
def test_loss_calculator_matches_shared_loss_functions(name, function, targets, predictions):
    result = calculate_loss_details(name, np.asarray(targets), np.asarray(predictions))
    assert result["loss"] == pytest.approx(function(np.asarray(targets), np.asarray(predictions)))
    assert result["formula"]
    assert result["substituted"]
    assert result["terms"].size > 0


def test_manual_gradients_match_existing_network_engine_and_shapes():
    x = np.array([0.4, -0.7])
    weights = [np.array([[0.2, -0.1], [0.5, 0.3]]), np.array([[0.6, -0.2]])]
    biases = [np.array([[0.1], [-0.3]]), np.array([[0.05]])]
    result = calculate_manual_gradients(x, [1.0], weights, biases, "Tanh", "Binary Cross-Entropy")
    reference = NeuralNetwork(2, (2,), 1, random_seed=3, hidden_activation="tanh")
    reference.weights = [value.copy() for value in weights]
    reference.biases = [value.copy() for value in biases]
    expected_prediction = reference.forward(x)
    expected_gradients = reference.backward(np.array([[1.0]]), loss="binary_cross_entropy")
    np.testing.assert_allclose(result["prediction"], expected_prediction)
    for expected, actual in zip(expected_gradients["weight_gradients"], result["gradients"]["weight_gradients"]):
        np.testing.assert_allclose(expected, actual)
    for gradient, weight in zip(result["gradients"]["weight_gradients"], weights):
        assert gradient.shape == weight.shape
        assert np.isfinite(gradient).all()
    assert result["input_gradient"].shape == x.shape


def test_categorical_gradient_is_rejected_for_sigmoid_output_network():
    with pytest.raises(ValueError, match="not supported"):
        calculate_manual_gradients([1.0], [1.0, 0.0], [np.ones((2, 1))], [np.zeros((2, 1))],
                                   "ReLU", "Categorical Cross-Entropy")


def test_parameter_update_previews_use_existing_optimizers_without_mutating_state():
    sgd_value, sgd = preview_parameter_update(1.0, 0.2, "SGD", 0.1)
    assert isinstance(sgd, SGD)
    assert sgd_value == pytest.approx(0.98)

    momentum = SGDMomentum(0.1, momentum=0.9)
    first_value, first_state = preview_parameter_update(1.0, 0.2, "Momentum", 0.1, momentum=0.9, optimizer_state=momentum)
    second_value, _ = preview_parameter_update(first_value, 0.2, "Momentum", 0.1, momentum=0.9, optimizer_state=first_state)
    assert first_value == pytest.approx(0.98)
    assert second_value == pytest.approx(0.942)
    assert momentum.velocities == []  # preview did not change the supplied optimizer

    adam_value, adam = preview_parameter_update(1.0, 0.2, "Adam", 0.1)
    assert isinstance(adam, Adam)
    assert adam_value == pytest.approx(0.9, abs=1e-7)


def test_explicit_update_changes_only_the_calculator_copy():
    weights = [np.array([[1.0, 2.0]])]
    biases = [np.array([[0.5]])]
    original_weight = weights[0].copy()
    updated_weights, updated_biases = apply_calculator_update(
        weights, biases, "Weight", 0, 0, 1, 0.25, SGD(0.1)
    )
    np.testing.assert_array_equal(weights[0], original_weight)
    np.testing.assert_allclose(updated_weights[0], [[1.0, 1.975]])
    np.testing.assert_array_equal(updated_biases[0], biases[0])


def test_math_calculator_is_available_from_navigation_and_persists_guided_step():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    next(button for button in app.button if button.key == "nav_math_calculator").click().run()
    assert not app.exception
    assert [title.value for title in app.title] == ["Neural Network Math Calculator"]
    assert len(app.tabs) == 5
    next(button for button in app.button if button.key == "mathcalc_guide_next").click().run()
    assert app.session_state["mathcalc_guide_step"] == 1
    assert not app.exception
