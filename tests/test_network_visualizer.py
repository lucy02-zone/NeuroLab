from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure
from streamlit.testing.v1 import AppTest

from src.neurolab.models import NeuralNetwork
from src.neurolab.network_visualizer import (
    architecture_summary,
    inspect_forward_pass,
    plot_network_diagram,
)


def test_architecture_summary_reflects_model_layer_shapes_and_activations():
    network = NeuralNetwork(2, [4, 3], 1, random_seed=32, hidden_activation="tanh")

    layers = architecture_summary(network)

    assert [layer["neurons"] for layer in layers] == [2, 4, 3, 1]
    assert [layer["activation"] for layer in layers] == ["Input values", "tanh", "tanh", "Sigmoid"]
    assert layers[1]["weight_shape"] == network.weights[0].shape
    assert layers[1]["bias_shape"] == network.biases[0].shape
    assert layers[-1]["weight_shape"] == network.weights[-1].shape


def test_forward_inspection_returns_the_networks_real_cached_values():
    network = NeuralNetwork(2, [3], 1, random_seed=12)
    sample = np.array([0.25, -0.4])

    trace = inspect_forward_pass(network, sample)

    np.testing.assert_allclose(trace["input"], network.activations[0][:, 0])
    np.testing.assert_allclose(trace["weighted_sums"][0], network.weighted_sums[0][:, 0])
    np.testing.assert_allclose(trace["activations"][0], network.activations[1][:, 0])
    np.testing.assert_allclose(trace["prediction"], network.forward(sample)[0])


def test_forward_inspection_rejects_multiple_samples_and_wrong_feature_count():
    network = NeuralNetwork(2, [2], 1, random_seed=4)

    with pytest.raises(ValueError, match="exactly one sample"):
        inspect_forward_pass(network, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="shape"):
        inspect_forward_pass(network, np.zeros(3))


def test_diagram_uses_actual_connections_and_marks_simplified_layers():
    network = NeuralNetwork(2, [10], 1, random_seed=7)

    figure, simplified = plot_network_diagram(network, max_neurons=4)

    assert isinstance(figure, Figure)
    assert simplified == ((1, 10),)
    # The input-to-hidden and hidden-to-output diagrams each use only visible nodes.
    assert len(figure.axes[0].lines) == 2 * 4 + 4 * 1


def test_visualizer_page_initializes_and_runs_a_forward_pass_without_training():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    _click(app, "nav_network_visualizer")
    assert not app.exception
    assert [title.value for title in app.title] == ["Network Visualizer"]

    _click(app, "visualizer_initialize")
    assert not app.exception
    network = app.session_state["network_visualizer_new_network"]
    initial_parameters = [parameter.copy() for parameter in network.trainable_parameters]

    _click(app, "visualizer_forward")
    assert not app.exception
    saved_trace = app.session_state["network_visualizer_forward_trace"]
    np.testing.assert_allclose(saved_trace["trace"]["prediction"], network.activations[-1][:, 0])
    stepper = next(widget for widget in app.select_slider if widget.key == "visualizer_step")
    stepper.set_value(1).run()
    assert not app.exception
    assert app.session_state["visualizer_step"] == 1
    assert all(
        np.array_equal(before, after)
        for before, after in zip(initial_parameters, network.trainable_parameters)
    )


def _click(app, key):
    next(button for button in app.button if button.key == key).click().run()
