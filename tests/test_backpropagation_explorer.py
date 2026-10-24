from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from src.neurolab.backpropagation_explorer import plot_backpropagation_network
from src.neurolab.models import NeuralNetwork


def test_backward_exposes_real_chain_rule_intermediates_in_sample_major_layout():
    network = NeuralNetwork(2, [3, 2], 1, random_seed=18, hidden_activation="tanh")
    sample = np.array([0.25, -0.75])
    target = np.array([1.0])
    prediction = network.forward(sample)
    gradients = network.backward(target)

    assert len(gradients["activation_gradients"]) == 3
    assert len(gradients["activation_derivatives"]) == 3
    assert len(gradients["preactivation_gradients"]) == 3
    for index in range(3):
        shape = (1, network.layer_sizes[index + 1])
        assert gradients["activation_gradients"][index].shape == shape
        assert gradients["activation_derivatives"][index].shape == shape
        assert gradients["preactivation_gradients"][index].shape == shape
    np.testing.assert_allclose(
        gradients["preactivation_gradients"][-1],
        gradients["activation_gradients"][-1] * gradients["activation_derivatives"][-1],
    )
    np.testing.assert_allclose(
        gradients["preactivation_gradients"][0],
        (gradients["preactivation_gradients"][1] @ network.weights[1]) * gradients["activation_derivatives"][0],
    )
    assert prediction.shape == (1, 1)
    assert np.isfinite(gradients["activation_gradients"][-1]).all()


def test_network_diagram_uses_initialized_forward_values():
    network = NeuralNetwork(2, [2], 1, random_seed=19)
    network.forward(np.array([0.3, -0.2]))

    figure = plot_backpropagation_network(network, np.array([0.3, -0.2]))

    assert figure.axes[0].collections
    assert len(figure.axes[0].lines) == sum(weight.size for weight in network.weights)


def test_streamlit_backprop_workflow_only_updates_on_explicit_request():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    app.radio[0].set_value("Backpropagation Explorer").run()
    assert not app.exception
    assert [title.value for title in app.title] == ["Backpropagation Explorer"]

    _click(app, "backprop_initialize")
    assert not app.exception
    state = app.session_state["backpropagation_explorer_state"]
    original = [parameter.copy() for parameter in state["network"].trainable_parameters]

    _click(app, "backprop_forward")
    assert not app.exception
    forward_prediction = float(state["network"].activations[-1][0, 0])
    assert state["stage"] == "forward"

    _click(app, "backprop_loss")
    assert not app.exception
    assert state["stage"] == "loss"
    assert np.isfinite(state["loss_value"])

    _click(app, "backprop_gradients")
    assert not app.exception
    assert state["stage"] == "gradients"
    assert all(np.array_equal(old, current) for old, current in zip(original, state["network"].trainable_parameters))
    gradients = [*state["gradients"]["weight_gradients"], *state["gradients"]["bias_gradients"]]

    _click(app, "backprop_update")
    assert not app.exception
    assert state["stage"] == "updated"
    for old, gradient, current in zip(original, gradients, state["network"].trainable_parameters):
        np.testing.assert_allclose(current, old - 0.05 * gradient)
    assert state["prediction_before_update"] == forward_prediction

    _click(app, "backprop_evaluate_after")
    assert not app.exception
    assert state["stage"] == "updated_forward"
    expected = float(state["network"].activations[-1][0, 0])
    assert state["prediction_after_update"] == expected
    assert np.isfinite(state["loss_after_update"])


def _click(app: AppTest, key: str) -> None:
    button = next(button for button in app.button if button.key == key)
    button.click().run()
