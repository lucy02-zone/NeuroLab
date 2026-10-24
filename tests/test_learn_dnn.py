from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.neurolab.learn_dnn import (
    MODULES,
    MODULE_IDS,
    classification_metrics,
    evaluate_quiz,
    forward_example,
    update_progress,
)


def test_forward_example_matches_layer_equations_and_interactive_inputs_change_output():
    result = forward_example(np.array([0.4, -0.2]), hidden_bias=0.1)
    network = result["network"]
    z_hidden = np.array([0.5 * 0.4 - 0.25 * -0.2 + 0.1, -0.4 * 0.4 + 0.6 * -0.2 - 0.1])
    np.testing.assert_allclose(result["hidden_z"].ravel(), z_hidden)
    np.testing.assert_allclose(result["hidden_a"].ravel(), np.tanh(z_hidden))
    expected_output_logit = 0.7 * np.tanh(z_hidden[0]) - 0.3 * np.tanh(z_hidden[1]) + 0.05
    assert result["output_z"][0, 0] == pytest.approx(expected_output_logit)
    changed = forward_example(np.array([-0.4, 0.7]), hidden_bias=0.1)
    assert changed["prediction"][0, 0] != result["prediction"][0, 0]
    assert network.activations[-1].shape == (1, 1)


def test_quiz_evaluation_and_progress_only_mark_correct_answers():
    assert evaluate_quiz("A", "A")
    assert not evaluate_quiz("B", "A")
    assert update_progress([], MODULE_IDS[0], False) == set()
    assert update_progress([], MODULE_IDS[0], True) == {MODULE_IDS[0]}
    assert update_progress([MODULE_IDS[0]], "unknown", True) == {MODULE_IDS[0]}


def test_binary_classification_metrics_and_confusion_matrix():
    result = classification_metrics(np.array([1, 0, 1, 1, 0, 0]), np.array([1, 1, 0, 1, 0, 0]))

    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(2 / 3)
    np.testing.assert_array_equal(result["confusion_matrix"], [[2, 1], [1, 2]])


def test_every_learning_module_renders_and_correct_quiz_updates_session_progress():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    next(button for button in app.button if button.key == "nav_learn_dnn").click().run()
    assert not app.exception
    assert [title.value for title in app.title] == ["Learn DNN"]

    for module in MODULES:
        next(widget for widget in app.selectbox if widget.key == "learn_dnn_module").set_value(module).run()
        assert not app.exception, f"Module failed to render: {module}"
        assert [title.value for title in app.title] == ["Learn DNN"]

    next(widget for widget in app.selectbox if widget.key == "learn_dnn_module").set_value(MODULES[0]).run()
    next(widget for widget in app.radio if widget.key == "learn_quiz_module_1").set_value("Shift its activation threshold")
    next(button for button in app.button if button.key == "learn_check_module_1").click().run()
    assert not app.exception
    assert "module_1" in app.session_state["learn_dnn_completed_modules"]
    next(button for button in app.button if button.key == "learn_link_network_1").click().run()
    assert not app.exception
    assert [title.value for title in app.title] == ["Network Playground"]
