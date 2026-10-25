"""State, validation, and full user workflow coverage for the Experiment Wizard."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.neurolab.experiment_store import ExperimentStore
from src.neurolab.experiment_wizard import (
    RESULT_KEY,
    new_wizard_state,
    training_is_authorized,
    validate_wizard_step,
    wizard_next,
    wizard_previous,
)


def _valid_config():
    return {
        "details": {"name": "XOR trial", "seed": 7},
        "dataset": {"name": "XOR", "samples": 100, "noise": 0.1,
                    "class_1_fraction": 0.5, "validation_fraction": 0.2},
        "network": {"hidden_layers": 1, "hidden_layer_sizes": [4],
                    "hidden_activation": "ReLU", "output_activation": "Sigmoid · binary classification",
                    "loss": "Binary cross-entropy"},
        "training": {"optimizer": "Adam", "learning_rate": 0.01, "epochs": 2,
                     "batch_size": 20, "momentum": 0.9, "beta1": 0.9,
                     "beta2": 0.999, "epsilon": 1e-8},
    }


def test_step_validation_rejects_missing_details_and_incompatible_network():
    config = _valid_config()
    config["details"]["name"] = "  "
    assert "Enter an experiment name" in validate_wizard_step(1, config)[0]
    config = _valid_config()
    config["network"]["output_activation"] = "Softmax"
    assert "requires sigmoid" in validate_wizard_step(3, config)[0]


def test_wizard_next_and_previous_preserve_state_and_require_valid_configuration():
    state = new_wizard_state()
    config = _valid_config()
    config["details"]["name"] = ""
    assert wizard_next(state, config)
    assert state["step"] == 0
    config["details"]["name"] = "Valid"
    assert wizard_next(state, config) == ()
    state["details"] = config["details"]
    wizard_previous(state)
    assert state["step"] == 0
    assert state["details"]["name"] == "Valid"
    assert not training_is_authorized(state)
    state.update(step=5, confirmed=True)
    assert training_is_authorized(state)


def test_wizard_trains_only_after_confirmation_and_saves_once(tmp_path, monkeypatch):
    database = tmp_path / "wizard.sqlite3"
    monkeypatch.setenv("NEUROLAB_EXPERIMENT_DB", str(database))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=60).run()
    _click(app, "nav_experiment_wizard")
    assert app.session_state["experiment_wizard"]["step"] == 0
    assert RESULT_KEY not in app.session_state

    # Required name blocks the first transition and configuration survives Back.
    _click(app, "wizard_next_0")
    assert app.session_state["experiment_wizard"]["step"] == 0
    _set_text(app, "wizard_name", "Guided XOR run")
    _set_number(app, "wizard_seed", 33)
    _click(app, "wizard_next_0")
    assert app.session_state["experiment_wizard"]["step"] == 1
    _click(app, "wizard_back_1")
    assert app.session_state["experiment_wizard"]["details"]["name"] == "Guided XOR run"
    _click(app, "wizard_next_0")

    _set_slider(app, "wizard_samples", 40)
    _click(app, "wizard_next_1")
    _set_slider(app, "wizard_layers", 0)
    _click(app, "wizard_next_2")
    _set_number(app, "wizard_epochs", 1)
    _set_number(app, "wizard_batch_size", 20)
    _click(app, "wizard_next_3")
    assert app.session_state["experiment_wizard"]["step"] == 4
    assert RESULT_KEY not in app.session_state

    _click(app, "wizard_confirm")
    assert app.session_state["experiment_wizard"]["step"] == 5
    assert RESULT_KEY not in app.session_state
    _click(app, "wizard_train")
    assert not app.exception
    assert app.session_state["experiment_wizard"]["step"] == 6
    completed_result = app.session_state[RESULT_KEY]
    assert completed_result["result"].epochs_completed == 1
    assert completed_result["result"].history["validation_accuracy"]

    # A regular Streamlit rerun preserves the result without invoking training again.
    app.run()
    assert app.session_state[RESULT_KEY] is completed_result
    _click(app, "wizard_save_bottom")
    assert any(button.key == "wizard_save" for button in app.button)
    _click(app, "wizard_save")
    assert not app.exception
    assert len(ExperimentStore(database).list()) == 1
    assert app.session_state["experiment_wizard"]["saved_id"] == 1

    # Reopening the save step presents the saved state, not a second save action.
    _click(app, "wizard_back_7")
    _click(app, "wizard_save_bottom")
    assert not any(button.key == "wizard_save" for button in app.button)
    assert len(ExperimentStore(database).list()) == 1


def _click(app, key):
    next(button for button in app.button if button.key == key).click().run()


def _set_text(app, key, value):
    next(widget for widget in app.text_input if widget.key == key).set_value(value).run()


def _set_number(app, key, value):
    next(widget for widget in app.number_input if widget.key == key).set_value(value).run()


def _set_slider(app, key, value):
    widgets = list(app.slider) + list(app.select_slider)
    next(widget for widget in widgets if widget.key == key).set_value(value).run()
