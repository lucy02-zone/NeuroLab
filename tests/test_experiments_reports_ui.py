from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.neurolab.experiment_store import ExperimentStore


def test_network_playground_can_save_a_completed_training_run(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROLAB_EXPERIMENT_DB", str(tmp_path / "network.sqlite3"))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    app.radio[0].set_value("Network Playground").run()
    _set_slider(app, "play_hidden_layer_count", 0)
    _set_number(app, "play_epochs", 1)
    _set_number(app, "play_batch_size", 20)
    _set_slider(app, "play_samples", 100)
    _click(app, "play_train_button")
    assert not app.exception
    _click(app, "network_playground_save")
    assert not app.exception
    records = ExperimentStore(tmp_path / "network.sqlite3").list()
    assert len(records) == 1
    assert records[0]["dataset"] == "XOR"
    assert records[0]["metrics"]["epochs_completed"] == 1


def test_optimizer_lab_can_save_a_real_optimizer_run(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROLAB_EXPERIMENT_DB", str(tmp_path / "optimizer.sqlite3"))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    app.radio[0].set_value("Optimizer Lab").run()
    for widget in app.multiselect:
        if widget.key == "optimizer_selection":
            widget.set_value(["SGD"])
    _set_slider(app, "optimizer_samples", 100)
    _set_slider(app, "optimizer_layers", 0)
    _set_number(app, "optimizer_epochs", 1)
    _set_number(app, "optimizer_batch", 20)
    _click(app, "optimizer_run")
    assert not app.exception
    _click(app, "optimizer_lab_save")
    assert not app.exception
    records = ExperimentStore(tmp_path / "optimizer.sqlite3").list()
    assert len(records) == 1
    assert records[0]["configuration"]["optimizer"] == "SGD"
    assert len(records[0]["history"]["training_loss"]) == 1


def test_reports_page_requires_explicit_delete_confirmation(tmp_path, monkeypatch):
    database = tmp_path / "reports.sqlite3"
    monkeypatch.setenv("NEUROLAB_EXPERIMENT_DB", str(database))
    store = ExperimentStore(database)
    record_id = store.save(
        name="Report trial", dataset="XOR",
        dataset_configuration={"n_samples": 30, "validation_fraction": 0.2, "split_seed": 3},
        configuration={"optimizer": "Adam", "hidden_layer_sizes": [3], "hidden_activation": "relu",
                       "learning_rate": 0.01, "batch_size": 8, "loss": "binary_cross_entropy"},
        history={"epoch": [1], "training_loss": [0.6], "validation_loss": [0.7],
                 "training_accuracy": [0.6], "validation_accuracy": [0.5]},
    )
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    app.radio[0].set_value("Experiments & Reports").run()
    assert not app.exception
    assert [title.value for title in app.title] == ["Experiments & Reports"]
    assert app.get("download_button")
    _click(app, "reports_delete_request")
    assert not app.exception
    assert ExperimentStore(database).get(record_id) is not None
    _click(app, "reports_delete_confirm")
    assert not app.exception
    assert ExperimentStore(database).get(record_id) is None


def _click(app, key):
    next(button for button in app.button if button.key == key).click().run()


def _set_number(app, key, value):
    for widget in app.number_input:
        if widget.key == key:
            widget.set_value(value)
            return
    raise AssertionError(f"Number input {key!r} was not found.")


def _set_slider(app, key, value):
    for widget in app.slider:
        if widget.key == key:
            widget.set_value(value)
            return
    raise AssertionError(f"Slider {key!r} was not found.")
