import numpy as np
from pathlib import Path
from streamlit.testing.v1 import AppTest

from src.neurolab.datasets import make_classification_dataset
from src.neurolab.optimizer_comparison import run_optimizer_comparison
from src.neurolab.visualization import plot_optimizer_history


def _comparison(optimizers=("SGD", "Momentum", "Adam")):
    features, labels = make_classification_dataset("moons", n_samples=80, noise=0.05, random_seed=17)
    return run_optimizer_comparison(
        features,
        labels,
        optimizers,
        hidden_layer_sizes=(5,),
        epochs=4,
        batch_size=13,
        learning_rate=0.02,
        validation_fraction=0.25,
        random_seed=23,
    )


def test_comparison_runs_only_selected_optimizers_with_real_history():
    comparison = _comparison(("SGD", "Adam"))

    assert tuple(comparison.runs) == ("SGD", "Adam")
    assert all(result.epochs_completed == 4 for result in comparison.runs.values())
    assert all(len(result.history["training_loss"]) == 4 for result in comparison.runs.values())
    assert all(len(result.history["validation_loss"]) == 4 for result in comparison.runs.values())
    assert all(len(result.history["validation_accuracy"]) == 4 for result in comparison.runs.values())


def test_models_start_equally_but_optimizer_objects_and_states_are_independent():
    comparison = _comparison()
    names = tuple(comparison.runs)
    first_weights, first_biases = comparison.initial_parameters[names[0]]
    for name in names[1:]:
        weights, biases = comparison.initial_parameters[name]
        assert all(np.array_equal(a, b) for a, b in zip(first_weights, weights))
        assert all(np.array_equal(a, b) for a, b in zip(first_biases, biases))
    assert len({id(result.network) for result in comparison.runs.values()}) == len(names)
    assert len({id(result.optimizer) for result in comparison.runs.values()}) == len(names)
    assert all(np.isfinite(result.history["training_loss"]).all() for result in comparison.runs.values())


def test_fixed_seed_repeats_training_history():
    first, second = _comparison(("SGD", "Adam")), _comparison(("SGD", "Adam"))
    for name in first.runs:
        for metric in first.runs[name].history:
            assert np.allclose(first.runs[name].history[metric], second.runs[name].history[metric])


def test_optimizer_comparison_plot_uses_recorded_history():
    comparison = _comparison(("SGD", "Adam"))
    figure = plot_optimizer_history(comparison.runs, "training_loss", "Training loss by epoch")
    assert [line.get_label() for line in figure.axes[0].lines] == ["SGD", "Adam"]
    for line, name in zip(figure.axes[0].lines, comparison.runs):
        assert np.allclose(line.get_ydata(), comparison.runs[name].history["training_loss"])


def test_optimizer_lab_runs_selected_subset_and_keeps_results_on_widget_rerun():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    next(button for button in app.button if button.key == "nav_optimizer_lab").click().run()
    assert not app.exception
    assert [title.value for title in app.title] == ["Optimizer Lab"]

    for widget in app.multiselect:
        if widget.key == "optimizer_selection":
            widget.set_value(["SGD", "Adam"])
    for widget in app.slider:
        if widget.key == "optimizer_samples":
            widget.set_value(100)
        if widget.key == "optimizer_layers":
            widget.set_value(0)
    for widget in app.number_input:
        if widget.key == "optimizer_epochs":
            widget.set_value(2)
        if widget.key == "optimizer_batch":
            widget.set_value(20)
    next(button for button in app.button if button.key == "optimizer_run").click().run()

    assert not app.exception
    saved = app.session_state["optimizer_lab_result"]
    assert tuple(saved["comparison"].runs) == ("SGD", "Adam")
    assert app.session_state["optimizer_lab_run_count"] == 1
    assert all(result.epochs_completed == 2 for result in saved["comparison"].runs.values())

    app.run()
    assert app.session_state["optimizer_lab_run_count"] == 1
