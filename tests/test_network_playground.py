"""Visualization and Streamlit integration checks for the Network Playground."""

import numpy as np
from pathlib import Path
from matplotlib.figure import Figure
from streamlit.testing.v1 import AppTest

from src.neurolab.datasets import make_classification_dataset, split_classification_data
from src.neurolab.models import NeuralNetwork
from src.neurolab.training import train_model
from src.neurolab.visualization import (
    plot_dataset_preview,
    plot_decision_boundary,
    plot_loss_history,
    plot_network_architecture,
)


def test_visualization_helpers_return_figures_for_dataset_architecture_history_and_boundary():
    features, labels = make_classification_dataset("moons", n_samples=80, random_seed=61)
    split = split_classification_data(features, labels, random_seed=61)
    result = train_model(
        split.X_train,
        split.y_train,
        hidden_layer_sizes=(4,),
        epochs=2,
        batch_size=16,
        optimizer="sgd",
        learning_rate=0.02,
        validation_data=split.validation_data,
        random_seed=61,
    )

    assert isinstance(plot_dataset_preview(features, labels), Figure)
    assert isinstance(plot_network_architecture((4, 3), hidden_activation="tanh"), Figure)
    assert isinstance(plot_loss_history(result.history), Figure)
    boundary = plot_decision_boundary(result.network, features, labels, scaler=split.scaler, resolution=30)
    assert isinstance(boundary, Figure)
    assert boundary.axes[0].collections


def test_streamlit_page_renders_and_training_runs_replace_session_results():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    next(button for button in app.button if button.key == "nav_network_playground").click().run()
    assert not app.exception
    assert [title.value for title in app.title] == ["Network Playground"]

    for widget in app.selectbox:
        if widget.key == "play_dataset":
            widget.set_value("XOR")
    for widget in app.slider:
        if widget.key == "play_samples":
            widget.set_value(100)
        if widget.key == "play_hidden_layer_count":
            widget.set_value(0)
    for widget in app.number_input:
        if widget.key == "play_epochs":
            widget.set_value(2)
        if widget.key == "play_batch_size":
            widget.set_value(20)
    next(button for button in app.button if button.key == "play_train_button").click().run()

    assert not app.exception
    first_run = app.session_state["network_playground_result"]
    assert app.session_state["network_playground_run_count"] == 1
    assert first_run["dataset_name"] == "XOR"
    assert len(first_run["result"].history["training_loss"]) == 2
    assert np.isfinite(first_run["result"].history["validation_loss"]).all()

    for widget in app.selectbox:
        if widget.key == "play_dataset":
            widget.set_value("Concentric circles")
    for widget in app.number_input:
        if widget.key == "play_epochs":
            widget.set_value(1)
    next(button for button in app.button if button.key == "play_train_button").click().run()

    assert not app.exception
    second_run = app.session_state["network_playground_result"]
    assert app.session_state["network_playground_run_count"] == 2
    assert second_run["dataset_name"] == "Concentric circles"
    assert len(second_run["result"].history["training_loss"]) == 1
    assert second_run is not first_run
