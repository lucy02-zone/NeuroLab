"""Dataset generation and Explorer UI coverage."""

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.neurolab.dataset_explorer import calculate_dataset_statistics
from src.neurolab.datasets import make_classification_dataset, split_classification_data


@pytest.mark.parametrize("name", ("xor", "moons", "circles", "spiral"))
def test_dataset_generators_are_two_dimensional_reproducible_and_respect_balance(name):
    first_X, first_y = make_classification_dataset(
        name, n_samples=120, noise=0.07, random_seed=29, class_1_fraction=0.65,
        circle_factor=0.4, spiral_turns=3.0,
    )
    second_X, second_y = make_classification_dataset(
        name, n_samples=120, noise=0.07, random_seed=29, class_1_fraction=0.65,
        circle_factor=0.4, spiral_turns=3.0,
    )
    assert first_X.shape == (120, 2)
    assert first_y.shape == (120,)
    assert np.array_equal(first_X, second_X)
    assert np.array_equal(first_y, second_y)
    assert np.count_nonzero(first_y == 1) == 78
    assert np.isfinite(first_X).all()


def test_dataset_statistics_reflect_actual_values_and_missing_values():
    X = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
    y = np.array([0, 1, 1])
    stats = calculate_dataset_statistics(X, y)
    assert stats["samples"] == 3
    assert stats["features"] == 2
    assert stats["classes"] == 2
    assert stats["class_counts"] == {0: 1, 1: 2}
    assert stats["missing_values"] == 1
    assert np.allclose(stats["mean"], [3.0, 4.0])
    assert np.allclose(stats["minimum"], [1.0, 2.0])
    assert np.allclose(stats["maximum"], [5.0, 6.0])


def test_split_is_reproducible_stratified_and_training_only_scaler():
    X, y = make_classification_dataset("moons", n_samples=200, noise=0.1, random_seed=7)
    split = split_classification_data(X, y, validation_fraction=0.25, random_seed=13, standardize=True)
    repeated = split_classification_data(X, y, validation_fraction=0.25, random_seed=13, standardize=True)
    unscaled = split_classification_data(X, y, validation_fraction=0.25, random_seed=13, standardize=False)
    assert split.X_train.shape[0] == 150
    assert split.X_validation.shape[0] == 50
    assert np.array_equal(split.X_train, repeated.X_train)
    assert np.array_equal(split.y_validation, repeated.y_validation)
    assert np.allclose(np.mean(split.X_train, axis=0), 0, atol=1e-12)
    assert split.scaler is not None
    assert np.allclose(split.scaler.mean_, np.mean(unscaled.X_train, axis=0))


def test_invalid_dataset_configuration_raises_clear_errors():
    with pytest.raises(ValueError, match="class_1_fraction"):
        make_classification_dataset("xor", n_samples=40, class_1_fraction=1.0)
    with pytest.raises(ValueError, match="circle_factor"):
        make_classification_dataset("circles", circle_factor=1.0)
    with pytest.raises(ValueError, match="spiral_turns"):
        make_classification_dataset("spiral", spiral_turns=0)


def test_dataset_explorer_page_renders_and_transfer_loads_network_playground():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    next(button for button in app.button if button.key == "nav_dataset_explorer").click().run()
    assert not app.exception
    assert [title.value for title in app.title] == ["Dataset Explorer"]
    next(button for button in app.button if button.key == "explorer_transfer").click().run()
    assert not app.exception
    assert [title.value for title in app.title] == ["Network Playground"]
    assert app.session_state["play_dataset"] == "XOR"
    assert app.session_state["play_samples"] == 300
    assert app.session_state["play_class_balance"] == "50% class 1"
