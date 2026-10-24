"""Tests for the end-to-end mini-batch training pipeline."""

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from src.neurolab.datasets import make_classification_dataset, split_classification_data
from src.neurolab.models import NeuralNetwork
from src.neurolab.models.losses import binary_cross_entropy
from src.neurolab.training import train_model


def test_training_history_has_expected_metrics_for_each_epoch():
    X = np.array([[-1.0], [-0.5], [0.5], [1.0]])
    y = np.array([0, 0, 1, 1])

    result = train_model(
        X,
        y,
        hidden_layer_sizes=(3,),
        epochs=3,
        batch_size=2,
        optimizer="sgd",
        learning_rate=0.1,
        validation_data=(X, y),
        random_seed=4,
    )

    assert result.history["epoch"] == [1, 2, 3]
    assert len(result.history["training_loss"]) == 3
    assert len(result.history["validation_loss"]) == 3
    assert len(result.history["training_accuracy"]) == 3
    assert len(result.history["validation_accuracy"]) == 3
    assert result.epochs_completed == 3


def test_mini_batches_cover_all_samples_including_incomplete_final_batch():
    X = np.arange(10, dtype=float).reshape(5, 2)
    y = np.array([0, 1, 0, 1, 0])
    network = NeuralNetwork(2, [], 1, random_seed=5)
    seen_batches = []
    original_train_step = network.train_step

    def recording_train_step(batch_x, batch_y, optimizer, loss="binary_cross_entropy"):
        seen_batches.append(batch_x.copy())
        return original_train_step(batch_x, batch_y, optimizer, loss)

    network.train_step = recording_train_step
    train_model(X, y, network=network, epochs=1, batch_size=2, optimizer="sgd", learning_rate=0.01)

    assert [len(batch) for batch in seen_batches] == [2, 2, 1]
    covered = np.vstack(seen_batches)
    np.testing.assert_array_equal(np.sort(covered[:, 0]), np.sort(X[:, 0]))


def test_training_is_reproducible_with_fixed_seed():
    X = np.array([[-1.0, 0.0], [-0.5, 0.3], [0.5, -0.1], [1.0, 0.2]])
    y = np.array([0, 0, 1, 1])
    kwargs = dict(hidden_layer_sizes=(4,), epochs=4, batch_size=3, optimizer="adam", random_seed=9)

    first = train_model(X, y, **kwargs)
    second = train_model(X, y, **kwargs)

    np.testing.assert_allclose(first.history["training_loss"], second.history["training_loss"])
    for first_parameter, second_parameter in zip(
        first.network.trainable_parameters, second.network.trainable_parameters
    ):
        np.testing.assert_array_equal(first_parameter, second_parameter)


def test_parameters_change_and_loss_decreases_on_separable_toy_data():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    network = NeuralNetwork(1, [], 1, random_seed=10)
    before = [parameter.copy() for parameter in network.trainable_parameters]

    result = train_model(
        X,
        y,
        network=network,
        epochs=80,
        batch_size=4,
        optimizer="sgd",
        learning_rate=0.2,
        random_seed=10,
    )

    assert result.history["training_loss"][-1] < result.history["training_loss"][0]
    assert any(not np.array_equal(old, new) for old, new in zip(before, network.trainable_parameters))


def test_validation_metrics_use_full_validation_dataset():
    X_train = np.array([[-1.0], [0.0], [1.0]])
    y_train = np.array([0, 0, 1])
    X_validation = np.array([[-2.0], [0.5], [2.0], [3.0]])
    y_validation = np.array([0, 0, 1, 1])
    result = train_model(
        X_train,
        y_train,
        hidden_layer_sizes=(),
        epochs=2,
        batch_size=2,
        optimizer="sgd",
        learning_rate=0.05,
        validation_data=(X_validation, y_validation),
        random_seed=11,
    )

    predictions = result.network.forward(X_validation)
    expected_loss = binary_cross_entropy(y_validation.reshape(-1, 1), predictions)
    expected_accuracy = np.mean((predictions[:, 0] >= 0.5) == y_validation.astype(bool))
    assert result.history["validation_loss"][-1] == pytest.approx(expected_loss)
    assert result.history["validation_accuracy"][-1] == pytest.approx(expected_accuracy)


def test_multiclass_one_hot_accuracy_uses_argmax_and_regression_has_no_accuracy():
    X = np.array([[-1.0], [0.0], [1.0]])
    one_hot = np.eye(3)
    multi_result = train_model(
        X, one_hot, output_size=3, hidden_layer_sizes=(), epochs=1, batch_size=3,
        optimizer="sgd", learning_rate=0.01, random_seed=12,
        validation_data=(X, np.array([0, 1, 2])),
    )
    assert len(multi_result.history["training_accuracy"]) == 1
    assert len(multi_result.history["validation_accuracy"]) == 1

    regression = train_model(
        X, np.array([-0.5, 0.2, 0.8]), hidden_layer_sizes=(), epochs=2, batch_size=2,
        optimizer="sgd", learning_rate=0.01, loss="mse", random_seed=13,
    )
    assert "training_accuracy" not in regression.history
    assert "validation_accuracy" not in regression.history


@pytest.mark.parametrize("name", ["xor", "moons", "circles", "spiral"])
def test_synthetic_dataset_generators_return_finite_binary_data(name):
    X, y = make_classification_dataset(name, n_samples=40, random_seed=14)

    assert X.shape == (40, 2)
    assert y.shape == (40,)
    assert set(np.unique(y)) == {0, 1}
    assert np.isfinite(X).all()


def test_train_validation_split_fits_scaler_on_training_only():
    X = np.arange(60, dtype=float).reshape(30, 2)
    y = np.array([0, 1] * 15)
    split = split_classification_data(X, y, validation_fraction=0.2, random_seed=15)
    raw_train, _, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=15, shuffle=True, stratify=y
    )

    np.testing.assert_allclose(split.scaler.mean_, raw_train.mean(axis=0))
    np.testing.assert_allclose(split.X_train.mean(axis=0), [0.0, 0.0], atol=1e-12)


def test_regression_history_omits_accuracy_and_validation_can_be_absent():
    result = train_model(
        np.array([[0.0], [1.0]]),
        np.array([0.2, 0.8]),
        epochs=1,
        batch_size=1,
        loss="mae",
        optimizer="adam",
        random_seed=16,
    )

    assert "validation_loss" not in result.history
    assert "training_accuracy" not in result.history
    assert len(result.history["training_loss"]) == 1


def test_cancellation_stops_cleanly_at_a_batch_boundary():
    calls = 0

    def should_stop():
        nonlocal calls
        calls += 1
        return calls > 2

    result = train_model(
        np.array([[0.0], [1.0], [2.0], [3.0]]),
        np.array([0, 0, 1, 1]),
        hidden_layer_sizes=(),
        epochs=5,
        batch_size=1,
        should_stop=should_stop,
        random_seed=17,
    )

    assert result.stopped_early
    assert result.epochs_completed == 1
    assert len(result.history["training_loss"]) == 1


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"epochs": 0}, "epochs"),
        ({"batch_size": 0}, "batch_size"),
        ({"learning_rate": 0}, "learning_rate"),
        ({"loss": "categorical_cross_entropy"}, "loss"),
    ],
)
def test_invalid_training_options_raise_clear_errors(kwargs, message):
    defaults = dict(
        X_train=np.array([[0.0], [1.0]]),
        y_train=np.array([0, 1]),
        epochs=1,
        batch_size=1,
        optimizer="sgd",
        random_seed=18,
    )
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=message):
        train_model(**defaults)


def test_invalid_features_and_validation_shapes_raise_clear_errors():
    with pytest.raises(ValueError, match="X_train"):
        train_model(np.array([[np.nan]]), np.array([0]), epochs=1)
    with pytest.raises(ValueError, match="y_train"):
        train_model(np.array([[0.0], [1.0]]), np.array([0]), epochs=1)
    with pytest.raises(ValueError, match="validation_data"):
        train_model(np.array([[0.0]]), np.array([0]), validation_data=(np.array([[0.0]]),), epochs=1)
