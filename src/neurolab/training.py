"""Mini-batch training orchestration for the NeuroLab NumPy network."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .models import NeuralNetwork
from .models import losses
from .models.optimizers import Adam, Optimizer, SGD, SGDMomentum


_LOSS_ALIASES = {
    "binary_cross_entropy": "binary_cross_entropy",
    "bce": "binary_cross_entropy",
    "mean_squared_error": "mean_squared_error",
    "mse": "mean_squared_error",
    "mean_absolute_error": "mean_absolute_error",
    "mae": "mean_absolute_error",
}
_LOSS_FUNCTIONS = {
    "binary_cross_entropy": losses.binary_cross_entropy,
    "mean_squared_error": losses.mean_squared_error,
    "mean_absolute_error": losses.mean_absolute_error,
}


@dataclass
class TrainingResult:
    """Trained model, optimizer state, and per-epoch metrics."""

    network: NeuralNetwork
    optimizer: Optimizer
    history: dict[str, list[int | float]]
    epochs_completed: int
    stopped_early: bool
    class_values: np.ndarray | None = None


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    network: NeuralNetwork | None = None,
    hidden_layer_sizes: Sequence[int] = (8, 8),
    hidden_activation: str = "relu",
    output_size: int | None = None,
    epochs: int = 100,
    batch_size: int = 32,
    optimizer: str | Optimizer = "adam",
    learning_rate: float | None = None,
    momentum: float = 0.9,
    beta1: float = 0.9,
    beta2: float = 0.999,
    optimizer_epsilon: float = 1e-8,
    loss: str = "binary_cross_entropy",
    validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    random_seed: int | None = 0,
    should_stop: Callable[[], bool] | None = None,
) -> TrainingResult:
    """Fit one network using shuffled mini-batches and collect whole-set metrics.

    ``binary_cross_entropy`` supports binary scalar targets (threshold at 0.5)
    and one-hot multiclass targets (predicted and true labels use ``argmax``
    across independent sigmoid outputs). Multi-output sigmoid values are not
    normalized probabilities. MSE and MAE are treated as regression losses,
    so no accuracy is reported for them. The current network has no softmax
    head; categorical cross-entropy is therefore not accepted here.

    If a ``network`` is supplied, its architecture and hidden activation are
    used. Otherwise they are created from ``hidden_layer_sizes``,
    ``hidden_activation``, and the inferred or explicit ``output_size``.
    When validation data is supplied, target classes are mapped using the
    training labels only. Feature preprocessing should be fitted beforehand
    with :func:`neurolab.datasets.split_classification_data`.

    ``should_stop`` is an optional cancellation callback checked between
    mini-batches. A partially completed epoch is evaluated against the full
    training and validation datasets before returning.
    """
    epoch_count = _positive_integer(epochs, "epochs")
    batch_width = _positive_integer(batch_size, "batch_size")
    seed = _validate_seed(random_seed)
    if should_stop is not None and not callable(should_stop):
        raise ValueError("should_stop must be callable or None.")
    loss_name = _normalize_loss(loss)
    X_values = _validate_features(X_train, "X_train")
    raw_y = _validate_targets(y_train, X_values.shape[0], "y_train")

    requested_output_size = output_size if output_size is not None else (network.output_size if network else None)
    y_values, inferred_output_size, class_values = _prepare_targets(
        raw_y, loss_name, requested_output_size, name="y_train"
    )
    resolved_output_size = requested_output_size or inferred_output_size
    if output_size is not None:
        resolved_output_size = _positive_integer(output_size, "output_size")

    if network is None:
        network = NeuralNetwork(
            input_feature_count=X_values.shape[1],
            hidden_layer_sizes=hidden_layer_sizes,
            output_size=resolved_output_size,
            random_seed=seed,
            hidden_activation=hidden_activation,
        )
    else:
        if network.input_feature_count != X_values.shape[1]:
            raise ValueError(
                f"X_train has {X_values.shape[1]} features but the network expects {network.input_feature_count}."
            )
        if network.output_size != resolved_output_size:
            raise ValueError(
                f"Targets have {resolved_output_size} outputs but the network expects {network.output_size}."
            )

    X_validation: np.ndarray | None = None
    y_validation: np.ndarray | None = None
    if validation_data is not None:
        if not isinstance(validation_data, (tuple, list)) or len(validation_data) != 2:
            raise ValueError("validation_data must be a pair (X_validation, y_validation).")
        X_validation = _validate_features(validation_data[0], "X_validation", X_values.shape[1])
        raw_validation_y = _validate_targets(validation_data[1], X_validation.shape[0], "y_validation")
        y_validation, validation_output_size, _ = _prepare_targets(
            raw_validation_y,
            loss_name,
            network.output_size,
            name="y_validation",
            known_classes=class_values,
        )
        if validation_output_size != network.output_size:
            raise ValueError("Validation target output dimensions do not match the network.")

    if isinstance(optimizer, Optimizer):
        selected_optimizer = optimizer
        if learning_rate is not None and not np.isclose(learning_rate, optimizer.learning_rate):
            raise ValueError("When passing an optimizer instance, configure its learning_rate on that instance.")
    elif isinstance(optimizer, str):
        name = optimizer.strip().lower().replace("-", "_").replace(" ", "_")
        default_lr = 0.001 if name == "adam" else 0.01
        rate = default_lr if learning_rate is None else learning_rate
        _validate_learning_rate(rate)
        if name == "sgd":
            selected_optimizer = SGD(rate)
        elif name in ("momentum", "sgd_momentum", "sgdwithmomentum"):
            selected_optimizer = SGDMomentum(rate, momentum=momentum)
        elif name == "adam":
            selected_optimizer = Adam(rate, beta1=beta1, beta2=beta2, epsilon=optimizer_epsilon)
        else:
            raise ValueError("optimizer must be 'sgd', 'momentum', 'adam', or an Optimizer instance.")
    else:
        raise ValueError("optimizer must be 'sgd', 'momentum', 'adam', or an Optimizer instance.")

    rng = np.random.default_rng(seed)
    history: dict[str, list[int | float]] = {"epoch": [], "training_loss": []}
    is_classification = loss_name == "binary_cross_entropy"
    if X_validation is not None:
        history["validation_loss"] = []
    if is_classification:
        history["training_accuracy"] = []
        if X_validation is not None:
            history["validation_accuracy"] = []

    stopped_early = False
    for epoch_index in range(epoch_count):
        permutation = rng.permutation(X_values.shape[0])
        processed_batch = False
        for start in range(0, X_values.shape[0], batch_width):
            if should_stop is not None and should_stop():
                stopped_early = True
                break
            batch_indices = permutation[start : start + batch_width]
            network.train_step(X_values[batch_indices], y_values[batch_indices], selected_optimizer, loss=loss_name)
            processed_batch = True
        if not processed_batch:
            break

        training_loss, training_predictions = _evaluate(network, X_values, y_values, loss_name, "training")
        epoch_number = epoch_index + 1
        history["epoch"].append(epoch_number)
        history["training_loss"].append(training_loss)
        if is_classification:
            history["training_accuracy"].append(_classification_accuracy(y_values, training_predictions))
        if X_validation is not None and y_validation is not None:
            validation_loss, validation_predictions = _evaluate(
                network, X_validation, y_validation, loss_name, "validation"
            )
            history["validation_loss"].append(validation_loss)
            if is_classification:
                history["validation_accuracy"].append(
                    _classification_accuracy(y_validation, validation_predictions)
                )
        if stopped_early:
            break

    return TrainingResult(
        network=network,
        optimizer=selected_optimizer,
        history=history,
        epochs_completed=len(history["epoch"]),
        stopped_early=stopped_early,
        class_values=class_values,
    )


def _evaluate(
    network: NeuralNetwork,
    features: np.ndarray,
    targets: np.ndarray,
    loss_name: str,
    split_name: str,
) -> tuple[float, np.ndarray]:
    predictions = network.forward(features)
    loss_value = _LOSS_FUNCTIONS[loss_name](targets, predictions)
    if not np.isfinite(loss_value) or not np.all(np.isfinite(predictions)):
        raise FloatingPointError(f"Non-finite {split_name} loss or predictions encountered during training.")
    return float(loss_value), predictions


def _classification_accuracy(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Use threshold 0.5 for scalar binary output and argmax for one-hot classes."""
    if predictions.shape[1] == 1:
        predicted_labels = predictions[:, 0] >= 0.5
        target_labels = targets[:, 0] >= 0.5
    else:
        predicted_labels = np.argmax(predictions, axis=1)
        target_labels = np.argmax(targets, axis=1)
    return float(np.mean(predicted_labels == target_labels))


def _prepare_targets(
    targets: np.ndarray,
    loss_name: str,
    output_size: int | None,
    *,
    name: str,
    known_classes: np.ndarray | None = None,
) -> tuple[np.ndarray, int, np.ndarray | None]:
    """Normalize scalar/regression or binary/one-hot classification targets."""
    values = np.asarray(targets, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    if values.ndim not in (1, 2):
        raise ValueError(f"{name} must be a 1D label vector or 2D target matrix.")

    if loss_name != "binary_cross_entropy":
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        inferred = values.shape[1]
        if output_size is not None and inferred != output_size:
            raise ValueError(f"{name} has {inferred} outputs; expected {output_size}.")
        return values, inferred, None

    if values.ndim == 2 and values.shape[1] > 1:
        inferred = values.shape[1]
        if output_size is not None and inferred != output_size:
            raise ValueError(f"{name} has {inferred} outputs; expected {output_size}.")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{name} classification targets must be in [0, 1].")
        if not np.all(np.isclose(values.sum(axis=1), 1.0)) or not np.all(
            np.isclose(values, 0.0) | np.isclose(values, 1.0)
        ):
            raise ValueError(f"{name} with multiple outputs must be one-hot encoded.")
        return values, inferred, None

    if values.ndim == 2:
        values = values[:, 0]
    unique_values = np.unique(values)

    if known_classes is not None:
        classes = np.asarray(known_classes, dtype=float)
        if output_size == 1:
            if np.all(np.isin(values, (0.0, 1.0))) and np.array_equal(classes, np.array([0.0, 1.0])):
                encoded = values.reshape(-1, 1)
            elif classes.size == 2:
                if not np.all(np.isin(values, classes)):
                    raise ValueError(f"{name} contains a class not present in training labels.")
                encoded = (values == classes[-1]).astype(float).reshape(-1, 1)
            else:
                raise ValueError(f"{name} cannot map its labels to a single binary output.")
            return encoded, 1, classes
        if classes.size != output_size or not np.all(np.isin(values, classes)):
            raise ValueError(f"{name} contains classes not present in the training labels.")
        encoded = np.column_stack([values == label for label in classes]).astype(float)
        return encoded, output_size, classes

    if output_size == 1 or (output_size is None and np.all(np.isin(unique_values, (0.0, 1.0)))):
        if np.all(np.isin(unique_values, (0.0, 1.0))):
            encoded = values.reshape(-1, 1)
            return encoded, 1, np.array([0.0, 1.0])
        if unique_values.size == 2 and output_size == 1:
            encoded = (values == unique_values[-1]).astype(float).reshape(-1, 1)
            return encoded, 1, unique_values
        raise ValueError(f"{name} for a single sigmoid output must contain binary labels 0 or 1.")

    if output_size is not None and output_size > 1:
        # One-hot training targets may be paired with integer class ids in
        # validation data; in that case class ids follow the 0..K-1 convention.
        integer_labels = np.all(np.equal(values, np.floor(values)))
        if integer_labels and np.all((values >= 0) & (values < output_size)):
            classes = np.arange(output_size, dtype=float)
        else:
            classes = unique_values
    else:
        classes = unique_values
    inferred = int(classes.size) if output_size is None else output_size
    if classes.size != inferred or inferred < 2:
        raise ValueError(f"{name} classes do not match the requested output_size={inferred}.")
    encoded = np.column_stack([values == label for label in classes]).astype(float)
    return encoded, inferred, classes


def _validate_features(
    features: np.ndarray, name: str, expected_feature_count: int | None = None
) -> np.ndarray:
    values = np.asarray(features, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D array (or one feature vector).")
    if expected_feature_count is not None and values.shape[1] != expected_feature_count:
        raise ValueError(f"{name} must contain {expected_feature_count} features; got {values.shape[1]}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values


def _validate_targets(targets: np.ndarray, sample_count: int, name: str) -> np.ndarray:
    values = np.asarray(targets, dtype=float)
    if values.ndim not in (1, 2) or values.shape[0] != sample_count or values.size == 0:
        raise ValueError(f"{name} must contain one target row for each of {sample_count} samples.")
    return values


def _normalize_loss(loss: str) -> str:
    if not isinstance(loss, str) or loss not in _LOSS_ALIASES:
        raise ValueError("loss must be binary_cross_entropy, mean_squared_error, or mean_absolute_error.")
    return _LOSS_ALIASES[loss]


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _validate_learning_rate(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        raise ValueError("learning_rate must be a finite positive number.")
    return float(value)


def _validate_seed(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("random_seed must be a non-negative integer or None.")
    return int(value)
