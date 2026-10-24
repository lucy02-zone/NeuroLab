"""Fair, reproducible comparisons using NeuroLab's existing trainer."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .datasets import DatasetSplit, split_classification_data
from .models import NeuralNetwork
from .training import TrainingResult, train_model


OPTIMIZERS = ("SGD", "Momentum", "Adam")


@dataclass
class OptimizerComparison:
    """Results and shared data from one optimizer comparison run."""

    runs: dict[str, TrainingResult]
    split: DatasetSplit
    features: np.ndarray
    labels: np.ndarray
    initial_parameters: dict[str, tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]]


def run_optimizer_comparison(
    features: np.ndarray,
    labels: np.ndarray,
    optimizers: Sequence[str] = OPTIMIZERS,
    *,
    hidden_layer_sizes: Sequence[int] = (8, 8),
    hidden_activation: str = "relu",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    validation_fraction: float = 0.2,
    random_seed: int = 0,
    momentum: float = 0.9,
    beta1: float = 0.9,
    beta2: float = 0.999,
    optimizer_epsilon: float = 1e-8,
) -> OptimizerComparison:
    """Train fresh, identically initialized models on one shared data split."""
    selected = tuple(optimizers)
    if not selected or len(set(selected)) != len(selected) or any(name not in OPTIMIZERS for name in selected):
        raise ValueError(f"optimizers must be a non-empty unique subset of {OPTIMIZERS}.")
    X = np.asarray(features, dtype=float)
    y = np.asarray(labels)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("Optimizer comparison requires features with shape (n_samples, 2).")
    split = split_classification_data(
        X, y, validation_fraction=validation_fraction, random_seed=random_seed, standardize=True
    )

    runs: dict[str, TrainingResult] = {}
    initial_parameters: dict[str, tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]] = {}
    for name in selected:
        # Construct a new network and optimizer for every run. The shared seed
        # gives identical starting weights while the trainer creates isolated
        # optimizer state and applies the same seeded minibatch order.
        network = NeuralNetwork(
            input_feature_count=split.X_train.shape[1],
            hidden_layer_sizes=tuple(hidden_layer_sizes),
            output_size=1,
            random_seed=random_seed,
            hidden_activation=hidden_activation,
        )
        initial_parameters[name] = (
            tuple(weight.copy() for weight in network.weights),
            tuple(bias.copy() for bias in network.biases),
        )
        runs[name] = train_model(
            split.X_train,
            split.y_train,
            network=network,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=name.lower(),
            learning_rate=learning_rate,
            momentum=momentum,
            beta1=beta1,
            beta2=beta2,
            optimizer_epsilon=optimizer_epsilon,
            loss="binary_cross_entropy",
            validation_data=split.validation_data,
            random_seed=random_seed,
        )
    return OptimizerComparison(runs, split, X.copy(), y.copy(), initial_parameters)
