"""Small synthetic classification datasets and leakage-safe train/validation splits."""

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetSplit:
    """A prepared split; the scaler is fitted on training samples only."""

    X_train: np.ndarray
    X_validation: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    scaler: StandardScaler | None

    @property
    def validation_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return validation features and labels in the trainer's expected form."""
        return self.X_validation, self.y_validation


def make_classification_dataset(
    name: str,
    n_samples: int = 300,
    noise: float = 0.1,
    random_seed: int | None = 0,
    class_1_fraction: float | None = None,
    circle_factor: float = 0.5,
    spiral_turns: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create XOR, moons, circles, or spiral data with binary labels.

    Returns features shaped ``(n_samples, 2)`` and integer labels shaped
    ``(n_samples,)``. Noise is interpreted as coordinate noise for XOR and
    spiral and passed to scikit-learn's generators for moons and circles.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or n_samples < 4:
        raise ValueError("n_samples must be an integer of at least 4.")
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be a finite non-negative number.")
    if class_1_fraction is not None:
        if not np.isfinite(class_1_fraction) or not 0.0 < class_1_fraction < 1.0:
            raise ValueError("class_1_fraction must be between 0 and 1.")
        class_1_count = int(round(int(n_samples) * float(class_1_fraction)))
        counts = (int(n_samples) - class_1_count, class_1_count)
        if min(counts) < 2:
            raise ValueError("class_1_fraction must leave at least two samples in each class.")
    else:
        counts = None
    if not np.isfinite(circle_factor) or not 0.0 < circle_factor < 1.0:
        raise ValueError("circle_factor must be between 0 and 1.")
    if not np.isfinite(spiral_turns) or spiral_turns <= 0:
        raise ValueError("spiral_turns must be a finite positive number.")
    rng = np.random.default_rng(random_seed)
    key = name.strip().lower()
    if key == "moons":
        features, labels = make_moons(n_samples=counts or int(n_samples), noise=noise, random_state=random_seed)
    elif key == "circles":
        features, labels = make_circles(
            n_samples=counts or int(n_samples), noise=noise, factor=circle_factor, random_state=random_seed
        )
    elif key == "xor":
        if counts is None:
            features = rng.uniform(-1.0, 1.0, size=(int(n_samples), 2))
            labels = np.logical_xor(features[:, 0] >= 0.0, features[:, 1] >= 0.0).astype(int)
        else:
            feature_groups = []
            label_groups = []
            for label, count in enumerate(counts):
                magnitudes = rng.uniform(0.05, 1.0, size=(count, 2))
                first_sign = rng.choice((-1.0, 1.0), size=count)
                second_sign = first_sign * (-1.0 if label else 1.0)
                feature_groups.append(magnitudes * np.column_stack((first_sign, second_sign)))
                label_groups.append(np.full(count, label, dtype=int))
            features = np.vstack(feature_groups)
            labels = np.concatenate(label_groups)
            permutation = rng.permutation(int(n_samples))
            features, labels = features[permutation], labels[permutation]
        if noise:
            features += rng.normal(0.0, noise, size=features.shape)
    elif key == "spiral":
        spiral_counts = counts or (int(n_samples) // 2, int(n_samples) - int(n_samples) // 2)
        feature_groups = []
        label_groups = []
        for label, count in enumerate(spiral_counts):
            radius = np.linspace(0.05, 1.0, count)
            angle = (2.0 * np.pi * spiral_turns) * radius + label * np.pi
            group = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
            group += rng.normal(0.0, noise, size=group.shape)
            feature_groups.append(group)
            label_groups.append(np.full(count, label, dtype=int))
        features = np.vstack(feature_groups)
        labels = np.concatenate(label_groups)
        permutation = rng.permutation(int(n_samples))
        features, labels = features[permutation], labels[permutation]
    else:
        raise ValueError("Unknown dataset. Choose from: xor, moons, circles, spiral.")
    return np.asarray(features, dtype=float), np.asarray(labels, dtype=int)


def split_classification_data(
    features: np.ndarray,
    labels: np.ndarray,
    validation_fraction: float = 0.2,
    random_seed: int | None = 0,
    standardize: bool = True,
) -> DatasetSplit:
    """Stratify a classification dataset and optionally standardize without leakage."""
    X = np.asarray(features, dtype=float)
    y = np.asarray(labels)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("features must be a non-empty 2D array with shape (n_samples, n_features).")
    if y.ndim not in (1, 2) or y.shape[0] != X.shape[0]:
        raise ValueError("labels must have one row per feature sample.")
    if not np.all(np.isfinite(X)):
        raise ValueError("features must contain only finite values.")
    if not np.isfinite(validation_fraction) or not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    class_labels = np.argmax(y, axis=1) if y.ndim == 2 else y
    unique, counts = np.unique(class_labels, return_counts=True)
    validation_count = int(np.ceil(X.shape[0] * validation_fraction))
    training_count = X.shape[0] - validation_count
    if unique.size < 2 or counts.min() < 2 or validation_count < unique.size or training_count < unique.size:
        raise ValueError("Each class needs at least two samples, and each split must contain every class.")

    X_train, X_validation, y_train, y_validation = train_test_split(
        X,
        y,
        test_size=validation_fraction,
        random_state=random_seed,
        shuffle=True,
        stratify=class_labels,
    )
    scaler: StandardScaler | None = None
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_validation = scaler.transform(X_validation)
    return DatasetSplit(X_train, X_validation, y_train, y_validation, scaler)
