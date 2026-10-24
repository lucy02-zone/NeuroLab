"""Reusable Matplotlib visualizations for NeuroLab experiments."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle

from src.neurolab.models import NeuralNetwork


NAVY = "#101827"
BLUE = "#648cff"
CYAN = "#39d5e7"
VIOLET = "#b18cff"
TEXT = "#eaf1ff"
MUTED = "#8b9bb2"
GRID = "#26364b"


def _style_axis(axis: plt.Axes, title: str) -> None:
    axis.set_facecolor(NAVY)
    axis.set_title(title, loc="left", color=TEXT, fontsize=12, pad=12)
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.xaxis.label.set_color(MUTED)
    axis.yaxis.label.set_color(MUTED)
    for spine in axis.spines.values():
        spine.set_color(GRID)
    axis.grid(color=GRID, linewidth=0.65, alpha=0.55)


def plot_dataset_preview(features: np.ndarray, labels: np.ndarray, title: str = "Dataset preview") -> plt.Figure:
    """Create a scatter plot for a two-dimensional labeled dataset."""
    X, y = _validate_plot_data(features, labels)
    figure, axis = plt.subplots(figsize=(7.0, 5.0), facecolor=NAVY)
    _style_axis(axis, title)
    for class_index, color, label in ((0, BLUE, "Class 0"), (1, CYAN, "Class 1")):
        mask = y == class_index
        if np.any(mask):
            axis.scatter(X[mask, 0], X[mask, 1], s=25, c=color, label=label,
                         edgecolors="#dce8ff55", linewidths=0.35, alpha=0.88)
    axis.set_xlabel("Feature 1")
    axis.set_ylabel("Feature 2")
    axis.legend(facecolor="#141f30", edgecolor=GRID, labelcolor=TEXT)
    figure.tight_layout()
    return figure


def plot_network_architecture(
    hidden_layer_sizes: tuple[int, ...] | list[int],
    input_size: int = 2,
    output_size: int = 1,
    hidden_activation: str = "relu",
) -> plt.Figure:
    """Draw fully connected layers and neuron nodes for the configured model."""
    layer_sizes = (input_size, *hidden_layer_sizes, output_size)
    largest_layer = max(layer_sizes)
    figure_height = max(4.0, min(8.0, 3.5 + largest_layer * 0.12))
    figure, axis = plt.subplots(figsize=(max(6.0, len(layer_sizes) * 1.7), figure_height), facecolor=NAVY)
    axis.set_facecolor(NAVY)
    positions: list[list[tuple[float, float]]] = []
    layer_x = np.linspace(0.08, 0.92, len(layer_sizes))
    for x_coord, size in zip(layer_x, layer_sizes):
        y_coords = np.linspace(0.18, 0.82, size) if size > 1 else np.array([0.5])
        positions.append([(float(x_coord), float(y_coord)) for y_coord in y_coords])

    for layer_index in range(len(positions) - 1):
        for x1, y1 in positions[layer_index]:
            for x2, y2 in positions[layer_index + 1]:
                axis.plot([x1, x2], [y1, y2], color="#516b91", alpha=0.3, linewidth=0.75, zorder=1)

    for layer_index, layer_positions in enumerate(positions):
        color = BLUE if layer_index == 0 else CYAN if layer_index == len(positions) - 1 else VIOLET
        for neuron_index, (x_coord, y_coord) in enumerate(layer_positions):
            radius = min(0.032, 0.3 / max(1, largest_layer - 1))
            axis.add_patch(Circle((x_coord, y_coord), radius, facecolor=color, edgecolor="#eaf1ff",
                                  linewidth=0.8, zorder=3))
            if layer_index == 0:
                axis.text(x_coord, y_coord, f"x{neuron_index + 1}", color=TEXT, fontsize=7,
                          ha="center", va="center", zorder=4)
    for index, (x_coord, size) in enumerate(zip(layer_x, layer_sizes)):
        label = "Input" if index == 0 else "Output · sigmoid" if index == len(layer_sizes) - 1 else f"Hidden · {hidden_activation}"
        axis.text(x_coord, 0.04, f"{label}\n{size} neuron{'s' if size != 1 else ''}",
                  color=TEXT if index != len(layer_sizes) - 1 else CYAN, fontsize=8,
                  ha="center", va="top")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-0.12, 1.0)
    axis.axis("off")
    figure.tight_layout()
    return figure


def plot_loss_history(history: dict[str, list[int | float]]) -> plt.Figure:
    """Plot actual training and optional validation loss history."""
    epochs = np.asarray(history.get("epoch", []), dtype=int)
    if epochs.size == 0 or len(history.get("training_loss", [])) != epochs.size:
        raise ValueError("history must contain matching non-empty epoch and training_loss values.")
    figure, axis = plt.subplots(figsize=(8.0, 4.3), facecolor=NAVY)
    _style_axis(axis, "Loss by epoch")
    axis.plot(epochs, history["training_loss"], color=BLUE, linewidth=2.3, label="Training loss")
    validation = history.get("validation_loss")
    if validation is not None:
        if len(validation) != epochs.size:
            raise ValueError("validation_loss must have one value per epoch.")
        axis.plot(epochs, validation, color=CYAN, linewidth=2.1, label="Validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.legend(facecolor="#141f30", edgecolor=GRID, labelcolor=TEXT)
    figure.tight_layout()
    return figure


def plot_decision_boundary(
    network: NeuralNetwork,
    features: np.ndarray,
    labels: np.ndarray,
    scaler: object | None = None,
    resolution: int = 180,
) -> plt.Figure:
    """Plot model-predicted regions and data points in original feature space.

    If a scaler is supplied, grid coordinates are transformed with that
    already-fitted training scaler before prediction.
    """
    X, y = _validate_plot_data(features, labels)
    if isinstance(resolution, bool) or not isinstance(resolution, (int, np.integer)) or resolution < 20:
        raise ValueError("resolution must be an integer of at least 20.")
    padding = max(0.15, float(np.ptp(X[:, 0])) * 0.08, float(np.ptp(X[:, 1])) * 0.08)
    x_values = np.linspace(float(X[:, 0].min()) - padding, float(X[:, 0].max()) + padding, resolution)
    y_values = np.linspace(float(X[:, 1].min()) - padding, float(X[:, 1].max()) + padding, resolution)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    model_grid = scaler.transform(grid) if scaler is not None else grid
    predictions = network.forward(model_grid)
    predicted_classes = (predictions[:, 0] >= 0.5).astype(int) if predictions.shape[1] == 1 else np.argmax(predictions, axis=1)
    regions = predicted_classes.reshape(grid_x.shape)

    figure, axis = plt.subplots(figsize=(7.0, 5.0), facecolor=NAVY)
    axis.set_facecolor(NAVY)
    axis.contourf(grid_x, grid_y, regions, levels=(-0.5, 0.5, 1.5),
                  cmap=ListedColormap(("#345fc766", "#159eae66")), alpha=0.72)
    _style_axis(axis, "Trained model decision boundary")
    for class_index, color, label in ((0, BLUE, "Class 0"), (1, CYAN, "Class 1")):
        mask = y == class_index
        if np.any(mask):
            axis.scatter(X[mask, 0], X[mask, 1], s=20, c=color, label=label,
                         edgecolors="#f4f7ff", linewidths=0.4, alpha=0.93)
    axis.set_xlabel("Feature 1")
    axis.set_ylabel("Feature 2")
    axis.legend(facecolor="#141f30", edgecolor=GRID, labelcolor=TEXT)
    figure.tight_layout()
    return figure


def _validate_plot_data(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(features, dtype=float)
    y = np.asarray(labels)
    if X.ndim != 2 or X.shape[1] != 2 or X.shape[0] == 0:
        raise ValueError("Visualization requires non-empty two-dimensional features with shape (n_samples, 2).")
    if y.ndim == 2 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)
    else:
        y = y.reshape(-1)
    if y.shape[0] != X.shape[0]:
        raise ValueError("labels must contain one class per feature sample.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("Visualization data must contain only finite values.")
    unique_labels = np.unique(y)
    if np.any(~np.isin(unique_labels, (0, 1))):
        raise ValueError("The network playground visualizations support binary classes 0 and 1.")
    return X, y.astype(int)
