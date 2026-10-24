"""A small NumPy feedforward network for forward propagation demonstrations."""

from collections.abc import Sequence

import numpy as np

from .activations import relu, sigmoid


class NeuralNetwork:
    """Feedforward network with ReLU hidden layers and sigmoid outputs.

    Inputs are provided as ``(n_samples, n_features)``. A single sample may be
    passed as ``(n_features,)``; in either case predictions have shape
    ``(n_samples, output_size)``. Hidden-layer weights use He initialization,
    which is suited to ReLU. The sigmoid output layer uses Xavier uniform
    initialization. Biases start at zero.
    """

    def __init__(
        self,
        input_feature_count: int,
        hidden_layer_sizes: Sequence[int],
        output_size: int,
        random_seed: int | None = None,
    ) -> None:
        """Create a network with the requested layer widths.

        Args:
            input_feature_count: Number of input features.
            hidden_layer_sizes: Width of each hidden layer; may be empty.
            output_size: Number of sigmoid output units.
            random_seed: Optional seed for reproducible parameters.

        Raises:
            ValueError: If a layer width is not a positive integer.
        """
        self.input_feature_count = self._validate_width(input_feature_count, "input_feature_count")
        self.hidden_layer_sizes = tuple(
            self._validate_width(size, f"hidden_layer_sizes[{index}]")
            for index, size in enumerate(hidden_layer_sizes)
        )
        self.output_size = self._validate_width(output_size, "output_size")
        self.layer_sizes = (self.input_feature_count, *self.hidden_layer_sizes, self.output_size)

        rng = np.random.default_rng(random_seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for layer_index, (fan_in, fan_out) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            if layer_index < len(self.hidden_layer_sizes):
                # He normal: preserves activation scale through ReLU layers.
                weight = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_out, fan_in))
            else:
                # Xavier uniform: balances scale for the sigmoid output layer.
                limit = np.sqrt(6.0 / (fan_in + fan_out))
                weight = rng.uniform(-limit, limit, size=(fan_out, fan_in))
            self.weights.append(weight)
            self.biases.append(np.zeros((fan_out, 1), dtype=float))

        # Caches are populated on every forward pass for future learning steps.
        self.activations: list[np.ndarray] = []
        self.weighted_sums: list[np.ndarray] = []

    @staticmethod
    def _validate_width(value: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer; got {value!r}.")
        return int(value)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Compute predictions for a sample or a batch of samples.

        The computation uses column-oriented activations internally, so each
        layer applies ``Z = W @ A + b``. Returned rows correspond to samples.
        """
        values = np.asarray(inputs, dtype=float)
        if values.ndim == 1:
            if values.shape[0] != self.input_feature_count:
                raise ValueError(
                    f"Expected a sample with {self.input_feature_count} features; got shape {values.shape}."
                )
            values = values.reshape(1, -1)
        elif values.ndim != 2:
            raise ValueError(
                "Inputs must be a 1D sample or a 2D batch with shape "
                f"(n_samples, {self.input_feature_count}); got shape {values.shape}."
            )

        if values.shape[1] != self.input_feature_count:
            raise ValueError(
                f"Expected {self.input_feature_count} input features in axis 1; got shape {values.shape}."
            )
        if values.shape[0] == 0:
            raise ValueError("Input batch must contain at least one sample.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Inputs must contain only finite numeric values.")

        current = values.T
        self.activations = [current.copy()]
        self.weighted_sums = []

        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z_value = weight @ current + bias
            current = relu(z_value) if index < len(self.hidden_layer_sizes) else sigmoid(z_value)
            self.weighted_sums.append(z_value)
            self.activations.append(current)

        return current.T.copy()
