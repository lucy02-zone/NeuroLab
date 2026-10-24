"""A small NumPy feedforward network for forward propagation demonstrations."""

from collections.abc import Sequence

import numpy as np

from . import losses
from .activations import (
    leaky_relu,
    leaky_relu_derivative,
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    tanh,
    tanh_derivative,
)


HIDDEN_ACTIVATIONS = {
    "relu": relu,
    "leaky_relu": leaky_relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
}


class NeuralNetwork:
    """Feedforward network with configurable hidden and sigmoid output layers.

    Inputs are provided as ``(n_samples, n_features)``. A single sample may be
    passed as ``(n_features,)``; in either case predictions have shape
    ``(n_samples, output_size)``. Hidden layers default to ReLU. ReLU and
    Leaky ReLU use He initialization; sigmoid/tanh hidden layers use Xavier
    initialization. The sigmoid output layer uses Xavier uniform
    initialization. Biases start at zero.
    """

    def __init__(
        self,
        input_feature_count: int,
        hidden_layer_sizes: Sequence[int],
        output_size: int,
        random_seed: int | None = None,
        hidden_activation: str = "relu",
    ) -> None:
        """Create a network with the requested layer widths.

        Args:
            input_feature_count: Number of input features.
            hidden_layer_sizes: Width of each hidden layer; may be empty.
            output_size: Number of sigmoid output units.
            random_seed: Optional seed for reproducible parameters.
            hidden_activation: Elementwise hidden activation. One of ``relu``,
                ``leaky_relu``, ``sigmoid``, or ``tanh``. The output remains
                sigmoid for binary classification.

        Raises:
            ValueError: If a layer width is not a positive integer.
        """
        self.input_feature_count = self._validate_width(input_feature_count, "input_feature_count")
        self.hidden_layer_sizes = tuple(
            self._validate_width(size, f"hidden_layer_sizes[{index}]")
            for index, size in enumerate(hidden_layer_sizes)
        )
        self.output_size = self._validate_width(output_size, "output_size")
        if hidden_activation not in HIDDEN_ACTIVATIONS:
            choices = ", ".join(HIDDEN_ACTIVATIONS)
            raise ValueError(f"hidden_activation must be one of: {choices}.")
        self.hidden_activation = hidden_activation
        self.layer_sizes = (self.input_feature_count, *self.hidden_layer_sizes, self.output_size)

        rng = np.random.default_rng(random_seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for layer_index, (fan_in, fan_out) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            if layer_index < len(self.hidden_layer_sizes) and hidden_activation in ("relu", "leaky_relu"):
                # He normal preserves activation scale for rectifier units.
                variance_scale = 2.0 if hidden_activation == "relu" else 2.0 / (1.0 + 0.01**2)
                weight = rng.normal(0.0, np.sqrt(variance_scale / fan_in), size=(fan_out, fan_in))
            else:
                # Xavier uniform: balances scale for the sigmoid output layer.
                limit = np.sqrt(6.0 / (fan_in + fan_out))
                weight = rng.uniform(-limit, limit, size=(fan_out, fan_in))
            self.weights.append(weight)
            self.biases.append(np.zeros((fan_out, 1), dtype=float))

        # Caches are populated on every forward pass for future learning steps.
        self.activations: list[np.ndarray] = []
        self.weighted_sums: list[np.ndarray] = []
        self._gradients_available = False

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
        self._gradients_available = False
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
            if index < len(self.hidden_layer_sizes):
                activation = HIDDEN_ACTIVATIONS[self.hidden_activation]
                current = activation(z_value)
            else:
                # Keep sigmoid output behavior for binary classification.
                current = sigmoid(z_value)
            self.weighted_sums.append(z_value)
            self.activations.append(current)

        return current.T.copy()

    def backward(self, targets: np.ndarray, loss: str = "binary_cross_entropy") -> dict[str, object]:
        """Calculate loss gradients without changing parameters.

        The loss derivatives already include their documented averaging
        factor. Since cached arrays are column-oriented ``(units, samples)``,
        this method sums ``dZ`` over the batch when forming ``dW`` and ``db``;
        it must not divide those gradients by the batch size again.

        Args:
            targets: Values matching the most recent forward output. For a
                one-output network, a batch vector of shape ``(n_samples,)``
                is also accepted.
            loss: ``binary_cross_entropy``, ``mean_squared_error``, or
                ``mean_absolute_error``. Categorical cross-entropy is not
                supported because this network's output is sigmoid, not
                softmax.

        Returns:
            A dictionary containing loss, weight and bias gradients, and
            their per-layer L2 norms. Also stores it as ``self.gradients``.
        """
        if not self.activations or not self.weighted_sums:
            raise RuntimeError("Call forward() before backward() so layer values are cached.")

        targets_array = self._prepare_targets(targets)
        predictions = self.activations[-1].T
        loss_functions = {
            "binary_cross_entropy": (losses.binary_cross_entropy, losses.binary_cross_entropy_derivative),
            "bce": (losses.binary_cross_entropy, losses.binary_cross_entropy_derivative),
            "mean_squared_error": (losses.mean_squared_error, losses.mean_squared_error_derivative),
            "mse": (losses.mean_squared_error, losses.mean_squared_error_derivative),
            "mean_absolute_error": (losses.mean_absolute_error, losses.mean_absolute_error_derivative),
            "mae": (losses.mean_absolute_error, losses.mean_absolute_error_derivative),
        }
        if loss not in loss_functions:
            supported = "binary_cross_entropy, mean_squared_error, mean_absolute_error"
            raise ValueError(f"Unsupported loss {loss!r}; choose one of: {supported}.")
        loss_function, loss_derivative = loss_functions[loss]
        loss_value = loss_function(targets_array, predictions)

        # BCE + sigmoid has dL/dZ = (prediction - target) / n_elements.
        # This combines the BCE probability derivative with the sigmoid
        # derivative algebraically, while retaining BCE's elementwise mean.
        if loss in ("binary_cross_entropy", "bce"):
            delta = (predictions - targets_array).T / predictions.size
        else:
            d_loss_d_output = loss_derivative(targets_array, predictions).T
            output_values = self.activations[-1]
            delta = d_loss_d_output * sigmoid_derivative(
                self.weighted_sums[-1], activation_output=output_values
            )

        weight_gradients: list[np.ndarray] = [np.empty_like(weight) for weight in self.weights]
        bias_gradients: list[np.ndarray] = [np.empty_like(bias) for bias in self.biases]
        # Keep chain-rule intermediates in layer order and present them in the
        # public sample-major layout ``(samples, units)`` for inspection.
        activation_gradients: list[np.ndarray] = [np.empty_like(value.T) for value in self.weighted_sums]
        activation_derivatives: list[np.ndarray] = [np.empty_like(value.T) for value in self.weighted_sums]
        preactivation_gradients: list[np.ndarray] = [np.empty_like(value.T) for value in self.weighted_sums]
        output_values = self.activations[-1]
        activation_derivatives[-1] = np.asarray(
            sigmoid_derivative(self.weighted_sums[-1], activation_output=output_values)
        ).T.copy()
        activation_gradients[-1] = loss_derivative(targets_array, predictions)
        if loss in ("binary_cross_entropy", "bce"):
            # BCE + sigmoid simplifies dL/dZ to (prediction-target)/n.
            # It is algebraically dL/dA multiplied by the sigmoid derivative.
            delta = (predictions - targets_array).T / predictions.size
        else:
            delta = activation_gradients[-1].T * activation_derivatives[-1].T
        for layer_index in range(len(self.weights) - 1, -1, -1):
            preactivation_gradients[layer_index] = delta.T.copy()
            previous_activation = self.activations[layer_index]
            # The loss derivative's averaging factor is already in delta.
            weight_gradients[layer_index] = delta @ previous_activation.T
            bias_gradients[layer_index] = np.sum(delta, axis=1, keepdims=True)
            if layer_index > 0:
                d_activation = self.weights[layer_index].T @ delta
                hidden_index = layer_index - 1
                activation_gradients[hidden_index] = d_activation.T.copy()
                z_value = self.weighted_sums[hidden_index]
                activation_output = self.activations[hidden_index + 1]
                if self.hidden_activation == "relu":
                    derivative = relu_derivative(z_value)
                elif self.hidden_activation == "leaky_relu":
                    derivative = leaky_relu_derivative(z_value)
                elif self.hidden_activation == "sigmoid":
                    derivative = sigmoid_derivative(z_value, activation_output=activation_output)
                else:
                    derivative = tanh_derivative(z_value, activation_output=activation_output)
                activation_derivatives[hidden_index] = np.asarray(derivative).T.copy()
                delta = d_activation * derivative

        self.gradients = {
            "loss": float(loss_value),
            "loss_name": loss,
            "weight_gradients": weight_gradients,
            "bias_gradients": bias_gradients,
            "weight_norms": [float(np.linalg.norm(gradient)) for gradient in weight_gradients],
            "bias_norms": [float(np.linalg.norm(gradient)) for gradient in bias_gradients],
            "activation_gradients": activation_gradients,
            "activation_derivatives": activation_derivatives,
            "preactivation_gradients": preactivation_gradients,
        }
        self.weight_gradients = weight_gradients
        self.bias_gradients = bias_gradients
        self._gradients_available = True
        return self.gradients

    def _prepare_targets(self, targets: np.ndarray) -> np.ndarray:
        """Validate targets against the batch cached by the latest forward pass."""
        values = np.asarray(targets, dtype=float)
        sample_count = self.activations[0].shape[1]
        expected_shape = (sample_count, self.output_size)
        if values.ndim == 1:
            if self.output_size == 1 and values.shape[0] == sample_count:
                values = values.reshape(sample_count, 1)
            elif sample_count == 1 and values.shape[0] == self.output_size:
                values = values.reshape(1, self.output_size)
        if values.shape != expected_shape:
            raise ValueError(f"Targets must have shape {expected_shape}; got {values.shape}.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Targets must contain only finite values.")
        return values

    @property
    def trainable_parameters(self) -> list[np.ndarray]:
        """Return weights followed by biases, matching the gradient ordering."""
        return [*self.weights, *self.biases]

    def apply_gradients(
        self,
        optimizer: object,
        gradients: dict[str, object] | None = None,
    ) -> None:
        """Apply previously calculated gradients through an optimizer.

        Gradient calculation and parameter updates remain separate: call
        ``backward`` first, then pass its result here. The forward cache is
        cleared after a successful update so it cannot be reused with new
        weights.
        """
        if gradients is None and not self._gradients_available:
            raise RuntimeError("Call backward() after the latest forward pass before apply_gradients().")
        selected_gradients = self.gradients if gradients is None else gradients
        if selected_gradients is None:
            raise RuntimeError("Call backward() before apply_gradients().")
        try:
            weight_gradients = selected_gradients["weight_gradients"]
            bias_gradients = selected_gradients["bias_gradients"]
        except (KeyError, TypeError) as error:
            raise ValueError("Gradients must contain weight_gradients and bias_gradients.") from error
        if len(weight_gradients) != len(self.weights) or len(bias_gradients) != len(self.biases):
            raise ValueError("Gradient layer counts must match the network parameters.")
        if not hasattr(optimizer, "step"):
            raise TypeError("optimizer must provide a step(parameters, gradients) method.")
        optimizer.step(
            self.trainable_parameters,
            [*weight_gradients, *bias_gradients],
        )
        self.activations = []
        self.weighted_sums = []
        self._gradients_available = False

    def train_step(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        optimizer: object,
        loss: str = "binary_cross_entropy",
    ) -> float:
        """Run one forward/backward/update step and return the pre-update loss."""
        self.forward(inputs)
        gradients = self.backward(targets, loss=loss)
        loss_before_update = float(gradients["loss"])
        self.apply_gradients(optimizer, gradients)
        return loss_before_update
