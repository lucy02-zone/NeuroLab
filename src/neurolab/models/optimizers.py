"""NumPy optimizers for updating model parameters from calculated gradients."""

from collections.abc import Sequence

import numpy as np


class Optimizer:
    """Base class for optimizers that update arrays in place."""

    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = _positive_finite(learning_rate, "learning_rate")

    def step(self, parameters: Sequence[np.ndarray], gradients: Sequence[np.ndarray]) -> None:
        """Apply one update after validating all parameters and gradients."""
        params, grads = _validate_parameter_gradient_pairs(parameters, gradients)
        with np.errstate(over="ignore", invalid="ignore"):
            candidates = [parameter - self.learning_rate * gradient for parameter, gradient in zip(params, grads)]
        _validate_candidates(candidates)
        for parameter, candidate in zip(params, candidates):
            parameter[...] = candidate


class SGD(Optimizer):
    """Stochastic gradient descent: ``parameter -= learning_rate * gradient``."""


class SGDMomentum(Optimizer):
    """SGD with a per-parameter velocity accumulator."""

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9) -> None:
        super().__init__(learning_rate)
        if not np.isfinite(momentum) or not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be finite and in [0, 1).")
        self.momentum = float(momentum)
        self.velocities: list[np.ndarray] = []

    def step(self, parameters: Sequence[np.ndarray], gradients: Sequence[np.ndarray]) -> None:
        params, grads = _validate_parameter_gradient_pairs(parameters, gradients)
        _validate_state_shapes(self.velocities, params, "momentum velocity")
        old_velocities = self.velocities or [np.zeros_like(parameter) for parameter in params]
        with np.errstate(over="ignore", invalid="ignore"):
            new_velocities = [self.momentum * velocity - self.learning_rate * gradient
                              for velocity, gradient in zip(old_velocities, grads)]
            candidates = [parameter + velocity for parameter, velocity in zip(params, new_velocities)]
        _validate_candidates([*new_velocities, *candidates])
        for parameter, candidate in zip(params, candidates):
            parameter[...] = candidate
        self.velocities = [velocity.copy() for velocity in new_velocities]


class Adam(Optimizer):
    """Adaptive moment estimation with bias-corrected first and second moments."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(learning_rate)
        if not np.isfinite(beta1) or not 0.0 <= beta1 < 1.0:
            raise ValueError("beta1 must be finite and in [0, 1).")
        if not np.isfinite(beta2) or not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must be finite and in [0, 1).")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = _positive_finite(epsilon, "epsilon")
        self.first_moments: list[np.ndarray] = []
        self.second_moments: list[np.ndarray] = []
        self.time_step = 0

    def step(self, parameters: Sequence[np.ndarray], gradients: Sequence[np.ndarray]) -> None:
        params, grads = _validate_parameter_gradient_pairs(parameters, gradients)
        _validate_state_shapes(self.first_moments, params, "Adam first moment")
        _validate_state_shapes(self.second_moments, params, "Adam second moment")
        old_first = self.first_moments or [np.zeros_like(parameter) for parameter in params]
        old_second = self.second_moments or [np.zeros_like(parameter) for parameter in params]
        next_step = self.time_step + 1
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            first = [self.beta1 * moment + (1.0 - self.beta1) * gradient
                     for moment, gradient in zip(old_first, grads)]
            second = [self.beta2 * moment + (1.0 - self.beta2) * np.square(gradient)
                      for moment, gradient in zip(old_second, grads)]
            corrected_first = [moment / (1.0 - self.beta1**next_step) for moment in first]
            corrected_second = [moment / (1.0 - self.beta2**next_step) for moment in second]
            candidates = [parameter - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                          for parameter, m_hat, v_hat in zip(params, corrected_first, corrected_second)]
        _validate_candidates([*first, *second, *candidates])
        for parameter, candidate in zip(params, candidates):
            parameter[...] = candidate
        self.first_moments = [moment.copy() for moment in first]
        self.second_moments = [moment.copy() for moment in second]
        self.time_step = next_step


# Readable aliases for callers who prefer the longer/conventional class names.
Momentum = SGDMomentum
SGDWithMomentum = SGDMomentum


def _positive_finite(value: float, name: str) -> float:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return float(value)


def _validate_parameter_gradient_pairs(
    parameters: Sequence[np.ndarray], gradients: Sequence[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    params = list(parameters)
    grads = [np.asarray(gradient, dtype=float) for gradient in gradients]
    if not params:
        raise ValueError("At least one parameter is required for an optimizer step.")
    if len(params) != len(grads):
        raise ValueError(f"Expected {len(params)} gradients for {len(params)} parameters; got {len(grads)}.")
    for index, (parameter, gradient) in enumerate(zip(params, grads)):
        if not isinstance(parameter, np.ndarray):
            raise TypeError(f"Parameter {index} must be a NumPy array.")
        if parameter.shape != gradient.shape:
            raise ValueError(
                f"Gradient {index} shape {gradient.shape} does not match parameter shape {parameter.shape}."
            )
        if not np.all(np.isfinite(parameter)):
            raise ValueError(f"Parameter {index} contains non-finite values.")
        if not np.all(np.isfinite(gradient)):
            raise ValueError(f"Gradient {index} contains non-finite values.")
    return params, grads


def _validate_state_shapes(state: list[np.ndarray], parameters: list[np.ndarray], name: str) -> None:
    if state and (len(state) != len(parameters) or any(
        value.shape != parameter.shape for value, parameter in zip(state, parameters)
    )):
        raise ValueError(f"{name} state no longer matches the parameter shapes.")


def _validate_candidates(values: Sequence[np.ndarray]) -> None:
    if any(not np.all(np.isfinite(value)) for value in values):
        raise ValueError("Optimizer update would create non-finite state or parameters; no update was applied.")
