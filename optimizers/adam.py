"""Adam optimizer."""

import numpy as np

from .optimizer import Optimizer, _validate_positive, _validate_unit_interval


class Adam(Optimizer):
    def __init__(self, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> None:
        super().__init__(learning_rate)
        _validate_unit_interval("beta1", beta1)
        _validate_unit_interval("beta2", beta2)
        _validate_positive("epsilon", epsilon)
        self.beta1, self.beta2, self.epsilon = float(beta1), float(beta2), float(epsilon)

    def _update(self, parameter, gradient, state) -> None:
        first = state.setdefault("first", np.zeros_like(parameter))
        second = state.setdefault("second", np.zeros_like(parameter))
        first *= self.beta1
        first += (1.0 - self.beta1) * gradient
        second *= self.beta2
        second += (1.0 - self.beta2) * np.square(gradient)
        first_corrected = first / (1.0 - self.beta1 ** self._step_count)
        second_corrected = second / (1.0 - self.beta2 ** self._step_count)
        parameter -= self.learning_rate * first_corrected / (np.sqrt(second_corrected) + self.epsilon)
