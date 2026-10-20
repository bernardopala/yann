"""RMSProp optimizer."""

import numpy as np

from .optimizer import Optimizer, _validate_positive, _validate_unit_interval


class RMSProp(Optimizer):
    def __init__(self, learning_rate: float, rho: float = 0.9, epsilon: float = 1e-8) -> None:
        super().__init__(learning_rate)
        _validate_unit_interval("rho", rho)
        _validate_positive("epsilon", epsilon)
        self.rho, self.epsilon = float(rho), float(epsilon)

    def _update(self, parameter, gradient, state) -> None:
        average = state.setdefault("average", np.zeros_like(parameter))
        average *= self.rho
        average += (1.0 - self.rho) * np.square(gradient)
        parameter -= self.learning_rate * gradient / (np.sqrt(average) + self.epsilon)
