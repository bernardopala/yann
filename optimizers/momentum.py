"""Momentum optimizer."""

import numpy as np

from .optimizer import Optimizer, _validate_unit_interval


class Momentum(Optimizer):
    def __init__(self, learning_rate: float, momentum: float) -> None:
        super().__init__(learning_rate)
        _validate_unit_interval("momentum", momentum)
        self.momentum = float(momentum)

    def _update(self, parameter, gradient, state) -> None:
        velocity = state.setdefault("velocity", np.zeros_like(parameter))
        velocity *= self.momentum
        velocity -= self.learning_rate * gradient
        parameter += velocity
