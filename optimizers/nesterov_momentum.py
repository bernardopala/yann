"""Nesterov momentum optimizer."""

import numpy as np

from .momentum import Momentum


class NesterovMomentum(Momentum):
    def _update(self, parameter, gradient, state) -> None:
        velocity = state.setdefault("velocity", np.zeros_like(parameter))
        previous_velocity = velocity.copy()
        velocity *= self.momentum
        velocity -= self.learning_rate * gradient
        parameter += -self.momentum * previous_velocity + (1.0 + self.momentum) * velocity
