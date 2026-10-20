"""AdamW optimizer."""

from .adam import Adam


class AdamW(Adam):
    def __init__(self, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, weight_decay: float = 0.01) -> None:
        super().__init__(learning_rate, beta1, beta2, epsilon)
        if weight_decay < 0:
            raise ValueError("weight_decay must not be negative")
        self.weight_decay = float(weight_decay)

    def _update(self, parameter, gradient, state) -> None:
        parameter *= 1.0 - self.learning_rate * self.weight_decay
        super()._update(parameter, gradient, state)
