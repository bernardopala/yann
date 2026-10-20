"""Stochastic gradient descent optimizer."""

from .optimizer import Optimizer


class SGD(Optimizer):
    def _update(self, parameter, gradient, state) -> None:
        parameter -= self.learning_rate * gradient
