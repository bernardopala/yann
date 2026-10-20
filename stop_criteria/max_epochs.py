"""Epoch-count stop criterion."""

import numpy as np

from .stop_criterion import StopCriterion


class MaxEpochs(StopCriterion):
    def __init__(self, n: int) -> None:
        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError("n must be a positive integer")

        self.n = int(n)

    def reached(self, epochs_completed: int, elapsed_seconds: float, loss: float | None = None) -> bool:
        return epochs_completed >= self.n
