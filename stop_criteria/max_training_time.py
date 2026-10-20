"""Elapsed-training-time stop criterion."""

from .stop_criterion import StopCriterion


class MaxTrainingTime(StopCriterion):
    def __init__(self, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError("seconds must be positive")

        self.seconds = float(seconds)

    def reached(self, epochs_completed: int, elapsed_seconds: float, loss: float | None = None) -> bool:
        return elapsed_seconds >= self.seconds
