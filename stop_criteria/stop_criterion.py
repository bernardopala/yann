"""Base interface for training stop criteria."""

from abc import ABC, abstractmethod


class StopCriterion(ABC):
    """Condition that determines when training should stop."""

    @abstractmethod
    def reached(self, epochs_completed: int, elapsed_seconds: float, loss: float | None = None) -> bool:
        pass
