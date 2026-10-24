"""Base evaluator interface and observer dispatch."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .events import EvaluationResult


class Evaluator(ABC):
    def __init__(self) -> None:
        self.observers: list[Any] = []

    def add_observer(self, observer: Any) -> None:
        self.observers.append(observer)

    @abstractmethod
    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> EvaluationResult:
        """Evaluate a model and return aggregated results."""

    def _notify(self, method_name: str, event: Any) -> None:
        for observer in self.observers:
            callback = getattr(observer, method_name, None)

            if callback is not None:
                callback(event)
