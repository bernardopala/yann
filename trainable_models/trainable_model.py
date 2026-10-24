"""Common interface for models that can be trained and evaluated."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class TrainableModel(ABC):
    @abstractmethod
    def train(self, X: np.ndarray, Y: np.ndarray) -> Any:
        """Train the model and return a training result."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for a batch of samples."""

    @abstractmethod
    def add_observer(self, observer: Any) -> None:
        """Subscribe an observer to the model's training events."""
