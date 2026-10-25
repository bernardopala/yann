"""Base contract for evaluation metrics."""

from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
    """Calculate one scalar quality measure for a complete dataset."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name used as a key in evaluation results."""

    @property
    def greater_is_better(self) -> bool:
        return True

    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the metric for all samples."""
