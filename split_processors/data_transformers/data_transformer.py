"""Base interface for data transformations."""

from abc import ABC, abstractmethod
import numpy as np


class DataTransformer(ABC):
    @abstractmethod
    def fit(self, values: np.ndarray) -> None:
        """Fit this transformation using training values only."""

    @abstractmethod
    def transform(self, values: np.ndarray) -> np.ndarray:
        """Transform values using fitted state."""

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values)
