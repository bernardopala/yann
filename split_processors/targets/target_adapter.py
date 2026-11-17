"""Base interface for split-local target adapters."""

from abc import ABC, abstractmethod

import numpy as np


class TargetAdapter(ABC):
    """Prepare targets and model predictions for one data split."""

    @property
    @abstractmethod
    def output_size(self) -> int:
        """Return the number of outputs required from the model."""

    @property
    def supports_multi_step(self) -> bool:
        """Whether a time-series processor may repeat this target over a horizon."""
        return False

    @abstractmethod
    def fit(self, y_train: np.ndarray) -> None:
        """Fit target preprocessing using training targets only."""

    @abstractmethod
    def targets_for_training(self, values: np.ndarray) -> np.ndarray:
        """Convert raw targets to the representation expected by the loss."""

    @abstractmethod
    def targets_for_scoring(self, values: np.ndarray) -> np.ndarray:
        """Convert raw targets to the representation expected by metrics."""

    @abstractmethod
    def predictions_for_scoring(self, predictions: np.ndarray) -> np.ndarray:
        """Convert model predictions to the representation expected by metrics."""
