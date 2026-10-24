"""Contracts for fold-level dataset preparation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreparedFold:
    X_train: np.ndarray
    Y_train: np.ndarray
    X_validation: np.ndarray
    scoring_Y_train: np.ndarray
    scoring_Y_validation: np.ndarray
    input_size: int
    output_size: int


class FoldProcessor(ABC):
    @abstractmethod
    def prepare(self, X_train: np.ndarray, Y_train: np.ndarray, X_validation: np.ndarray, Y_validation: np.ndarray) -> PreparedFold:
        """Prepare one fold using only the training partition for fitting."""

    @abstractmethod
    def predictions_for_scoring(self, predictions: np.ndarray) -> np.ndarray:
        """Convert model predictions to the scale expected by evaluation metrics."""
