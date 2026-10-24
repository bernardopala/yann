"""Contracts for split-level dataset preparation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreparedSplit:
    X_train: np.ndarray
    Y_train: np.ndarray
    Y_train_for_scoring: np.ndarray
    X_validation: np.ndarray
    Y_validation_for_scoring: np.ndarray
    input_size: int
    output_size: int


class SplitProcessor(ABC):
    def __init__(self, X: np.ndarray, Y: np.ndarray, train_indices: np.ndarray, validation_indices: np.ndarray) -> None:
        X = np.asarray(X)
        Y = np.asarray(Y)
        train_indices = np.asarray(train_indices)
        validation_indices = np.asarray(validation_indices)

        if X.ndim < 1 or Y.ndim < 1:
            raise ValueError("X and Y must contain a sample dimension")
        if len(X) != len(Y):
            raise ValueError("X and Y must contain the same number of samples")
        if train_indices.ndim != 1 or validation_indices.ndim != 1:
            raise ValueError("fold indices must be one-dimensional")
        if not np.issubdtype(train_indices.dtype, np.integer) or not np.issubdtype(validation_indices.dtype, np.integer):
            raise TypeError("fold indices must be integers")
        if len(train_indices) == 0 or len(validation_indices) == 0:
            raise ValueError("train and validation partitions cannot be empty")
        if np.any(train_indices < 0) or np.any(train_indices >= len(X)) or np.any(validation_indices < 0) or np.any(validation_indices >= len(X)):
            raise IndexError("fold indices are out of bounds")
        if np.unique(train_indices).size != len(train_indices) or np.unique(validation_indices).size != len(validation_indices):
            raise ValueError("fold indices must be unique within each partition")
        if np.intersect1d(train_indices, validation_indices).size:
            raise ValueError("train and validation partitions must be disjoint")

        self._X_train = X[train_indices]
        self._Y_train = Y[train_indices]
        self._X_validation = X[validation_indices]
        self._Y_validation = Y[validation_indices]

    @abstractmethod
    def prepare(self) -> PreparedSplit:
        """Prepare one fold using only the training partition for fitting."""

    @abstractmethod
    def predictions_for_scoring(self, predictions: np.ndarray) -> np.ndarray:
        """Convert model predictions to the scale expected by evaluation metrics."""
