"""Sparse categorical cross-entropy metric."""

import numpy as np

from .metric import Metric


class SparseCategoricalCrossEntropy(Metric):
    _EPSILON = 1e-7

    @property
    def name(self) -> str:
        return "cross_entropy"

    @property
    def greater_is_better(self) -> bool:
        return False

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        expected_classes = np.asarray(y_true, dtype=np.int64).reshape(-1)
        probabilities = np.asarray(y_pred)[np.arange(len(expected_classes)), expected_classes]
        return float(-np.mean(np.log(np.clip(probabilities, self._EPSILON, 1.0))))
