"""Accuracy metric for sparse multiclass predictions."""

import numpy as np

from .metric import Metric


class Accuracy(Metric):
    @property
    def name(self) -> str:
        return "accuracy"

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        expected_classes = np.asarray(y_true, dtype=np.int64).reshape(-1)
        predicted_classes = np.argmax(np.asarray(y_pred), axis=1)
        return float(np.mean(expected_classes == predicted_classes))
