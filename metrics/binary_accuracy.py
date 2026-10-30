"""Accuracy metric for binary classification probabilities."""

import numpy as np

from .metric import Metric


class BinaryAccuracy(Metric):
    """Calculate accuracy after thresholding probabilities at 0.5."""

    @property
    def name(self) -> str:
        return "accuracy"

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        expected = np.asarray(y_true, dtype=np.float32).reshape(-1)
        probabilities = np.asarray(y_pred, dtype=np.float32).reshape(-1)
        predicted = probabilities >= 0.5
        return float(np.mean(predicted == (expected >= 0.5)))
