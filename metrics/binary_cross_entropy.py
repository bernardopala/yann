"""Binary cross-entropy metric."""

import numpy as np

from .metric import Metric


class BinaryCrossEntropy(Metric):
    """Calculate binary cross-entropy for binary or multilabel probabilities."""

    _EPSILON = 1e-7

    @property
    def name(self) -> str:
        return "cross_entropy"

    @property
    def greater_is_better(self) -> bool:
        return False

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        targets = np.asarray(y_true, dtype=np.float32)
        probabilities = np.clip(np.asarray(y_pred, dtype=np.float32), self._EPSILON, 1.0 - self._EPSILON)
        return float(-np.mean(targets * np.log(probabilities) + (1.0 - targets) * np.log(1.0 - probabilities)))
