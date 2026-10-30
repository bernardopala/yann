"""Mean squared error loss."""

import numpy as np

from .loss import Loss


class MSE(Loss):
    """Mean squared error for single- and multi-output regression."""

    def value_batch(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        return float(np.mean(np.square(np.asarray(predictions, dtype=np.float32) - np.asarray(targets, dtype=np.float32))))

    def gradient_batch(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        predictions = np.asarray(predictions, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        return 2.0 * (predictions - targets) / predictions.size
