"""Mean squared error loss."""

import numpy as np

from .loss import Loss


class MSE(Loss):
    """Mean squared error for single- and multi-output regression."""

    def value(self, prediction: np.ndarray, target: np.ndarray) -> float:
        target = np.asarray(target, dtype=np.float32)
        return float(np.mean(np.square(prediction - target)))

    def gradient(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=np.float32)
        return 2.0 * (prediction - target) / prediction.size
