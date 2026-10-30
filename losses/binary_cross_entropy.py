"""Binary cross entropy loss."""

import numpy as np

from .loss import Loss


class BinaryCrossEntropy(Loss):
    """Binary cross entropy for binary and multilabel classification."""

    _EPSILON = 1e-7

    def validate_targets(self, targets: np.ndarray, output_size: int) -> np.ndarray:
        targets = super().validate_targets(targets, output_size)

        if np.any((targets < 0) | (targets > 1)):
            raise ValueError("BinaryCrossEntropy targets must be in [0, 1]")

        return targets

    def value_batch(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        targets = np.asarray(targets, dtype=np.float32)
        probabilities = np.clip(np.asarray(predictions, dtype=np.float32), self._EPSILON, 1.0 - self._EPSILON)
        return float(-np.mean(targets * np.log(probabilities) + (1.0 - targets) * np.log(1.0 - probabilities)))

    def gradient_batch(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        targets = np.asarray(targets, dtype=np.float32)
        probabilities = np.clip(np.asarray(predictions, dtype=np.float32), self._EPSILON, 1.0 - self._EPSILON)
        return (probabilities - targets) / (probabilities * (1.0 - probabilities) * probabilities.size)
