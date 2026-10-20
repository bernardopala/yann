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

    def value(self, prediction: np.ndarray, target: np.ndarray) -> float:
        target = np.asarray(target, dtype=np.float32)
        probabilities = np.clip(prediction, self._EPSILON, 1.0 - self._EPSILON)
        return float(-np.mean(target * np.log(probabilities) + (1.0 - target) * np.log(1.0 - probabilities)))

    def gradient(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=np.float32)
        probabilities = np.clip(prediction, self._EPSILON, 1.0 - self._EPSILON)
        return (probabilities - target) / (probabilities * (1.0 - probabilities) * prediction.size)
