"""Sparse categorical cross entropy loss."""

import numpy as np

from .loss import Loss


class SparseCategoricalCrossEntropy(Loss):
    """Cross entropy for integer class labels and a probability-vector output."""

    _EPSILON = 1e-7

    def validate_targets(self, targets: np.ndarray, output_size: int) -> np.ndarray:
        targets = np.asarray(targets)

        if targets.ndim != 1 or not np.issubdtype(targets.dtype, np.integer):
            raise ValueError("SparseCategoricalCrossEntropy requires one integer label per sample")

        if np.any(targets < 0) or np.any(targets >= output_size):
            raise ValueError(f"Sparse class labels must be in [0, {output_size})")

        return targets

    def value(self, prediction: np.ndarray, target: int) -> float:
        return float(-np.log(np.clip(prediction[int(target)], self._EPSILON, 1.0)))

    def gradient(self, prediction: np.ndarray, target: int) -> np.ndarray:
        result = np.zeros_like(prediction)
        result[int(target)] = -1.0 / np.clip(prediction[int(target)], self._EPSILON, 1.0)
        return result
