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

    def value_batch(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        predictions = np.asarray(predictions, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.intp)
        probabilities = np.clip(predictions[np.arange(len(targets)), targets], self._EPSILON, 1.0)
        return float(-np.mean(np.log(probabilities)))

    def gradient_batch(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        predictions = np.asarray(predictions, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.intp)
        result = np.zeros_like(predictions)
        result[np.arange(len(targets)), targets] = -1.0 / np.clip(predictions[np.arange(len(targets)), targets], self._EPSILON, 1.0)
        return result / len(targets)
