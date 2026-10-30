"""Base interface for training loss functions."""

from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    """A differentiable objective evaluated for a batch of training examples."""

    def validate_targets(self, targets: np.ndarray, output_size: int) -> np.ndarray:
        """Validate and normalize a batch of targets for the network output."""
        targets = np.asarray(targets)

        if output_size == 1 and targets.ndim == 1:
            targets = targets.reshape(-1, 1)

        expected_shape = (len(targets), output_size)

        if targets.shape != expected_shape:
            raise ValueError(f"Y must have shape {expected_shape}")

        return targets.astype(np.float32)

    @abstractmethod
    def value_batch(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Return the mean loss for a batch of examples."""

    @abstractmethod
    def gradient_batch(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Return the gradient of the mean batch loss with respect to predictions."""
