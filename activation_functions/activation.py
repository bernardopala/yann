"""Base interface for activation functions."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):
    """Base class for an activation and its derivative."""

    def __call__(self, x):
        return self.forward(x)

    @abstractmethod
    def forward(self, x):
        """Compute the activation for a scalar or an array."""

    @abstractmethod
    def backward(self, weighted_sum, output, upstream_gradient):
        """Return ``dL/dz`` from ``dL/doutput`` for one neuron/vector."""

    def initialize_weights(self, num_inputs: int, num_outputs: int,
                           rng: np.random.Generator) -> np.ndarray:
        """Return one neuron's default Xavier-initialized input weights."""
        limit = np.sqrt(6.0 / (num_inputs + num_outputs))
        return rng.uniform(-limit, limit, size=num_inputs)
