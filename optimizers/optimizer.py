"""Base interface and shared validation helpers for optimizers."""

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np


class Optimizer(ABC):
    """Base class for optimizers which update parameters in place."""

    def __init__(self, learning_rate: float) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.learning_rate = float(learning_rate)
        self._state: dict[tuple[int, str], dict[str, np.ndarray]] = {}
        self._step_count = 0

    def reset_state(self) -> None:
        self._state.clear()
        self._step_count = 0

    def step(self, parameters: Iterable[tuple[tuple[int, str], np.ndarray, np.ndarray]]) -> None:
        self._step_count += 1
        for key, parameter, gradient in parameters:
            state = self._state.setdefault(key, {})
            self._update(parameter, gradient, state)

    @abstractmethod
    def _update(self, parameter: np.ndarray, gradient: np.ndarray, state: dict[str, np.ndarray]) -> None:
        pass


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_unit_interval(name: str, value: float) -> None:
    if not 0 <= value < 1:
        raise ValueError(f"{name} must be in [0, 1)")
