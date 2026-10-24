"""Base splitter interface."""

from abc import ABC, abstractmethod

import numpy as np


class Splitter(ABC):
    @property
    @abstractmethod
    def label(self) -> str:
        """Return the human-readable name used in evaluator notifications."""

    def format_split_label(self, split_index: int) -> str:
        return f"{self.label} {split_index}"

    @abstractmethod
    def split(self, X: np.ndarray, Y: np.ndarray):
        """Yield train and validation index arrays."""
