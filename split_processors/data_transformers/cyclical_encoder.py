"""Encode periodic numeric values with sine and cosine features."""

import numpy as np
from .data_transformer import DataTransformer


class CyclicalEncoder(DataTransformer):
    def __init__(self, period: float) -> None:
        if period <= 0:
            raise ValueError("period must be positive")

        self.period = float(period)

    def fit(self, values: np.ndarray) -> None:
        return None

    def transform(self, values: np.ndarray) -> np.ndarray:
        numeric_values = np.asarray(values, dtype=np.float32).reshape(-1)
        angle = 2.0 * np.pi * numeric_values / self.period
        return np.column_stack((np.sin(angle), np.cos(angle))).astype(np.float32)
