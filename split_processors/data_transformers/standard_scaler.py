"""Standardize numeric values."""

import numpy as np
from .data_transformer import DataTransformer


class StandardScaler(DataTransformer):
    def fit(self, values: np.ndarray) -> None:
        numeric_values = np.asarray(values, dtype=np.float32)
        scale = np.std(numeric_values, axis=0)

        self.mean = np.mean(numeric_values, axis=0)
        self.scale = np.where(scale == 0.0, 1.0, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        numeric_values = np.asarray(values, dtype=np.float32)
        return (numeric_values - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * self.scale + self.mean
