"""Pass input values through unchanged."""

import numpy as np
from .data_transformer import DataTransformer


class Passthrough(DataTransformer):
    def fit(self, values: np.ndarray) -> None:
        return None

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values)
