"""Encode categorical labels as integer values."""

import numpy as np
from .data_transformer import DataTransformer


class LabelEncoder(DataTransformer):
    def fit(self, values: np.ndarray) -> None:
        self.categories = np.unique(np.asarray(values))
        self.mapping = {category: index for index, category in enumerate(self.categories)}

    def transform(self, values: np.ndarray) -> np.ndarray:
        encoded = []

        for value in np.asarray(values).reshape(-1):
            if value not in self.mapping:
                raise ValueError(f"Unknown category {value!r}")

            encoded.append(self.mapping[value])

        return np.asarray(encoded, dtype=np.int64)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values)
