"""Encode categorical values as one-hot vectors."""

import numpy as np
from .data_transformer import DataTransformer


class OneHotEncoder(DataTransformer):
    def __init__(self, handle_unknown: str = "error") -> None:
        if handle_unknown not in {"error", "ignore"}:
            raise ValueError("handle_unknown must be 'error' or 'ignore'")

        self.handle_unknown = handle_unknown

    def fit(self, values: np.ndarray) -> None:
        self.categories = np.unique(np.asarray(values))
        self.category_indices = {category: index for index, category in enumerate(self.categories)}

    def transform(self, values: np.ndarray) -> np.ndarray:
        result = np.zeros((len(np.asarray(values).reshape(-1)), len(self.categories)), dtype=np.float32)

        for row, value in enumerate(np.asarray(values).reshape(-1)):
            if value not in self.category_indices:
                if self.handle_unknown == "ignore":
                    continue
                raise ValueError(f"Unknown category {value!r}")
            result[row, self.category_indices[value]] = 1.0

        return result
