"""Extract calendar features from datetime values."""

import numpy as np
from .data_transformer import DataTransformer


class DateTimeFeatures(DataTransformer):
    def __init__(self, features: list[str]) -> None:
        allowed_features = {"hour", "day_of_week", "month"}
        if not features or not set(features).issubset(allowed_features):
            raise ValueError("features must contain hour, day_of_week or month")

        self.features = list(features)

    def fit(self, values: np.ndarray) -> None:
        return None

    def transform(self, values: np.ndarray) -> np.ndarray:
        dates = np.asarray(values).reshape(-1).astype("datetime64[h]")
        hours = dates.astype(np.int64)
        days = dates.astype("datetime64[D]").astype(np.int64)
        months = dates.astype("datetime64[M]").astype(np.int64)
        available = {
            "hour": hours % 24,
            "day_of_week": (days + 3) % 7,
            "month": (months % 12) + 1,
        }
        return np.column_stack([available[feature] for feature in self.features]).astype(np.float32)
