"""Encode categorical values as ordinal floating-point values."""

import numpy as np
from .label_encoder import LabelEncoder


class OrdinalEncoder(LabelEncoder):
    def transform(self, values: np.ndarray) -> np.ndarray:
        return super().transform(values).astype(np.float32)
