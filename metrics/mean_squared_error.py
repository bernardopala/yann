"""Mean squared error metric."""

import numpy as np

from .metric import Metric


class MeanSquaredError(Metric):
    @property
    def name(self) -> str:
        return "mse"

    @property
    def greater_is_better(self) -> bool:
        return False

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.square(np.asarray(y_true) - np.asarray(y_pred))))
