"""Mean absolute error metric."""

import numpy as np

from .metric import Metric


class MeanAbsoluteError(Metric):
    @property
    def name(self) -> str:
        return "mae"

    @property
    def greater_is_better(self) -> bool:
        return False

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
