"""Root mean squared error metric."""

import numpy as np

from .mean_squared_error import MeanSquaredError
from .metric import Metric


class RootMeanSquaredError(Metric):
    @property
    def name(self) -> str:
        return "rmse"

    @property
    def greater_is_better(self) -> bool:
        return False

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(MeanSquaredError().compute(y_true, y_pred)))
