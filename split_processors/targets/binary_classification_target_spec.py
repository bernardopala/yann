"""Binary-classification target specification and adapter."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .target_adapter import TargetAdapter
from .target_spec import TargetSpec


def _require_hashable(value: Any, name: str) -> None:
    try:
        hash(value)
    except TypeError as error:
        raise TypeError(f"{name} must be hashable") from error


def _single_target(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        return values
    if values.ndim == 2 and values.shape[1] == 1:
        return values[:, 0]
    raise ValueError("target must have shape (samples,) or (samples, 1)")


@dataclass
class BinaryClassificationTargetSpec(TargetSpec):
    """Configuration for a single binary target."""

    negative_class: Any = 0
    positive_class: Any = 1

    def __post_init__(self) -> None:
        _require_hashable(self.negative_class, "negative_class")
        _require_hashable(self.positive_class, "positive_class")
        if self.negative_class == self.positive_class:
            raise ValueError("negative_class and positive_class must be different")

    @property
    def classes(self) -> tuple[Any, Any]:
        return self.negative_class, self.positive_class

    def create_adapter(self) -> TargetAdapter:
        return _BinaryClassificationTargetAdapter(self.negative_class, self.positive_class)


class _BinaryClassificationTargetAdapter(TargetAdapter):
    def __init__(self, negative_class: Any, positive_class: Any) -> None:
        self.negative_class = negative_class
        self.positive_class = positive_class
        self._fitted = False

    @property
    def output_size(self) -> int:
        return 1

    def fit(self, y_train: np.ndarray) -> None:
        self._encode(y_train)
        self._fitted = True

    def targets_for_training(self, values: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return self._encode(values)

    def targets_for_scoring(self, values: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return self._encode(values)

    def predictions_for_scoring(self, predictions: np.ndarray) -> np.ndarray:
        self._require_fitted()
        predictions = np.asarray(predictions, dtype=np.float32)
        if predictions.ndim == 2 and predictions.shape[1] == 1:
            return predictions[:, 0]
        if predictions.ndim != 1:
            raise ValueError("binary predictions must have shape (samples,) or (samples, 1)")
        return predictions

    def _encode(self, values: np.ndarray) -> np.ndarray:
        values = _single_target(values)
        encoded = np.empty(len(values), dtype=np.float32)

        for index, value in enumerate(values):
            if value == self.negative_class:
                encoded[index] = 0.0
            elif value == self.positive_class:
                encoded[index] = 1.0
            else:
                raise ValueError(f"Unknown binary class {value!r}")

        return encoded

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("target adapter must be fitted before use")
