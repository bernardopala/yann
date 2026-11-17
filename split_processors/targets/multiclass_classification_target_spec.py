"""Multiclass target specification and adapter."""

from copy import deepcopy
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


def _validate_classes(classes: list[Any], minimum: int) -> None:
    if not isinstance(classes, list):
        raise TypeError("classes must be a list")
    if len(classes) < minimum:
        raise ValueError(f"at least {minimum} classes are required")
    for class_value in classes:
        _require_hashable(class_value, "class values")
    if len(set(classes)) != len(classes):
        raise ValueError("classes must be unique")


def _single_target(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        return values
    if values.ndim == 2 and values.shape[1] == 1:
        return values[:, 0]
    raise ValueError("target must have shape (samples,) or (samples, 1)")


@dataclass
class MulticlassClassificationTargetSpec(TargetSpec):
    """Configuration for one mutually-exclusive multiclass target."""

    classes: list[Any] | None = None

    def __post_init__(self) -> None:
        if self.classes is not None:
            _validate_classes(self.classes, minimum=2)

    def create_adapter(self) -> TargetAdapter:
        return _MulticlassClassificationTargetAdapter(deepcopy(self.classes))


class _MulticlassClassificationTargetAdapter(TargetAdapter):
    def __init__(self, classes: list[Any] | None) -> None:
        self.classes = classes
        self.mapping: dict[Any, int] = {}
        self._fitted = False

    @property
    def output_size(self) -> int:
        if not self._fitted:
            raise RuntimeError("target adapter must be fitted before output_size is available")
        return len(self.classes)

    def fit(self, y_train: np.ndarray) -> None:
        values = _single_target(y_train)
        if self.classes is None:
            self.classes = list(np.unique(values))
            _validate_classes(self.classes, minimum=2)

        self.mapping = {class_value: index for index, class_value in enumerate(self.classes)}
        self._encode(values)
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
        if predictions.ndim != 2 or predictions.shape[1] != self.output_size:
            raise ValueError(f"multiclass predictions must have shape (samples, {self.output_size})")
        return predictions

    def _encode(self, values: np.ndarray) -> np.ndarray:
        values = _single_target(values)
        encoded = []

        for value in values:
            if value not in self.mapping:
                raise ValueError(f"Unknown multiclass class {value!r}")
            encoded.append(self.mapping[value])

        return np.asarray(encoded, dtype=np.int64)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("target adapter must be fitted before use")
