"""Multilabel target specification and adapter."""

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


def _label_sets(values: np.ndarray) -> list[list[Any]]:
    rows = list(values)
    result = []
    for row in rows:
        if isinstance(row, (str, bytes)):
            raise ValueError("multilabel targets must contain label collections, not scalar strings")
        try:
            result.append(list(row))
        except TypeError as error:
            raise ValueError("multilabel targets must be a binary matrix or label collections") from error
    return result


def _unique_labels(rows: list[list[Any]]) -> list[Any]:
    unique = []
    seen = set()
    for row in rows:
        for label in row:
            _require_hashable(label, "multilabel classes")
            if label not in seen:
                seen.add(label)
                unique.append(label)
    return unique


@dataclass
class MultiLabelClassificationTargetSpec(TargetSpec):
    """Configuration for independent binary labels."""

    classes: list[Any] | None = None

    def __post_init__(self) -> None:
        if self.classes is not None:
            _validate_classes(self.classes, minimum=1)

    def create_adapter(self) -> TargetAdapter:
        return _MultiLabelClassificationTargetAdapter(deepcopy(self.classes))


class _MultiLabelClassificationTargetAdapter(TargetAdapter):
    def __init__(self, classes: list[Any] | None) -> None:
        self.classes = classes
        self.mapping: dict[Any, int] = {}
        self._matrix_input = False
        self._fitted = False

    @property
    def output_size(self) -> int:
        if not self._fitted:
            raise RuntimeError("target adapter must be fitted before output_size is available")
        return len(self.classes)

    def fit(self, y_train: np.ndarray) -> None:
        values = np.asarray(y_train)
        self._matrix_input = values.ndim == 2 and values.dtype != object

        if self._matrix_input:
            if self.classes is None:
                self.classes = list(range(values.shape[1]))
            if values.shape[1] != len(self.classes):
                raise ValueError("multilabel target width must match the number of classes")
            self._validate_matrix(values)
        else:
            rows = _label_sets(y_train)
            if self.classes is None:
                self.classes = _unique_labels(rows)
                _validate_classes(self.classes, minimum=1)
            self.mapping = {class_value: index for index, class_value in enumerate(self.classes)}
            self._encode_sets(rows)

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
            raise ValueError(f"multilabel predictions must have shape (samples, {self.output_size})")
        return predictions

    def _encode(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim == 2 and array.dtype != object:
            if array.shape[1] != self.output_size:
                raise ValueError("multilabel target width must match the number of classes")
            self._validate_matrix(array)
            return array.astype(np.float32)
        return self._encode_sets(_label_sets(values))

    def _encode_sets(self, rows: list[list[Any]]) -> np.ndarray:
        encoded = np.zeros((len(rows), len(self.classes)), dtype=np.float32)
        mapping = {class_value: index for index, class_value in enumerate(self.classes)}

        for row_index, labels in enumerate(rows):
            for label in labels:
                if label not in mapping:
                    raise ValueError(f"Unknown multilabel class {label!r}")
                encoded[row_index, mapping[label]] = 1.0

        return encoded

    @staticmethod
    def _validate_matrix(values: np.ndarray) -> None:
        numeric = np.asarray(values, dtype=np.float32)
        if np.any((numeric != 0.0) & (numeric != 1.0)):
            raise ValueError("multilabel matrix targets must contain only 0 and 1")

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("target adapter must be fitted before use")
