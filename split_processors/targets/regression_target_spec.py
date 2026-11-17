"""Regression target specification and adapter."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .regression_output_spec import RegressionOutputSpec
from .target_adapter import TargetAdapter
from .target_spec import TargetSpec


def _regression_columns(values: np.ndarray, output_size: int) -> list[np.ndarray]:
    values = np.asarray(values)
    if values.ndim == 1:
        if output_size != 1:
            raise ValueError(f"Y must contain {output_size} columns")
        return [values]
    if values.ndim != 2 or values.shape[1] != output_size:
        raise ValueError(f"Y must have shape (samples, {output_size})")
    return [values[:, index] for index in range(output_size)]


def _prediction_columns(predictions: np.ndarray, output_size: int) -> list[np.ndarray]:
    predictions = np.asarray(predictions)
    if predictions.ndim == 1:
        if output_size != 1:
            raise ValueError(f"predictions must contain {output_size} columns")
        return [predictions]
    if predictions.ndim != 2 or predictions.shape[1] != output_size:
        raise ValueError(f"predictions must have shape (samples, {output_size})")
    return [predictions[:, index] for index in range(output_size)]


def _single_output(values: np.ndarray, source: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        return values
    if values.ndim == 2 and values.shape[1] == 1:
        return values[:, 0]
    raise ValueError(f"{source} must preserve one value per regression output")


def _combine_columns(columns: list[np.ndarray], squeeze: bool, dtype: np.dtype) -> np.ndarray:
    result = np.column_stack([np.asarray(column).reshape(-1) for column in columns]).astype(dtype)
    if squeeze:
        return result[:, 0]
    return result


@dataclass
class RegressionTargetSpec(TargetSpec):
    """Configuration for one- or multi-output numeric regression."""

    outputs: list[RegressionOutputSpec]

    def __post_init__(self) -> None:
        if not isinstance(self.outputs, list):
            raise TypeError("outputs must be a list of RegressionOutputSpec")
        if not self.outputs:
            raise ValueError("outputs cannot be empty")
        if not all(isinstance(output, RegressionOutputSpec) for output in self.outputs):
            raise TypeError("outputs must contain RegressionOutputSpec objects")

    def create_adapter(self) -> TargetAdapter:
        return _RegressionTargetAdapter(deepcopy(self.outputs))


class _RegressionTargetAdapter(TargetAdapter):
    def __init__(self, outputs: list[RegressionOutputSpec]) -> None:
        self.outputs = outputs
        self._fitted = False

    @property
    def output_size(self) -> int:
        return len(self.outputs)

    @property
    def supports_multi_step(self) -> bool:
        return True

    def fit(self, y_train: np.ndarray) -> None:
        columns = _regression_columns(y_train, self.output_size)

        for column, output in zip(columns, self.outputs):
            current = column
            for transformer in output.transformers:
                transformer.fit(current)
                current = transformer.transform(current)
                _single_output(current, "regression target transformer")

        self._fitted = True

    def targets_for_training(self, values: np.ndarray) -> np.ndarray:
        self._require_fitted()
        columns = _regression_columns(values, self.output_size)
        transformed = []

        for column, output in zip(columns, self.outputs):
            current = column
            for transformer in output.transformers:
                current = transformer.transform(current)
            transformed.append(_single_output(current, "regression target transformer"))

        return _combine_columns(transformed, squeeze=self.output_size == 1, dtype=np.float32)

    def targets_for_scoring(self, values: np.ndarray) -> np.ndarray:
        self._require_fitted()
        columns = _regression_columns(values, self.output_size)
        return _combine_columns(columns, squeeze=self.output_size == 1, dtype=np.float32)

    def predictions_for_scoring(self, predictions: np.ndarray) -> np.ndarray:
        self._require_fitted()
        columns = _prediction_columns(predictions, self.output_size)
        restored = []

        for column, output in zip(columns, self.outputs):
            current = column
            for transformer in reversed(output.transformers):
                current = transformer.inverse_transform(current)
            restored.append(_single_output(current, "inverse target transformer"))

        return _combine_columns(restored, squeeze=self.output_size == 1, dtype=np.float32)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("target adapter must be fitted before use")
