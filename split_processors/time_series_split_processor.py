"""Schema-driven processor for chronological windowed data."""

from typing import Callable

import numpy as np

from .specs import ColumnSpec, WindowSpec
from .split_processor import PreparedSplit, SplitProcessor
from .table_transformer import TableTransformer


class TimeSeriesSplitProcessor(SplitProcessor):
    def __init__(self, X: np.ndarray, Y: np.ndarray, train_indices: np.ndarray, validation_indices: np.ndarray, x_schema: list[ColumnSpec], y_schema: list[ColumnSpec], window_spec: WindowSpec) -> None:
        super().__init__(X, Y, train_indices, validation_indices)
        self.x_transformer = TableTransformer(x_schema)
        self.y_transformer = TableTransformer(y_schema, squeeze_single_output=True, output_dtype=None)
        self.window_spec = window_spec

    def prepare(self) -> PreparedSplit:
        self.x_transformer.fit(self._X_train)
        self.y_transformer.fit(self._Y_train)

        if self.window_spec.horizon > 1 and self.y_transformer.has_categorical_output:
            raise ValueError("horizon > 1 currently supports regression targets only")

        X_train_rows = self.x_transformer.transform(self._X_train)
        X_validation_rows = self.x_transformer.transform(self._X_validation)
        Y_train_rows = self.y_transformer.transform(self._Y_train)
        Y_validation_rows = self.y_transformer.transform(self._Y_validation)
        Y_train_scoring_rows = self.y_transformer.transform_for_scoring(self._Y_train)
        Y_validation_scoring_rows = self.y_transformer.transform_for_scoring(self._Y_validation)

        X_train, Y_train = self._make_windows(X_train_rows, Y_train_rows)
        X_validation, Y_validation = self._make_windows(X_validation_rows, Y_validation_rows)
        _, Y_train_for_scoring = self._make_windows(X_train_rows, Y_train_scoring_rows)
        _, Y_validation_for_scoring = self._make_windows(X_validation_rows, Y_validation_scoring_rows)
        output_size = self.y_transformer.model_output_size * self.window_spec.horizon

        return PreparedSplit(
            X_train=X_train,
            Y_train=Y_train,
            Y_train_for_scoring=Y_train_for_scoring,
            X_validation=X_validation,
            Y_validation_for_scoring=Y_validation_for_scoring,
            input_size=X_train.shape[1],
            output_size=output_size,
        )

    def predictions_for_scoring(self, predictions: np.ndarray) -> np.ndarray:
        predictions = np.asarray(predictions)
        if self.window_spec.horizon == 1:
            return self.y_transformer.inverse_transform(predictions)

        target_width = self.y_transformer.model_output_size
        expected_width = self.window_spec.horizon * target_width
        if predictions.ndim != 2 or predictions.shape[1] != expected_width:
            raise ValueError(f"prediction width does not match expected multi-step output width {expected_width}")

        step_predictions = predictions.reshape(len(predictions), self.window_spec.horizon, target_width)
        restored_steps = [self.y_transformer.inverse_transform(step_predictions[:, step]) for step in range(self.window_spec.horizon)]
        return np.stack(restored_steps, axis=1).reshape(len(predictions), -1)

    def _make_windows(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X)
        Y = np.asarray(Y)
        sample_count = len(X) - self.window_spec.lookback - self.window_spec.horizon + 1

        if sample_count <= 0:
            raise ValueError("not enough samples for the requested time-series window")

        starts = range(0, sample_count, self.window_spec.stride)
        features = [X[start:start + self.window_spec.lookback] for start in starts]
        targets = [Y[start + self.window_spec.lookback:start + self.window_spec.lookback + self.window_spec.horizon] for start in starts]
        features = np.asarray(features, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)

        if self.window_spec.horizon == 1:
            targets = targets[:, 0]
        elif targets.ndim == 3:
            targets = targets.reshape(len(targets), -1)

        # The current MLP-based training pipeline accepts only two-dimensional samples, so every time-series window is flattened.
        features = features.reshape(len(features), -1)

        return features, targets


def create_time_series_split_processor_factory(x_schema: list[ColumnSpec], y_schema: list[ColumnSpec], window_spec: WindowSpec) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], TimeSeriesSplitProcessor]:
    def factory(X: np.ndarray, Y: np.ndarray, train_indices: np.ndarray, validation_indices: np.ndarray) -> TimeSeriesSplitProcessor:
        return TimeSeriesSplitProcessor(X, Y, train_indices, validation_indices, x_schema, y_schema, window_spec)

    return factory
