"""Schema-driven processor for tabular train/validation splits."""

from typing import Callable

import numpy as np

from .specs import ColumnSpec
from .split_processor import PreparedSplit, SplitProcessor
from .table_transformer import TableTransformer


class TabularSplitProcessor(SplitProcessor):
    def __init__(self, X: np.ndarray, Y: np.ndarray, train_indices: np.ndarray, validation_indices: np.ndarray, x_schema: list[ColumnSpec], y_schema: list[ColumnSpec]) -> None:
        super().__init__(X, Y, train_indices, validation_indices)
        self.x_transformer = TableTransformer(x_schema)
        self.y_transformer = TableTransformer(y_schema, squeeze_single_output=True, output_dtype=None)

    def prepare(self) -> PreparedSplit:
        self.x_transformer.fit(self._X_train)
        self.y_transformer.fit(self._Y_train)

        X_train = self.x_transformer.transform(self._X_train)
        Y_train = self.y_transformer.transform(self._Y_train)
        Y_train_for_scoring = self.y_transformer.transform_for_scoring(self._Y_train)
        X_validation = self.x_transformer.transform(self._X_validation)
        Y_validation_for_scoring = self.y_transformer.transform_for_scoring(self._Y_validation)
        output_size = self.y_transformer.model_output_size

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
        return self.y_transformer.inverse_transform(predictions)


def create_tabular_split_processor_factory(x_schema: list[ColumnSpec], y_schema: list[ColumnSpec]) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], TabularSplitProcessor]:
    def factory(X: np.ndarray, Y: np.ndarray, train_indices: np.ndarray, validation_indices: np.ndarray) -> TabularSplitProcessor:
        return TabularSplitProcessor(X, Y, train_indices, validation_indices, x_schema, y_schema)

    return factory
