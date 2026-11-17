"""Schema-driven processor for tabular train/validation splits."""

from typing import Callable

import numpy as np

from .specs import ColumnSpec
from .split_processor import PreparedSplit, SplitProcessor
from .table_transformer import TableTransformer
from .targets import TargetAdapter, TargetSpec


class TabularSplitProcessor(SplitProcessor):
    def __init__(self, X: np.ndarray, Y: np.ndarray, train_indices: np.ndarray, validation_indices: np.ndarray, x_schema: list[ColumnSpec], target_spec: TargetSpec) -> None:
        super().__init__(X, Y, train_indices, validation_indices)
        if not isinstance(target_spec, TargetSpec):
            raise TypeError("target_spec must be a TargetSpec")
        self.x_transformer = TableTransformer(x_schema)
        self.target_adapter: TargetAdapter = target_spec.create_adapter()

    def prepare(self) -> PreparedSplit:
        self.x_transformer.fit(self._X_train)
        self.target_adapter.fit(self._Y_train)

        X_train = self.x_transformer.transform(self._X_train)
        Y_train = self.target_adapter.targets_for_training(self._Y_train)
        Y_train_for_scoring = self.target_adapter.targets_for_scoring(self._Y_train)
        X_validation = self.x_transformer.transform(self._X_validation)
        Y_validation_for_scoring = self.target_adapter.targets_for_scoring(self._Y_validation)

        return PreparedSplit(
            X_train=X_train,
            Y_train=Y_train,
            Y_train_for_scoring=Y_train_for_scoring,
            X_validation=X_validation,
            Y_validation_for_scoring=Y_validation_for_scoring,
            input_size=X_train.shape[1],
            output_size=self.target_adapter.output_size,
        )

    def predictions_for_scoring(self, predictions: np.ndarray) -> np.ndarray:
        return self.target_adapter.predictions_for_scoring(predictions)


def create_tabular_split_processor_factory(x_schema: list[ColumnSpec], target_spec: TargetSpec) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], TabularSplitProcessor]:
    def factory(X: np.ndarray, Y: np.ndarray, train_indices: np.ndarray, validation_indices: np.ndarray) -> TabularSplitProcessor:
        return TabularSplitProcessor(X, Y, train_indices, validation_indices, x_schema, target_spec)

    return factory
