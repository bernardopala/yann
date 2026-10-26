"""Apply independent transformation pipelines to selected columns."""

from copy import deepcopy

import numpy as np

from .specs import ColumnSpec
from .data_transformers import LabelEncoder, OneHotEncoder


class TableTransformer:
    def __init__(self, schema: list[ColumnSpec], squeeze_single_output: bool = False, output_dtype: np.dtype | None = np.dtype(np.float32)) -> None:
        if not isinstance(schema, list) or not schema:
            raise ValueError("schema must be a non-empty list of ColumnSpec")
        if not all(isinstance(spec, ColumnSpec) for spec in schema):
            raise TypeError("schema must contain ColumnSpec objects")
        if len({spec.index for spec in schema}) != len(schema):
            raise ValueError("column indices must be unique")

        self.schema = deepcopy(schema)
        self.squeeze_single_output = squeeze_single_output
        self.output_dtype = output_dtype
        self.output_widths: dict[int, int] = {}
        self.model_output_size = 0
        self.has_categorical_output = False

    def fit(self, values: np.ndarray) -> None:
        for spec in self.schema:
            current = self._column(values, spec.index)

            for transformer in spec.transformers:
                transformer.fit(current)
                current = transformer.transform(current)
            self.output_widths[spec.index] = self._as_2d(current).shape[1]
            if any(isinstance(transformer, LabelEncoder) for transformer in spec.transformers):
                self.has_categorical_output = True
                encoder = next(transformer for transformer in spec.transformers if isinstance(transformer, LabelEncoder))
                self.model_output_size += len(encoder.categories)
            elif any(isinstance(transformer, OneHotEncoder) for transformer in spec.transformers):
                self.has_categorical_output = True
                encoder = next(transformer for transformer in spec.transformers if isinstance(transformer, OneHotEncoder))
                self.model_output_size += len(encoder.categories)
            else:
                self.model_output_size += self.output_widths[spec.index]

    def transform(self, values: np.ndarray) -> np.ndarray:
        outputs = [self._transform_spec(values, spec) for spec in self.schema]
        outputs = [output for output in outputs if output.shape[1] > 0]

        if not outputs:
            raise ValueError("schema produced no output columns")

        result = np.concatenate(outputs, axis=1)
        if self.output_dtype is not None:
            result = result.astype(self.output_dtype)

        if self.squeeze_single_output and result.shape[1] == 1:
            return result[:, 0]

        return result

    def transform_for_scoring(self, values: np.ndarray) -> np.ndarray:
        outputs = []

        for spec in self.schema:
            if spec.score_transformed:
                output = self._transform_spec(values, spec)
            else:
                output = self._column(values, spec.index)
                output = self._as_2d(output)
            if output.shape[1] > 0:
                outputs.append(output)

        if not outputs:
            raise ValueError("schema produced no scoring columns")

        result = np.concatenate(outputs, axis=1)

        if self.squeeze_single_output and result.shape[1] == 1:
            return result[:, 0]

        return result

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        current_values = self._as_2d(values)
        if self.has_categorical_output:
            return np.asarray(values)
        result = []
        offset = 0

        for spec in self.schema:
            width = self._output_width(spec)
            if width == 0:
                continue
            current = current_values[:, offset:offset + width]
            offset += width

            for transformer in reversed(spec.transformers):
                current = self._as_2d(transformer.inverse_transform(current))
            result.append(current)

        if offset != current_values.shape[1]:
            raise ValueError("prediction width does not match schema output width")

        restored = np.concatenate(result, axis=1)
        if self.squeeze_single_output and restored.shape[1] == 1:
            return restored[:, 0]

        return restored

    def _transform_spec(self, values: np.ndarray, spec: ColumnSpec) -> np.ndarray:
        current = self._column(values, spec.index)
        for transformer in spec.transformers:
            current = transformer.transform(current)

        return self._as_2d(current)

    def _output_width(self, spec: ColumnSpec) -> int:
        if spec.index not in self.output_widths:
            raise RuntimeError("The table transformer must be fitted before inverse_transform")

        return self.output_widths[spec.index]

    @staticmethod
    def _column(values: np.ndarray, index: int) -> np.ndarray:
        values = np.asarray(values)
        if values.ndim == 1:
            if index != 0:
                raise IndexError("one-dimensional data has only column index 0")
            return values

        if values.ndim != 2:
            raise ValueError("tabular data must be one- or two-dimensional")

        return values[:, index]

    @staticmethod
    def _as_2d(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        if values.ndim == 1:
            return values.reshape(-1, 1)

        if values.ndim != 2:
            raise ValueError("transformer output must be one- or two-dimensional")

        return values
