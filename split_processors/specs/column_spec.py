"""Column-level transformation specification."""

from dataclasses import dataclass
from ..data_transformers import DataTransformer


@dataclass(frozen=True)
class ColumnSpec:
    index: int
    transformers: list[DataTransformer]
    score_transformed: bool = False
