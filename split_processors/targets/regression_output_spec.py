"""Configuration for preprocessing one regression output."""

from dataclasses import dataclass, field

from ..data_transformers import DataTransformer


@dataclass
class RegressionOutputSpec:
    """Preprocessing pipeline for one positional regression output."""

    transformers: list[DataTransformer] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.transformers, list):
            raise TypeError("transformers must be a list of DataTransformer")
        if not all(isinstance(transformer, DataTransformer) for transformer in self.transformers):
            raise TypeError("transformers must contain DataTransformer objects")
