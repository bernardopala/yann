"""Target specifications and split-local target adapters."""

from .binary_classification_target_spec import BinaryClassificationTargetSpec
from .multiclass_classification_target_spec import MulticlassClassificationTargetSpec
from .multi_label_classification_target_spec import MultiLabelClassificationTargetSpec
from .regression_output_spec import RegressionOutputSpec
from .regression_target_spec import RegressionTargetSpec
from .target_adapter import TargetAdapter
from .target_spec import TargetSpec


__all__ = [
    "TargetAdapter",
    "TargetSpec",
    "RegressionOutputSpec",
    "RegressionTargetSpec",
    "BinaryClassificationTargetSpec",
    "MulticlassClassificationTargetSpec",
    "MultiLabelClassificationTargetSpec",
]
