"""Events and results emitted by evaluators."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FoldEpochEndEvent:
    split_label: str
    fold_index: int
    epoch_event: Any
    training_scores: dict[str, float]
    validation_scores: dict[str, float]


@dataclass(frozen=True)
class FoldEndEvent:
    split_label: str
    fold_index: int
    training_result: Any
    training_scores: dict[str, float]
    scores: dict[str, float]
    train_size: int
    validation_size: int


@dataclass(frozen=True)
class SplitResult:
    split_label: str
    fold_index: int
    training_result: Any
    training_scores: dict[str, float]
    scores: dict[str, float]


@dataclass(frozen=True)
class EvaluationResult:
    splits: list[SplitResult]

    def scores(self, metric_name: str) -> np.ndarray:
        return np.asarray([split.scores[metric_name] for split in self.splits], dtype=np.float64)

    def training_scores(self, metric_name: str) -> np.ndarray:
        return np.asarray([split.training_scores[metric_name] for split in self.splits], dtype=np.float64)

    def mean_validation_score(self, metric_name: str) -> float:
        return float(np.mean(self.scores(metric_name)))

    def std_validation_score(self, metric_name: str) -> float:
        return float(np.std(self.scores(metric_name)))

    def mean_training_score(self, metric_name: str) -> float:
        return float(np.mean(self.training_scores(metric_name)))

    def std_training_score(self, metric_name: str) -> float:
        return float(np.std(self.training_scores(metric_name)))
