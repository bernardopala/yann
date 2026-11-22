"""Events and results emitted by evaluators."""

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class SplitEpochEndEvent:
    split_label: str
    split_index: int
    epoch_event: Any
    training_scores: dict[str, float]
    validation_scores: dict[str, float]


@dataclass(frozen=True)
class SplitEndEvent:
    """Summary emitted after one split has been trained and scored.

    ``raw_*`` sizes count records selected by the splitter before preparation.
    ``prepared_*`` sizes count samples actually passed to the model and metrics.
    For time-series processors, prepared samples are windows.
    """

    split_label: str
    split_index: int
    training_result: Any
    training_scores: dict[str, float]
    scores: dict[str, float]
    raw_train_size: int
    raw_validation_size: int
    prepared_train_size: int
    prepared_validation_size: int


@dataclass(frozen=True)
class SplitResult:
    """Final scores and size metadata for one evaluated split."""

    split_label: str
    split_index: int
    training_result: Any
    training_scores: dict[str, float]
    scores: dict[str, float]
    raw_train_size: int
    raw_validation_size: int
    prepared_train_size: int
    prepared_validation_size: int


@dataclass(frozen=True)
class EvaluationResult:
    """Scores and descriptive split-level aggregates for one evaluation.

    The aggregate methods below are macro-averages over splits: every split
    contributes equally, regardless of its raw or prepared size. They are not
    scores computed from pooled out-of-fold predictions. Standard deviations
    describe the evaluated split scores and are not confidence intervals.
    """

    splits: list[SplitResult]

    def scores(self, metric_name: str) -> np.ndarray:
        """Return validation scores in split order."""
        return np.asarray([split.scores[metric_name] for split in self.splits], dtype=np.float64)

    def training_scores(self, metric_name: str) -> np.ndarray:
        """Return training scores in split order."""
        return np.asarray([split.training_scores[metric_name] for split in self.splits], dtype=np.float64)

    def mean_validation_score(self, metric_name: str) -> float:
        """Return the unweighted arithmetic mean across validation splits.

        Each split has equal weight, independent of its validation size. This
        is a macro-average over splits, not a pooled out-of-fold score.
        """
        return float(np.mean(self.scores(metric_name)))

    def std_validation_score(self, metric_name: str) -> float:
        """Return population standard deviation across validation splits.

        Uses ``ddof=0`` and describes split-to-split variability. It is not a
        standard error or a confidence interval.
        """
        return float(np.std(self.scores(metric_name), ddof=0))

    def mean_training_score(self, metric_name: str) -> float:
        """Return the unweighted arithmetic mean across training splits."""
        return float(np.mean(self.training_scores(metric_name)))

    def std_training_score(self, metric_name: str) -> float:
        """Return population standard deviation across training splits.

        Uses ``ddof=0`` and describes split-to-split variability, not a
        standard error or confidence interval.
        """
        return float(np.std(self.training_scores(metric_name), ddof=0))
