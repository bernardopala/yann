"""Evaluator for holdout, k-fold and time-series resampling strategies."""

from typing import Callable, Mapping

import numpy as np

from trainable_models import TrainableModel

from .evaluator import Evaluator
from .events import EvaluationResult, FoldEndEvent, FoldEpochEndEvent, SplitResult
from .splitters import Splitter


Metric = Callable[[np.ndarray, np.ndarray], float]
ModelFactory = Callable[[np.random.Generator], TrainableModel]


class _EpochRelay:
    def __init__(self, evaluator, fold_index, model, X_train, Y_train, X_validation, Y_validation):
        self.evaluator = evaluator
        self.fold_index = fold_index
        self.split_label = evaluator.splitter.format_split_label(fold_index)
        self.model = model
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_validation = X_validation
        self.Y_validation = Y_validation

    def on_epoch_end(self, event):
        training_predictions = self.model.predict(self.X_train)
        predictions = self.model.predict(self.X_validation)
        training_scores = self.evaluator._score(training_predictions, self.Y_train)
        scores = self.evaluator._score(predictions, self.Y_validation)

        self.evaluator._notify(
            "on_fold_epoch_end",
            FoldEpochEndEvent(
                split_label=self.split_label,
                fold_index=self.fold_index,
                epoch_event=event,
                training_scores=training_scores,
                validation_scores=scores,
            ),
        )


class ResamplingEvaluator(Evaluator):
    def __init__(self, splitter: Splitter, model_factory: ModelFactory, metrics: Mapping[str, Metric], rng: np.random.Generator | None = None) -> None:
        super().__init__()

        if not isinstance(splitter, Splitter):
            raise TypeError("splitter must be a Splitter")

        if not callable(model_factory):
            raise TypeError("model_factory must be callable")

        if not metrics:
            raise ValueError("metrics cannot be empty")

        if not all(isinstance(name, str) and name for name in metrics):
            raise TypeError("metric names must be non-empty strings")

        if not all(callable(metric) for metric in metrics.values()):
            raise TypeError("metrics must contain callable values")

        self.splitter = splitter
        self.model_factory = model_factory
        self.metrics = dict(metrics)
        self.rng = np.random.default_rng() if rng is None else rng

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> EvaluationResult:
        X = np.asarray(X)
        Y = np.asarray(Y)

        if len(X) != len(Y):
            raise ValueError("X and Y must contain the same number of samples")

        split_results = []

        for fold_index, (train_indices, validation_indices) in enumerate(self.splitter.split(X, Y), start=1):
            model_rng = np.random.default_rng(self.rng.integers(0, 2**63 - 1))
            model = self.model_factory(model_rng)

            if not isinstance(model, TrainableModel):
                raise TypeError("model_factory must return a TrainableModel")

            model.add_observer(
                _EpochRelay(
                    evaluator=self,
                    fold_index=fold_index,
                    model=model,
                    X_train=X[train_indices],
                    Y_train=Y[train_indices],
                    X_validation=X[validation_indices],
                    Y_validation=Y[validation_indices],
                )
            )

            training_result = model.train(X[train_indices], Y[train_indices])
            training_predictions = model.predict(X[train_indices])
            predictions = model.predict(X[validation_indices])
            training_scores = self._score(training_predictions, Y[train_indices])
            scores = self._score(predictions, Y[validation_indices])

            split_result = SplitResult(
                split_label=self.splitter.format_split_label(fold_index),
                fold_index=fold_index,
                training_result=training_result,
                training_scores=training_scores,
                scores=scores,
            )

            split_results.append(split_result)

            self._notify(
                "on_fold_end",
                FoldEndEvent(
                    split_label=self.splitter.format_split_label(fold_index),
                    fold_index=fold_index,
                    training_result=training_result,
                    training_scores=training_scores,
                    scores=scores,
                    train_size=len(train_indices),
                    validation_size=len(validation_indices),
                ),
            )

        result = EvaluationResult(split_results)
        self._notify("on_evaluation_end", result)
        return result

    def _score(self, predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
        return {
            name: float(metric(predictions, targets))
            for name, metric in self.metrics.items()
        }
