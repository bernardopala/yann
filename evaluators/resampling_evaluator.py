"""Evaluator for holdout, k-fold and time-series resampling strategies."""

from typing import Callable, Mapping

import numpy as np

from fold_processors import FoldProcessor, PreparedFold
from trainable_models import TrainableModel

from .evaluator import Evaluator
from .events import EvaluationResult, FoldEndEvent, FoldEpochEndEvent, SplitResult
from .splitters import Splitter


Metric = Callable[[np.ndarray, np.ndarray], float]
ModelFactory = Callable[[np.random.Generator, int, int], TrainableModel]
ProcessorFactory = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], FoldProcessor]


class _EpochRelay:
    def __init__(self, evaluator: "ResamplingEvaluator", fold_index: int, model: TrainableModel, processor: FoldProcessor, prepared_fold: PreparedFold) -> None:
        self.evaluator = evaluator
        self.fold_index = fold_index
        self.split_label = evaluator.splitter.format_split_label(fold_index)
        self.model = model
        self.processor = processor
        self.prepared_fold = prepared_fold

    def on_epoch_end(self, event) -> None:
        training_predictions = self.processor.predictions_for_scoring(self.model.predict(self.prepared_fold.X_train))
        predictions = self.processor.predictions_for_scoring(self.model.predict(self.prepared_fold.X_validation))
        training_scores = self.evaluator._score(training_predictions, self.prepared_fold.Y_train_for_scoring)
        scores = self.evaluator._score(predictions, self.prepared_fold.Y_validation_for_scoring)

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
    def __init__(self, splitter: Splitter, model_factory: ModelFactory, processor_factory: ProcessorFactory, metrics: Mapping[str, Metric], rng: np.random.Generator | None = None) -> None:
        super().__init__()

        if not isinstance(splitter, Splitter):
            raise TypeError("splitter must be a Splitter")

        if not callable(model_factory):
            raise TypeError("model_factory must be callable")

        if not callable(processor_factory):
            raise TypeError("processor_factory must be callable")

        if not metrics:
            raise ValueError("metrics cannot be empty")

        if not all(isinstance(name, str) and name for name in metrics):
            raise TypeError("metric names must be non-empty strings")

        if not all(callable(metric) for metric in metrics.values()):
            raise TypeError("metrics must contain callable values")

        self.splitter = splitter
        self.model_factory = model_factory
        self.processor_factory = processor_factory
        self.metrics = dict(metrics)
        self.rng = np.random.default_rng() if rng is None else rng

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> EvaluationResult:
        X = np.asarray(X)
        Y = np.asarray(Y)

        if len(X) != len(Y):
            raise ValueError("X and Y must contain the same number of samples")

        split_results = []

        for fold_index, (train_indices, validation_indices) in enumerate(self.splitter.split(X, Y), start=1):
            processor = self.processor_factory(X, Y, train_indices, validation_indices)

            if not isinstance(processor, FoldProcessor):
                raise TypeError("processor_factory must return a FoldProcessor")

            prepared_fold = processor.prepare()

            model_rng = np.random.default_rng(self.rng.integers(0, 2**63 - 1))
            model = self.model_factory(model_rng, prepared_fold.input_size, prepared_fold.output_size)

            if not isinstance(model, TrainableModel):
                raise TypeError("model_factory must return a TrainableModel")

            model.add_observer(
                _EpochRelay(
                    evaluator=self,
                    fold_index=fold_index,
                    model=model,
                    processor=processor,
                    prepared_fold=prepared_fold,
                )
            )

            training_result = model.train(prepared_fold.X_train, prepared_fold.Y_train)
            training_predictions = processor.predictions_for_scoring(model.predict(prepared_fold.X_train))
            predictions = processor.predictions_for_scoring(model.predict(prepared_fold.X_validation))
            training_scores = self._score(training_predictions, prepared_fold.Y_train_for_scoring)
            scores = self._score(predictions, prepared_fold.Y_validation_for_scoring)

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
