"""Evaluator for holdout, k-fold and time-series resampling strategies."""

from typing import Callable

import numpy as np

from split_processors import PreparedSplit, SplitProcessor
from trainable_models import TrainableModel
from metrics import Metric

from .evaluator import Evaluator
from .events import EvaluationResult, FoldEndEvent, FoldEpochEndEvent, SplitResult
from .splitters import Splitter


ModelFactory = Callable[[np.random.Generator, int, int], TrainableModel]
SplitProcessorFactory = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], SplitProcessor]


class _EpochRelay:
    def __init__(self, evaluator: "ResamplingEvaluator", fold_index: int, model: TrainableModel, processor: SplitProcessor, prepared_split: PreparedSplit) -> None:
        self.evaluator = evaluator
        self.fold_index = fold_index
        self.split_label = evaluator.splitter.format_split_label(fold_index)
        self.model = model
        self.processor = processor
        self.prepared_split = prepared_split

    def on_epoch_end(self, event) -> None:
        training_predictions = self.processor.predictions_for_scoring(self.model.predict(self.prepared_split.X_train))
        predictions = self.processor.predictions_for_scoring(self.model.predict(self.prepared_split.X_validation))
        training_scores = self.evaluator._score(self.prepared_split.Y_train_for_scoring, training_predictions)
        scores = self.evaluator._score(self.prepared_split.Y_validation_for_scoring, predictions)

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
    def __init__(self, splitter: Splitter, model_factory: ModelFactory, split_processor_factory: SplitProcessorFactory, metrics: list[Metric], rng: np.random.Generator | None = None) -> None:
        super().__init__()

        if not isinstance(splitter, Splitter):
            raise TypeError("splitter must be a Splitter")

        if not callable(model_factory):
            raise TypeError("model_factory must be callable")

        if not callable(split_processor_factory):
            raise TypeError("split_processor_factory must be callable")

        if not isinstance(metrics, list):
            raise TypeError("metrics must be a list of Metric")

        if not metrics:
            raise ValueError("metrics cannot be empty")

        if not all(isinstance(metric, Metric) for metric in metrics):
            raise TypeError("metrics must contain Metric objects")

        if len({metric.name for metric in metrics}) != len(metrics):
            raise ValueError("metric names must be unique")

        self.splitter = splitter
        self.model_factory = model_factory
        self.split_processor_factory = split_processor_factory
        self.metrics: list[Metric] = list(metrics)
        self.rng = np.random.default_rng() if rng is None else rng

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> EvaluationResult:
        X = np.asarray(X)
        Y = np.asarray(Y)

        if len(X) != len(Y):
            raise ValueError("X and Y must contain the same number of samples")

        split_results = []

        for fold_index, (train_indices, validation_indices) in enumerate(self.splitter.split(X, Y), start=1):
            processor = self.split_processor_factory(X, Y, train_indices, validation_indices)

            if not isinstance(processor, SplitProcessor):
                raise TypeError("split_processor_factory must return a SplitProcessor")

            prepared_split = processor.prepare()

            model_rng = np.random.default_rng(self.rng.integers(0, 2**63 - 1))
            model = self.model_factory(model_rng, prepared_split.input_size, prepared_split.output_size)

            if not isinstance(model, TrainableModel):
                raise TypeError("model_factory must return a TrainableModel")

            model.add_observer(
                _EpochRelay(
                    evaluator=self,
                    fold_index=fold_index,
                    model=model,
                    processor=processor,
                    prepared_split=prepared_split,
                )
            )

            training_result = model.train(prepared_split.X_train, prepared_split.Y_train)
            training_predictions = processor.predictions_for_scoring(model.predict(prepared_split.X_train))
            predictions = processor.predictions_for_scoring(model.predict(prepared_split.X_validation))
            training_scores = self._score(prepared_split.Y_train_for_scoring, training_predictions)
            scores = self._score(prepared_split.Y_validation_for_scoring, predictions)

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

    def _score(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        scores = {metric.name: metric.compute(y_true, y_pred) for metric in self.metrics}

        if not all(np.isscalar(score) and np.isfinite(score) for score in scores.values()):
            raise ValueError("metrics must return finite scalar values")

        return {name: float(score) for name, score in scores.items()}
