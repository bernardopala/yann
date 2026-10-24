"""History observer for fold and per-epoch validation events."""

from .events import EvaluationResult, FoldEndEvent, FoldEpochEndEvent


class HistoryObserver:
    def __init__(self) -> None:
        self.epochs: dict[int, list[FoldEpochEndEvent]] = {}
        self.folds: list[FoldEndEvent] = []
        self.evaluation: EvaluationResult | None = None

    def on_fold_epoch_end(self, event: FoldEpochEndEvent) -> None:
        self.epochs.setdefault(event.fold_index, []).append(event)

    def on_fold_end(self, event: FoldEndEvent) -> None:
        self.folds.append(event)

    def on_evaluation_end(self, event: EvaluationResult) -> None:
        self.evaluation = event
