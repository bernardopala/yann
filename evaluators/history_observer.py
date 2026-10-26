"""History observer for split and per-epoch validation events."""

from .events import EvaluationResult, SplitEndEvent, SplitEpochEndEvent


class HistoryObserver:
    def __init__(self) -> None:
        self.epochs: dict[int, list[SplitEpochEndEvent]] = {}
        self.splits: list[SplitEndEvent] = []
        self.evaluation: EvaluationResult | None = None

    def on_split_epoch_end(self, event: SplitEpochEndEvent) -> None:
        self.epochs.setdefault(event.split_index, []).append(event)

    def on_split_end(self, event: SplitEndEvent) -> None:
        self.splits.append(event)

    def on_evaluation_end(self, event: EvaluationResult) -> None:
        self.evaluation = event
