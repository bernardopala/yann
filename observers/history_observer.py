"""Observer that keeps a lightweight per-epoch loss history."""

from neural_network import EpochEndEvent

from .backpropagation_observer import BackpropagationObserver


class HistoryObserver(BackpropagationObserver):
    """Collect epoch losses without retaining network snapshots."""

    def __init__(self) -> None:
        self.losses: list[float] = []
        self.epochs_completed = 0
        self.elapsed_seconds = 0.0

    def on_epoch_end(self, event: EpochEndEvent) -> None:
        self.losses.append(event.loss)
        self.epochs_completed = event.epoch
        self.elapsed_seconds = event.elapsed_seconds
