"""Observer interface for backpropagation training events."""

from abc import ABC, abstractmethod

from neural_network import EpochEndEvent


class BackpropagationObserver(ABC):
    @abstractmethod
    def on_epoch_end(self, event: EpochEndEvent) -> None:
        """Handle an event emitted after an epoch."""
