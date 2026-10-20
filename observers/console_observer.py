"""Console observer for backpropagation training."""

import sys
from typing import TextIO

from neural_network import EpochEndEvent

from .backpropagation_observer import BackpropagationObserver


class ConsoleObserver(BackpropagationObserver):
    """Print epoch progress to a text stream."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = sys.stdout if stream is None else stream

    def on_epoch_end(self, event: EpochEndEvent) -> None:
        print(f"Epoch {event.epoch}: loss={event.loss:.6f}; elapsed={event.elapsed_seconds:.2f}s", file=self.stream,flush=True)
