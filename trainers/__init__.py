"""Training algorithms available to the network API."""

from .backpropagation_trainer import BackpropagationOptions, BackpropagationResult, BackpropagationTrainer
from observers import BackpropagationObserver, ConsoleObserver, HistoryObserver
from neural_network import EpochEndEvent

__all__ = [
    "BackpropagationOptions",
    "BackpropagationResult",
    "BackpropagationTrainer",
    "BackpropagationObserver",
    "ConsoleObserver",
    "HistoryObserver",
    "EpochEndEvent",
]
