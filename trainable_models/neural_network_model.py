"""TrainableModel adapter for the existing NeuralNetwork API."""

from typing import Any

import numpy as np

from neural_network import NeuralNetwork
from trainers import BackpropagationTrainer

from .trainable_model import TrainableModel


class NeuralNetworkModel(TrainableModel):
    def __init__(self, network: NeuralNetwork, trainer: BackpropagationTrainer) -> None:
        self.network = network
        self.trainer = trainer

    def train(self, X: np.ndarray, Y: np.ndarray) -> Any:
        return self.network.train(self.trainer, X, Y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 2:
            raise ValueError("X must have shape (samples, features)")

        return self.network.compute_outputs(X)

    def add_observer(self, observer: Any) -> None:
        self.trainer.add_observer(observer)
