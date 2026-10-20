"""Backpropagation trainer and its training options."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from losses import Loss
from neural_network import EpochEndEvent, NeuralNetwork, NeuralNetworkLayer
from optimizers import Optimizer
from stop_criteria import StopCriterion

from observers import BackpropagationObserver


class BackpropagationTrainer:
    def __init__(self, loss: Loss, optimizer: Optimizer, options: BackpropagationOptions, rng: np.random.Generator | None = None) -> None:
        if not isinstance(loss, Loss):
            raise TypeError("loss must be a Loss")

        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be an Optimizer")

        if not isinstance(options, BackpropagationOptions):
            raise TypeError("options must be a BackpropagationOptions")

        self.loss, self.optimizer, self.options = loss, optimizer, options
        self.rng = np.random.default_rng() if rng is None else rng
        self.observers: list[BackpropagationObserver] = []

    def add_observer(self, observer: BackpropagationObserver) -> None:
        self.observers.append(observer)

    def train(self, network: NeuralNetwork, X: np.ndarray, Y: np.ndarray) -> BackpropagationResult:
        X, Y = self._validate_data(network, X, Y)

        started_at = perf_counter()
        epochs_completed = 0
        average_loss : float | None = None
        time_elapsed = perf_counter() - started_at
        sample_count = len(X)

        while not self._should_stop(epochs_completed, time_elapsed, average_loss):
            indices = self._prepare_indices(sample_count)
            epoch_loss, processed_samples = 0.0, 0

            for start in range(0, sample_count, self.options.batch_size):
                batch_indices = indices[start:start + self.options.batch_size]
                batch_weight_gradients, batch_bias_gradients = self._zero_gradients(network)

                for index in batch_indices:
                    prediction = network.compute_outputs(X[index])
                    epoch_loss += self.loss.value(prediction, Y[index])
                    sample_weight_gradients, sample_bias_gradients = self._backpropagate(network, self.loss.gradient(prediction, Y[index]))
                    self._accumulate_gradients(batch_weight_gradients, batch_bias_gradients, sample_weight_gradients, sample_bias_gradients)
                    processed_samples += 1

                self._update_network_weights_and_biases(network, batch_weight_gradients, batch_bias_gradients, len(batch_indices))

            epochs_completed += 1
            time_elapsed = perf_counter() - started_at
            average_loss = float(epoch_loss / processed_samples)

            if self.observers:
                event = EpochEndEvent(epoch=epochs_completed, loss=average_loss, elapsed_seconds=time_elapsed, snapshot=network.create_snapshot())

                for observer in self.observers:
                    observer.on_epoch_end(event)

        return BackpropagationResult(average_loss, epochs_completed, time_elapsed)

    def _prepare_indices(self, sample_count: int) -> np.ndarray:
        if self.options.shuffle:
            return self.rng.permutation(sample_count)

        return np.arange(sample_count)

    def _validate_data(self, network: NeuralNetwork, X: np.ndarray, Y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y)

        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != network.network_structure[0]:
            raise ValueError(f"X must have shape (samples, {network.network_structure[0]}) with at least one sample")

        if len(Y) != len(X):
            raise ValueError("X and Y must contain the same number of samples")

        outputs = network.network_structure[-1]

        return X, self.loss.validate_targets(Y, outputs)

    def _should_stop(
        self,
        epochs_completed: int,
        elapsed_seconds: float,
        loss: float | None,
    ) -> bool:
        return any(
            sc.reached(epochs_completed, elapsed_seconds, loss)
            for sc in self.options.stop_criteria
        )

    def _zero_gradients(self, network: NeuralNetwork) -> tuple[list[np.ndarray], list[np.ndarray]]:
        weight_gradients = [np.zeros_like(layer.weights) for layer in network.layers]
        bias_gradients = [np.zeros_like(layer.biases)for layer in network.layers]
        return weight_gradients, bias_gradients

    def _backpropagate(self, network: NeuralNetwork, output_gradient: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        gradient = np.asarray(output_gradient, dtype=np.float32)
        weight_gradients, bias_gradients = self._zero_gradients(network)

        for layer_index in range(len(network.layers) - 1, -1, -1):
            layer = network.layers[layer_index]
            delta = self._compute_layer_delta(layer, gradient)
            weight_gradients[layer_index][...] = np.outer(layer.inputs, delta)
            bias_gradients[layer_index][...] = delta
            gradient = layer.weights @ delta

        return weight_gradients, bias_gradients

    def _accumulate_gradients(self, batch_weight_gradients: list[np.ndarray], batch_bias_gradients: list[np.ndarray], sample_weight_gradients: list[np.ndarray], sample_bias_gradients: list[np.ndarray]) -> None:
        for layer_index in range(len(batch_weight_gradients)):
            batch_weight_gradients[layer_index] += sample_weight_gradients[layer_index]
            batch_bias_gradients[layer_index] += sample_bias_gradients[layer_index]

    def _update_network_weights_and_biases(self, network: NeuralNetwork, weight_gradients: list[np.ndarray], bias_gradients: list[np.ndarray], batch_size: int) -> None:
        weights_and_biases = []

        for index, layer in enumerate(network.layers):
            weights_and_biases.append(((index, "weights"),layer.weights,weight_gradients[index] / batch_size))
            weights_and_biases.append(((index, "biases"),layer.biases,bias_gradients[index] / batch_size))

        self.optimizer.step(weights_and_biases)

    def _compute_layer_delta(self, layer: NeuralNetworkLayer, output_gradient: np.ndarray) -> np.ndarray:
        return np.asarray(layer.activation.backward(layer.weighted_sums, layer.outputs, output_gradient), dtype=np.float32)


class BackpropagationOptions:
    def __init__(self, batch_size: int, stop_criteria: list[StopCriterion], shuffle: bool = True) -> None:
        if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        if not isinstance(stop_criteria, list):
            raise TypeError("stop_criteria must be a list of StopCriterion")

        if not stop_criteria or not all(isinstance(item, StopCriterion) for item in stop_criteria):
            raise TypeError("stop_criteria must contain at least one StopCriterion")

        if not isinstance(shuffle, (bool, np.bool_)):
            raise TypeError("shuffle must be a boolean")

        self.batch_size = int(batch_size)
        self.stop_criteria = stop_criteria.copy()
        self.shuffle = bool(shuffle)


@dataclass(frozen=True)
class BackpropagationResult:
    final_loss: float | None
    epochs_completed: int
    elapsed_seconds: float


