from dataclasses import dataclass

import numpy as np

from activation_functions import Activation


@dataclass(frozen=True)
class LayerSnapshot:
    """Immutable-by-convention copy of one network layer's state."""

    weights: np.ndarray
    biases: np.ndarray
    activation_name: str


@dataclass(frozen=True)
class NetworkSnapshot:
    """Copy of the network architecture and parameters at an epoch boundary."""

    network_structure: list[int]
    layers: list[LayerSnapshot]


@dataclass(frozen=True)
class EpochEndEvent:
    """Information emitted after a completed backpropagation epoch."""

    epoch: int
    loss: float
    elapsed_seconds: float
    snapshot: NetworkSnapshot


class NeuralNetworkLayer:
    def __init__(self, num_of_inputs: int, num_of_outputs: int, activation: Activation,
                 rng: np.random.Generator) -> None:
        self.inputs = np.zeros((0, num_of_inputs), dtype=np.float32)
        self.weights = np.zeros((num_of_inputs, num_of_outputs), dtype=np.float32)
        self.biases = np.zeros(num_of_outputs, dtype=np.float32)
        self.weighted_sums = np.zeros((0, num_of_outputs), dtype=np.float32)
        self.activation = activation
        self.outputs = np.zeros((0, num_of_outputs), dtype=np.float32)

        self.initialize_weights(rng)

    def set_weights(self, weights: np.ndarray) -> None:
        weights = np.asarray(weights, dtype=self.weights.dtype)

        if weights.shape != self.weights.shape:
            raise ValueError(f"Expected weights with shape {self.weights.shape}, got {weights.shape}")

        self.weights[...] = weights

    def set_biases(self, biases: np.ndarray) -> None:
        biases = np.asarray(biases, dtype=self.biases.dtype)

        if biases.shape != self.biases.shape:
            raise ValueError(f"Expected biases with shape {self.biases.shape}, got {biases.shape}")

        self.biases[...] = biases

    def initialize_weights(self, rng: np.random.Generator | None = None ) -> None:
        if rng is None:
            rng = np.random.default_rng()

        num_inputs, num_outputs = self.weights.shape
        new_weights = np.empty_like(self.weights)

        for neuron_index in range(num_outputs):
            new_weights[:, neuron_index] = self.activation.initialize_weights(num_inputs, num_outputs, rng)

        self.set_weights(new_weights)
        self.set_biases(np.zeros_like(self.biases))

    def compute_outputs(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = np.asarray(inputs, dtype=np.float32)

        if self.inputs.ndim != 2 or self.inputs.shape[1] != self.weights.shape[0]:
            raise ValueError(f"Expected inputs with shape (batch, {self.weights.shape[0]}), got {self.inputs.shape}")

        z = self.inputs @ self.weights + self.biases
        y = np.asarray(self.activation.forward(z), dtype=np.float32)

        self.weighted_sums = z
        self.outputs = y

        return self.outputs


class NeuralNetwork:
    def __init__(self, network_structure: list[int], activation_functions: list[Activation], rng: np.random.Generator | None = None) -> None:
        if len(network_structure) < 2:
            raise AttributeError("Network structure should have given number of inputs and outputs at least")

        if any(size <= 0 for size in network_structure):
            raise ValueError("Layer sizes must be positive")

        number_of_layers = len(network_structure) - 1

        if len(activation_functions) != number_of_layers:
            raise ValueError(f"Network has {number_of_layers} layers, but {len(activation_functions)} activation specifications were provided.")

        if rng is None:
            rng = np.random.default_rng()

        self.layers = list[NeuralNetworkLayer]()
        self.network_structure = network_structure

        for i in range(len(self.network_structure) - 1):
            self.layers.append(NeuralNetworkLayer(self.network_structure[i], self.network_structure[i + 1], activation_functions[i], rng))

    def set_weights_and_biases(self, weights_and_biases: list[tuple[np.ndarray, np.ndarray]]) -> None:
        if len(weights_and_biases) != len(self.layers):
            raise ValueError(f"Expected weights and biases for {len(self.layers)} layers, got {len(weights_and_biases)}")

        for layer_index, weights_and_biases_of_layer in enumerate(weights_and_biases):
            if not isinstance(weights_and_biases_of_layer, (tuple, list)):
                raise ValueError(f"Weights and biases for layer {layer_index} must be a tuple or list")

            if len(weights_and_biases_of_layer) != 2:
                raise ValueError(f"Weights and biases for layer {layer_index} must contain exactly two elements")

        for layer_index, (layer, (weights, biases)) in enumerate(zip(self.layers, weights_and_biases)):
            try:
                layer.set_weights(weights)
                layer.set_biases(biases)
            except ValueError as error:
                raise ValueError(f"Invalid weights or biases in layer {layer_index}: {error}") from error

    def compute_outputs(self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.asarray(inputs, dtype=np.float32)
        if inputs.ndim != 2 or inputs.shape[1] != self.network_structure[0]:
            raise ValueError(f'Expected inputs with shape (batch, {self.network_structure[0]}), got {inputs.shape}')

        data = inputs
        for i in range(len(self.layers)):
            data = self.layers[i].compute_outputs(data)

        return data

    def train(self, trainer, X: np.ndarray, Y: np.ndarray):
        return trainer.train(self, X, Y)

    def create_snapshot(self) -> NetworkSnapshot:
        return NetworkSnapshot(
            network_structure=list(self.network_structure),
            layers=[LayerSnapshot(weights=layer.weights.copy(), biases=layer.biases.copy(), activation_name=type(layer.activation).__name__) for layer in self.layers],
        )

