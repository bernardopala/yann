import numpy as np

from activation_functions import relu, sigmoid, linear, tanh


class NeuralNetworkLayer:
    def __init__(self, num_of_inputs: int, num_of_outputs: int, activation_spec, rng: np.random.Generator) -> None:
        self.inputs = np.zeros(num_of_inputs, dtype=np.float32)
        self.weights = np.zeros((num_of_inputs, num_of_outputs), dtype=np.float32)
        self.biases = np.zeros(num_of_outputs, dtype=np.float32)
        self.weighted_sums = np.zeros(num_of_outputs, dtype=np.float32)

        if isinstance(activation_spec, list):
            if len(activation_spec) != num_of_outputs:
                raise ValueError("Number of activation functions must match number of neurons")

            self.activation_functions = activation_spec
        else:
            self.activation_functions = [activation_spec for _ in range(num_of_outputs)]

        self.outputs = np.zeros(num_of_outputs, dtype=np.float32)

        self.initialize_weights(rng)

        # self.gradients = np.zeros(num_of_outputs, dtype=np.float32)
        # self.prev_weights_delta = np.zeros((num_of_inputs, num_of_outputs), dtype=np.float32)
        # self.prev_biases_delta = np.zeros(num_of_outputs, dtype=np.float32)

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

        xavier_limit = np.sqrt(6.0 / (num_inputs + num_outputs))
        he_std = np.sqrt(2.0 / num_inputs)

        for neuron_index, activation in enumerate(self.activation_functions):
            if activation is relu:
                new_weights[:, neuron_index] = rng.normal(loc=0.0, scale=he_std, size=num_inputs)

            elif activation is sigmoid or activation is tanh:
                new_weights[:, neuron_index] = rng.uniform(low=-xavier_limit, high=xavier_limit, size=num_inputs)

            elif activation is linear:
                new_weights[:, neuron_index] = rng.uniform(low=-xavier_limit, high=xavier_limit, size=num_inputs)

            else:
                raise ValueError(f"No initialization rule for activation {activation}")

        self.set_weights(new_weights)
        self.set_biases(np.zeros_like(self.biases))

    def compute_outputs(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs

        W = self.weights
        x = self.inputs
        b = self.biases

        z = W.T @ x + b
        y = np.array([activation(value) for activation, value in zip(self.activation_functions, z)])

        self.weighted_sums = z
        self.outputs = y

        return self.outputs

    # def back_propagate(self):


class NeuralNetwork:
    def __init__(self, network_structure: list[int], activation_functions: list, rng: np.random.Generator | None = None) -> None:
        if len(network_structure) < 2:
            raise AttributeError("Network structure should have given number of inputs and outputs at least")
        if any(size <= 0 for size in network_structure):
            raise ValueError("Layer sizes must be positive")
        number_of_layers = len(network_structure) - 1

        if not isinstance(activation_functions, list):
            raise ValueError("Define one activation function per layer or one activation function for each neuron in every layer.")

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

    def propagate_forward(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.size != self.network_structure[0]:
            raise ValueError(f'Number of passed inputs ({inputs.size}) is different '
                             f'than number of network inputs {self.network_structure[0]}!')

        data = inputs
        for i in range(len(self.layers)):
            data = self.layers[i].compute_outputs(data)

        return data

