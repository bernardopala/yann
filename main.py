import numpy as np
from neural_network import NeuralNetwork
from activation_functions import sigmoid, relu, tanh, linear

if __name__ == '__main__':
    network_structure = [3, 4, 2]
    activation_functions = [
        [
            linear,
            relu,
            sigmoid,
            tanh
        ],
        linear
    ]

    nn = NeuralNetwork(network_structure, activation_functions, np.random.default_rng(42))

    inputs = np.array([2.1, 3.2, 5.3])
    outputs = nn.propagate_forward(inputs)

    print(f'Output: {outputs}')