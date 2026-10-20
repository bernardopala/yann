import numpy as np
from neural_network import NeuralNetwork
from activation_functions import Linear, ReLU

if __name__ == '__main__':
    network_structure = [3, 4, 2]
    activation_functions = [
        ReLU(),
        Linear()
    ]

    nn = NeuralNetwork(network_structure, activation_functions, np.random.default_rng(42))

    inputs = np.array([2.1, 3.2, 5.3])
    outputs = nn.compute_outputs(inputs)

    print(f'Output: {outputs}')
