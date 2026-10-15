import numpy as np

from .activation import Activation


class ReLU(Activation):
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, weighted_sum, output, upstream_gradient):
        return upstream_gradient * np.where(np.asarray(weighted_sum) > 0, 1.0, 0.0)

    def initialize_weights(self, num_inputs, num_outputs, rng):
        return rng.normal(0.0, np.sqrt(2.0 / num_inputs), size=num_inputs)
