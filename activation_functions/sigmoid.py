import numpy as np

from .activation import Activation


class Sigmoid(Activation):
    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def backward(self, weighted_sum, output, upstream_gradient):
        return upstream_gradient * output * (1.0 - output)
