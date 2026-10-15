import numpy as np

from .activation import Activation


class Tanh(Activation):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, weighted_sum, output, upstream_gradient):
        return upstream_gradient * (1.0 - output * output)
