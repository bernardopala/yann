import numpy as np

from .activation import Activation


class Sigmoid(Activation):
    def forward(self, x):
        x = np.asarray(x)

        return np.exp(-np.logaddexp(0.0, -x))

    def backward(self, weighted_sum, output, upstream_gradient):
        return upstream_gradient * output * (1.0 - output)
