import numpy as np

from .activation import Activation


class Softmax(Activation):
    def forward(self, x):
        x = np.asarray(x)
        shifted = x - np.max(x)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials)

    def backward(self, weighted_sum, output, upstream_gradient):
        # J_softmax.T @ upstream_gradient without constructing the Jacobian.
        return output * (upstream_gradient - np.dot(upstream_gradient, output))
