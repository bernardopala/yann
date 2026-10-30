import numpy as np

from .activation import Activation


class Softmax(Activation):
    def forward(self, x):
        x = np.asarray(x)
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials, axis=-1, keepdims=True)

    def backward(self, weighted_sum, output, upstream_gradient):
        # J_softmax.T @ upstream_gradient without constructing the Jacobian.
        output = np.asarray(output)
        upstream_gradient = np.asarray(upstream_gradient)
        projection = np.sum(upstream_gradient * output, axis=-1, keepdims=True)
        return output * (upstream_gradient - projection)
