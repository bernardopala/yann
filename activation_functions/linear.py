from .activation import Activation


class Linear(Activation):
    def forward(self, x):
        return x

    def backward(self, weighted_sum, output, upstream_gradient):
        return upstream_gradient
