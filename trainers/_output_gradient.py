"""Resolve stable output gradients for supported loss/activation pairs."""

import numpy as np

from activation_functions import Activation, Sigmoid, Softmax
from losses import BinaryCrossEntropy, Loss, SparseCategoricalCrossEntropy


def _bce_sigmoid_delta(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    return (prediction - target) / prediction.size


def _scce_softmax_delta(prediction: np.ndarray, target: int) -> np.ndarray:
    delta = np.asarray(prediction, dtype=np.float32).copy()
    delta[int(target)] -= 1.0

    return delta


def resolve_output_gradient(loss: Loss, activation: Activation, prediction: np.ndarray, target: np.ndarray | int) -> tuple[np.ndarray, bool]:
    if type(loss) is BinaryCrossEntropy and type(activation) is Sigmoid:
        return _bce_sigmoid_delta(prediction, target), True

    if type(loss) is SparseCategoricalCrossEntropy and type(activation) is Softmax:
        return _scce_softmax_delta(prediction, target), True

    return loss.gradient(prediction, target), False
