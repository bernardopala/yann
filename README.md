# YANN — Yet Another Neural Network (from Scratch in NumPy)

YANN is an educational neural-network framework implemented from scratch with NumPy. It provides a compact, modular pipeline for training multilayer perceptrons, evaluating models with common resampling strategies, and preparing tabular or time-series data.

The project is intended for learning and experimentation: its implementation keeps forward propagation, backpropagation, optimization, preprocessing, and evaluation directly accessible instead of hiding them behind a high-level deep-learning framework.

Example:
```python
import numpy as np

from activation_functions import Sigmoid, Tanh
from losses import BinaryCrossEntropy
from neural_network import NeuralNetwork
from optimizers import Adam
from stop_criteria import MaxEpochs
from trainers import BackpropagationOptions, BackpropagationTrainer


# Logical XOR training set.
X = np.array(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ],
    dtype=np.float32,
)
Y = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)

network = NeuralNetwork(
    network_structure=[2, 4, 1],
    activation_functions=[Tanh(), Sigmoid()],
    rng=np.random.default_rng(42),
)

trainer = BackpropagationTrainer(
    loss=BinaryCrossEntropy(),
    optimizer=Adam(learning_rate=0.05),
    options=BackpropagationOptions(
        batch_size=4,
        stop_criteria=[MaxEpochs(100)],
        shuffle=False,
    ),
)

result = network.train(trainer, X, Y)

probabilities = network.compute_outputs(X).ravel()
predictions = (probabilities >= 0.5).astype(int)

print(f"Epochs: {result.epochs_completed}")
print(f"Final loss: {result.final_loss:.6f}")
print("Probabilities:", np.round(probabilities, 3))
print("Predictions:", predictions)
```
