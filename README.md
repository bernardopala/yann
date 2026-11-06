# YANN — Yet Another Neural Network (from Scratch in NumPy)

YANN is an educational neural-network framework implemented from scratch with
NumPy. It makes the mechanics of a multilayer perceptron explicit: dense-layer
forward propagation, backpropagation, parameter initialization, optimization,
data preprocessing, resampling, and model evaluation remain visible and easy to
inspect.

The project is designed for learning and controlled experiments rather than as
a replacement for a production deep-learning framework. YANN currently focuses
on fully connected feed-forward networks and runs on the CPU through NumPy.

## Why YANN?

- **Readable mechanics** — weights, biases, activations, cached layer values,
  gradients, and optimizer updates are represented directly by NumPy arrays.
- **Vectorized training** — forward and backward passes operate on complete
  mini-batches instead of iterating over individual samples.
- **Modular components** — activations, losses, optimizers, stopping criteria,
  observers, metrics, transformers, and splitters have small public contracts.
- **Reproducible experiments** — network initialization, batch shuffling,
  resampling, and model creation accept explicit NumPy random generators.
- **Leakage-aware evaluation** — transformations are fitted on each training
  partition and then applied to its validation partition.
- **Tabular and time-series workflows** — schema-driven column transformations,
  chronological splits, flattened lookback windows, and multi-step regression
  are included.

## Features at a glance

| Area | Available components |
|---|---|
| Activations | `Linear`, `ReLU`, `Sigmoid`, `Softmax`, `Tanh` |
| Losses | `MSE`, `BinaryCrossEntropy`, `SparseCategoricalCrossEntropy` |
| Optimizers | `SGD`, `Momentum`, `NesterovMomentum`, `RMSProp`, `Adam`, `AdamW` |
| Stopping criteria | `MaxEpochs`, `MaxTrainingTime`, `TargetLoss` |
| Training observers | `ConsoleObserver`, `HistoryObserver`, epoch-end network snapshots |
| Metrics | accuracy, binary accuracy, binary/sparse categorical cross-entropy, MAE, MSE, RMSE |
| Resampling | holdout, K-fold, stratified K-fold, expanding or rolling time-series splits |
| Data transformations | passthrough, standard scaling, label/ordinal/one-hot encoding, cyclical and date-time features |
| Extension points | activations, losses, optimizers, stopping criteria, observers, metrics, transformers, splitters, and trainable models |

YANN uses numerically stable combined output gradients for the standard
`BinaryCrossEntropy` + `Sigmoid` and
`SparseCategoricalCrossEntropy` + `Softmax` pairs.

## Quick start: learn XOR

XOR contains all four combinations of two binary inputs. Its target is 1 when the inputs differ and 0 otherwise. Because XOR is not linearly separable, solving it requires a hidden layer.

The following example builds a `2 → 4 → 1` MLP and trains it as a binary
classifier. The first number in `network_structure` is the input width; every
subsequent number creates a dense layer with its own weights, biases, and
activation.

```python
import numpy as np

from activation_functions import Sigmoid, Tanh
from losses import BinaryCrossEntropy
from neural_network import NeuralNetwork
from observers import HistoryObserver
from optimizers import Adam
from stop_criteria import MaxEpochs
from trainers import BackpropagationOptions, BackpropagationTrainer


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
    rng=np.random.default_rng(43),
)

result = network.train(trainer, X, Y)

probabilities = network.compute_outputs(X).ravel()
predictions = (probabilities >= 0.5).astype(int)

print(f"Epochs: {result.epochs_completed}")
print(f"Final loss: {result.final_loss:.6f}")
print("Probabilities:", np.round(probabilities, 3))
print("Predictions:", predictions)
```

A clean run produces output similar to:

```text
Epochs: 100
Final loss: 0.005161
Probabilities: [0.006 0.997 0.994 0.005]
Predictions: [0 1 1 0]
```

`network.compute_outputs(X)` always returns the raw output of the last
activation. Converting it into a task-level prediction is explicit:

- regression: use the output directly;
- binary or multilabel classification: apply a chosen threshold;
- multiclass classification: use `np.argmax(probabilities, axis=1)`.

## Advanced example

This example demonstrates a more complete YANN workflow by combining preprocessing, model construction, training, holdout evaluation, and metrics in a single reproducible pipeline.

The example below uses a small synthetic regression dataset. It contains 120
evenly spaced input values from -2 to 2, with targets generated by the nonlinear
function $y = 0.4x^3 - x + 0.1$. The dataset is noise-free so that the example
focuses on the YANN workflow rather than on data cleaning or uncertainty.

The evaluator reserves 20% of the observations for validation. Input and target
scalers are fitted only on the training partition, and validation RMSE is
reported in the original target scale.

```python
import numpy as np

from activation_functions import Linear, Tanh
from evaluators import HoldoutSplitter, ResamplingEvaluator
from losses import MSE
from metrics import RootMeanSquaredError
from neural_network import NeuralNetwork
from optimizers import Adam
from split_processors import (
    ColumnSpec,
    StandardScaler,
    create_tabular_split_processor_factory,
)
from stop_criteria import MaxEpochs
from trainable_models import NeuralNetworkModel
from trainers import BackpropagationOptions, BackpropagationTrainer


X = np.linspace(-2.0, 2.0, 120, dtype=np.float32).reshape(-1, 1)
Y = (0.4 * X[:, 0] ** 3 - X[:, 0] + 0.1).astype(np.float32)

processor_factory = create_tabular_split_processor_factory(
    x_schema=[ColumnSpec(0, [StandardScaler()])],
    y_schema=[ColumnSpec(0, [StandardScaler()])],
)


def model_factory(rng, input_size, output_size):
    network_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
    trainer_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))

    network = NeuralNetwork(
        [input_size, 16, output_size],
        [Tanh(), Linear()],
        rng=network_rng,
    )
    trainer = BackpropagationTrainer(
        loss=MSE(),
        optimizer=Adam(learning_rate=0.01),
        options=BackpropagationOptions(
            batch_size=16,
            stop_criteria=[MaxEpochs(300)],
            shuffle=True,
        ),
        rng=trainer_rng,
    )
    return NeuralNetworkModel(network, trainer)


evaluator = ResamplingEvaluator(
    splitter=HoldoutSplitter(
        test_size=0.2,
        shuffle=True,
        rng=np.random.default_rng(100),
    ),
    model_factory=model_factory,
    split_processor_factory=processor_factory,
    metrics=[RootMeanSquaredError()],
    rng=np.random.default_rng(101),
)

evaluation = evaluator.evaluate(X, Y)
print("Validation RMSE:", evaluation.mean_validation_score("rmse"))
```

## License

YANN is available under the [MIT License](LICENSE).
