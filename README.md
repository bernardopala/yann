# YANN — Yet Another Neural Network (from scratch in NumPy)

YANN to edukacyjny framework sieci neuronowych zaimplementowany od podstaw przy
użyciu NumPy. W przejrzysty sposób przedstawia mechanizmy działania perceptronu
wielowarstwowego: propagację w przód w warstwach gęstych, propagację wsteczną,
inicjalizację parametrów, optymalizację, wstępne przetwarzanie danych,
resampling oraz ocenę modelu — wszystko pozostaje widoczne i łatwe do
przeanalizowania.

Projekt jest przeznaczony do nauki i kontrolowanych eksperymentów, a nie jako
zamiennik produkcyjnego frameworka deep learning. Obecnie YANN koncentruje się
na w pełni połączonych sieciach jednokierunkowych i wykonuje obliczenia na CPU
za pośrednictwem NumPy.

## Kurs wprowadzający

[**1. YANN: Budowanie i trenowanie sieci**](https://github.com/bernardopala/yann/blob/main/notebooks/yann_crash_course.ipynb)

[**2. YANN: Walidacja i rzetelna ocena modelu**](https://github.com/bernardopala/yann/blob/main/notebooks/yann_evaluation.ipynb) 

## Dlaczego YANN?

- **Czytelne mechanizmy** - wagi, obciążenia, aktywacje, buforowane wartości
  warstw, gradienty i aktualizacje optymalizatora są reprezentowane bezpośrednio
  przez tablice NumPy.
- **Wektoryzowane trenowanie** - przebiegi w przód i wstecz operują na całych
  mini-batchach zamiast iterować po pojedynczych próbkach.
- **Modułowe komponenty** - aktywacje, funkcje straty, optymalizatory, kryteria
  zatrzymania, obserwatory, metryki, transformatory i strategie podziału danych
  mają niewielkie, jasno określone interfejsy publiczne.
- **Powtarzalne eksperymenty** - inicjalizacja sieci, tasowanie batchów,
  resampling i tworzenie modeli przyjmują jawnie przekazane generatory liczb
  losowych NumPy.
- **Ocena zapobiegająca wyciekowi danych** - transformacje są dopasowywane do
  każdego zbioru treningowego, a następnie stosowane do odpowiadającego mu
  zbioru walidacyjnego.
- **Obsługa danych tabelarycznych i szeregów czasowych** - dostępne są
  transformacje kolumn oparte na schemacie, podziały chronologiczne, spłaszczone
  okna danych historycznych oraz regresja wielokrokowa.

## Najważniejsze funkcje

| Obszar | Dostępne komponenty                                                                                                                                       |
|---|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Funkcje aktywacji | `Linear`, `ReLU`, `Sigmoid`, `Softmax`, `Tanh`                                                                                                            |
| Funkcje straty | `MSE`, `BinaryCrossEntropy`, `SparseCategoricalCrossEntropy`                                                                                              |
| Optymalizatory | `SGD`, `Momentum`, `NesterovMomentum`, `RMSProp`, `Adam`, `AdamW`                                                                                         |
| Kryteria zatrzymania | `MaxEpochs`, `MaxTrainingTime`, `TargetLoss`                                                                                                              |
| Obserwatory trenowania | `ConsoleObserver`, `HistoryObserver`, snapshoty sieci na koniec epoki                                                                                     |
| Metryki | accuracy, binary accuracy, binary/sparse categorical cross-entropy, MAE, MSE, RMSE                                                                        |
| Resampling | holdout, K-fold, stratified K-fold, splity dla szeregów czasowych typu expanding lub rolling                                                 |
| Transformacje danych | passthrough, standard scaling, label/ordinal/one-hot encoding, cechy cykliczne oraz cechy daty i czasu                                    |
| Punkty rozszerzeń | aktywacje, funkcje straty, optymalizatory, kryteria zatrzymania, obserwatory, metryki, transformatory, strategie podziału i modele podlegające trenowaniu |

YANN wykorzystuje stabilne numerycznie, połączone gradienty wyjściowe dla
standardowych par `BinaryCrossEntropy` + `Sigmoid` oraz
`SparseCategoricalCrossEntropy` + `Softmax`.

## Szybki start: nauka funkcji XOR

XOR obejmuje wszystkie cztery kombinacje dwóch wejść binarnych. Wartość
docelowa wynosi 1, gdy wejścia są różne, oraz 0 w przeciwnym przypadku. Ponieważ
funkcja XOR nie jest liniowo separowalna, do jej odwzorowania potrzebna jest
warstwa ukryta.

Poniższy przykład tworzy perceptron wielowarstwowy `2 → 4 → 1` i trenuje go jako
klasyfikator binarny. Pierwsza liczba w `network_structure` określa rozmiar
wejścia, a każda kolejna tworzy warstwę gęstą z własnymi wagami, obciążeniami i
funkcją aktywacji.

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

Prawidłowe uruchomienie daje wynik podobny do poniższego:

```text
Epochs: 100
Final loss: 0.005161
Probabilities: [0.006 0.997 0.994 0.005]
Predictions: [0 1 1 0]
```

`network.compute_outputs(X)` zawsze zwraca surowe wyjście ostatniej funkcji
aktywacji. Jego przekształcenie we właściwą predykcję dla danego zadania odbywa
się jawnie:

- regresja: użyj wyjścia bezpośrednio;
- klasyfikacja binarna lub wieloetykietowa: zastosuj wybrany próg;
- klasyfikacja wieloklasowa: użyj `np.argmax(probabilities, axis=1)`.

## Pełny przepływ pracy 

Ten przykład przedstawia kompletny przepływ pracy w YANN, łącząc
wstępne przetwarzanie danych, budowę modelu, trenowanie, ocenę metodą holdout i
metryki w jednym powtarzalnym pipeline.

Poniższy przykład wykorzystuje niewielki syntetyczny zbiór danych regresyjnych.
Zawiera on 120 równomiernie rozmieszczonych wartości wejściowych od -2 do 2, a
wartości docelowe są generowane przez nieliniową funkcję
$y = 0.4x^3 - x + 0.1$. Zbiór nie zawiera szumu, dzięki czemu przykład skupia
się na przepływie pracy w YANN, a nie na czyszczeniu danych ani na niepewności.

Moduł oceniający rezerwuje 20% obserwacji na walidację. Skalery wejścia i
wartości docelowych są dopasowywane wyłącznie do zbioru treningowego, a
walidacyjny błąd RMSE jest podawany w oryginalnej skali wartości docelowej.

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

Prawidłowe uruchomienie daje wynik podobny do poniższego:

```text
Validation RMSE: 0.126783
```

## Licencja

YANN jest udostępniany na warunkach [licencji MIT](LICENSE).
