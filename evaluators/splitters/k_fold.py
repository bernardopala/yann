import numpy as np

from .splitter import Splitter


class KFoldSplitter(Splitter):
    def __init__(self, n_folds: int, shuffle: bool = True, rng: np.random.Generator | None = None) -> None:
        if not isinstance(n_folds, (int, np.integer)) or n_folds < 2:
            raise ValueError("n_folds must be at least 2")

        self.n_folds = int(n_folds)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng() if rng is None else rng

    @property
    def label(self) -> str:
        return "Fold"

    def split(self, X: np.ndarray, Y: np.ndarray):
        if len(X) != len(Y):
            raise ValueError("X and Y must contain the same number of samples")

        if self.n_folds > len(X):
            raise ValueError("n_folds cannot exceed the number of samples")

        indices = np.arange(len(X))

        if self.shuffle:
            indices = self.rng.permutation(indices)

        folds = np.array_split(indices, self.n_folds)

        for fold_index, validation_indices in enumerate(folds):
            train_indices = np.concatenate([fold for index, fold in enumerate(folds) if index != fold_index])
            yield train_indices, validation_indices
