import numpy as np

from .splitter import Splitter


class StratifiedKFoldSplitter(Splitter):
    def __init__(self, n_folds: int, shuffle: bool = True, rng: np.random.Generator | None = None) -> None:
        if not isinstance(n_folds, (int, np.integer)) or n_folds < 2:
            raise ValueError("n_folds must be at least 2")

        self.n_folds = int(n_folds)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng() if rng is None else rng

    @property
    def label(self) -> str:
        return "Stratified fold"

    def split(self, X: np.ndarray, Y: np.ndarray):
        if len(X) != len(Y):
            raise ValueError("X and Y must contain the same number of samples")

        targets = np.asarray(Y).reshape(-1)
        folds = [[] for _ in range(self.n_folds)]

        for class_value in np.unique(targets):
            class_indices = np.flatnonzero(targets == class_value)

            if len(class_indices) < self.n_folds:
                raise ValueError("Every class must contain at least n_folds samples")

            if self.shuffle:
                class_indices = self.rng.permutation(class_indices)

            for position, sample_index in enumerate(class_indices):
                folds[position % self.n_folds].append(int(sample_index))

        folds = [np.asarray(sorted(fold), dtype=np.int64) for fold in folds]

        for fold_index, validation_indices in enumerate(folds):
            train_indices = np.concatenate([fold for index, fold in enumerate(folds) if index != fold_index])
            yield train_indices, validation_indices
