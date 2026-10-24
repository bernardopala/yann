import numpy as np

from .splitter import Splitter


class HoldoutSplitter(Splitter):
    def __init__(self, test_size: float = 0.2, shuffle: bool = True, rng: np.random.Generator | None = None) -> None:
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1")

        self.test_size = float(test_size)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng() if rng is None else rng

    @property
    def label(self) -> str:
        return "Holdout split"

    def format_split_label(self, split_index: int) -> str:
        return self.label

    def split(self, X: np.ndarray, Y: np.ndarray):
        if len(X) != len(Y):
            raise ValueError("X and Y must contain the same number of samples")

        indices = np.arange(len(X))

        if self.shuffle:
            indices = self.rng.permutation(indices)

        validation_count = max(1, int(len(X) * self.test_size))
        yield indices[:-validation_count], indices[-validation_count:]
