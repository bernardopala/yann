import numpy as np

from .splitter import Splitter


class TimeSeriesSplitter(Splitter):
    def __init__(self, n_splits: int, validation_size: int, gap: int = 0, train_size: int | None = None, expanding: bool = True) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be positive")
        if validation_size < 1:
            raise ValueError("validation_size must be positive")
        if gap < 0:
            raise ValueError("gap cannot be negative")
        if train_size is not None and train_size < 1:
            raise ValueError("train_size must be positive")

        self.n_splits = int(n_splits)
        self.validation_size = int(validation_size)
        self.gap = int(gap)
        self.train_size = train_size
        self.expanding = bool(expanding)

    @property
    def label(self) -> str:
        return "Time-series split"

    def split(self, X: np.ndarray, Y: np.ndarray):
        if len(X) != len(Y):
            raise ValueError("X and Y must contain the same number of samples")

        initial_train_size = self.train_size

        if initial_train_size is None:
            initial_train_size = len(X) - self.gap - self.n_splits * self.validation_size

        if initial_train_size < 1:
            raise ValueError("Not enough samples for requested splits")

        for split_index in range(self.n_splits):
            train_end = initial_train_size + split_index * self.validation_size
            validation_start = train_end + self.gap
            validation_end = validation_start + self.validation_size

            if validation_end > len(X):
                raise ValueError("Not enough samples for requested splits")

            train_start = 0 if self.expanding else train_end - initial_train_size
            yield np.arange(train_start, train_end), np.arange(validation_start, validation_end)
