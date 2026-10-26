"""Time-series window specification."""

from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class WindowSpec:
    lookback: int
    horizon: int
    stride: int = 1

    def __post_init__(self) -> None:
        parameters = (self.lookback, self.horizon, self.stride)
        if any(not isinstance(parameter, Integral) or isinstance(parameter, bool) for parameter in parameters):
            raise TypeError("lookback, horizon and stride must be integers")
        if self.lookback < 1 or self.horizon < 1 or self.stride < 1:
            raise ValueError("lookback, horizon and stride must be positive")
