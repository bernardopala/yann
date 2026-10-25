"""Dataset-specific fold processors used by evaluator tests."""

from .air_quality import AirQualitySplitProcessor
from .iris import IrisSplitProcessor
from .wine_quality import WineQualitySplitProcessor

__all__ = ["AirQualitySplitProcessor", "IrisSplitProcessor", "WineQualitySplitProcessor"]
