"""Base interface for target specifications."""

from abc import ABC, abstractmethod

from .target_adapter import TargetAdapter


class TargetSpec(ABC):
    """Configuration used to create split-local adapters."""

    @abstractmethod
    def create_adapter(self) -> TargetAdapter:
        """Create a fresh, unfitted adapter."""
