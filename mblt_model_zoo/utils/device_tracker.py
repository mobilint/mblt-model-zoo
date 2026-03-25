from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class BaseDeviceTracker(ABC):
    """Abstract base class for device trackers."""

    def __init__(self, interval: float):
        if interval <= 0:
            raise ValueError("interval must be positive")
        self._interval = interval

    @abstractmethod
    def start(self) -> None:
        """Start tracking device metrics."""

    @abstractmethod
    def stop(self) -> None:
        """Stop tracking device metrics."""

    @abstractmethod
    def get_metric(self) -> Dict:
        """Return summarized device metrics."""

    @abstractmethod
    def get_trace(self) -> List[Tuple[float, float]]:
        """Return sampled power trace as (timestamp, power_watt) pairs."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal sampled data."""
