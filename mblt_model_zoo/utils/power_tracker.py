from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class BasePowerTracker(ABC):
    """Abstract base class for power trackers."""

    def __init__(self, interval: float):
        if interval <= 0:
            raise ValueError("interval must be positive")
        self._interval = interval

    @abstractmethod
    def start(self) -> None:
        """Start tracking power consumption."""

    @abstractmethod
    def stop(self) -> None:
        """Stop tracking power consumption."""

    @abstractmethod
    def get_power_metric(self) -> Dict:
        """Return summarized power metric."""

    @abstractmethod
    def get_power_trace(self) -> List[Tuple[float, float]]:
        """Return sampled power trace as (timestamp, power_watt) pairs."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal sampled data."""
