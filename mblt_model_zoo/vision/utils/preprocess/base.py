from abc import ABC, abstractmethod


class PreBase(ABC):
    """Base class for preprocess."""

    @abstractmethod
    def __call__(self, x):
        pass
