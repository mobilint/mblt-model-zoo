from abc import ABC, abstractmethod
from typing import List, Union

import torch


class PreOps(ABC):
    """Abstract base class for individual preprocessing operations."""

    def __init__(self):
        """Initializes the preprocessing operation."""
        super().__init__()
        self.device = torch.device("cpu")

    @abstractmethod
    def __call__(self, x):
        """Executes the preprocess operation.

        Args:
            x (Any): Input data to be processed.

        Returns:
            Any: Processed data.
        """

    def to(self, device: Union[str, torch.device]):
        """Move the operation to the specified device.
        Args:
            device (Union[str, torch.device]): Device to move the operation to.
        """
        if isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise TypeError(f"Got unexpected type for device={type(device)}.")
        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(self.device))


class PreBase:
    """Base class for orchestrating a series of preprocessing operations."""

    def __init__(self, Ops: List[PreOps]):
        """Initializes the PreBase class with a list of operations.

        Args:
            Ops (List[PreOps]): List of ordered PreOps instances to be applied.
        """
        self.Ops = Ops
        self._check_ops()
        self.device = torch.device("cpu")

    def _check_ops(self):
        """Check if the operations are valid."""
        for op in self.Ops:
            if not isinstance(op, PreOps):
                raise TypeError(f"Got unsupported type={type(op)}.")

    def __call__(self, x):
        """Applies the sequence of preprocessing operations to the input.

        Args:
            x (Any): Initial input data.

        Returns:
            Any: Fully processed data.
        """
        for op in self.Ops:
            x = op(x)
        return x

    def to(self, device: Union[str, torch.device]):
        """Move the operations to the specified device.
        Args:
            device (Union[str, torch.device]): Device to move the operations to.
        """
        if isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise TypeError(f"Got unexpected type for device={type(device)}.")
        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(self.device))
        for op in self.Ops:
            op.to(self.device)
