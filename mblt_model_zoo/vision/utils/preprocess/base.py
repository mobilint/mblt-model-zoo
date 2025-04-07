from abc import ABC, abstractmethod
import torch
from typing import Union, List


class PreOps(ABC):
    """Base class for preprocess operations."""

    def __init__(self):
        """Initialize the PreOps class."""
        super().__init__()
        self.device = torch.device("cpu")

    @abstractmethod
    def __call__(self, x):
        pass

    def to(self, device: Union[str, torch.device]):
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
    """Base class for preprocess."""

    def __init__(self, Ops: List[PreOps]):
        """Initialize the PreBase class.

        Args:
            Ops (list): List of operations to be applied.
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
        """Apply the operations to the input.

        Args:
            x: Input data.

        Returns:
            x: Processed data.
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
