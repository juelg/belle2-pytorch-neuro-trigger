from typing import Optional
from torch import nn
import torch


def pad(f: int) -> int:
    """Same-padding for convolutions
    -- Currently not in use as no convolutions are used --
    Args:
        f (int): filter / kernel size
    Returns:
        int: padding
    """
    return int((f - 1) / 2)


class BaselineModel(nn.Module):
    def __init__(self, inp: int = 27, out: int = 2, act: Optional[nn.Module]=None):
        """Recreation of the model developed in BASF2

        Args:
            inp (int, optional): Number of inputs (input neurons). Defaults to 27.
            out (int, optional): Number of outputs (output neurons). Setting this to one will only train on Z. Defaults to 2.
            act (Optional[nn.Module], optional): Activation function. Defaults to nn.Tanh.
        """
        super().__init__()
        act = act or nn.Tanh()
        self.net = nn.Sequential(
            nn.Linear(inp, 81),
            act,
            nn.Linear(81, out),
            act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class BaselineModelBN(nn.Module):
    def __init__(self, inp: int = 27, out: int = 2, act: Optional[nn.Module]=None):
        """Baseline model from BASF extended with batchnorm to stabalize the training

        Args:
            inp (int, optional): Number of inputs (input neurons). Defaults to 27.
            out (int, optional): Number of outputs (output neurons). Setting this to one will only train on Z. Defaults to 2.
            act (Optional[nn.Module], optional): Activation function. Defaults to nn.Tanh.
        """
        super().__init__()
        act = act or nn.Tanh()
        self.net = nn.Sequential(
            nn.BatchNorm1d(inp),
            nn.Linear(inp, 81),
            act,
            nn.BatchNorm1d(81),
            nn.Linear(81, out),
            act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleModel(nn.Module):
    def __init__(self, inp: int = 27, out: int = 2, act: Optional[nn.Module]=None):
        """Model with 5 layers with batchnorm in between.

        Args:
            inp (int, optional): Number of inputs (input neurons). Defaults to 27.
            out (int, optional): Number of outputs (output neurons). Setting this to one will only train on Z. Defaults to 2.
            act (Optional[nn.Module], optional): Activation function. Defaults to nn.Tanh.
        """
        super().__init__()
        act = act or nn.Tanh()
        self.net = nn.Sequential(
            nn.Linear(inp, 50),
            nn.BatchNorm1d(50),
            act,
            nn.Linear(50, 100),
            nn.BatchNorm1d(100),
            act,
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            act,
            nn.Linear(50, 10),
            nn.BatchNorm1d(10),
            act,
            nn.Linear(10, out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # get models dtype for conversion:
        # next(self.parameters()).dtype
        return self.net(x)
