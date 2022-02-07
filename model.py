from torch import nn
import torch


def pad(f):
    return int((f - 1) / 2)


class BaselineModel(nn.Module):
    def __init__(self, inp: int = 27, out: int = 2, act=nn.Tanh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 81),
            act(),
            nn.Linear(81, out),
            act(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleModel(nn.Module):
    def __init__(self, inp: int = 27, out: int = 2, act=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 50),
            nn.BatchNorm1d(50),
            act(),
            nn.Linear(50, 100),
            nn.BatchNorm1d(100),
            act(),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            act(),
            nn.Linear(50, 10),
            nn.BatchNorm1d(10),
            act(),
            nn.Linear(10, out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # get models dtype for conversion:
        # next(self.parameters()).dtype
        return self.net(x)
