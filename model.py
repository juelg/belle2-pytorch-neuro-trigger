from torch import nn

def pad(f):
    return int((f - 1) / 2)

class BaselineModel(nn.Module):
    def __init__(self, inp=27, out=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 81),
            nn.Tanh(),
            nn.Linear(81, out),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)

class SimpleModel(nn.Module):
    def __init__(self, inp=27, out=2, act=nn.ReLU):
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

    def forward(self, x):
        return self.net(x)

