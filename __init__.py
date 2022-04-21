# Version to check with the config, only when versions match can we proceed
__version__ = 0.4
import torch
import model

class LambdaModule(torch.nn.Module):
    def __init__(self, f) -> None:
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


crits = {
    "MSELoss": torch.nn.MSELoss()
}


models = {
    "BaselineModel": model.BaselineModel,
    "BaselineModelBN": model.BaselineModelBN,
    "SimpleModel": model.SimpleModel,
}

act_fun = {
    "tanh": torch.nn.Tanh(),
    "tanh/2": LambdaModule(lambda x: torch.tanh(x/2)),
    "softsign": torch.nn.Softsign(),
    "relu": torch.nn.ReLU(),
    "sigmoid": torch.nn.Sigmoid(),
    "leaky_relu": torch.nn.LeakyReLU(),
    "selu": torch.nn.SELU(),
}

