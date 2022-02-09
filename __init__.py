# Version to check with the config, only when versions match can we proceed
__version__ = 0.3
import torch
import model


crits = {
    "MSELoss": torch.nn.MSELoss()
}


models = {
    "BaselineModel": model.BaselineModel,
    "SimpleModel": model.SimpleModel,
}

act_fun = {
    "tanh": torch.nn.Tanh(),
    "tanh/2": lambda x: torch.functional.tanh(x/2),
    "relu": torch.nn.ReLU(),
    "sigmoid": torch.nn.Sigmoid(),
    "leaky_relu": torch.nn.LeakyReLU(),
    "selu": torch.nn.SELU(),
}

