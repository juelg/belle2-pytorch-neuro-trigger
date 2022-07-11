# Version to check with the config, only when versions match can we proceed
__version__ = 0.4
import torch
import neuro_trigger.pytorch.model as model
from scipy.stats import norm, uniform
from easydict import EasyDict

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

# definition of supported optimizers
supported_optimizers = ("Adam", "Rprob", "SGD")

act_fun = {
    "tanh": torch.nn.Tanh(),
    "tanh/2": LambdaModule(lambda x: torch.tanh(x/2)),
    "softsign": torch.nn.Softsign(),
    "relu": torch.nn.ReLU(),
    "sigmoid": torch.nn.Sigmoid(),
    "leaky_relu": torch.nn.LeakyReLU(),
    "selu": torch.nn.SELU(),
}


def get_dist_func(conf_key: EasyDict):
    if "norm" in conf_key:
        return norm(loc=conf_key["norm"]["mean"], scale=conf_key["norm"]["std"])
    elif "uniform" in conf_key:
        return uniform(loc=conf_key["uniform"]["lower"], scale=(conf_key["uniform"]["upper"] - conf_key["uniform"]["lower"]))






