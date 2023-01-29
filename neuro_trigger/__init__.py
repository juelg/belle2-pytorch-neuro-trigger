"""
 Copyright (c) 2021-2023 Tobias Juelg

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
 """

__version__ = 0.4
import torch
from easydict import EasyDict
from scipy.stats import norm, uniform

import neuro_trigger.pytorch.model as model


class LambdaModule(torch.nn.Module):
    def __init__(self, f) -> None:
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


crits = {"MSELoss": torch.nn.MSELoss()}


models = {
    "BaselineModel": model.BaselineModel,
    "BaselineModelBN": model.BaselineModelBN,
    "SimpleModel": model.SimpleModel,
}

# definition of supported optimizers
supported_optimizers = ("Adam", "Rprob", "SGD")

act_fun = {
    "tanh": torch.nn.Tanh(),
    "tanh/2": LambdaModule(lambda x: torch.tanh(x / 2)),
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
        return uniform(
            loc=conf_key["uniform"]["lower"],
            scale=(conf_key["uniform"]["upper"] - conf_key["uniform"]["lower"]),
        )
