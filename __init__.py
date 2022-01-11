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
