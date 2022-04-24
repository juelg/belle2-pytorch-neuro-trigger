from typing import Dict, List
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import numpy as np


from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment


class MeanLoggerExp(LightningLoggerBase):
    """Lighting logger for each expert that is just an interface between lighting and
    MeanTBLogger. Thus, all logs are just forwarded to the MeanTBLogger object.
    """
    def __init__(self, version: int, mean_tb_logger: 'MeanTBLogger', expert: int = -1):
        super().__init__()
        self.expert = expert
        self._version = version
        self.mean_tb_logger = mean_tb_logger


    @property
    def name(self):
        return f"MeanLoggerExp{self.expert}"

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        self.mean_tb_logger

    @property
    def version(self):
        # Return the experiment version, int or str.
        return self._version

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        self.mean_tb_logger.log_dict(self.expert, metrics, step)

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        # If you implement this, remember to call `super().save()`
        # at the start of the method (important for aggregation of metrics)
        super().save()

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass




class MeanTBLogger:
    """Wrapper around TensorBoardLogger

    Gathers metrics from all experts, waits until all are there then logs the mean over all experts.
    """
    def __init__(self, path: str, experts: List[int], name=""):
        self.logger = TensorBoardLogger(path, name=name)
        # expert (this will be dict as -1 as key is possible), metric, step
        self.log_data = {expert: {} for expert in experts}

    def check_log(self, metric, step, amount_log_entries):
        lens = [len(self.log_data[expert][metric]) for expert in self.log_data]
        if all([l>=amount_log_entries for l in lens]):
            self.logger.log_metrics({metric: np.mean([self.log_data[expert][metric][amount_log_entries-1] 
                                                        for expert in self.log_data])}, step)


    def log(self, expert: int, metric: str, step: int, value: float):
        # dynamically add metric to log dict
        if metric not in self.log_data[expert]:
            for exp in self.log_data:
                self.log_data[exp][metric] = []
        self.log_data[expert][metric].append(value)

        self.check_log(metric, step, len(self.log_data[expert][metric]))

    def log_dict(self, expert: int, metric_dict: Dict[str, float], step: int):
        for metric, value in metric_dict.items():
            self.log(expert, metric, step, value)

