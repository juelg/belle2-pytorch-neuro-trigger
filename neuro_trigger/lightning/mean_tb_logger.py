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

from multiprocessing import Queue
from threading import Thread
from typing import List

import numpy as np
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only


class MeanLoggerExp(LightningLoggerBase):
    """Lighting logger for each expert that is just an interface between lighting and
    MeanTBLogger. Thus, all logs are just forwarded to the MeanTBLogger object.
    """

    def __init__(
        self,
        queue: Queue,
        version: int,
        mean_tb_logger: "MeanTBLogger",
        expert: int = -1,
    ):
        super().__init__()
        self.expert = expert
        self._version = version
        self.mean_tb_logger = mean_tb_logger
        self.queue = queue

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
        for metric, value in metrics.items():
            self.queue.put((self.expert, metric, step, value))

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


class MeanTBLogger(Thread):
    """Wrapper around TensorBoardLogger

    Gathers metrics from all experts, waits until all are there then logs the mean over all experts.
    """

    def __init__(self, path: str, experts: List[int], name: str = ""):
        Thread.__init__(self)
        self.logger = TensorBoardLogger(path, name=name)
        # expert (this will be dict as -1 as key is possible), metric, step
        self.log_data = {expert: {} for expert in experts}
        self.queue = Queue()
        self.running = False

    def start_thread(self):
        """Starts the logging thread"""
        self.running = True
        self.start()

    def stop_thread(self):
        """Stops the logging thread by sending a stop signal"""
        self.running = False
        self.queue.put("stop")
        self.join()

    def log(self, expert: int, metric: str, step: int, value: float):
        """Logs to the mean tensorboard metric graph

        Args:
            expert (int): expert number
            metric (str): metric name
            step (int): logging step
            value (float): value to log
        """
        if metric not in self.log_data[expert]:
            for exp in self.log_data:
                self.log_data[exp][metric] = []
        self.log_data[expert][metric].append(value)

        vals = self.get_ith_metric_value(metric, step)
        self.logger.log_metrics({metric: np.mean(vals)}, step)

    def get_ith_metric_value(self, metric: str, step) -> np.ndarray:
        step = int(step)
        vals = []
        for expert in self.log_data:
            if len(self.log_data[expert][metric]) > step:
                vals.append(self.log_data[expert][metric][step])

            elif len(self.log_data[expert][metric]) != 0:
                vals.append(self.log_data[expert][metric][-1])

        return np.array(vals)

    def run(self):
        while self.running:
            data = self.queue.get(block=True)
            if isinstance(data, str):
                return
            expert, metric, step, value = data
            self.log(expert, metric, step, value)
