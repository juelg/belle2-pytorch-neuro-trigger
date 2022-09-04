from typing import List
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from multiprocessing import Queue
from threading import Thread


from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment


class MeanLoggerExp(LightningLoggerBase):
    """Lighting logger for each expert that is just an interface between lighting and
    MeanTBLogger. Thus, all logs are just forwarded to the MeanTBLogger object.
    """
    def __init__(self, queue: Queue, version: int, mean_tb_logger: 'MeanTBLogger', expert: int = -1):
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
        self.running = True
        self.start()
    
    def stop_thread(self):
        self.running = False
        self.queue.put("stop")
        self.join()

    def check_log(self, metric: str, step: int, amount_log_entries: int):
        lens = [len(self.log_data[expert][metric]) for expert in self.log_data]
        if all([l>=amount_log_entries for l in lens]):
            self.logger.log_metrics({metric: np.mean([self.log_data[expert][metric][amount_log_entries-1] 
                                                        for expert in self.log_data])}, step)

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

        self.check_log(metric, step, len(self.log_data[expert][metric]))

    def run(self):
        while self.running:
            data = self.queue.get(block=True)
            if isinstance(data, str):
                return
            expert, metric, step, value = data
            self.log(expert, metric, step, value)

