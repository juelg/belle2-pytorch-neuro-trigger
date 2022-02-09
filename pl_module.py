from ctypes import Union
import json
import os
from typing import Dict, Optional, List, Tuple
import pytorch_lightning as pl
import torch
from torch import optim
from dataset import BelleIIBetter, BelleIIBetterExpert
from torch.utils.data import DataLoader
from model import BaselineModel, SimpleModel
from visualize import Visualize
import logging
from easydict import EasyDict
import copy
from __init__ import crits, models, act_fun
import numpy as np

def init_weights(m: torch.Module, act: str):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain(act))
        # leave bias as it is


class NeuroTrigger(pl.LightningModule):

    def __init__(self, hparams: EasyDict, data: BelleIIBetter, log_path: Optional[str] = None, expert: int = -1):
        super().__init__()
        self.expert = expert
        self.log_path = log_path
        self.hparams.update(self.extract_expert_hparams(hparams))
        self.model = models[self.hparams.model](
            hparams.in_size, hparams.out_size, act=act_fun[self.hparams.act])
        # self.model.apply(init_weights, self.hparams.act)
        self.file_logger = logging.getLogger()

        if self.expert == -1:
            self.data = [BelleIIBetter(
                data[i], logger=self.file_logger, out_dim=hparams.out_size) for i in range(3)]
        else:
            self.data = [BelleIIBetterExpert(self.expert,
                                             data[i], logger=self.file_logger, out_dim=hparams.out_size) for i in range(3)]

        # to see model and crit have a look into the dict defined in __init__.py
        self.crit = crits[self.hparams.loss]
        self.save_hyperparameters()
        self.visualize = Visualize(self, self.data[1])
        self.file_logger.debug(
            f"DONE init expert {self.expert} with loss '{self.hparams.loss}' and model '{self.hparams.model}'")

    def extract_expert_hparams(self, hparams: Union[Dict, EasyDict]):
        expert_hparams = copy.deepcopy(hparams)
        new_expert_hparams = hparams.get(self.exp_str, {})
        expert_hparams.update(new_expert_hparams)
        return expert_hparams

    @property
    def exp_str(self):
        return f"expert_{self.expert}"

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int):
        x, y = batch[0], batch[1]
        y_hat = self.model(x)
        # TODO set more weight on z for learning and the loss function
        loss = self.crit(y_hat, y)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor], batch_idx: int):
        x, y = batch[0], batch[1]
        y_hat = self.model(x)
        loss = self.crit(y_hat, y)
        self.log("val_loss", loss)
        # TODO: find out samples with very high loss

        y_hat_old = batch[2]
        loss_old = self.crit(y_hat_old, y)
        val_loss_vs_old_loss = loss/loss_old
        self.log("val_loss_vs_old_loss", val_loss_vs_old_loss)
        val_z_diff_std = torch.std(y[:,0]-y_hat[:,0])
        self.log("val_z_diff_std", val_z_diff_std)
        self.log("val_z_diff_std_vs_old", val_z_diff_std/torch.std(y[:,0]-y_hat_old[:,0]))

        return y, y_hat, loss, val_loss_vs_old_loss

    def validate(self, path: str, mode: str = "val"):
        # data = {"train": self.train_dataloader, "eval": self.val_dataloader, "test": self.test_dataloader}[mode]()
        mode = {"train": 0, "val": 1, "test": 2}[mode]
        # output dataset -> no, do this for all experts outside of the training
        # create plots
        outputs = []
        with torch.no_grad():
            d = DataLoader(self.data[mode], batch_size=10000, num_workers=0, drop_last=False)
            for i in d:
                x, y, y_hat_old = i[0], i[1], i[2]
                y_hat = self.model(x)
                outputs.append((y, y_hat, y_hat_old))
        pred_data = torch.cat(outputs)

        loss = self.crit(pred_data[:,0], pred_data[:,1])
        loss_old = self.crit(pred_data[:,0], pred_data[:,2])
        val_loss_vs_old_loss = loss/loss_old
        val_z_diff_std = torch.std(pred_data[:,0,0]-pred_data[:,1,0])
        to_save = {"loss": loss.mean(), "loss_old": loss_old.mean(),
                    "val_loss_vs_old_loss": val_loss_vs_old_loss.mean(),
                    "val_z_diff_std": val_z_diff_std.mean()}
        # output final scores in json
        with open(os.path.join(path, "result.csv")) as f:
            json.dump(to_save, f)

        self.visualize.create_plots(
                pred_data[:,0], pred_data[:,1], save=os.path.join(path, "post_training_plots"), create_baseline_plots=True)


    def test_step(self, batch: Tuple[torch.Tensor], batch_idx: int):
        x, y = batch[0], batch[1]
        y_hat = self.model(x)
        loss = self.crit(y_hat, y)
        self.log("test_loss", loss)

    def validation_epoch_end(self, outputs: List[Tuple[torch.Tensor]]):
        # outputs are a list of the tuples that have been returned by the validation
        self.visualize.create_plots(
            torch.cat([i[0] for i in outputs]), torch.cat([i[1] for i in outputs]))
        self.file_logger.info(
            f"{self.exp_str}: epoch #{self.current_epoch} finished with val {np.mean(([i[2] for i in outputs])):.{3}f} and {np.mean(([i[3] for i in outputs])):.{3}f} vs old")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.data[0], batch_size=self.hparams.batch_size, num_workers=self.hparams.workers,
                          drop_last=True, pin_memory=True, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.data[1], batch_size=self.hparams.batch_size, num_workers=self.hparams.workers,
                          drop_last=True, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.data[2], batch_size=self.hparams.batch_size, num_workers=self.hparams.workers,
                          drop_last=True, pin_memory=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # definition of supported optimizers
        if self.hparams.optim == "Adam":
            return optim.Adam(self.model.parameters(), self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optim == "Rprob":
            return optim.Rprop(self.model.parameters(), self.hparams.learning_rate)
        else:
            raise RuntimeError(f"Optimizer {self.hparams.optim} is not defined!")

