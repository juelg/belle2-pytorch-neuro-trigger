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

def init_weights(m, act):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain(act))
        # leave bias as it is



class NeuroTrigger(pl.LightningModule):

    def __init__(self, hparams: EasyDict, data, expert=-1):
        super().__init__()
        self.expert = expert
        self.hparams.update(self.extract_expert_hparams(hparams))
        # self.model = SimpleModel(hparams.in_size, hparams.out_size)
        # self.model = BaselineModel(hparams.in_size, hparams.out_size)
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

        self.crit = crits[self.hparams.loss]  # torch.nn.MSELoss()
        self.save_hyperparameters()
        self.visualize = Visualize(self, self.data[1])
        self.file_logger.debug(
            f"DONE init expert {self.expert} with loss '{self.hparams.loss}' and model '{self.hparams.model}'")

    def extract_expert_hparams(self, hparams):
        expert_hparams = copy.deepcopy(hparams)
        new_expert_hparams = hparams.get(f"expert_{self.expert}", {})
        expert_hparams.update(new_expert_hparams)
        return expert_hparams

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.model(x)
        # TODO set more weight on z for learning and the loss function
        loss = self.crit(y_hat, y)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
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

    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.model(x)
        loss = self.crit(y_hat, y)
        self.log("test_loss", loss)

    def validation_epoch_end(self, outputs):
        self.visualize.create_plots(
            torch.cat([i[0] for i in outputs]), torch.cat([i[1] for i in outputs]))
        self.file_logger.info(
            f"expert_{self.expert}: epoch #{self.current_epoch} finished with val {np.mean(([i[2] for i in outputs])):.{3}f} and {np.mean(([i[3] for i in outputs])):.{3}f} vs old")

    def train_dataloader(self):
        return DataLoader(self.data[0], batch_size=self.hparams.batch_size, num_workers=self.hparams.workers,
                          drop_last=True, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data[1], batch_size=self.hparams.batch_size, num_workers=self.hparams.workers,
                          drop_last=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.data[2], batch_size=self.hparams.batch_size, num_workers=self.hparams.workers,
                          drop_last=True, pin_memory=True)

    def configure_optimizers(self):
        # definition of supported optimizers
        if self.hparams.optim == "Adam":
            return optim.Adam(self.model.parameters(), self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optim == "Rprob":
            return optim.Rprop(self.model.parameters(), self.hparams.learning_rate)
        else:
            raise RuntimeError(f"Optimizer {self.hparams.optim} is not defined!")

