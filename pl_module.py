import copy
import os
from typing import Dict, Optional, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import optim
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from functools import partial
from dataset import BelleII, BelleIIBetter, BelleIIExpert, BelleIIBetterExpert
from model import BaselineModel, SimpleModel
import numpy as np
from visualize import Visualize
import logging

DEBUG = False
# NUM_WORKERS = 4 #os.cpu_count() if not DEBUG else 0



class NeuroTrigger(pl.LightningModule):

    def __init__(self, hparams: Dict, data, expert=-1):
        super().__init__()
        self.hparams.update(hparams)
        self.model = SimpleModel(hparams["in_size"], hparams["out_size"])
        # self.model = BaselineModel(hparams["in_size"], hparams["out_size"])
        self.expert = expert
        self.file_logger = logging.getLogger(f"expert_{expert}")

        if self.expert == -1:
            self.data = [BelleIIBetter(data[i], logger=self.file_logger) for i in range(3)]
        else:
            self.data = [BelleIIBetterExpert(data[i], logger=self.file_logger, expert=self.expert) for i in range(3)]

        self.crit = torch.nn.MSELoss()
        self.save_hyperparameters()
        self.visualize = Visualize(self, self.data[1])
        self.file_logger.debug("DONE init")


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.model(x)
        #if self.hparams.get("noise") is not None:
        #    bs = x.shape[0]
        #    y = y+torch.normal(0, 1*0.005, size=(bs,), device=self.device)
        # TODO set more weight on z for learning and the loss function
        loss = self.crit(y_hat, y)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.model(x)
        loss = self.crit(y_hat, y)
        self.log("val_loss", loss)
        # TODO: compare loss to other networks output
        # find out samples with very high loss
        # compute and display loss of the other networks output

        y_hat_old = batch[2]
        loss_old = self.crit(y_hat_old, y)
        self.log("val_loss_vs_old_loss", loss/loss_old)

        return y, y_hat

    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.model(x)
        loss = self.crit(y_hat, y)
        self.log("test_loss", loss)

    def validation_epoch_end(self, outputs):
        # todo, use outputs

        self.visualize.create_plots(torch.cat([i[0] for i in outputs]), torch.cat([i[1] for i in outputs]))
        # self.visualize.y = (torch.cat([i[0] for i in outputs]), torch.cat([i[1] for i in outputs]))
        # self.visualize.z_plot()
        # self.visualize.hist_plot()


    def train_dataloader(self):
        return DataLoader(self.data[0], batch_size=self.hparams["batch_size"], num_workers=self.hparams["workers"],
                              drop_last=True, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data[1], batch_size=self.hparams["batch_size"], num_workers=self.hparams["workers"],
                              drop_last=True, pin_memory=True)


    def test_dataloader(self):
        return DataLoader(self.data[2], batch_size=self.hparams["batch_size"], num_workers=self.hparams["workers"],
                              drop_last=True, pin_memory=True)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), self.hparams['learning_rate'], weight_decay=self.hparams.get('weight_decay', 0))
    
