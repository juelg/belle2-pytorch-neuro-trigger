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
from dataset import BelleII
from model import SimpleModel
import numpy as np
from visualize import Visualize

DEBUG = False
MULTI_GPU = False
# NUM_WORKERS = 4 #os.cpu_count() if not DEBUG else 0




class AutoModule(pl.LightningModule):

    def __init__(self, hparams: Dict, data):
        super().__init__()
        self.hparams.update(hparams)
        self.model = SimpleModel(hparams["in_size"], hparams["out_size"])

        self.data = [BelleII(data[i]) for i in range(3)]

        self.crit = torch.nn.MSELoss()
        self.save_hyperparameters()
        self.visualize = Visualize(self, self.data[1])
        print("DONE init")


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        #if self.hparams.get("noise") is not None:
        #    bs = x.shape[0]
        #    y = y+torch.normal(0, 1*0.005, size=(bs,), device=self.device)
        loss = self.crit(y_hat, y)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.crit(y_hat, y)
        self.log("val_loss", loss)
        # todo return dict, which is output -> vis does not need a forward pass

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.crit(y_hat, y)
        self.log("test_loss", loss)

    def validation_epoch_end(self, outputs):
        # todo, use outputs
        self.visualize.create_plot()


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
    
