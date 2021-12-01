import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pl_module import AutoModule
import os
from torchvision.transforms import transforms
import numpy as np
from torchvision.transforms.functional import crop
import torch

hparams = {"learning_rate": 1e-3, "batch_size": 2048, "weight_decay": 1e-6, "in_size": 27, "out_size": 2, "workers": 6,
    "noise": None}
gpu_idx = 0
epochs = 50

train = "/home/tobi/neurotrigger/train1"
val = "/home/tobi/neurotrigger/valid1"
test = "/home/tobi/neurotrigger/test1"

data = (train, val, test)



class LambdaTrans():
    def __init__(self, f):
        self.f = f

    def __call__(self, x):
        return self.f(x)


transforms_compose = transforms.Compose([
        LambdaTrans(lambda x: crop(x, 160-100, 0, 100, 320)), # crop out the upper part of the image -> 72 x 320
        transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(9, sigma=(0.1, 2))]), p=0.8),
        
        transforms.ToTensor(),
        # transforms.Normalize(mean, std),

        # LambdaTrans(lambda x: x/255)
    ])


if __name__ == "__main__":


        
    pl_module = AutoModule(hparams, data)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        strict=True,
        verbose=True,
        mode='min'
    )
    model_checkpoint = ModelCheckpoint(
                monitor='val_loss',
                save_last=True,
                save_top_k=1,
    )
    # callbacks = [early_stop_callback, model_checkpoint]
    callbacks = [model_checkpoint]

    trainer = pl.Trainer(
        #row_log_interval=1,
        #track_grad_norm=2,
        # weights_summary=None,
        #distributed_backend='dp',
        callbacks=callbacks,
        max_epochs=epochs,
        deterministic=True,
        #profiler=True,
        #fast_dev_run=True,
        gpus=[gpu_idx], #[0, 1],
        #default_root_dir="lightning_logs" #os.path.join(results_path, "supervised", "loss_dist_sphere_fix_radius", "asdf"),
        #auto_select_gpus=True,
        #enable_pl_optimizer=True,
    )
    trainer.fit(pl_module)
