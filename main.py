from multiprocessing import Pool
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, base
from dataset import BelleIIExpert
from pl_module import NeuroTrigger
import os
from torchvision.transforms import transforms
import numpy as np
from torchvision.transforms.functional import crop
import torch
from configs import configs
from pathlib import Path
import logging
import sys



config = "baseline_v1"
base_log = "log"
gpu_idx = 0
experts = [-1] #[0, 1, 2, 3, 4]

# train = "/home/tobi/neurotrigger/train1"
# val = "/home/tobi/neurotrigger/valid1"
# test = "/home/tobi/neurotrigger/test1"

train = "/remote/neurobelle/data/dqmNeuro/dqmNeuro_mpp34_exp20_430-459/lt100reco/idhist_10170_default/section_fp/random1.gz"
val = "/remote/neurobelle/data/dqmNeuro/dqmNeuro_mpp34_exp20_430-459/lt100reco/idhist_10170_default/section_fp/random2.gz"
test = "/remote/neurobelle/data/dqmNeuro/dqmNeuro_mpp34_exp20_430-459/lt100reco/idhist_10170_default/section_fp/random3.gz"


data = (train, val, test)
hparams = configs[config]
hparams["config"] = config


# check the latest version
version = max([int(str(i).split("_")[-1]) for i in (Path(base_log) / config).glob("version_*")], default=-1) + 1
print(version)

log_folder = os.path.join(base_log, config, f"version_{version}")
# todo maybe create folder
if not Path(log_folder).exists():
    Path(log_folder).mkdir(parents=True)



# logging.basicConfig(filename='myapp.log', level=logging.DEBUG)
logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
        filename=os.path.join(log_folder, 'main.log'),
        filemode='w'
)

# create file logger
for expert in experts:
    logger = logging.getLogger(f'expert_{expert}')
    fh = logging.FileHandler(os.path.join(log_folder, f"expert_{expert}.log"), mode="w")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

class LambdaTrans():
    def __init__(self, f):
        self.f = f

    def __call__(self, x):
        return self.f(x)


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass



if __name__ == "__main__":



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


    trainers_modules = []
    for expert in experts:
        pl_module = NeuroTrigger(hparams, data, expert=expert)
        trainer = pl.Trainer(
            #row_log_interval=1,
            #track_grad_norm=2,
            # weights_summary=None,
            #distributed_backend='dp',
            callbacks=callbacks,
            max_epochs=hparams["epochs"],
            deterministic=True,
            #profiler=True,
            #fast_dev_run=True,
            # gpus=[gpu_idx], #[0, 1],
            default_root_dir=os.path.join(log_folder, f"expert_{expert}"),
            #auto_select_gpus=True,
            #enable_pl_optimizer=True,
        )
        trainers_modules.append((trainer, pl_module))

    def fit(trainer_module):
        # normal call: trainer.fit(pl_module)
        # expert = trainer_module[1].expert
        # logger = logging.getLogger(f'expert_{expert}')
        # fh = logging.FileHandler(os.path.join(log_folder, f"expert_{expert}"))
        # logger.addHandler(fh)
        sys.stdout = StreamToLogger(logger,logging.INFO)
        sys.stderr = StreamToLogger(logger,logging.ERROR)
        trainer_module[0].fit(trainer_module[1])



    fit(trainer_module=trainers_modules[0])
    # TODO lookup how pool can allow child processes
    # with Pool(len(experts)) as p:
    #     p.map(fit, trainers_modules)