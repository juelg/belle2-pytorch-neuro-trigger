import threading
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pl_module import NeuroTrigger
import os
from configs import get_hyperpar_by_name
from pathlib import Path
import logging


# config = "baseline_v1"
config = "only_z"
base_log = "log"
gpu_idx = 0
experts = [0, 1, 2, 3, 4] # [-1] #[0, 1, 2, 3, 4]
enable_progress_bar = False

# train = "/home/tobi/neurotrigger/train1"
# val = "/home/tobi/neurotrigger/valid1"
# test = "/home/tobi/neurotrigger/test1"

# train = "/remote/neurobelle/data/dqmNeuro/dqmNeuro_mpp34_exp20_430-459/lt100reco/idhist_10170_default/section_fp/neuroresults_random1.gz"
# val = "/remote/neurobelle/data/dqmNeuro/dqmNeuro_mpp34_exp20_430-459/lt100reco/idhist_10170_default/section_fp/neuroresults_random2.gz"
# test = "/remote/neurobelle/data/dqmNeuro/dqmNeuro_mpp34_exp20_430-459/lt100reco/idhist_10170_default/section_fp/neuroresults_random3.gz"

# sshfs juelg@neurobelle.mpp.mpg.de:/mnt/scratch/data data
train = "/home/iwsatlas1/juelg/data/dqmNeuro/dqmNeuro_mpp34_exp20_430-459/lt100reco/idhist_10170_default/section_fp/neuroresults_random1.gz"
val   = "/home/iwsatlas1/juelg/data/dqmNeuro/dqmNeuro_mpp34_exp20_430-459/lt100reco/idhist_10170_default/section_fp/neuroresults_random2.gz"
test  = "/home/iwsatlas1/juelg/data/dqmNeuro/dqmNeuro_mpp34_exp20_430-459/lt100reco/idhist_10170_default/section_fp/neuroresults_random3.gz"


data = (train, val, test)
hparams = get_hyperpar_by_name(config)

experts_str = [f"expert_{i}" for i in experts]
logger = logging.getLogger()

# check the latest version
version = max([int(str(i).split("_")[-1]) for i in (Path(base_log) / config).glob("version_*")], default=-1) + 1

log_folder = os.path.join(base_log, config, f"version_{version}")
# todo maybe create folder
if not Path(log_folder).exists():
    Path(log_folder).mkdir(parents=True)



logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
        # filename=os.path.join(log_folder, 'main.log'),
        # filemode='w'
)

# mute other libs debugging output
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True
logging.getLogger('matplotlib.colorbar').disabled = True
logging.getLogger('PIL.PngImagePlugin').disabled = True
logging.getLogger('h5py._conv').disabled = True

class ThreadLogFilter(logging.Filter):
    """
    This filter only show log entries for specified thread name
    """

    def __init__(self, thread_name, *args, **kwargs):
        logging.Filter.__init__(self, *args, **kwargs)
        self.thread_name = thread_name

    def filter(self, record):
        return record.threadName == self.thread_name

# general logger which logs everything
fh = logging.FileHandler(os.path.join(log_folder, f"app.log"), mode="w")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(threadName)s:%(levelname)s:%(name)s:%(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

# create file loggers
for expert in experts:
    fh = logging.FileHandler(os.path.join(log_folder, f"expert_{expert}.log"), mode="w")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s:%(threadName)s:%(levelname)s:%(name)s:%(message)s')
    fh.setFormatter(formatter)
    fh.addFilter(ThreadLogFilter(f'expert_{expert}'))
    logger.addHandler(fh)

logger.info(f"Using config {config} in version {version}")

def snap_source_state(log_folder):
    # get git commit id
    os.system(f'git log --format="%H" -n 1 > {os.path.join(log_folder, "git_id.txt")}')
    # get git diff
    os.system(f'git diff > {os.path.join(log_folder, "git_diff.txt")}')



if __name__ == "__main__":


    # save git commit and git diff in file
    snap_source_state(log_folder)

    trainers_modules = []
    for expert in experts:
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=10,
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
            enable_progress_bar=enable_progress_bar,
        )
        trainers_modules.append((trainer, pl_module))

    def fit(trainer_module):
        trainer_module[0].fit(trainer_module[1])

    if len(experts) == 1:
        fit(trainer_module=trainers_modules[0])
    else:
        for trainer_module, expert in zip(trainers_modules, experts_str):
            t = threading.Thread(target=fit,
                                name=expert,
                                args=[trainer_module])
            t.start()
