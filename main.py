import threading
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from mean_tb_logger import MeanLoggerExp, MeanTBLogger
from pl_module import NeuroTrigger
import os
from configs import get_hyperpar_by_name
from pathlib import Path
import logging
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import torch

from utils import ThreadLogFilter, create_dataset_with_predictions, expert_weights_json, save_predictions_pickle, snap_source_state

debug = True
config = "baseline_v4_softsign"
base_log = "/tmp/nt_pytorch_debug_log" if debug else "log" 
gpu_idx = 0
enable_progress_bar = False


# train = "/home/tobi/neurotrigger/train1"
# val = "/home/tobi/neurotrigger/valid1"
# test = "/home/tobi/neurotrigger/test1"

# train = "/remote/neurobelle/data/dqmNeuro/dqmNeuro_mpp34_exp20_430-459/lt100reco/idhist_10170_default/section_fp/neuroresults_random1.gz"
# val = "/remote/neurobelle/data/dqmNeuro/dqmNeuro_mpp34_exp20_430-459/lt100reco/idhist_10170_default/section_fp/neuroresults_random2.gz"
# test = "/remote/neurobelle/data/dqmNeuro/dqmNeuro_mpp34_exp20_430-459/lt100reco/idhist_10170_default/section_fp/neuroresults_random3.gz"

# sshfs juelg@neurobelle.mpp.mpg.de:/mnt/scratch/data data
if debug:
    train = "data/dqmNeuro/dqmNeuro_mpp34_exp20_400-944/lt100reco/idhist_10170_default/section_correct_fp/neuroresults_random1.gz"
    val =   "data/dqmNeuro/dqmNeuro_mpp34_exp20_400-944/lt100reco/idhist_10170_default/section_correct_fp/neuroresults_random2.gz"
    test =  "data/dqmNeuro/dqmNeuro_mpp34_exp20_400-944/lt100reco/idhist_10170_default/section_correct_fp/neuroresults_random3.gz"
else:
    train = "data/dqmNeuro/dqmNeuro_mpp34_exp20_400-944/lt100reco/idhist_10170_default/section_correct_fp/neuroresults_random1.gz"
    val =   "data/dqmNeuro/dqmNeuro_mpp34_exp20_400-944/lt100reco/idhist_10170_default/section_correct_fp/neuroresults_random2.gz"
    test =  "data/dqmNeuro/dqmNeuro_mpp34_exp20_400-944/lt100reco/idhist_10170_default/section_correct_fp/neuroresults_random3.gz"

def fit(trainer_module, logger):
    try:
        # train
        logger.info(f"Expert {trainer_module[1].expert} start training.")
        trainer_module[0].fit(trainer_module[1])
    except ValueError:
        # needed to avoid signal error from pytorch lightning in threads
        logger.info(f"Expert {trainer_module[1].expert} has finished training.")

    # load the best weights for evaluation
    ckpt_path = os.path.join(trainer_module[1].log_path, "ckpts")
    best_ckpt = os.path.join(ckpt_path, [i for i in os.listdir(ckpt_path) if i.startswith("epoch")][0])
    trainer_module[1].load_state_dict(torch.load(best_ckpt)["state_dict"])

    # create eval plots
    trainer_module[1].validate(path=trainer_module[1].log_path, mode="val")
    logger.info(f"Expert {trainer_module[1].expert} done creating val plots, finished.")


data = (train, val, test)
hparams = get_hyperpar_by_name(config)
if debug:
    hparams["epochs"] = 3
experts = hparams.experts if not debug else [-1] #[0, 1, 2, 3, 4]

experts_str = [f"expert_{i}" for i in experts]
logger = logging.getLogger()

# check the latest version
version = max([int(str(i).split("_")[-1])
              for i in (Path(base_log) / config).glob("version_*")], default=-1) + 1

log_folder = os.path.join(base_log, config, f"version_{version}")
if not Path(log_folder).exists():
    Path(log_folder).mkdir(parents=True)

if not debug:
    # force a short experiment description
    with open(os.path.join(log_folder, "desc.txt"), "w") as f:
        f.write(f"Config description: {hparams.description}")
    os.system(f"vim {os.path.join(log_folder, 'desc.txt')}")


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



# general logger which logs everything
fh = logging.FileHandler(os.path.join(log_folder, f"app.log"), mode="w")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s:%(threadName)s:%(levelname)s:%(name)s:%(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

# create file loggers
for expert in experts:
    fh = logging.FileHandler(os.path.join(
        log_folder, f"expert_{expert}.log"), mode="w")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s:%(threadName)s:%(levelname)s:%(name)s:%(message)s')
    fh.setFormatter(formatter)
    fh.addFilter(ThreadLogFilter(f'expert_{expert}'))
    logger.addHandler(fh)

logger.info(f"Using config {config} in version {version}")



if __name__ == "__main__":
    # save git commit and git diff in file
    snap_source_state(log_folder)

    trainers_modules = []
    for expert in experts:
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=30,
            strict=True,
            verbose=True,
            mode='min'
        )
        model_checkpoint = ModelCheckpoint(
            os.path.join(log_folder, f"expert_{expert}", "ckpts"),
            monitor='val_loss',
            save_last=True,
            save_top_k=1,
        )
        callbacks = [early_stop_callback, model_checkpoint]

        mean_tb_logger = MeanTBLogger(os.path.join(log_folder, "mean_expert"), experts)

        pl_module = NeuroTrigger(hparams, data, expert=expert, log_path=os.path.join(log_folder, f"expert_{expert}"))
        trainer = pl.Trainer(
            logger=[TensorBoardLogger(os.path.join(log_folder, f"expert_{expert}"), "tb"), 
                        CSVLogger(os.path.join(log_folder, f"expert_{expert}"), "csv"),
                        MeanLoggerExp(version, mean_tb_logger, expert)],
            # row_log_interval=1,
            # track_grad_norm=2,
            # weights_summary=None,
            # distributed_backend='dp',
            callbacks=callbacks,
            max_epochs=hparams["epochs"],
            deterministic=True,
            # log_every_n_steps=1,
            # profiler=True,
            # fast_dev_run=True,
            # gpus=[gpu_idx], #[0, 1],
            default_root_dir=os.path.join(log_folder, f"expert_{expert}"),
            # auto_select_gpus=True,
            # enable_pl_optimizer=True,
            enable_progress_bar=enable_progress_bar,
        )
        trainers_modules.append((trainer, pl_module))


    if len(experts) == 1:
        fit(trainer_module=trainers_modules[0], logger=logger)
    else:
        threads = []
        for trainer_module, expert in zip(trainers_modules, experts_str):
            t = threading.Thread(target=fit,
                                 name=expert,
                                 args=[trainer_module, logger])
            t.start()
            threads.append(t)
    # wait for all threads to finish training
    if len(experts) != 1:
        for t in threads:
            t.join()
    # create dataset with predictions
    expert_modules = [i[1] for i in trainers_modules]
    create_dataset_with_predictions(expert_modules, path=log_folder, mode="test")
    expert_weights_json(expert_modules, path=log_folder)
    save_predictions_pickle(expert_modules, path=log_folder, mode="train")
    save_predictions_pickle(expert_modules, path=log_folder, mode="val")
    save_predictions_pickle(expert_modules, path=log_folder, mode="test")

