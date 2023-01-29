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

import argparse
import itertools
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from easydict import EasyDict
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from neuro_trigger import utils
from neuro_trigger.configs import get_hyperpar_by_name
from neuro_trigger.lightning.mean_tb_logger import MeanLoggerExp, MeanTBLogger
from neuro_trigger.lightning.pl_module import NeuroTrigger
from neuro_trigger.pytorch.dataset import BelleIIDataManager
from neuro_trigger.pytorch.dataset_filters import IdentityFilter
from neuro_trigger.utils import (ThreadLogFilter,
                                 create_dataset_with_predictions_per_expert,
                                 expert_weights_json, get_loss,
                                 load_json_weights_to_module,
                                 save_csv_dataset_with_predictions,
                                 save_predictions_pickle, snap_source_state)

# Data needs to be mounted to the data folder, e.g. with sshfs:
# sshfs juelg@neurobelle.mpp.mpg.de:/mnt/scratch/data data

train = [
    "data/dqmNeuro/dqmNeuro_mpp34_exp20_400-944/lt100reco/idhist_10170_default/section_correct_fp/neuroresults_random1.gz"
]
val = [
    "data/dqmNeuro/dqmNeuro_mpp34_exp20_400-944/lt100reco/idhist_10170_default/section_correct_fp/neuroresults_random2.gz"
]
test = [
    "data/dqmNeuro/dqmNeuro_mpp34_exp20_400-944/lt100reco/idhist_10170_default/section_correct_fp/neuroresults_random3.gz"
]

DATA_DEBUG = (train, val, test)

train = [
    "data/dqmNeuro/dqmNeuro_mpp34_exp20_400-944/lt100reco/idhist_10170_default/section_correct_fp/neuroresults_random1.gz"
]
val = [
    "data/dqmNeuro/dqmNeuro_mpp34_exp20_400-944/lt100reco/idhist_10170_default/section_correct_fp/neuroresults_random2.gz"
]
test = [
    "data/dqmNeuro/dqmNeuro_mpp34_exp20_400-944/lt100reco/idhist_10170_default/section_correct_fp/neuroresults_random3.gz"
]
DATA_PROD = (train, val, test)


def fit(trainer_module: Tuple[pl.Trainer, NeuroTrigger], logger: logging.Logger):
    """Runs training and validation

    Args:
        trainer_module (Tuple[pl.Trainer, NeuroTrigger]): pytorch lightning trainer and lightning module
        logger (logging.Logger): logger to use for the training
    """
    try:
        # train
        logger.info(f"Expert {trainer_module[1].expert} start training.")
        trainer_module[0].fit(trainer_module[1])
    except ValueError:
        # needed to avoid signal error from pytorch lightning in threads
        logger.info(f"Expert {trainer_module[1].expert} has finished training.")

    # load the best weights for evaluation
    ckpt_path = os.path.join(trainer_module[1].log_path, "ckpts")
    best_ckpt = os.path.join(
        ckpt_path, [i for i in os.listdir(ckpt_path) if i.startswith("epoch")][0]
    )
    trainer_module[1].load_state_dict(torch.load(best_ckpt)["state_dict"])

    # create eval plots
    trainer_module[1].validate(path=trainer_module[1].log_path, mode="val")
    logger.info(f"Expert {trainer_module[1].expert} done creating val plots, finished.")


def create_trainer_pl_module(
    expert_i: int,
    experts: List[int],
    log_folder: str,
    hparams: EasyDict,
    data_mgrs: List[BelleIIDataManager],
    version: int,
    mean_tb_logger: MeanTBLogger,
    fast_dev_run: bool = False,
    overfit_batches: Union[int, float] = 0.0,
    debug: bool = False,
) -> Tuple[pl.Trainer, NeuroTrigger]:
    """Initializes the pytorch lightning module and the pytorch lightning trainer.

    This function should be executed for each expert.

    Args:
        expert_i (int): ith expert
        experts (List[int]): list of experts
        log_folder (str): _description_
        hparams (EasyDict): hyperparameter dict
        data_mgrs: List[BelleIIDataManager]: Data manager objects which load the data. One object for train, validation and test.
        version (int): training version
        mean_tb_logger (MeanTBLogger): tensor board logger object
        fast_dev_run (bool, optional): Whether to let pytorch lightning perform a fast dev run. Only used for debugging or in tests. Defaults to False.
        overfit_batches (Union[int, float], optional): If pytorch lightning should overfit this amount of batches or percentage. See pl.Trainer docu for further information. Only used for debugging or in tests. Defaults to 0.0.
        debug (bool, optional): Whether we run in debug mode. Defaults to False.

    Returns:
        Tuple[pl.Trainer, NeuroTrigger]: pytorch lightning trainer and module
    """

    # this seems unnecassary
    expert = experts[expert_i]
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=30, strict=True, verbose=True, mode="min"
    )
    model_checkpoint = ModelCheckpoint(
        os.path.join(log_folder, f"expert_{expert}", "ckpts"),
        monitor="val_loss",
        save_last=True,
        save_top_k=1,
    )
    callbacks = [early_stop_callback, model_checkpoint]

    pl_module = NeuroTrigger(
        hparams,
        data_mgrs,
        expert=expert,
        log_path=os.path.join(log_folder, f"expert_{expert}"),
    )
    trainer = pl.Trainer(
        logger=[
            TensorBoardLogger(os.path.join(log_folder, f"expert_{expert}"), "tb"),
            CSVLogger(os.path.join(log_folder, f"expert_{expert}"), "csv"),
            MeanLoggerExp(mean_tb_logger.queue, version, mean_tb_logger, expert),
        ],
        # row_log_interval=1,
        # track_grad_norm=2,
        # weights_summary=None,
        # distributed_backend='dp',
        # gradient_clip_val=..,
        callbacks=callbacks,
        max_epochs=hparams["epochs"],
        deterministic=True,
        log_every_n_steps=30,
        profiler="simple" if debug else None,
        fast_dev_run=fast_dev_run,
        overfit_batches=overfit_batches,
        # gpus=[gpu_idx], #[0, 1],
        default_root_dir=os.path.join(log_folder, f"expert_{expert}"),
        # auto_select_gpus=True,
        # enable_pl_optimizer=True,
        # enable_progress_bar=debug,
    )
    return trainer, pl_module


def write_global_journal(base_log: str, config: str, journal_name: str = "log.txt"):
    """Writes timestamp and log path to a global training log

    Args:
        base_log (str): base log folder
        config (str): key of the config to use defined in `configs.py`
        journal_name (str, optional): path to the global log file. Defaults to "log.txt".
    """
    with open(os.path.join(base_log, journal_name), "a") as f:
        f.write(f"{datetime.now()}: {config}\n")


def prepare_vars(
    config: str,
    debug: bool = False,
    solo_expert: bool = False,
    overwrite_hparams: Optional[Dict[str, Any]] = None,
) -> Tuple[EasyDict, str, List[int], int, List[str], logging.Logger]:
    """Creates logging folder, initializes logger per expert and writes the run to a global training log file

    Args:
        config (str): key of the config to use defined in `configs.py`
        debug (bool, optional): Whether debug mode is enable. If True the training will log to a folder in `/tmp`. Defaults to False.
        solo_expert (bool, optional): Whether to only train on a single expert with all data. Defaults to False.
        overwrite_hparams (Optional[Dict[str, Any]]): Overwrite hyperparameters of the current used config.
            Use dotted syntax to access nested parameters. Defaults to None.

    Returns:
        Tuple[EasyDict, str, List[int], int, List[str], logging.Logger]: Hyperparameters dict, folder to log to,
            List of expert numbers, experiment version (used in the log folder), List of expert names e.g. `expert_0`, python logger to log debug messages to
    """
    base_log = "/tmp/nt_pytorch_debug_log" if debug else "log"
    hparams = get_hyperpar_by_name(config, overwrite_hparams)
    if debug:
        hparams["epochs"] = 10
    experts = hparams.experts if not solo_expert else [-1]  # [0, 1, 2, 3, 4]

    experts_str = [f"expert_{i}" for i in experts]
    logger = logging.getLogger()

    # check the latest version
    version = (
        max(
            [
                int(str(i).split("_")[-1])
                for i in (Path(base_log) / config).glob("version_*")
            ],
            default=-1,
        )
        + 1
    )

    log_folder = os.path.join(base_log, config, f"version_{version}")
    if not Path(log_folder).exists():
        Path(log_folder).mkdir(parents=True)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
    )

    # mute other libs debugging output
    logging.getLogger("matplotlib.font_manager").disabled = True
    logging.getLogger("matplotlib.ticker").disabled = True
    logging.getLogger("matplotlib.colorbar").disabled = True
    logging.getLogger("PIL.PngImagePlugin").disabled = True
    logging.getLogger("h5py._conv").disabled = True

    # general logger which logs everything
    fh = logging.FileHandler(os.path.join(log_folder, f"app.log"), mode="w")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s:%(threadName)s:%(levelname)s:%(name)s:%(message)s"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # create file loggers
    for expert in experts:
        fh = logging.FileHandler(
            os.path.join(log_folder, f"expert_{expert}.log"), mode="w"
        )
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s:%(threadName)s:%(levelname)s:%(name)s:%(message)s"
        )
        fh.setFormatter(formatter)
        fh.addFilter(ThreadLogFilter(f"expert_{expert}"))
        logger.addHandler(fh)

    logger.info(f"Using config {config} in version {version}")

    write_global_journal(base_log, log_folder)

    return hparams, log_folder, experts, version, experts_str, logger


def main(
    config: str,
    data: Tuple[List[str], List[str], List[str]],
    debug: bool = False,
    solo_expert: bool = False,
    run_description: Optional[str] = None,
    overwrite_hparams: Optional[Dict[str, Any]] = None,
) -> str:
    """Wirtes commit id and diff, runs training and creates output dataset

    Args:
        config (str): key of the config to use defined in `configs.py`
        data (Tuple[List[str], List[str], List[str]]): datasets as train, validation and test datasets
        debug (bool, optional): _description_. Defaults to False.
        solo_expert (bool, optional): Whether to only train on a single expert with all data. Defaults to False.
        run_description (Optional[str]): Experiment description. Defaults to None.
        overwrite_hparams (Optional[Dict[str, Any]]): Overwrite hyperparameters of the current used config.
            Use dotted syntax to access nested parameters. Defaults to None.

    Returns:
        str: Log folder path
    """
    hparams, log_folder, experts, version, experts_str, logger = prepare_vars(
        config, debug, solo_expert, overwrite_hparams
    )

    # save git commit and git diff in file
    hparams["git_id"] = snap_source_state(log_folder)

    if not debug and run_description is None:
        # force a short experiment description
        hparams["run_description"] = input("Experiment run description: ")
    else:
        hparams["run_description"] = run_description

    with open(os.path.join(log_folder, "summary.json"), "w") as f:
        json.dump(hparams, f, indent=2, sort_keys=True)

    mean_tb_logger = MeanTBLogger(os.path.join(log_folder, "mean_expert"), experts)
    mean_tb_logger.start_thread()

    compare_to = utils.get_compare_to_path(hparams)

    data_mgrs = [
        BelleIIDataManager(data[i], out_dim=hparams.out_size, compare_to=compare_to[i])
        for i in range(3)
    ]

    trainers_modules = [
        create_trainer_pl_module(
            expert_i,
            experts,
            log_folder,
            hparams,
            data_mgrs,
            version,
            mean_tb_logger,
            debug=debug,
        )
        for expert_i in range(len(experts))
    ]

    # config: load_pre_trained_weights with path to json weights in order to train with preinialized weights
    if hparams.get("load_pre_trained_weights"):
        pl_modules = [pl_module for _, pl_module in trainers_modules]
        load_json_weights_to_module(hparams.load_pre_trained_weights, pl_modules)

    if len(experts) == 1:
        fit(trainer_module=trainers_modules[0], logger=logger)
    else:
        threads = []
        for trainer_module, expert in zip(trainers_modules, experts_str):
            t = threading.Thread(target=fit, name=expert, args=[trainer_module, logger])
            t.start()
            threads.append(t)

    # wait for all threads to finish training
    if len(experts) != 1:
        for t in threads:
            t.join()
    mean_tb_logger.stop_thread()
    # create dataset with predictions
    expert_modules = [i[1] for i in trainers_modules]

    logger.info("Creating prediction datasets")
    loss = {
        "train": {"filtered": {}, "unfiltered": {}},
        "val": {"filtered": {}, "unfiltered": {}},
        "test": {"filtered": {}, "unfiltered": {}},
    }

    expert_weights_json(expert_modules, path=log_folder)

    for filtered, mode in itertools.product(
        [True, False] if hparams.get("filter") else [False], ["train", "val", "test"]
    ):
        name_extension = "_filtered" if filtered else ""
        preds = create_dataset_with_predictions_per_expert(
            expert_modules, mode=mode, filter=None if filtered else IdentityFilter()
        )
        save_csv_dataset_with_predictions(
            expert_modules,
            preds,
            path=log_folder,
            mode=mode,
            name_extension=name_extension,
        )
        save_predictions_pickle(
            expert_modules,
            preds,
            path=log_folder,
            mode=mode,
            name_extension=name_extension,
        )
        # loss_overall, std_overall, loss, std
        (
            loss[mode]["filtered" if filtered else "unfiltered"]["loss_overall"],
            loss[mode]["filtered" if filtered else "unfiltered"]["std_overall"],
            loss[mode]["filtered" if filtered else "unfiltered"]["loss"],
            loss[mode]["filtered" if filtered else "unfiltered"]["std"],
        ) = get_loss(expert_modules, preds)

    logger.info("Writing summary")
    with open(os.path.join(log_folder, "summary.json"), "r") as f:
        summary = json.load(f)
    summary["loss"] = loss
    summary["data_path"] = data
    with open(os.path.join(log_folder, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    return log_folder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tool to start the neuro trigger training."
    )
    parser.add_argument(
        "mode", type=str, help="Config mode to use, must be defined in config.py"
    )
    # if not production then logs will go to /tmp and only one expert will be used
    parser.add_argument(
        "-p",
        "--production",
        help="If not given code will run in debug mode",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--solo_expert",
        help="Whether one expert or several experts are used for training. One expert is good for debugging as there is not multi processing",
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--description",
        help="Run description",
        default=None,
    )
    parser.add_argument(
        "-m",
        "--hparams",
        nargs="+",
        help="Overwrite config parameters, e.g. -p expert_2.bach_size=16 act=softsign",
        default=[],
    )
    args = parser.parse_args()
    args.hparams = {i.split("=")[0]: eval(i.split("=")[1]) for i in args.hparams}
    return args


if __name__ == "__main__":
    args = parse_args()
    debug = not args.production
    if debug:
        logging.info("Running in debug mode.")
    main(
        config=args.mode,
        data=DATA_DEBUG if debug else DATA_PROD,
        debug=debug,
        solo_expert=args.solo_expert,
        run_description=args.description,
        overwrite_hparams=args.m__hparams,
    )
