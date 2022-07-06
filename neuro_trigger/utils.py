from collections import OrderedDict
import copy
import json
import logging
from lzma import MODE_FAST
import os
from typing import Iterable, List, Dict, Optional
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
import numpy as np

import torch

from neuro_trigger.lightning.pl_module import NeuroTrigger


CSV_HEAD = """Experiment      Run     Subrun  Event   Track   nTracks Expert  iNodes  oNodes  SL0-relID       SL0-driftT      SL0-alpha       SL1-relID       SL1-driftT      SL1-alpha       SL2-relID       SL2-driftT      SL2-alpha       SL3-relID       SL3-driftT      SL3-alpha       SL4-relID       SL4-driftT      SL4-alpha       SL5-relID       SL5-driftT      SL5-alpha       SL6-relID       SL6-driftT      SL6-alpha       SL7-relID       SL7-driftT      SL7-alpha       SL8-relID       SL8-driftT      SL8-alpha       RecoZ   RecoTheta       ScaleZ  RawZ    ScaleTheta      RawTheta        NewZ    NewTheta

"""

MODE2IN = {"train": 0, "val": 1, "test": 2}

PREDICTIONS_DATASET_FILENAME = "prediction_random{}{}.pt"

class ThreadLogFilter(logging.Filter):
    """
    This filter only show log entries for specified thread name
    """

    def __init__(self, thread_name: str, *args, **kwargs):
        logging.Filter.__init__(self, *args, **kwargs)
        self.thread_name = thread_name

    def filter(self, record):
        return record.threadName == self.thread_name


def snap_source_state(log_folder: str):
    # get git commit id
    os.system(
        f'git log --format="%H" -n 1 > {os.path.join(log_folder, "git_id.txt")}')
    # get git diff
    os.system(f'git diff > {os.path.join(log_folder, "git_diff.txt")}')

    with open(os.path.join(log_folder, "git_id.txt"), "r") as f:
        return f.read()


def create_dataset_with_predictions_per_expert(expert_pl_modules: List[LightningModule], mode="val", filter=None) -> Dict[int, torch.tensor]:
    # None means original filters
    mode = MODE2IN[mode]
    preds = {}
    for expert in expert_pl_modules:
        preds[expert.expert] = []
        expert.eval()
        with torch.no_grad():

            d = DataLoader(expert.get_expert_dataset(split=mode, filter=filter), batch_size=10000, num_workers=0, drop_last=False)
            for i in d:
                x, y, y_hat_old, idx = i
                y_hat = expert(x)
                preds[expert.expert].append(torch.cat([idx.unsqueeze(1), y_hat, y], dim=1))

    for expert in expert_pl_modules:
        # cat and not stack because we have batches -> bullshit: stack adds to the same dimension, cat creates a new one (like tuple) -> bullshit bullshit
        # first one was correct
        preds[expert.expert] = torch.cat(preds[expert.expert])
    return preds


def save_csv_dataset_with_predictions(expert_pl_modules: List[LightningModule], preds: Dict[int, torch.tensor], path: str, mode="val", name_extension=""):
    mode = MODE2IN[mode]
    idxs = torch.cat([preds[expert.expert][:,0] for expert in expert_pl_modules])
    data = torch.cat([preds[expert.expert][:,1:3] for expert in expert_pl_modules])

    data_arr = expert_pl_modules[0].data_mgrs[mode].open()
    new_arr = np.zeros((data_arr.shape[0], data_arr.shape[1] + 2))
    new_arr[:,:-2] = data_arr
    for i in range(len(data)):
        new_arr[idxs[i].int().item(),-2:] = data[i]

    fname = f"pred_data_random{mode+1}{name_extension}.csv"

    np.savetxt(os.path.join(path, fname), new_arr, delimiter="", fmt="\t".join(['%i'for _ in range(9)] + ["%f" for _ in range(33)] + ["%.16f", "%.16f"]))

    with open(os.path.join(path, fname), 'r+') as file:
        content = file.read()
        file.seek(0)
        file.write(CSV_HEAD + content)

def save_predictions_pickle(expert_pl_modules: List[LightningModule], preds: Dict[int, torch.tensor], path: str, mode="val", name_extension=""):
    mode = MODE2IN[mode]
    dataset = torch.cat([preds[expert.expert] for expert in expert_pl_modules])
    dataset = sorted(dataset, key=lambda x: x[0])
    # stack needed as sorted creates python list
    data = torch.stack(dataset)

    with open(os.path.join(path, PREDICTIONS_DATASET_FILENAME.format(mode+1, name_extension)), 'wb') as file:
        torch.save(data, file)

def get_loss(expert_pl_modules: List[LightningModule], preds: Dict[int, torch.tensor]):
    loss = {}
    std = {}
    for expert in expert_pl_modules:
        loss[expert.expert] = expert.crit(preds[expert.expert][:,1], preds[expert.expert][:,2]).item()
        std[expert.expert] = torch.std(preds[expert.expert][:,1] - preds[expert.expert][:,2]).item()

    overall = torch.cat([preds[expert.expert] for expert in expert_pl_modules])

    loss_overall = expert.crit(overall[:,1], overall[:,2])
    std_overall = torch.std(overall[:,1] - overall[:,2])

    return loss_overall.item(), std_overall.item(), loss, std


def expert_weights_json(expert_pl_modules: List[LightningModule], path: str):
    exps = OrderedDict()
    for expert_module in expert_pl_modules:
        desc = {}
        weights = {}
        for key, value in expert_module.state_dict().items():
            desc[key] = value.shape
            weights[key] = value.tolist()

        exps[expert_module.exp_str] = {"shapes": desc, "weights": weights}

    with open(os.path.join(path, "weights.json"), "w") as f:
        json.dump(exps, f)


def load_from_checkpoint(config: str, version="version_1", experts: Optional[List] = None):
    from neuro_trigger.lightning.pl_module import NeuroTrigger    
    experts = experts or [f"expert_{i}" for i in range(5)]

    expert_paths = [os.path.join("log", config, version, expert, "ckpts") for expert in experts]

    checkpoints = []
    for expert in expert_paths:
        checkpoints.append([i for i in os.listdir(expert) if i.startswith("epoch")][0])


    models = []
    for expert, path, checkpoint in zip(experts, expert_paths, checkpoints):
        c_path = os.path.join(path, checkpoint)
        model = NeuroTrigger.load_from_checkpoint(c_path) # TODO: maybe add data
        model.eval()
        models.append(model)
    return models


def load_from_json(json_path: str, config: str, version="version_1", experts: Optional[List] = None):
    models  = load_from_checkpoint(config, version, experts)

    return load_json_weights_to_module(json_path, models)

def load_json_weights_to_module(json_path: str, models: NeuroTrigger):
    for model in models:
        expert = model.expert
        # load json dict
        with open(json_path, "r") as f:
            wb = json.load(f)
        exp_wb = wb[expert]
        wb = exp_wb["weights"]
        wb = {key: torch.tensor(value) for key, value in wb.items()}
        model.load_state_dict(wb)
    return models


def create_figures(path, models, mode=2):
    for model in models:
        outputs = []
        model.visualize.folder = os.path.join(path, model.exp_str())
        with torch.no_grad():
            d = DataLoader(model.data[mode], batch_size=10000, num_workers=0, drop_last=False)
            for i in d:
                x, y, y_hat_old, idx = i
                y_hat = model(x)
                outputs.append((y, y_hat))

        model.visualize.create_plots(
                torch.cat([i[0] for i in outputs]), torch.cat([i[1] for i in outputs]))