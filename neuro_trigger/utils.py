from ast import Dict
from collections import OrderedDict
import copy
import json
import logging
from lzma import MODE_FAST
import os
from typing import List
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
import numpy as np

import torch    

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


def create_dataset_with_predictions_per_expert(expert_pl_modules: List[LightningModule], mode="val", re_init=False) -> Dict[int, torch.tensor]:
    mode = MODE2IN[mode]
    preds = {}
    for expert in expert_pl_modules:
        preds[expert.expert] = []
        expert.eval()
        with torch.no_grad():
            if re_init:
                da = copy.deepcopy(expert.data[mode])
                # if compare_to no set, it will take the very original one
                da.init_data(filter=None, compare_to=da.compare_to)
                d = DataLoader(da, batch_size=10000, num_workers=0, drop_last=False)
            else:
                d = DataLoader(expert.data[mode], batch_size=10000, num_workers=0, drop_last=False)
            for i in d:
                x, y, y_hat_old, idx = i
                y_hat = expert(x)
                # dataset.append((idx, y_hat))
                preds[expert.expert].append((idx, y_hat, y, expert))

    for expert in expert_pl_modules:
        # cat and not stack because we have batches
        preds[expert.expert] = torch.cat(preds[expert.expert])
    return preds


def save_csv_dataset_with_predictions(expert_pl_modules: List[LightningModule], preds: Dict[int, torch.tensor], path: str, mode="val", name_extension=""):
    mode = MODE2IN[mode]
    # dataset = []
    # for expert in expert_pl_modules:
    #     expert.eval()
    #     with torch.no_grad():
    #         if re_init:
    #             da = copy.deepcopy(expert.data[mode])
    #             # if compare_to no set, it will take the very original one
    #             da.init_data(filter=None, compare_to=da.compare_to)
    #             d = DataLoader(da, batch_size=10000, num_workers=0, drop_last=False)
    #         else:
    #             d = DataLoader(expert.data[mode], batch_size=10000, num_workers=0, drop_last=False)
    #         for i in d:
    #             x, y, y_hat_old, idx = i
    #             y_hat = expert(x)
    #             dataset.append((idx, y_hat))
    # idxs = torch.cat([i[0] for i in dataset])
    # data = torch.cat([i[1] for i in dataset])

    idxs = torch.cat([preds[expert.expert][:,0] for expert in expert_pl_modules])
    data = torch.cat([preds[expert.expert][:,1] for expert in expert_pl_modules])

    data_arr = expert_pl_modules[0].data[mode].get_data_array()
    new_arr = np.zeros((data_arr.shape[0], data_arr.shape[1] + 2))
    new_arr[:,:-2] = data_arr
    for i in range(len(data)):
        new_arr[idxs[i],-2:] = data[i]

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
        loss[expert.expert] = expert.crit(preds[expert.expert][:,1], preds[expert.expert][:,2])
        std[expert.expert] = torch.std(preds[expert.expert][:,1,0] - preds[expert.expert][:,2,0])

    overall = torch.cat([preds[expert.expert] for expert in expert_pl_modules])

    loss_overall = expert.crit(overall[:,1], overall[:,2])
    std_overall = torch.std(overall[:,1,0] - overall[:,2,0])

    return loss_overall, std_overall, loss, std


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

# TODO: add function to load network from checkpoint and from json

