import json
import logging
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from neuro_trigger.lightning.pl_module import NeuroTrigger
from neuro_trigger.pytorch.dataset_filters import Filter

CSV_HEAD = """Experiment      Run     Subrun  Event   Track   nTracks Expert  iNodes  oNodes  SL0-relID       SL0-driftT      SL0-alpha       SL1-relID       SL1-driftT      SL1-alpha       SL2-relID       SL2-driftT      SL2-alpha       SL3-relID       SL3-driftT      SL3-alpha       SL4-relID       SL4-driftT      SL4-alpha       SL5-relID       SL5-driftT      SL5-alpha       SL6-relID       SL6-driftT      SL6-alpha       SL7-relID       SL7-driftT      SL7-alpha       SL8-relID       SL8-driftT      SL8-alpha       RecoZ   RecoTheta       ScaleZ  RawZ    ScaleTheta      RawTheta        NewZ    NewTheta

"""

MODE2IN = {"train": 0, "val": 1, "test": 2}
IN2MODE = ["train", "val", "test"]

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


def get_compare_to_path(hparams: Dict) -> List[Optional[str]]:
    if hparams.compare_to:
        return [
            os.path.join(
                "log",
                hparams.compare_to,
                PREDICTIONS_DATASET_FILENAME.format(i + 1, ""),
            )
            for i in range(3)
        ]
    else:
        return [None, None, None]


def snap_source_state(log_folder: str) -> str:
    """reads out the git commit id of the last commit and writes it a a log file
    clalled "git_id.txt" in the given `log_folger`.

    Args:
        log_folder (str): Folder where to save the file with the commit id.

    Returns:
        str: commit id for further processing
    """
    # get git commit id
    os.system(f'git log --format="%H" -n 1 > {os.path.join(log_folder, "git_id.txt")}')
    # get git diff
    os.system(f'git diff > {os.path.join(log_folder, "git_diff.txt")}')

    with open(os.path.join(log_folder, "git_id.txt"), "r") as f:
        return f.read().split("\n")[0]


def create_dataset_with_predictions_per_expert(
    expert_pl_modules: List[LightningModule],
    mode: str = "val",
    filter: Optional[Filter] = None,
) -> Dict[int, torch.tensor]:
    """Create predictions for a specific dataset using a specific filter for all experts

    Args:
        expert_pl_modules (List[LightningModule]): Expert lightning modules
        mode (str, optional): Dataset mode: "train", "val" and "test" are possible. Defaults to "val".
        filter (Optional[Filter], optional): Filter which should be applied to the dataset. If None than no filter is applied. Defaults to None.

    Returns:
        Dict[int, torch.tensor]: Prediction dataset for each expert which includes idx, y_hat and y
    """
    mode = MODE2IN[mode]
    preds = {}
    for expert in expert_pl_modules:
        preds[expert.expert] = []
        expert.eval()
        with torch.no_grad():

            d = DataLoader(
                expert.get_expert_dataset(split=mode, filter=filter),
                batch_size=10000,
                num_workers=0,
                drop_last=False,
            )
            for i in d:
                x, y, y_hat_old, idx = i
                y_hat = expert(x)
                # Attention: this implicitly converts the index to float32!
                preds[expert.expert].append(
                    torch.cat([idx.unsqueeze(1), y_hat, y, y_hat_old], dim=1)
                )

    for expert in expert_pl_modules:
        # cat and not stack because we have batches
        preds[expert.expert] = torch.cat(preds[expert.expert])

    return preds


def save_csv_dataset_with_predictions(
    expert_pl_modules: List[LightningModule],
    preds: Dict[int, torch.tensor],
    path: str,
    mode: str = "val",
    name_extension: str = "",
):
    """Saves given prediction to a CSV file

    Args:
        expert_pl_modules (List[LightningModule]): Expert lightning modules
        preds (Dict[int, torch.tensor]): prediction dictionary with predictions for each expert
        path (str): path to store the csv file excluding the file's name.
        mode (str, optional): Dataset mode: "train", "val" and "test" are possible. Defaults to "val".
        name_extension (str, optional): Extension that should be added to the file's name. Defaults to "".
    """
    mode = MODE2IN[mode]
    idxs = torch.cat([preds[expert.expert][:, 0] for expert in expert_pl_modules])
    data = torch.cat([preds[expert.expert][:, 1:3] for expert in expert_pl_modules])

    data_arr = expert_pl_modules[0].data_mgrs[mode].get_data_array()
    new_arr = np.zeros((data_arr.shape[0], data_arr.shape[1] + 2))
    new_arr[:, :-2] = data_arr
    for i in range(len(data)):
        new_arr[idxs[i].int().item(), -2:] = data[i]

    fname = f"pred_data_random{mode+1}{name_extension}.csv"

    np.savetxt(
        os.path.join(path, fname),
        new_arr,
        delimiter="",
        fmt="\t".join(
            ["%i" for _ in range(9)] + ["%f" for _ in range(33)] + ["%.16f", "%.16f"]
        ),
    )

    with open(os.path.join(path, fname), "r+") as file:
        content = file.read()
        file.seek(0)
        file.write(CSV_HEAD + content)


def save_predictions_pickle(
    expert_pl_modules: List[LightningModule],
    preds: Dict[int, torch.tensor],
    path: str,
    mode: str = "val",
    name_extension: str = "",
):
    """Saves predictions as pickle files / torch binary format

    _extended_summary_

    Args:
        expert_pl_modules (List[LightningModule]): Expert lightning modules
        preds (Dict[int, torch.tensor]): prediction dictionary with predictions for each expert
        path (str): path to store the pickle's file excluding the file's name.
        mode (str, optional): Dataset mode: "train", "val" and "test" are possible. Defaults to "val".
        name_extension (str, optional): Extension that should be added to the file's name. Defaults to "".
    """
    mode = MODE2IN[mode]
    dataset = torch.cat([preds[expert.expert] for expert in expert_pl_modules])
    dataset = sorted(dataset, key=lambda x: x[0])
    # stack needed as sorted creates python list
    data = torch.stack(dataset)

    with open(
        os.path.join(
            path, PREDICTIONS_DATASET_FILENAME.format(mode + 1, name_extension)
        ),
        "wb",
    ) as file:
        torch.save(data[:, 1:3], file)


def get_loss(
    expert_pl_modules: List[LightningModule], preds: Dict[int, torch.tensor]
) -> Tuple[float, float, Dict[int, float], Dict[int, float]]:
    """Calculates loss and the standard deviation of the z difference for each expert and also
    averaged over all experts

    Args:
        expert_pl_modules (List[LightningModule]): Expert lightning modules
        preds (Dict[int, torch.tensor]): prediction dictionary with predictions for each expert

    Returns:
        Tuple[float, float, float, float]: loss averaged over all experts,
            standard deviation of the z difference averaged over all experts,
            loss per expert (dict), standard deviation of the z difference per expert (dict)
    """
    loss = {}
    z_diff_std = {}
    for expert in expert_pl_modules:
        loss[expert.expert] = expert.crit(
            preds[expert.expert][:, 1:3], preds[expert.expert][:, 3:5]
        ).item()
        z_diff_std[expert.expert] = torch.std(
            preds[expert.expert][:, 1] - preds[expert.expert][:, 3]
        ).item()

    overall = torch.cat([preds[expert.expert] for expert in expert_pl_modules])

    loss_overall = expert.crit(overall[:, 1:3], overall[:, 3:5])
    z_diff_std_overall = torch.std(overall[:, 1] - overall[:, 3])

    return loss_overall.item(), z_diff_std_overall.item(), loss, z_diff_std


def expert_weights_json(expert_pl_modules: List[LightningModule], path: str):
    """Saves weights of the pytorch networks to a json file

    The json file will have the following format (only in case the BaselineModel is used):

    {
        "expert_x": {
            "shapes": {
                "model.net.0.weight": [81, 27],
                "model.net.0.bias": [81],
                "model.net.2.weight": [2, 81],
                "model.net.2.bias": [2]
            },
            "weights": {
                "model.net.0.weight": [[...]],
                "model.net.0.bias": [...],
                "model.net.2.weight": [[...]],
                "model.net.2.bias": [...]
            }
        }
    }
    where "..." refers to numeric data. "shapes" tells what dimensions the data arrays have
    and "weights" gives the actual data for these weight tensors.

    Note that the number jumps from 0 to 2 as there is an activation layer in between which does
    not contain any weights

    Args:
        expert_pl_modules (List[LightningModule]): Expert lightning modules
        path (str): path where the json file should be saved excluding the file name
    """
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


def load_from_checkpoint(
    config: str, version: str = "version_1", experts: Optional[List[str]] = None
) -> List[NeuroTrigger]:
    """Loads the pytorch lightning module initialized with the weights from a given checkpoint

    Args:
        config (str): (hyperparameter's) configuration that was used for the training of the checkpoint that should be loaded
        version (str, optional): Training set version. Have a look in the log/log.txt file to find out the version number quickly.
            Defaults to "version_1".
        experts (Optional[List], optional): String name of the experts that where used during training. None means that experts 0 to 4 are tried to load.
            Defaults to None.

    Returns:
        List[NeuroTrigger]: List of the pytorch lightning modules
    """

    experts = experts or [f"expert_{i}" for i in range(5)]

    expert_paths = [
        os.path.join("log", config, version, expert, "ckpts") for expert in experts
    ]

    checkpoints = []
    for expert in expert_paths:
        checkpoints.append([i for i in os.listdir(expert) if i.startswith("epoch")][0])

    models = []
    for expert, path, checkpoint in zip(experts, expert_paths, checkpoints):
        c_path = os.path.join(path, checkpoint)
        model = NeuroTrigger.load_from_checkpoint(c_path)  # TODO: maybe add data
        model.eval()
        models.append(model)
    return models


def load_from_json(
    json_path: str,
    config: str,
    version: str = "version_1",
    experts: Optional[List] = None,
) -> List[NeuroTrigger]:
    """Loads the pytorch lightning module with initialized weights from a given json file
    (produced with the `expert_weights_json` function or at least with the same format)

    Args:
        json_path (str): path to the json file including the weights
        config (str): (hyperparemter's) config name used for the training
        version (str, optional): Training set version. Have a look in the log/log.txt file to find out the version number quickly.
            Defaults to "version_1".
        experts (Optional[List], optional): String name of the experts that where used during training. None means that experts 0 to 4 are tried to load.
            Defaults to None.

    Returns:
        List[NeuroTrigger]: _description_
    """

    models = load_from_checkpoint(config, version, experts)

    return load_json_weights_to_module(json_path, models)


def load_json_weights_to_module(
    json_path: str, models: List[NeuroTrigger]
) -> List[NeuroTrigger]:
    """Loads weights contained in a given json file (produced by or having the same format as
    the `expert_weights_json` function) into given pytorch lighting modules

    Args:
        json_path (str): path to the json file
        models (List[NeuroTrigger]): list of pytorch lightning expert modules

    Returns:
        List[NeuroTrigger]: _description_
    """
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


def create_figures(path: str, models: List[NeuroTrigger], mode: int = 2):
    """Creates and saves figures using the loaded plots classes in the pytorch lightning's visualize class.

    Args:
        path (str): Path where the figures should be saved
        models (List[NeuroTrigger]): pytorch lightning expert modules
        mode (int, optional): Dataset mode: 1 means train, 2 means validation and 3 means test. Defaults to 2".
    """
    for model in models:
        outputs = []
        model.visualize.folder = os.path.join(path, model.exp_str())
        with torch.no_grad():
            d = DataLoader(
                model.data[mode], batch_size=10000, num_workers=0, drop_last=False
            )
            for i in d:
                x, y, _, _ = i
                y_hat = model(x)
                outputs.append((y, y_hat))

        model.visualize.create_plots(
            torch.cat([i[0] for i in outputs]), torch.cat([i[1] for i in outputs])
        )
