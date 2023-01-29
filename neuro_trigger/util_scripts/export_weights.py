# %%
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

# Utility script that exports weights from trained net checkpoints to a json file
import json
import os
from typing import OrderedDict

import torch


def create_json_for_conf(conf, version=0):
    experts = [f"expert_{i}" for i in range(5)]
    exptert_paths = [
        os.path.join("log", conf, f"version_{version}", expert, "ckpts")
        for expert in experts
    ]

    fn = []
    for expert in exptert_paths:
        fn.append([i for i in os.listdir(expert) if i.startswith("epoch")][0])

    exps = OrderedDict()
    for expert, path, fn in zip(experts, exptert_paths, fn):

        # new_model = pl_module.NeuroTrigger.load_from_checkpoint(checkpoint_path=exptert_paths[0])
        checkpoint = torch.load(os.path.join(path, fn))

        # model = model.BaselineModel()
        # model.load_state_dict(checkpoint["state_dict"])
        desc = {}
        for key, value in checkpoint["state_dict"].items():
            desc[key] = value.shape
            checkpoint["state_dict"][key] = value.tolist()
        # print(checkpoint["state_dict"])
        exps[expert] = {"shapes": desc, "weights": checkpoint["state_dict"]}

        # with open(os.path.join(path, "wights.json"), "w") as f:
        #     json.dump(exps[expert], f)

    # with open(os.path.join("log", conf, "version_0", "wights.json"), "w") as f:
    with open(os.path.join("json_weights", "wights_new.json"), "w") as f:
        json.dump(exps, f)


conf = "baseline_v2"
create_json_for_conf(conf, version=3)


# %%
