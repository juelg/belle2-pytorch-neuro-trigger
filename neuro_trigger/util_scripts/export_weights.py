# %%
# Utility script that exports weights from trained net checkpoints to a json file
from typing import OrderedDict
import os
import torch
import json


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
