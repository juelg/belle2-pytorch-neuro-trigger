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

import json
import os

import torch
from torch.utils.data import DataLoader

from neuro_trigger.lightning.pl_module import NeuroTrigger

# j_path = "json_weights/tobias1_10.json"
j_path = "json_weights/felix10_10.json"
conf = "baseline_v1"
experts = [f"expert_{i}" for i in range(5)]
expert_paths = [
    os.path.join("log", conf, "version_1", expert, "ckpts") for expert in experts
]

fn = []
for expert in expert_paths:
    fn.append([i for i in os.listdir(expert) if i.startswith("epoch")][0])

#%%

models = []
for expert, path, fn in zip(experts, expert_paths, fn):
    c_path = os.path.join(path, fn)
    model = NeuroTrigger.load_from_checkpoint(
        c_path,
        data=(
            "/mnt/scratch/juelg/neuro-trigger-v2/data/dqmNeuro/dqmNeuro_mpp34_exp20_400-944/lt100reco/idhist_10170_default/section_fp/neuroresults_random1.gz",
            "/mnt/scratch/juelg/neuro-trigger-v2/data/dqmNeuro/dqmNeuro_mpp34_exp20_400-944/lt100reco/idhist_10170_default/section_fp/neuroresults_random2.gz",
            "/mnt/scratch/juelg/neuro-trigger-v2/data/dqmNeuro/dqmNeuro_mpp34_exp20_400-944/lt100reco/idhist_10170_default/section_fp/neuroresults_random3.gz",
        ),
    )
    print(path)
    model.eval()
    models.append(model)
#%%

for expert, model in zip(experts, models):
    # load json dict
    with open(j_path, "r") as f:
        wb = json.load(f)
    exp_wb = wb[expert]
    wb = exp_wb["weights"]
    wb = {key: torch.tensor(value) for key, value in wb.items()}

    # load json weights
    # for key, value in wb.items():
    #     model.state_dict()[key] = value
    model.load_state_dict(wb)
#%%
# make eval pass
for expert, model in zip(experts, models):
    outputs = []
    model.visualize.folder = f"json_weights/felix/{expert}"
    with torch.no_grad():
        d = DataLoader(model.data[2], batch_size=10000, num_workers=0, drop_last=False)
        for i in d:
            x, y, y_hat_old, idx = i
            y_hat = model(x)
            outputs.append((y, y_hat))

    model.visualize.create_plots(
        torch.cat([i[0] for i in outputs]), torch.cat([i[1] for i in outputs])
    )


#     model.eval()
#     with torch.no_grad():
#         d = DataLoader(model.data[2], batch_size=10000, num_workers=0, drop_last=False)
#         for i in d:
#             x, y, y_hat_old, idx = i
#             y_hat = model(x)
#             dataset.append((idx, y_hat))
# # %%
# idxs = torch.cat([i[0] for i in dataset])
# data = torch.cat([i[1] for i in dataset])
# %%
