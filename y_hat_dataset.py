# %%
import os
from pl_module import NeuroTrigger
import torch
from torch.utils.data import DataLoader

conf = "baseline_v1"
experts = [f"expert_{i}" for i in range(5)]
expert_paths = [os.path.join("log", conf, "version_1", expert, "ckpts") for expert in experts]

fn = []
for expert in expert_paths:
    fn.append([i for i in os.listdir(expert) if i.startswith("epoch")][0])

#%%

dataset = []
for expert, path, fn in zip(experts, expert_paths, fn):
    c_path = os.path.join(path, fn)
    model = NeuroTrigger.load_from_checkpoint(c_path, data=(
    '/mnt/scratch/juelg/neuro-trigger-v2/data/dqmNeuro/dqmNeuro_mpp34_exp20_400-944/lt100reco/idhist_10170_default/section_fp/neuroresults_random1.gz',
    '/mnt/scratch/juelg/neuro-trigger-v2/data/dqmNeuro/dqmNeuro_mpp34_exp20_400-944/lt100reco/idhist_10170_default/section_fp/neuroresults_random2.gz',
    '/mnt/scratch/juelg/neuro-trigger-v2/data/dqmNeuro/dqmNeuro_mpp34_exp20_400-944/lt100reco/idhist_10170_default/section_fp/neuroresults_random3.gz'))
    print(path)
    model.eval()
    with torch.no_grad():
        d = DataLoader(model.data[2], batch_size=10000, num_workers=0, drop_last=False)
        for i in d:
            x, y, y_hat_old, idx = i
            y_hat = model(x)
            dataset.append((idx, y_hat))
# %%
idxs = torch.cat([i[0] for i in dataset])
data = torch.cat([i[1] for i in dataset])

# %%
import numpy as np
data_arr = model.data[2].shared_data['/mnt/scratch/juelg/neuro-trigger-v2/data/dqmNeuro/dqmNeuro_mpp34_exp20_400-944/lt100reco/idhist_10170_default/section_fp/neuroresults_random3.gz']
new_arr = np.zeros((data_arr.shape[0], data_arr.shape[1] + 2))
new_arr[:,:-2] = data_arr
# %%
for i in range(len(data)):
    # new_arr[i,-2:] = data[idxs[i]]
    new_arr[idxs[i],-2:] = data[i]
# %%

np.savetxt("baselinev1_data_output.csv", new_arr, delimiter="", fmt="\t".join(['%i'for _ in range(9)] + ["%f" for _ in range(33)] + ["%.16f", "%.16f"]))
# %%
