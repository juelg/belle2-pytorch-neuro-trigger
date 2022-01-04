from easydict import EasyDict
import copy

configs = {
    "baseline_v1": {
        "version": 0.1,
        "learning_rate": 1e-3,
        "batch_size": 2048,
        "weight_decay": 1e-6,
        "in_size": 27,
        "out_size": 2,
        "workers": 6,
        "noise": None,
        "epochs": 50
    },
    "only_z": {
        "version": 0.1,
        "extends": "base",
        "in_size": 27,
        "out_size": 1,
    },
    "base": {
        "version": 0.1,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "weight_decay": 1e-6,
        "in_size": 27,
        "out_size": 2,
        "workers": 5,
        "epochs": 50
    }


}

def extend(use_dict):
    if use_dict.get("extends"):
        name = use_dict.get("extends")
        extended = extend(configs.get(name, {}))
        extended.update(use_dict)
        return extended
    else:
        return copy.deepcopy(use_dict)


def get_hyperpar_by_name(name):
    hparams = configs[name]
    hparams["config"] = name
    return EasyDict(extend(hparams))


# TODO: 
# - [x] add extending of other config
# - [x] add training for only z component
# - [x] add easy dict
# - [ ] add categories in config
# - [ ] safe git diff and commit id into log
# - [ ] add change log
# - [ ] per expert hparams -> which overwrite the default value
# - [ ] issue with singal shutdown in thread