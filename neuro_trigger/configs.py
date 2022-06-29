from easydict import EasyDict
import copy

configs = {
    "baseline_v4_softsign": {
        # improving: 
        "extends": "baseline_v3",
        # should be similar to "baseline_v2/version_4"
        "compare_to": "baseline_v4/version_0",
        "description": "Baseline arch with v3 (softsign) and batchnorm before the layers",
        "model": "BaselineModelBN",
    },
    "baseline_v4_tanh": {
        # improving: 
        "extends": "baseline_v1",
        # should be similar to "baseline_v2/version_4"
        "compare_to": "baseline_v4/version_0",
        "description": "Baseline arch with v1 (tanh) and batchnorm before the layers",
        "model": "BaselineModelBN",
    },
    "baseline_v4": {
        # improving: yes definatly upon baseline v2
        "extends": "baseline_v2",
        # should be similar to "baseline_v2/version_4"
        "compare_to": "baseline_v2/version_4",
        "description": "Baseline arch with v2 (tanh/2) and batchnorm before the layers",
        "model": "BaselineModelBN",
    },
    "baseline_v1_comp_v2": {
        # imporving: yes -> tanh better than tanh/2
        "extends": "baseline_v1",
        # should be similar to "baseline_v2/version_4"
        "compare_to": "baseline_v2/version_4",
        "description": "Baseline arch compared to baseline_v2 (tanh/2)",
        "act": "tanh",
    },
    "baseline_v2_sgd": {
        # imporving: no definatly worse
        "extends": "baseline_v2",
        # should be similar to "baseline_v2/version_4"
        "compare_to": "baseline_v2/version_4",
        "description": "Baseline arch with SGD optimizer",
        "optim": "SGD",
    },
    "baseline_v3.1": {
        # improving: yes upon baseline v2
        "extends": "baseline_v3",
        "batch_size": 512,
        # should be similar to "baseline_v2/version_4"
        "compare_to": "baseline_v2/version_4",
        "description": "Baseline arch with softsign activation function, comp b_v2/v4, bs 512, del dup events",
    },
    "baseline_v3": {
        # improving: yes upon baseline v2
        "extends": "baseline_v2",
        # should be similar to "baseline_v2/version_4"
        "compare_to": "baseline_v2/version_4",
        "description": "Baseline arch with softsign activation function",
        "act": "softsign",
    },
    "baseline_v2": {
        "extends": "baseline_v1",
        "description": "Baseline arch with tanh/2",
        "act": "tanh/2",
    },

    "simple_model_v1": {
        "extends": "baseline_v1",
        "act": "relu",
        "model": "SimpleModel",
    },
    "simple_model_v1_only_z": {
        "extends": "simple_model_v1",
        # only train on the z target
        "out_size": 1,
        "description": "like baseline_v1 but only trains on the z target",
    },

    "baseline_v1": {
        "extends": "base",
        "learning_rate": 1e-3,
        "weight_decay": 1e-6,
        "in_size": 27,
        "out_size": 2,
        "epochs": 1000,
        "description": "Baseline arch with tanh",
        "act": "tanh",
    },
    "baseline_v1_only_z": {
        "extends": "baseline_v1",
        # only train on the z target
        "out_size": 1,
        "description": "like baseline_v1 but only trains on the z target",
    },
    "base": {
        "version": 0.2,
        "description": "Base version to inherit from",
        "learning_rate": 1e-3,
        "batch_size": 2048,
        "weight_decay": 1e-6,
        "in_size": 27,
        # out_size must be in {1, 2} (if out_size=1 then we only train on z)
        "out_size": 2,
        "workers": 5,
        "epochs": 10,
        "model": "BaselineModel",
        "loss": "MSELoss",
        "optim": "Adam",
        "act": "relu",
        "experts": [0, 1, 2, 3, 4],
        # should be similar to "baseline_v2/version_3"
        "compare_to": None,
        # example for expert specific parameters
        # "expert_0": {
        #     "batch_size": 2048,
        # },
        # "expert_1": {
        #     "batch_size": 2048,
        # },
        # "expert_2": {
        #     "batch_size": 16,
        # },
        # "expert_3": {
        #     "batch_size": 128,
        # },
        # "expert_4": {
        #     "batch_size": 32,
        # },
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