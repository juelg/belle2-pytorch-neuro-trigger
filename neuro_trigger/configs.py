from typing import Any, Dict, Optional
from easydict import EasyDict
import copy
from flatten_dict import flatten, unflatten

configs = {
    "filter_combine_max2_dupl_events": {
        "extends": "baseline_v2",
        "compare_to": "baseline_v2/version_4",
        "description": "Baseline_v2 (tanh/2) with max 2 events and duplicative events filter combinded",
        "filter": "dataset_filters.ConCatFilter([dataset_filters.Max2EventsFilter(), dataset_filters.DuplicateEventsFilter()])",
    },
    "filter_max2_events": {
        "extends": "baseline_v2",
        "compare_to": "baseline_v2/version_4",
        "description": "Baseline_v2 (tanh/2) with max 2 events filter",
        "filter": "dataset_filters.Max2EventsFilter()",
    },
    "filter_dupl_events": {
        "extends": "baseline_v2",
        "compare_to": "baseline_v2/version_4",
        "description": "Baseline_v2 (tanh/2) with duplicative events filter",
        "filter": "dataset_filters.DuplicateEventsFilter()",
    },
    "reweight_uniform": {
        "extends": "baseline_v2",
        "compare_to": "baseline_v2/version_4",
        "description": "Baseline_v2 (tanh/2) with uniform distribution dataloader",
        "workers": 20,
        "dist": {
            "n_buckets": 11,
            "uniform": {
                "lower": -1,
                "upper": 1,
            },
        },
    },
    "reweight_normal_11_04_inf_bounds": {
        "extends": "baseline_v2",
        "compare_to": "baseline_v2/version_4",
        "description": "Baseline_v2 (tanh/2) with normal distribution dataloader with 11 buckets and std=0.4, with infinite bounds -> prob. will go up at the edges",
        "workers": 20,
        "dist": {
            "n_buckets": 11,
            "inf_bounds": True,
            "norm": {
                "mean": 0,
                "std": 0.4,
            },
        },
    },
    "reweight_normal_11_04_no_inf_bounds": {
        "extends": "baseline_v2",
        # "compare_to": "baseline_v2/version_4",
        "description": "Baseline_v2 (tanh/2) with normal distribution dataloader with 11 buckets and std=0.4, no infinite bounds -> reweighting",
        # "workers": 20,
        "dist": {
            "n_buckets": 11,
            "inf_bounds": False,
            "norm": {
                "mean": 0,
                "std": 0.4,
            },
        },
    },
    "baseline_v4_softsign": {
        # improving: no, worse than tanh
        "extends": "baseline_v3",
        # should be similar to "baseline_v2/version_4"
        "compare_to": "baseline_v4/version_0",
        "description": "Baseline arch with v3 (softsign) and batchnorm before the layers",
        "model": "BaselineModelBN",
    },
    # idea: add more data, maybe train on random split 1 and 2 and eval on 3
    "baseline_v4_tanh_per_expert": {
        # improving:
        "extends": "baseline_v4_tanh",
        # should be similar to "baseline_v2/version_4"
        "compare_to": "baseline_v4_tanh/version_0",
        "description": "Baseline arch with v1 (tanh) and batchnorm before the layers, different batch sizes and epochs",
        "model": "BaselineModelBN",
        "expert_0": {
            "batch_size": 2048,
            "epochs": 1000,
        },
        "expert_1": {
            "batch_size": 2048,
            "epochs": 1000,
        },
        "expert_2": {
            "batch_size": 16,
            "epochs": 4000,
        },
        "expert_3": {
            "batch_size": 128,
            "epochs": 2000,
        },
        "expert_4": {
            "batch_size": 16,
            "epochs": 4000,
        },
    },
    "baseline_v4_tanh": {
        # improving: yes but only little and not for all experts
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
    "baseline_v2_norm_dist": {
        # imporving: no, it is even worse
        "extends": "baseline_v1",
        # should be similar to "baseline_v2/version_4"
        "compare_to": "baseline_v2/version_4",
        "description": "Baseline_v2 (tanh/2) with new dits dataloader",
        "act": "tanh",
        "dist": "norm",
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
    "baseline_v3_per_expert": {
        # improving: yes upon baseline v2
        "extends": "baseline_v3",
        # should be similar to "baseline_v2/version_4"
        "compare_to": "baseline_v3/version_1",
        "description": "Baseline arch with softsign activation function and per expert batchsizes and epochs",
        "expert_0": {
            "batch_size": 1024,
            "epochs": 1000,
        },
        "expert_1": {
            "batch_size": 1024,
            "epochs": 1000,
        },
        "expert_2": {
            "batch_size": 16,
            "epochs": 4000,
        },
        "expert_3": {
            "batch_size": 128,
            "epochs": 2000,
        },
        "expert_4": {
            "batch_size": 16,
            "epochs": 4000,
        },
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
    # idea: add more data, maybe train on random split 1 and 2 and eval on 3
    "baseline_v2_tanh_per_expert": {
        # improving: yes all but expert 1
        "extends": "baseline_v2",
        # should be similar to "baseline_v2/version_4"
        "compare_to": "baseline_v2/version_4",
        "description": "Baseline arch with v2 (tanh/2), different batch sizes and epochs",
        "expert_0": {
            "batch_size": 2048,
            "epochs": 1000,
        },
        "expert_1": {
            "batch_size": 2048,
            "epochs": 1000,
        },
        "expert_2": {
            "batch_size": 16,
            "epochs": 4000,
        },
        "expert_3": {
            "batch_size": 128,
            "epochs": 2000,
        },
        "expert_4": {
            "batch_size": 16,
            "epochs": 4000,
        },
    },
    "baseline_v2_batchnorm": {
        "extends": "baseline_v1",
        "description": "Baseline with BN arch with tanh/2",
        "act": "tanh/2",
        "model": "BaselineModelBN",
    },
    "baseline_v2_batchnorm_tanh": {
        "extends": "baseline_v1",
        "description": "Baseline arch with BN with tanh",
        "act": "tanh",
        "model": "BaselineModelBN",
    },
    "baseline_v2_batchnorm_relu": {
        "extends": "baseline_v1",
        "description": "Baseline with BN arch with relu act",
        "act": "relu",
        "model": "BaselineModelBN",
    },
    "baseline_v2_tanh": {
        "extends": "baseline_v1",
        "description": "Baseline arch with tanh",
        "act": "tanh",
    },
    "baseline_v2_softsign": {
        "extends": "baseline_v1",
        "description": "Baseline arch with softsign act",
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
        # path for pre trained weights that one wants to start with:
        "load_pre_trained_weights": None,
    },
}


def extend(use_dict: Dict, flattened_configs: Dict) -> Dict:
    """Extend a config with its parent config recursively.

    Args:
        use_dict (Dict): Config dict to extend.
        flattened_configs (Dict): Parent config dict which should be used to extend the config.

    Returns:
        Dict: Extended config dict.
    """
    if use_dict.get("extends"):
        name = use_dict.get("extends")
        extended = extend(flattened_configs.get(name, {}), flattened_configs)
        extended.update(use_dict)
        return extended
    else:
        return copy.deepcopy(use_dict)


def get_hyperpar_by_name(
    name: str, overwrite_hparams: Optional[Dict[str, Any]] = None
) -> EasyDict:
    """Get hyperparameter config by name.

    Gets the config with the given name, extends it with its parent config
    recursively and overwrites the given overwrite_hparams.

    Args:
        name (str): Config name to use.
        overwrite_hparams (Optional[Dict[str, Any]], optional): Flat parameter dict to overwrite
            the config with. Defaults to None.

    Returns:
        EasyDict: Extended and overwritten config dict accessible via dot notation.
    """
    overwrite_hparams = overwrite_hparams or {}
    flattened_configs = {
        key: flatten(value, reducer="dot") for key, value in configs.items()
    }

    hparams = flattened_configs[name]
    hparams["config"] = name
    extened_hparams = extend(hparams, flattened_configs)
    extened_hparams.update(overwrite_hparams)
    return EasyDict(unflatten(extened_hparams, splitter="dot"))
