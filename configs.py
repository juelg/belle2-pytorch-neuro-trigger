

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
        "learning_rate": 1e-3,
        "batch_size": 32,
        "weight_decay": 1e-6,
        "in_size": 27,
        "out_size": 1,
        "workers": 5,
        "noise": None,
        "epochs": 1000
    }


}