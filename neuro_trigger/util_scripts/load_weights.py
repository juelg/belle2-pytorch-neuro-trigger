import gzip
import json
from typing import Dict, Tuple

import numpy as np


def load_dataset(path: str) -> Tuple[np.array]:
    """Loads a reco dataset from the given path.

    The dataset must be in Felix's fromat.

    Args:
        path (str): path to the dataset

    Returns:
        Tuple[np.array]: train data with shape [N, 27], targets with shape [N, 2], expert numbers
            with shape [N, 1]
    """
    with gzip.open(path, "rt") as f:
        dt = np.loadtxt(path, skiprows=2)
    data = {
        "x": dt[:, 9:36],
        "y": dt[:, 36:38],
        "expert": dt[:, 6],
        "y_hat_old": dt[:, -4:-1:2],
    }
    return data["x"], data["y"], data["expert"]


def forward_numpy(expert: int, weights: Dict, x: np.array) -> np.array:
    """Calculates a forward pass of the network with tanh/2 activation funcitons.

    Args:
        expert (int): Number of the expert that the weights and x belong to
        weights (Dict): Weights to use as python dictionary with the values [w0, b0, w1, b1] for the
            weights and biases of the respective layers
        x (np.array): Network input of shape [27] (not batched)

    Returns:
        np.array: Network output of shape [2]
    """
    # "@" means matrix multiplication in numpy and "+" is component wise
    x = weights[expert]["w0"] @ x + weights[expert]["b0"]
    x = np.tanh(x / 2)
    x = weights[expert]["w1"] @ x + weights[expert]["b1"]
    x = np.tanh(x / 2)
    return x


def load_json_weights(path: str) -> Dict:
    """load the network weights from a json file

    The input dictionary must have the following format:
    ["expert_{0, 1, 2, 3, 4}"]["weights"]["model.net.{0, 2}.{weight, bias}"]
    Note that the layer numbers are 0 and 2 (instead of 1) because in PyTorch the activation function
    is also considered a layer.

    Args:
        path (str): Path to the json file

    Returns:
        Dict: Weights as python dictionary. The dictionary is given as follows: [<expert_nr>][<layer>]
            where layer is out of {"w0", "b0", "w1", "b0"} where wX means the weights of between layer
            X and X+1 and bX means the bias between layer X and X+1
    """
    # load the json file
    with open(path, "r") as f:
        dic = json.load(f)

    # convert to more confinient dictionary structure and to numpy arrays
    weights = {}
    for expert in range(5):
        weights[expert] = {}
        weights[expert]["w0"] = np.array(
            dic[f"expert_{expert}"]["weights"]["model.net.0.weight"]
        )
        weights[expert]["b0"] = np.array(
            dic[f"expert_{expert}"]["weights"]["model.net.0.bias"]
        )
        weights[expert]["w1"] = np.array(
            dic[f"expert_{expert}"]["weights"]["model.net.2.weight"]
        )
        weights[expert]["b1"] = np.array(
            dic[f"expert_{expert}"]["weights"]["model.net.2.bias"]
        )
    return weights


if __name__ == "__main__":
    # TODO: paths need to be adapted
    weights = load_json_weights("log/baseline_v2/version_3/weights.json")
    x, y, expert = load_dataset(
        "data/dqmNeuro/dqmNeuro_mpp34_exp20_400-944/lt100reco/idhist_10170_default/section_correct_fp/neuroresults_random1.gz"
    )
    expert_to_use = 0
    # get the data belonging to the respective expert
    x_expert0 = x[expert == expert_to_use]
    y = forward_numpy(expert_to_use, weights, x_expert0[0])
    print(y)
