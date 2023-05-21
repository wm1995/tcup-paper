import json

import numpy as np


def write_dataset(name, data, info):
    output = {
        "data": data,
        "info": info,
    }
    with open(f"data/{name}.json", "w") as f:
        json.dump(output, f)


def load_dataset(filename):
    with open(filename, "r") as f:
        dataset = json.load(f)

    data = {key: np.array(val) for key, val in dataset["data"].items()}
    params = dataset.get("info")
    return data, params
